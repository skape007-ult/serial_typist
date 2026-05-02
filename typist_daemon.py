"""
Serial Typist daemon — continuous keystroke capture with session boundary
detection.

Runs as a long-lived background process (typically auto-started by launchd).
Monitors keystrokes via pynput. When typing activity is detected after an
idle period, starts a new session. When activity stops for the configured
threshold, ends the session and triggers the analysis pipeline.

Optionally spawns capture_p1.py as a subprocess for video capture when the
recording flag is enabled.

State files:
  ~/serial_typist/daemon.state         JSON, current state (read by CLI)
  ~/serial_typist/commands.jsonl       Pending CLI commands (written by CLI)
  ~/serial_typist/daemon.log           Rolling log
  ~/serial_typist/sessions/            Captured sessions
  ~/serial_typist/recording_active     Empty file existence = recording flag on
  ~/.config/serial_typist/config.yml   Configuration (auto-created with defaults)

Run manually for debugging:
    python typist_daemon.py
"""

import csv
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock

# Defer pynput import so this module can be imported for testing without it
try:
    from pynput import keyboard
except ImportError:
    print("ERROR: pynput not installed. Run: pip install pynput")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None  # config falls back to defaults if yaml not installed

# ==========================================
# PATHS
# ==========================================
HOME = Path.home()
DATA_DIR = HOME / 'serial_typist'
SESSIONS_DIR = DATA_DIR / 'sessions'
STATE_FILE = DATA_DIR / 'daemon.state'
COMMAND_FILE = DATA_DIR / 'commands.jsonl'
LOG_FILE = DATA_DIR / 'daemon.log'
RECORDING_FLAG = DATA_DIR / 'recording_active'
CONFIG_DIR = HOME / '.config' / 'serial_typist'
CONFIG_FILE = CONFIG_DIR / 'config.yml'

DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# DEFAULT CONFIG
# ==========================================
DEFAULT_CONFIG = {
    'idle_threshold_seconds': 300,  # 5 min — session ends after this much idle
    'min_letter_keystrokes': 200,  # session must have this many letters to qualify
    'min_letter_fraction': 0.60,  # at least this fraction must be letter keys
    'tick_interval_seconds': 1.0,  # main loop polling interval
    'flush_interval_seconds': 30,  # buffer-to-disk flush rate during session
    'capture_script_path': str(Path(__file__).parent / 'capture_p1.py'),
    'analysis_script_path': str(Path(__file__).parent / 'analyze.py'),
    'fusion_script_path': str(Path(__file__).parent / 'fuse_v2.py'),
    'feedback_script_path': str(Path(__file__).parent / 'feedback.py'),
    'python_path': sys.executable,
}


def load_config():
    """Load config from YAML, creating it with defaults if missing."""
    if not CONFIG_FILE.exists():
        if yaml:
            with open(CONFIG_FILE, 'w') as f:
                yaml.safe_dump(DEFAULT_CONFIG, f, default_flow_style=False)
        return dict(DEFAULT_CONFIG)
    if not yaml:
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_FILE) as f:
        loaded = yaml.safe_load(f) or {}
    return {**DEFAULT_CONFIG, **loaded}


CONFIG = load_config()

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger('typist')


# ==========================================
# STATE
# ==========================================
class DaemonState:
    """Daemon state. All access serialized via lock since pynput callback
    runs on its own thread and the main loop reads/writes too."""

    def __init__(self):
        self._lock = Lock()
        self.state = 'IDLE'  # IDLE | ACTIVE | ENDING
        self.session_dir = None  # Path of current session
        self.session_started_at = None  # monotonic time
        self.last_keystroke_at = None  # monotonic time
        self.session_press_count = 0
        self.session_letter_count = 0
        self.session_buffer = []  # list of {'t_mono', 't_wall', 'event', 'key'}
        self.last_flush_at = None  # monotonic time of last buffer flush
        self.video_subprocess = None  # Popen handle if recording
        self.force_start_pending = False
        self.force_stop_pending = False
        self.daemon_started_at = time.monotonic()

    def snapshot(self):
        """Return a dict snapshot for the state file (read by CLI)."""
        with self._lock:
            uptime_s = time.monotonic() - self.daemon_started_at
            session_age = (time.monotonic() - self.session_started_at
                           if self.session_started_at else 0)
            idle_s = (time.monotonic() - self.last_keystroke_at
                      if self.last_keystroke_at else None)
            return {
                'state': self.state,
                'uptime_seconds': round(uptime_s, 1),
                'session_id': self.session_dir.name if self.session_dir else None,
                'session_age_seconds': round(session_age, 1) if session_age else 0,
                'session_press_count': self.session_press_count,
                'session_letter_count': self.session_letter_count,
                'idle_seconds': round(idle_s, 1) if idle_s is not None else None,
                'recording_active': RECORDING_FLAG.exists(),
                'video_subprocess_running': (
                        self.video_subprocess is not None
                        and self.video_subprocess.poll() is None
                ),
                'pid': os.getpid(),
                'updated_at': datetime.now().isoformat(),
            }


STATE = DaemonState()


# ==========================================
# KEYSTROKE LISTENER
# ==========================================
def normalize_key(key):
    """Convert a pynput Key/KeyCode to a consistent string representation."""
    if hasattr(key, 'char') and key.char is not None:
        return key.char
    name = str(key).replace('Key.', '')
    return name


def is_letter_key(key_str):
    return len(key_str) == 1 and key_str.isalpha()


def on_press(key):
    _record_keystroke(key, 'press')


def on_release(key):
    _record_keystroke(key, 'release')


def _record_keystroke(key, event):
    """Append keystroke to the buffer. Triggered from pynput's listener thread."""
    t_mono = time.monotonic()
    t_wall = time.time()
    key_str = normalize_key(key)

    with STATE._lock:
        STATE.last_keystroke_at = t_mono
        if STATE.state == 'ACTIVE':
            STATE.session_buffer.append({
                't_mono': t_mono, 't_wall': t_wall,
                'event': event, 'key': key_str,
            })
            if event == 'press':
                STATE.session_press_count += 1
                if is_letter_key(key_str):
                    STATE.session_letter_count += 1


# ==========================================
# COMMAND FILE
# ==========================================
def consume_commands():
    """Read pending commands and clear the file. Returns list of command dicts."""
    if not COMMAND_FILE.exists():
        return []
    try:
        with open(COMMAND_FILE) as f:
            lines = f.readlines()
        # Truncate atomically
        COMMAND_FILE.write_text('')
    except Exception as e:
        log.error(f"Failed to read command file: {e}")
        return []

    commands = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            commands.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning(f"Malformed command, skipping: {line!r}")
    return commands


def handle_command(cmd):
    """Apply a CLI command to daemon state."""
    action = cmd.get('action')
    log.info(f"Command received: {action}")

    if action == 'start':
        with STATE._lock:
            STATE.force_start_pending = True
    elif action == 'stop':
        with STATE._lock:
            STATE.force_stop_pending = True
    elif action == 'record_on':
        already = RECORDING_FLAG.exists()
        RECORDING_FLAG.touch()
        log.info("Recording flag enabled.")
        # If a session is already in progress and no subprocess is running,
        # spawn one now. Loses the typing that already happened from being
        # captured on video, but starts the rest from this moment forward —
        # which is what the user expects from a "turn recording on" command.
        if not already:
            with STATE._lock:
                in_active_session = (STATE.state == 'ACTIVE'
                                     and STATE.session_dir is not None
                                     and (STATE.video_subprocess is None
                                          or STATE.video_subprocess.poll() is not None))
                session_dir = STATE.session_dir
            if in_active_session:
                try:
                    STATE.video_subprocess = subprocess.Popen(
                        [CONFIG['python_path'], CONFIG['capture_script_path'],
                         '--output-dir', str(session_dir),
                         '--no-keystroke-capture'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    log.info(f"Mid-session video capture spawned (PID "
                             f"{STATE.video_subprocess.pid}). "
                             f"Note: pre-flag typing won't be on video.")
                except Exception as e:
                    log.error(f"Failed to spawn mid-session video capture: {e}")
                    STATE.video_subprocess = None
    elif action == 'record_off':
        if RECORDING_FLAG.exists():
            RECORDING_FLAG.unlink()
        log.info("Recording flag disabled.")
    else:
        log.warning(f"Unknown command: {action!r}")


# ==========================================
# SESSION LIFECYCLE
# ==========================================
def start_session(forced=False):
    """Transition IDLE → ACTIVE. Creates session directory and (optionally)
    spawns video capture subprocess."""
    with STATE._lock:
        if STATE.state != 'IDLE':
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = SESSIONS_DIR / ts
        session_dir.mkdir(exist_ok=True)

        STATE.state = 'ACTIVE'
        STATE.session_dir = session_dir
        STATE.session_started_at = time.monotonic()
        STATE.session_press_count = 0
        STATE.session_letter_count = 0
        STATE.session_buffer = []
        STATE.last_flush_at = time.monotonic()
        STATE.force_start_pending = False

    log.info(f"SESSION START: {session_dir.name} {'(forced)' if forced else ''}")

    # Spawn video capture if recording flag on
    if RECORDING_FLAG.exists():
        try:
            STATE.video_subprocess = subprocess.Popen(
                [CONFIG['python_path'], CONFIG['capture_script_path'],
                 '--output-dir', str(session_dir),
                 '--no-keystroke-capture'],  # daemon handles keystrokes
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info(f"Video capture subprocess spawned (PID {STATE.video_subprocess.pid})")
        except Exception as e:
            log.error(f"Failed to spawn video capture: {e}")
            STATE.video_subprocess = None


def flush_buffer_to_disk():
    """Append buffered keystrokes to keystrokes.csv. Called periodically and
    at session end."""
    with STATE._lock:
        if STATE.state != 'ACTIVE' or not STATE.session_buffer:
            return
        if not STATE.session_dir:
            return
        buf = STATE.session_buffer
        STATE.session_buffer = []
        STATE.last_flush_at = time.monotonic()
        session_dir = STATE.session_dir

    csv_path = session_dir / 'keystrokes.csv'
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t_mono', 't_wall', 'event', 'key'])
        if write_header:
            w.writeheader()
        for row in buf:
            w.writerow(row)


def end_session(forced=False):
    """Transition ACTIVE → IDLE, flushing data, stopping video, and triggering
    analysis if the session passes filters."""
    with STATE._lock:
        if STATE.state != 'ACTIVE':
            return
        STATE.state = 'ENDING'
        session_dir = STATE.session_dir
        press_count = STATE.session_press_count
        letter_count = STATE.session_letter_count
        video_proc = STATE.video_subprocess
        STATE.force_stop_pending = False

    log.info(f"SESSION END: {session_dir.name} "
             f"({press_count} presses, {letter_count} letters) "
             f"{'(forced)' if forced else ''}")

    # Stop video subprocess
    if video_proc is not None and video_proc.poll() is None:
        try:
            video_proc.send_signal(signal.SIGTERM)
            video_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("Video subprocess didn't exit cleanly, killing.")
            video_proc.kill()
        except Exception as e:
            log.error(f"Error stopping video subprocess: {e}")

    # Final flush
    flush_buffer_to_disk()

    # Write metadata
    duration = (time.monotonic() - STATE.session_started_at
                if STATE.session_started_at else 0)
    metadata = {
        'session_id': session_dir.name,
        'start_time_iso': datetime.now().isoformat(),
        'duration_wall_seconds': round(duration, 1),
        'press_count': press_count,
        'letter_press_count': letter_count,
        'capture_method': 'daemon',
        'video_recorded': video_proc is not None,
        'forced_end': forced,
    }
    with open(session_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Filter check
    letter_fraction = letter_count / max(press_count, 1)
    qualifies = (letter_count >= CONFIG['min_letter_keystrokes']
                 and letter_fraction >= CONFIG['min_letter_fraction'])

    if qualifies:
        log.info(f"Session qualifies for analysis (letters={letter_count}, "
                 f"fraction={letter_fraction:.2f}). Triggering pipeline.")
        trigger_analysis_pipeline(session_dir, has_video=video_proc is not None)
    else:
        log.info(f"Session below threshold (letters={letter_count}, "
                 f"fraction={letter_fraction:.2f}). Archiving without analysis.")
        (session_dir / 'DISCARDED.txt').write_text(
            f"Session below filter thresholds.\n"
            f"Letter keystrokes: {letter_count} (min: {CONFIG['min_letter_keystrokes']})\n"
            f"Letter fraction: {letter_fraction:.2f} (min: {CONFIG['min_letter_fraction']})\n"
        )

    # Reset state
    with STATE._lock:
        STATE.state = 'IDLE'
        STATE.session_dir = None
        STATE.session_started_at = None
        STATE.video_subprocess = None


def adapt_daemon_session_for_pipeline(session_dir):
    """Convert daemon-captured session data into the format the analysis
    pipeline expects.

    The daemon's keystrokes.csv uses absolute time.monotonic() values; the
    pipeline expects t_mono relative to video start. Daemon's metadata.json
    is missing the `transform` field that downstream tools want.

    This function backs up the daemon-format files and rewrites them in
    the pipeline-compatible format. Idempotent — won't re-backup if rerun.
    """
    video_meta_path = session_dir / 'video_metadata.json'
    if not video_meta_path.exists():
        log.warning(f"No video_metadata.json in {session_dir.name}; "
                    f"can't adapt for biomechanical pipeline.")
        return False

    daemon_meta_path = session_dir / 'metadata.json'
    keystrokes_path = session_dir / 'keystrokes.csv'

    daemon_meta_backup = session_dir / 'metadata.daemon.json'
    keystrokes_backup = session_dir / 'keystrokes.daemon.csv'

    if not daemon_meta_backup.exists():
        daemon_meta_backup.write_text(daemon_meta_path.read_text())
    if not keystrokes_backup.exists():
        keystrokes_backup.write_text(keystrokes_path.read_text())

    daemon_meta = json.loads(daemon_meta_backup.read_text())
    video_meta = json.loads(video_meta_path.read_text())

    # Merged metadata: daemon's session-level fields + video subprocess's
    # transform/fps/source fields. Mirrors the original capture_p1.py format
    # so landmark_extractor_p2 / fuse_v2 / analyze treat it as a normal session.
    merged = {
        "session_id": daemon_meta["session_id"],
        "start_time_iso": video_meta.get("start_time_iso", daemon_meta.get("start_time_iso")),
        "end_time_iso": video_meta.get("end_time_iso"),
        "duration_wall_seconds": daemon_meta.get("duration_wall_seconds"),
        "t_zero_wall": video_meta.get("t_zero_wall"),
        "stop_reason": video_meta.get("stop_reason", "daemon_stop"),
        "press_count": daemon_meta.get("press_count"),
        "release_count": None,
        "frames_written": video_meta.get("frames_written"),
        "video_source": video_meta.get("video_source"),
        "target_fps": video_meta.get("target_fps", 30),
        "pre_roll_seconds": video_meta.get("pre_roll_seconds", 0),
        "post_roll_seconds": video_meta.get("post_roll_seconds", 0),
        "files": {
            "video": "session.mp4",
            "keystrokes": "keystrokes.csv",
            "frame_timestamps": "frame_timestamps.csv",
        },
        "transform": video_meta["transform"],
        "notes": "Adapted from daemon capture: t_mono normalized against "
                 "video_metadata.t_zero_wall.",
    }
    daemon_meta_path.write_text(json.dumps(merged, indent=2))

    # Re-zero keystroke t_mono using wall-clock difference from video t_zero.
    t_zero_wall = video_meta["t_zero_wall"]
    rows_in = list(csv.DictReader(open(keystrokes_backup)))
    with open(keystrokes_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_mono', 't_wall', 'event', 'key'])
        for r in rows_in:
            t_wall = float(r['t_wall'])
            new_t_mono = t_wall - t_zero_wall
            w.writerow([f"{new_t_mono:.6f}", r['t_wall'], r['event'], r['key']])
    log.info(f"Adapted session {session_dir.name} for biomechanical pipeline "
             f"({len(rows_in)} keystrokes re-zeroed).")
    return True


def trigger_analysis_pipeline(session_dir, has_video):
    """Run the appropriate pipeline on the session, async.

    With video → adapt formats, run landmark extraction → fuse_v2 → analyze.
    Without video → synthesize keystroke-only events_v2.csv, run analyze.

    Runs in a daemon thread so the daemon main loop stays responsive.
    Total wall-time for the video path is ~60-90 seconds; the daemon
    keeps accepting commands during that time.
    """

    def run_step(label, cmd, timeout):
        log.info(f"[{session_dir.name}] {label} starting...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout)
            if result.returncode != 0:
                log.error(f"[{session_dir.name}] {label} FAILED: "
                          f"{result.stderr.strip()[-500:]}")
                return False
            log.info(f"[{session_dir.name}] {label} done.")
            return True
        except subprocess.TimeoutExpired:
            log.error(f"[{session_dir.name}] {label} timed out after {timeout}s.")
            return False
        except Exception as e:
            log.error(f"[{session_dir.name}] {label} error: {e}")
            return False

    def run_pipeline():
        try:
            analysis_succeeded = False

            if has_video and (session_dir / 'session.mp4').exists():
                # Step 1: data-format adapter
                if not adapt_daemon_session_for_pipeline(session_dir):
                    log.warning(f"Adapter failed for {session_dir.name}; "
                                f"falling back to keystroke-only analysis.")
                    create_keystroke_only_events_csv(session_dir)
                    analysis_succeeded = run_step(
                        'analyze',
                        [CONFIG['python_path'], CONFIG['analysis_script_path'],
                         str(session_dir)],
                        timeout=120)
                else:
                    # Step 2: landmark extraction. Throughput is ~50fps regardless
                    # of session length. 900s (15 min) covers up to ~45,000 frames
                    # which is ~25 minutes of real-time video — well past anything
                    # you'd realistically capture in one session.
                    if not run_step('landmark_extractor_p2',
                                    [CONFIG['python_path'],
                                     str(Path(CONFIG['fusion_script_path']).parent
                                         / 'landmark_extractor_p2.py'),
                                     str(session_dir)],
                                    timeout=900):
                        return

                    # Step 3: fusion
                    if not run_step('fuse_v2',
                                    [CONFIG['python_path'],
                                     CONFIG['fusion_script_path'],
                                     str(session_dir)],
                                    timeout=120):
                        return

                    # Step 4: analyze (now with real fusion data)
                    analysis_succeeded = run_step(
                        'analyze',
                        [CONFIG['python_path'], CONFIG['analysis_script_path'],
                         str(session_dir)],
                        timeout=120)
                    if analysis_succeeded:
                        log.info(f"Full biomechanical pipeline complete for "
                                 f"{session_dir.name}.")
            else:
                # Keystroke-only path
                create_keystroke_only_events_csv(session_dir)
                analysis_succeeded = run_step(
                    'analyze',
                    [CONFIG['python_path'], CONFIG['analysis_script_path'],
                     str(session_dir)],
                    timeout=120)
                if analysis_succeeded:
                    log.info(f"Keystroke-only analysis complete for "
                             f"{session_dir.name}.")

            # Step 5: feedback note + email (gated on analyze succeeding so we
            # don't try to read a missing analysis.json). Feedback failure is
            # non-fatal — log and move on, the analysis artifacts already exist.
            if analysis_succeeded:
                feedback_path = Path(CONFIG['feedback_script_path'])
                if feedback_path.exists():
                    run_step(
                        'feedback',
                        [CONFIG['python_path'], str(feedback_path),
                         str(session_dir)],
                        timeout=60)
                else:
                    log.warning(f"feedback.py not found at {feedback_path}; "
                                f"skipping coaching note generation.")
        except Exception as e:
            log.error(f"Pipeline error for {session_dir.name}: {e}")

    # Fire-and-forget thread so the daemon stays responsive
    import threading
    threading.Thread(target=run_pipeline, daemon=True).start()


def create_keystroke_only_events_csv(session_dir):
    """For keystroke-only sessions (no video), create a minimal events_v2.csv
    so analyze.py's bigram and timing analysis work. Per-finger fields are
    empty since fusion can't run without video."""
    keystrokes_path = session_dir / 'keystrokes.csv'
    events_path = session_dir / 'events_v2.csv'
    if events_path.exists():
        return  # fusion ran; don't overwrite

    rows = []
    with open(keystrokes_path) as f:
        reader = csv.DictReader(f)
        idx = 0
        for r in reader:
            if r['event'] != 'press':
                continue
            rows.append({
                'event_idx': idx,
                't_mono': r['t_mono'],
                'key': r['key'],
                'frame_index': '',
                'hand': '',
                'finger': '',
                'method': 'keystroke_only',
                'matches_canonical': '',
                'p1_finger': '',
                'p1_tip_x': '',
                'p1_tip_y': '',
            })
            idx += 1

    with open(events_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'event_idx', 't_mono', 'key', 'frame_index',
            'hand', 'finger', 'method', 'matches_canonical',
            'p1_finger', 'p1_tip_x', 'p1_tip_y',
        ])
        w.writeheader()
        w.writerows(rows)


# ==========================================
# MAIN LOOP
# ==========================================
def main_loop():
    """Daemon main loop: poll commands, handle state transitions, periodic flush."""
    log.info(f"Daemon started (PID {os.getpid()}). Watching for keystrokes.")
    log.info(f"Sessions will be written to {SESSIONS_DIR}")
    log.info(f"Recording flag: {'ON' if RECORDING_FLAG.exists() else 'OFF'}")

    # Start keystroke listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    def write_state_file():
        """Write the current state snapshot to disk. Cheap (<500 bytes), safe
        to call frequently. Called immediately after any state transition so
        the wrapper / typist CLI sees fresh state without waiting for a
        periodic tick."""
        try:
            STATE_FILE.write_text(json.dumps(STATE.snapshot(), indent=2))
        except Exception as e:
            log.error(f"Failed to write state file: {e}")

    # Write initial state immediately so the wrapper sees fresh IDLE before
    # any commands flow. Avoids the wrapper trusting a stale state file from
    # a previous daemon's lifetime.
    write_state_file()

    last_state_write = time.monotonic()

    try:
        while True:
            # Process pending CLI commands. Track whether anything happened
            # so we can write state right after, not on the periodic schedule.
            commands_processed = False
            for cmd in consume_commands():
                handle_command(cmd)
                commands_processed = True

            # State machine tick
            with STATE._lock:
                state = STATE.state
                last_ks = STATE.last_keystroke_at
                force_start = STATE.force_start_pending
                force_stop = STATE.force_stop_pending
                started_at = STATE.session_started_at
                last_flush = STATE.last_flush_at

            now = time.monotonic()
            state_changed = False

            # IDLE → ACTIVE: ONLY on explicit `start` command.
            # No more auto-detection from keystroke activity — the user controls
            # when sessions begin. Keystrokes typed in IDLE state are silently
            # discarded (the listener is still running so the daemon stays
            # responsive, but nothing is buffered).
            if state == 'IDLE':
                if force_start:
                    start_session(forced=True)
                    state_changed = True

            # ACTIVE → ENDING: ONLY on explicit `stop` command.
            # No more idle-timeout auto-end — sessions run until the user says
            # stop. Buffer is still flushed periodically so a daemon crash
            # doesn't lose more than ~30s of in-progress data.
            elif state == 'ACTIVE':
                if force_stop:
                    end_session(forced=True)
                    state_changed = True
                else:
                    # Periodic flush
                    if (last_flush is None
                            or (now - last_flush) >= CONFIG['flush_interval_seconds']):
                        flush_buffer_to_disk()

            # Write state file: immediately after any command or state change,
            # OR every 2 seconds as a periodic update for live press counts.
            if state_changed or commands_processed:
                write_state_file()
                last_state_write = now
            elif now - last_state_write >= 2.0:
                write_state_file()
                last_state_write = now

            time.sleep(CONFIG['tick_interval_seconds'])

    except KeyboardInterrupt:
        log.info("Daemon interrupted, ending any active session.")
        with STATE._lock:
            active = STATE.state == 'ACTIVE'
        if active:
            end_session(forced=True)
    finally:
        listener.stop()
        log.info("Daemon shutting down.")


if __name__ == '__main__':
    main_loop()