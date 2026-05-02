"""
Unified capture pipeline.

Two threads running concurrently:
  1. Video thread: continuously reads from IP Webcam stream into a rolling buffer.
  2. Keyboard thread: pynput listener logging keystrokes with monotonic timestamps.

The first keystroke flips recording 'hot': buffered frames are flushed to disk,
then every subsequent frame is written live until the session ends.

Session ends on either:
  - INACTIVITY_TIMEOUT seconds with no keypress, or
  - ESC keypress, or
  - SIGTERM (when run as a subprocess by the daemon).

After end, records a few seconds of trailing frames (post-roll) so the last
keystroke has finger-lift data in the video.

Daemon-mode flags (used when typist_daemon spawns this as a subprocess):
  --output-dir <path>       Write session files into <path> instead of
                            sessions/<timestamp>/. The daemon already chose
                            the directory.
  --no-keystroke-capture    Skip pynput listener — daemon owns keystroke
                            logging. Video starts recording immediately
                            (no waiting for first keystroke), and the script
                            stops on SIGTERM rather than ESC/inactivity.
"""

import argparse
import cv2
import os
import time
import csv
import json
import signal
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from pynput import keyboard
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


# ==========================================
# CLI ARGS (daemon-spawned mode)
# ==========================================
_arg_parser = argparse.ArgumentParser(
    description="Video + keystroke capture pipeline.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_arg_parser.add_argument('--output-dir', type=Path, default=None,
                         help="Write session files into this directory instead "
                              "of creating sessions/<timestamp>/. Used by daemon.")
_arg_parser.add_argument('--no-keystroke-capture', action='store_true',
                         help="Skip pynput keystroke logging. Video records "
                              "immediately and stops on SIGTERM. Used by daemon.")
ARGS = _arg_parser.parse_args()
DAEMON_MODE = ARGS.no_keystroke_capture or ARGS.output_dir is not None

# ==========================================
# CONFIGURATION
# ==========================================
IP_WEBCAM_URL = os.getenv('IP_WEBCAM_URL')
if not IP_WEBCAM_URL:
    raise RuntimeError("IP_WEBCAM_URL is not set. Add it to your .env file.")
CROP_Y_START, CROP_Y_END = 180, 1000  # widened bottom edge to full frame so wrists stay in frame
CROP_X_START, CROP_X_END = 700, 1340
ROTATION = cv2.ROTATE_90_CLOCKWISE  # or None to disable
SESSIONS_DIR = Path("sessions")
PRE_ROLL_SECONDS = 2.0       # video buffer duration before first keystroke
POST_ROLL_SECONDS = 2.0      # keep recording after final keystroke
INACTIVITY_TIMEOUT = 5.0     # end session after this many seconds of no keypresses
TARGET_FPS = 30              # what we expect from the stream; used for VideoWriter

# ==========================================
# SESSION SETUP
# ==========================================
session_start_wall = datetime.now()
if ARGS.output_dir is not None:
    # Daemon-spawned: use the directory the daemon already chose.
    session_dir = ARGS.output_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    session_id = session_dir.name
else:
    SESSIONS_DIR.mkdir(exist_ok=True)
    session_id = session_start_wall.strftime("%Y%m%d_%H%M%S")
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(exist_ok=True)

# In daemon mode, the daemon owns metadata.json and keystrokes.csv. We write
# our own video-only metadata under a different name to avoid clobbering.
keys_csv_path = session_dir / "keystrokes.csv"
video_path = session_dir / "session.mp4"
metadata_path = (session_dir / "video_metadata.json"
                 if DAEMON_MODE else session_dir / "metadata.json")

T_ZERO_MONO = time.monotonic()  # shared clock reference

# a monotonic clock is a stric stopwatch that never goes backward and cannot be changed by the OS.

# ==========================================
# SHARED STATE (thread-safe via lock)
# ==========================================
state_lock = threading.Lock()
state = {
    "recording_hot": False,       # has the first keystroke fired?
    "stop_requested": False,      # has a termination signal been received?
    "last_keystroke_mono": None,  # timestamp of most recent keypress
    "first_keystroke_mono": None, # timestamp of the one that flipped us hot
    "stop_reason": None,          # why we stopped (esc / inactivity / error)
    "frames_written": 0,
    "press_count": 0,
    "release_count": 0,
}

# Frame buffer: deque of (t_mono, frame) tuples. maxlen auto-drops old frames.
frame_buffer = deque(maxlen=int(PRE_ROLL_SECONDS * TARGET_FPS * 1.5))  # 50% headroom
# double-ended queue that only holds 90 pictures and automatically discards the older frames

def transform_frame(frame):
    """Apply rig-specific crop and rotation. Called on every frame before storage."""
    frame = frame[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]
    if ROTATION is not None:
        frame = cv2.rotate(frame, ROTATION)
    return frame

# ==========================================
# VIDEO CAPTURE THREAD
# ==========================================
def video_thread():
    """
    Reads frames continuously from the stream. Before recording goes hot, frames
    accumulate in frame_buffer (rolling). After going hot, frames are written
    straight to the VideoWriter.
    """
    print(f"[VIDEO] Connecting to {IP_WEBCAM_URL}...")
    cap = cv2.VideoCapture(IP_WEBCAM_URL)

    if not cap.isOpened():
        print("[VIDEO] ERROR: Could not open stream. Aborting.")
        with state_lock:
            state["stop_requested"] = True
            state["stop_reason"] = "video_connect_failed"
        return

    # Probe actual frame dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("[VIDEO] ERROR: Could not read first frame. Aborting.")
        cap.release()
        with state_lock:
            state["stop_requested"] = True
            state["stop_reason"] = "video_read_failed"
        return

    raw_h, raw_w = test_frame.shape[:2]
    test_frame_transformed = transform_frame(test_frame)
    h, w = test_frame_transformed.shape[:2]
    print(f"[VIDEO] Connected. Raw: {raw_w}x{raw_h} → Transformed: {w}x{h}. Filling pre-roll buffer...")

    writer = None
    frame_timestamps = []  # logged to CSV-like separate file for precise alignment

    try:
        while True:
            with state_lock:
                if state["stop_requested"]:
                    # Check post-roll
                    if state["last_keystroke_mono"] is not None:
                        if (time.monotonic() - T_ZERO_MONO) - state["last_keystroke_mono"] > POST_ROLL_SECONDS:
                            break
                    else:
                        break
                hot = state["recording_hot"]

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            t_mono = time.monotonic() - T_ZERO_MONO
            frame = transform_frame(frame)  # <-- crop + rotate before anything else

            if not hot:
                frame_buffer.append((t_mono, frame))
            else:
                # Recording is live
                if writer is None:
                    # First hot frame — initialize writer and flush buffer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(video_path), fourcc, TARGET_FPS, (w, h))

                    # Flush buffered pre-roll frames
                    # Flush buffered pre-roll frames
                    buffered = list(frame_buffer)
                    print(f"[VIDEO] Going HOT. Flushing {len(buffered)} pre-roll frames.")
                    for buf_t, buf_frame in buffered:
                        writer.write(buf_frame)
                        frame_timestamps.append(buf_t)
                    with state_lock:
                        state["frames_written"] += len(buffered)
                    frame_buffer.clear()

                writer.write(frame)
                frame_timestamps.append(t_mono)
                with state_lock:
                    state["frames_written"] += 1

    finally:
        if writer is not None:
            writer.release()
        cap.release()

        # Write per-frame timestamps to a sidecar file
        # (VideoWriter assumes fixed FPS but actual capture jitters — these are truth)
        timestamps_path = session_dir / "frame_timestamps.csv"
        with open(timestamps_path, 'w', newline='') as f:
            w_csv = csv.writer(f)
            w_csv.writerow(['frame_index', 't_mono'])
            for i, t in enumerate(frame_timestamps):
                w_csv.writerow([i, f"{t:.6f}"])

        print(f"[VIDEO] Thread ending. Wrote {len(frame_timestamps)} frames total.")

# ==========================================
# INACTIVITY MONITOR THREAD
# ==========================================
def inactivity_thread():
    """Polls for last-keystroke-time and triggers stop after INACTIVITY_TIMEOUT."""
    while True:
        time.sleep(0.5)
        with state_lock:
            if state["stop_requested"]:
                return
            last = state["last_keystroke_mono"]
            hot = state["recording_hot"]

        if hot and last is not None:
            now = time.monotonic() - T_ZERO_MONO
            if now - last > INACTIVITY_TIMEOUT:
                print(f"[MONITOR] Inactivity timeout reached ({INACTIVITY_TIMEOUT}s). Stopping.")
                with state_lock:
                    state["stop_requested"] = True
                    state["stop_reason"] = "inactivity"
                return

# ==========================================
# KEYSTROKE LOGGING (skipped in daemon mode — daemon owns keystrokes)
# ==========================================
key_file_lock = threading.Lock()
last_flush_mono = 0.0
if DAEMON_MODE:
    key_file = None
    key_writer = None
else:
    key_file = open(keys_csv_path, 'w', newline='')
    key_writer = csv.writer(key_file)
    key_writer.writerow(['t_mono', 't_wall', 'event', 'key'])

def serialize_key(key):
    try:
        return key.char
    except AttributeError:
        return str(key).replace('Key.', '')

def write_key_row(t_mono, event, key_str):
    global last_flush_mono
    if key_writer is None:
        return  # daemon mode — no local keystroke file
    t_wall = time.time()
    with key_file_lock:
        key_writer.writerow([f"{t_mono:.6f}", f"{t_wall:.6f}", event, key_str])
        if t_mono - last_flush_mono > 2.0:
            key_file.flush()
            last_flush_mono = t_mono

def on_press(key):
    t_mono = time.monotonic() - T_ZERO_MONO
    key_str = serialize_key(key)

    with state_lock:
        if not state["recording_hot"]:
            state["recording_hot"] = True
            state["first_keystroke_mono"] = t_mono
            print(f"[KEYBOARD] First keystroke '{key_str}' at t={t_mono:.3f}s — recording HOT.")
        state["last_keystroke_mono"] = t_mono
        state["press_count"] += 1

    write_key_row(t_mono, 'press', key_str)

def on_release(key):
    t_mono = time.monotonic() - T_ZERO_MONO
    key_str = serialize_key(key)

    with state_lock:
        state["release_count"] += 1

    write_key_row(t_mono, 'release', key_str)

    if key == keyboard.Key.esc:
        print("[KEYBOARD] ESC pressed. Stopping.")
        with state_lock:
            state["stop_requested"] = True
            state["stop_reason"] = "esc"
        return False  # stops the pynput listener

# ==========================================
# MAIN
# ==========================================
print("=" * 60)
print(f"SESSION: {session_id}")
print(f"Directory: {session_dir}")
print(f"Video source: {IP_WEBCAM_URL}")
if DAEMON_MODE:
    print("DAEMON MODE — keystroke logging disabled, immediate hot, "
          "stops on SIGTERM.")
else:
    print(f"Inactivity timeout: {INACTIVITY_TIMEOUT}s | Post-roll: {POST_ROLL_SECONDS}s")
print("=" * 60)
if DAEMON_MODE:
    # Skip the pynput listener entirely. Recording goes hot immediately so
    # frames flow to disk without waiting for a first keystroke (which the
    # daemon already received).
    with state_lock:
        state["recording_hot"] = True
        state["first_keystroke_mono"] = 0.0

    def _handle_sigterm(signum, _frame):
        print(f"[MAIN] Received signal {signum}, stopping.")
        with state_lock:
            state["stop_requested"] = True
            state["stop_reason"] = "sigterm"
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
else:
    print("\nStart typing whenever ready. Press ESC to end the session manually.")
    print("Recording will begin automatically on your first keystroke.\n")

# Launch video thread (always)
video_t = threading.Thread(target=video_thread, daemon=True)
video_t.start()
# Small delay so the video stream has a moment to start buffering before keystrokes register
time.sleep(0.5)

# Inactivity monitor + keyboard listener: only when NOT in daemon mode
monitor_t = None
listener = None
if not DAEMON_MODE:
    monitor_t = threading.Thread(target=inactivity_thread, daemon=True)
    monitor_t.start()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# Main thread waits for stop signal
try:
    while True:
        time.sleep(0.5)
        with state_lock:
            if state["stop_requested"]:
                break
except KeyboardInterrupt:
    print("\n[MAIN] KeyboardInterrupt — forcing stop.")
    with state_lock:
        state["stop_requested"] = True
        state["stop_reason"] = "keyboard_interrupt"

# Graceful shutdown
if listener is not None:
    listener.stop()
print("[MAIN] Waiting for video thread to finish post-roll...")
video_t.join(timeout=POST_ROLL_SECONDS + 3.0)
if monitor_t is not None:
    monitor_t.join(timeout=1.0)

# Close keystroke CSV (only if we opened it)
if key_file is not None:
    with key_file_lock:
        key_file.flush()
        key_file.close()

# Write session metadata
session_end_wall = datetime.now()
with state_lock:
    final_state = dict(state)

metadata = {
    "session_id": session_id,
    "start_time_iso": session_start_wall.isoformat(),
    "end_time_iso": session_end_wall.isoformat(),
    "duration_wall_seconds": (session_end_wall - session_start_wall).total_seconds(),
    "t_zero_wall": T_ZERO_MONO + (time.time() - time.monotonic()),  # approx wall time of t=0
    "stop_reason": final_state["stop_reason"],
    "first_keystroke_mono": final_state["first_keystroke_mono"],
    "last_keystroke_mono": final_state["last_keystroke_mono"],
    "press_count": final_state["press_count"],
    "release_count": final_state["release_count"],
    "frames_written": final_state["frames_written"],
    "video_source": IP_WEBCAM_URL,
    "target_fps": TARGET_FPS,
    "pre_roll_seconds": PRE_ROLL_SECONDS,
    "post_roll_seconds": POST_ROLL_SECONDS,
    "inactivity_timeout_seconds": INACTIVITY_TIMEOUT,
    "files": {
        "video": "session.mp4",
        "keystrokes": "keystrokes.csv",
        "frame_timestamps": "frame_timestamps.csv",
    },
    "transform": {
        "crop_y_start": CROP_Y_START,
        "crop_y_end": CROP_Y_END,
        "crop_x_start": CROP_X_START,
        "crop_x_end": CROP_X_END,
        "rotation": "ROTATE_90_CLOCKWISE" if ROTATION == cv2.ROTATE_90_CLOCKWISE else None,
    },
    "notes": ""
}
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("SESSION COMPLETE")
print(f"Stop reason: {final_state['stop_reason']}")
print(f"Presses: {final_state['press_count']} | Releases: {final_state['release_count']}")
print(f"Frames: {final_state['frames_written']}")
print(f"Output: {session_dir}")
print("=" * 60)
