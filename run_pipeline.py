"""
End-to-end session pipeline.

Runs the three stages in order, aborting if any one fails:
    1. capture_p1.py            - records a new session into sessions/<id>/
    2. landmark_extractor_p2.py - writes landmarks.csv for that session
    3. render_overlay.py        - writes session_overlay.mp4 for that session

Each stage is invoked as a subprocess using the same Python interpreter that
ran this script, so the .venv is respected automatically.

Usage:
    python run_pipeline.py              # full pipeline: capture -> extract -> render
    python run_pipeline.py --skip-capture sessions/20260417_211944
                                        # skip capture, run the other two on an
                                        # existing session directory
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
PYTHON = sys.executable  # same interpreter the user launched with
SESSIONS_DIR = PROJECT_ROOT / "sessions"

CAPTURE_SCRIPT = PROJECT_ROOT / "capture_p1.py"
EXTRACT_SCRIPT = PROJECT_ROOT / "landmark_extractor_p2.py"
OVERLAY_SCRIPT = PROJECT_ROOT / "render_overlay.py"


def run_stage(name: str, cmd: list[str]) -> None:
    """Run a subprocess; print a clean banner around it; abort on non-zero exit."""
    print("\n" + "#" * 60)
    print(f"# STAGE: {name}")
    print(f"# CMD:   {' '.join(cmd)}")
    print("#" * 60 + "\n")

    start = time.time()
    # Inherit stdout/stderr so the child script's prints stream to this terminal live
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[PIPELINE] ✗ {name} failed (exit {result.returncode}) after {elapsed:.1f}s. Aborting.")
        sys.exit(result.returncode)

    print(f"\n[PIPELINE] ✓ {name} finished in {elapsed:.1f}s.")


def find_latest_session() -> Path:
    """Return the newest sessions/<id>/ directory by modification time."""
    if not SESSIONS_DIR.exists():
        print(f"[PIPELINE] ERROR: {SESSIONS_DIR} does not exist. Did capture run?")
        sys.exit(1)

    candidates = [p for p in SESSIONS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        print(f"[PIPELINE] ERROR: No session directories found under {SESSIONS_DIR}.")
        sys.exit(1)

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def main() -> None:
    args = sys.argv[1:]
    skip_capture = False
    explicit_session: Path | None = None

    if args and args[0] == "--skip-capture":
        skip_capture = True
        if len(args) != 2:
            print("Usage: python run_pipeline.py --skip-capture <session_directory>")
            sys.exit(1)
        explicit_session = Path(args[1]).resolve()
        if not explicit_session.is_dir():
            print(f"[PIPELINE] ERROR: {explicit_session} is not a directory.")
            sys.exit(1)
    elif args:
        print("Usage:")
        print("  python run_pipeline.py")
        print("  python run_pipeline.py --skip-capture <session_directory>")
        sys.exit(1)

    # ---------- Stage 1: capture ----------
    if not skip_capture:
        run_stage("capture_p1", [PYTHON, str(CAPTURE_SCRIPT)])
        session_dir = find_latest_session()
        print(f"\n[PIPELINE] Detected new session: {session_dir}")
    else:
        session_dir = explicit_session
        print(f"\n[PIPELINE] Skipping capture. Using session: {session_dir}")

    # Sanity-check that the session actually has the files we need before moving on
    required = ["session.mp4", "keystrokes.csv", "frame_timestamps.csv", "metadata.json"]
    missing = [f for f in required if not (session_dir / f).exists()]
    if missing:
        print(f"[PIPELINE] ERROR: Session {session_dir.name} is missing: {missing}")
        sys.exit(1)

    # ---------- Stage 2: landmark extraction ----------
    run_stage("landmark_extractor_p2", [PYTHON, str(EXTRACT_SCRIPT), str(session_dir)])

    # ---------- Stage 3: overlay rendering ----------
    run_stage("render_overlay", [PYTHON, str(OVERLAY_SCRIPT), str(session_dir)])

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Session: {session_dir}")
    print("Outputs:")
    print(f"  video:      {session_dir}/session.mp4")
    print(f"  keystrokes: {session_dir}/keystrokes.csv")
    print(f"  landmarks:  {session_dir}/landmarks.csv")
    print(f"  overlay:    {session_dir}/session_overlay.mp4")
    print("=" * 60)


if __name__ == "__main__":
    main()
