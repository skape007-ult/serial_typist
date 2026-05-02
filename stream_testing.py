"""
Stream tester — shows live IP Webcam feed in a window.

Quick utility to verify your phone stream is reachable and producing usable
video before starting a real recording session. Press 'q' to quit.

Usage:
    python stream_test.py                          # uses URL from capture_p1.py
    python stream_test.py http://192.168.x.x:8080/video
"""

import os
import sys
import time
import cv2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DEFAULT_URL = os.getenv('IP_WEBCAM_URL')
if not DEFAULT_URL:
    raise RuntimeError("IP_WEBCAM_URL is not set. Add it to your .env file.")
CROP_Y_START, CROP_Y_END = 180, 1000  # widened bottom edge to full frame so wrists stay in frame
CROP_X_START, CROP_X_END = 700, 1340
ROTATION = cv2.ROTATE_90_CLOCKWISE

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f"Connecting to: {url}")

    cap = cv2.VideoCapture(url)

    # Don't wait forever — fail fast if stream isn't there
    if not cap.isOpened():
        print("ERROR: Could not open stream.")
        print("  Check that IP Webcam is running on your phone.")
        print("  Check the URL — try opening it in your browser first.")
        print(f"  Default URL is {DEFAULT_URL}; override with: python stream_test.py <url>")
        sys.exit(1)

    print("Stream opened. Showing live feed. Press 'q' to quit.")
    print()

    frame_count = 0
    fps_window_start = time.monotonic()
    fps_window_frames = 0
    last_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"  Stream returned empty frame at frame #{frame_count}. "
                  f"Phone may have disconnected.")
            break

        frame = frame[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]
        frame = cv2.rotate(frame, ROTATION)

        frame_count += 1
        fps_window_frames += 1

        # Update FPS once a second
        elapsed = time.monotonic() - fps_window_start
        if elapsed >= 1.0:
            last_fps = fps_window_frames / elapsed
            fps_window_start = time.monotonic()
            fps_window_frames = 0

        # Overlay diagnostics on the frame
        h, w = frame.shape[:2]
        overlay_text = f"{w}x{h} @ {last_fps:.1f} fps  |  frame #{frame_count}  |  q=quit"
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(frame, overlay_text, (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Stream Test (q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nClosed. Saw {frame_count} frames at ~{last_fps:.1f} fps near the end.")


if __name__ == '__main__':
    main()