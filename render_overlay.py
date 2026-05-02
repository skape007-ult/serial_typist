"""
Landmark overlay renderer.

Reads a session's saved video and its landmarks.csv, draws the 21-landmark
skeleton per hand on every frame, and writes an annotated video.

No MediaPipe inference here — we're just visualizing the landmarks we already
extracted, which means (a) it's fast, and (b) any drift you see in the overlay
is real landmark drift, not re-inference variance.
"""

import cv2
import csv
import sys
from collections import defaultdict
from pathlib import Path

# ==========================================
# INPUT
# ==========================================
if len(sys.argv) != 2:
    print("Usage: python render_overlay.py <session_directory>")
    sys.exit(1)

session_dir = Path(sys.argv[1])
video_in = session_dir / "session.mp4"
landmarks_csv = session_dir / "landmarks.csv"
video_out = session_dir / "session_overlay.mp4"

for required in [video_in, landmarks_csv]:
    if not required.exists():
        print(f"ERROR: Missing required file: {required}")
        sys.exit(1)

# ==========================================
# HAND SKELETON CONNECTIONS
# ==========================================
# Matches MediaPipe's HAND_CONNECTIONS — each tuple is (landmark_index_a, landmark_index_b)
# connected by a bone. Drawing these as lines gives you the skeleton.
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGERTIP_IDS = {4, 8, 12, 16, 20}  # highlighted in a different color

# Color per hand (BGR for OpenCV)
COLORS = {
    'Left': {'bone': (0, 255, 0), 'joint': (0, 200, 0), 'tip': (0, 255, 255)},
    'Right': {'bone': (255, 100, 0), 'joint': (200, 80, 0), 'tip': (0, 255, 255)},
}

# ==========================================
# LOAD LANDMARKS INDEXED BY FRAME
# ==========================================
# Schema: frame_landmarks[frame_idx] = [ {handedness, confidence, landmarks: [(x,y,z), ...]}, ... ]
frame_landmarks = defaultdict(list)

with open(landmarks_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame_idx = int(row['frame_index'])
        hand = {
            'handedness': row['handedness'],
            'confidence': float(row['detection_confidence']),
            'landmarks': [],
        }
        for i in range(21):
            x = float(row[f'x{i}'])
            y = float(row[f'y{i}'])
            z = float(row[f'z{i}'])
            hand['landmarks'].append((x, y, z))
        frame_landmarks[frame_idx].append(hand)

print(f"Loaded landmarks for {len(frame_landmarks)} frames.")

# ==========================================
# VIDEO I/O SETUP
# ==========================================
cap = cv2.VideoCapture(str(video_in))
if not cap.isOpened():
    print(f"ERROR: Cannot open {video_in}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(video_out), fourcc, fps, (w, h))

print(f"Rendering {total_frames} frames at {w}x{h} @ {fps:.1f} FPS...")

# ==========================================
# RENDER LOOP
# ==========================================
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw every hand that was detected for this frame
    for hand in frame_landmarks.get(frame_idx, []):
        handedness = hand['handedness']
        lms = hand['landmarks']
        colors = COLORS.get(handedness, COLORS['Right'])

        # Convert normalized (0-1) coords to pixel coords
        pts = [(int(x * w), int(y * h)) for x, y, z in lms]

        # Draw bones first (so joints render on top)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], colors['bone'], 2)

        # Draw joints
        for i, (px, py) in enumerate(pts):
            if i in FINGERTIP_IDS:
                # Fingertips get bigger highlighted circles
                cv2.circle(frame, (px, py), 6, colors['tip'], -1)
                cv2.circle(frame, (px, py), 8, (0, 0, 0), 1)  # black outline
            else:
                cv2.circle(frame, (px, py), 4, colors['joint'], -1)

        # Label near the wrist
        wrist_x, wrist_y = pts[0]
        label = f"{handedness} ({hand['confidence']:.2f})"
        cv2.putText(frame, label, (wrist_x + 10, wrist_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['bone'], 1, cv2.LINE_AA)

    # Frame index in corner — useful when scrubbing to find a specific moment
    cv2.putText(frame, f"#{frame_idx}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    writer.write(frame)

    if frame_idx % 100 == 0:
        print(f"  Frame {frame_idx}/{total_frames}")

    frame_idx += 1

cap.release()
writer.release()

print(f"\n✓  Overlay rendered: {video_out}")
print(f"   {frame_idx} frames written.")