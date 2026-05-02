"""
Analysis-phase landmark extractor.

Reads a captured session, runs MediaPipe hand landmarking on every frame,
and writes the 21 landmarks per detected hand to landmarks.csv.

Joins back to frame_timestamps.csv via frame_index for timeline alignment.
Runs offline — no real-time constraints, uses model_complexity=1 for accuracy.
"""

import cv2
import mediapipe as mp
import csv
import json
import sys
import time
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1

# ==========================================
# INPUT VALIDATION
# ==========================================
if len(sys.argv) != 2:
    print("Usage: python extract_landmarks.py <session_directory>")
    sys.exit(1)

session_dir = Path(sys.argv[1])
if not session_dir.is_dir():
    print(f"ERROR: {session_dir} is not a directory")
    sys.exit(1)

video_path = session_dir / "session.mp4"
timestamps_path = session_dir / "frame_timestamps.csv"
metadata_path = session_dir / "metadata.json"
output_path = session_dir / "landmarks.csv"

for required in [video_path, timestamps_path, metadata_path]:
    if not required.exists():
        print(f"ERROR: Missing required file: {required}")
        sys.exit(1)

# ==========================================
# LOAD METADATA AND TIMESTAMPS
# ==========================================
with open(metadata_path) as f:
    metadata = json.load(f)

frame_times = {}
with open(timestamps_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame_times[int(row['frame_index'])] = float(row['t_mono'])

print(f"Session: {metadata['session_id']}")
print(f"Expected frames: {len(frame_times)}")
print(f"Video dimensions after transform: applied at capture time")
print(f"Output will be written to: {output_path}")

# ==========================================
# MEDIAPIPE SETUP
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # treat as video sequence (uses tracking)
    max_num_hands=2,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

# ==========================================
# CSV HEADER
# ==========================================
# One row per detected hand per frame.
# If a frame has 0 hands, it contributes 0 rows (gaps inferrable via frame_index).
# If 2 hands, 2 rows (distinguished by handedness column).
header = ['frame_index', 't_mono', 'handedness', 'detection_confidence']
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

output_file = open(output_path, 'w', newline='')
writer = csv.writer(output_file)
writer.writerow(header)

# ==========================================
# PROCESS VIDEO
# ==========================================
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"ERROR: Could not open {video_path}")
    sys.exit(1)

total_expected = len(frame_times)
frame_idx = 0
frames_with_hands = 0
total_hand_detections = 0
missing_timestamp_count = 0

start_time = time.time()

print("\nProcessing frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lookup the capture-time timestamp for this frame index
    t_mono = frame_times.get(frame_idx)
    if t_mono is None:
        # Shouldn't happen if the capture script behaved, but guard anyway
        missing_timestamp_count += 1
        t_mono = -1.0

    # MediaPipe expects RGB; OpenCV reads BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perf hint: marking the array non-writeable lets MediaPipe skip a copy
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if results.multi_hand_landmarks:
        frames_with_hands += 1
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            total_hand_detections += 1

            # Get handedness label and its confidence.
            # MediaPipe assumes a selfie-view camera and mirrors its output labels.
            # For our top-down rig the labels come out inverted, so we swap them
            # here so the CSV reflects the user's physical left/right hands.
            handedness_info = results.multi_handedness[hand_idx].classification[0]
            raw_label = handedness_info.label  # 'Left' or 'Right' as MediaPipe reports it
            handedness = 'Right' if raw_label == 'Left' else 'Left'
            detection_conf = handedness_info.score  # confidence in the L/R label

            # Flatten 21 landmarks into a single row
            row = [frame_idx, f"{t_mono:.6f}", handedness, f"{detection_conf:.4f}"]
            for lm in hand_landmarks.landmark:
                row.extend([f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}"])

            writer.writerow(row)

    # Progress indicator every 50 frames
    if frame_idx % 50 == 0 and frame_idx > 0:
        elapsed = time.time() - start_time
        fps = frame_idx / elapsed
        eta = (total_expected - frame_idx) / fps if fps > 0 else 0
        print(f"  Frame {frame_idx}/{total_expected} | {fps:.1f} fps | ETA {eta:.0f}s")

    frame_idx += 1

cap.release()
hands.close()
output_file.flush()
output_file.close()

# ==========================================
# SUMMARY
# ==========================================
elapsed_total = time.time() - start_time

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print(f"Frames processed: {frame_idx}")
print(f"Frames with at least one hand: {frames_with_hands} ({frames_with_hands / max(frame_idx, 1) * 100:.1f}%)")
print(f"Total hand detections: {total_hand_detections}")
print(f"Avg hands per frame: {total_hand_detections / max(frame_idx, 1):.2f}")
print(f"Missing timestamps: {missing_timestamp_count}")
print(f"Processing time: {elapsed_total:.1f}s ({frame_idx / max(elapsed_total, 0.001):.1f} fps)")
print(f"Output: {output_path}")
print("=" * 60)

# Quick sanity check — tracking rate is the single most important quality metric
tracking_rate = frames_with_hands / max(frame_idx, 1)
if tracking_rate < 0.70:
    print(f"\n⚠  WARNING: Only {tracking_rate * 100:.1f}% of frames had hand detection.")
    print(f"   Check lighting, crop region, or camera position.")
elif tracking_rate < 0.90:
    print(f"\n   Tracking rate {tracking_rate * 100:.1f}% — acceptable but could be improved.")
else:
    print(f"\n✓  Tracking rate {tracking_rate * 100:.1f}% — excellent.")