"""
Phase 3 — Fusion.

For every keystroke press in the session, determine which fingertip pressed it
using hand-landmark motion data.

Core heuristic: the fingertip with the largest downward Z-velocity (toward the
keyboard, away from the camera) in the window around the press is the finger
responsible. Z is computed relative to a smoothed wrist position to remove
wrist jitter.

Design decisions (see DECISIONS.md for rationale):
  - Window: 5 frames before + 1 frame after each press.
  - Modifiers/F-keys/tab/esc: skipped, marked as rule-assigned.
  - Spacebar: marked as 'thumb' with hand unknown, skipped for motion fusion.
  - Backspace: tried via fusion (right pinky in principle, but verify from data).
  - Confidence score: ratio of top finger's velocity to second-best.

Outputs:
  - events.csv: one row per keystroke, lean columns for analysis.
  - events_debug/: one JSON per fused keystroke with the full landmark window.

Usage:
    python fuse.py sessions/20260418_XXXXXX
"""

import csv
import json
import sys
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Time window (in frames) around each keystroke used to compute velocity
FRAMES_BEFORE = 5
FRAMES_AFTER = 1

# Wrist smoothing window size (odd number for centered average)
WRIST_SMOOTH_WINDOW = 5

# Keys that skip motion fusion entirely and get rule-based assignment
FILTERED_KEYS = {
    'shift', 'shift_r', 'shift_l',
    'ctrl', 'ctrl_r', 'ctrl_l',
    'alt', 'alt_r', 'alt_l', 'alt_gr',
    'cmd', 'cmd_r', 'cmd_l',
    'caps_lock', 'tab', 'esc',
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
    'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
}

# Spacebar — handled separately (thumb, hand unknown)
SPACE_KEYS = {'space', ' '}

# Fingertip landmark indices in MediaPipe's 21-point hand model
FINGERTIPS = {
    4: 'thumb',
    8: 'index',
    12: 'middle',
    16: 'ring',
    20: 'pinky',
}

# ==========================================
# INPUT
# ==========================================
if len(sys.argv) != 2:
    print("Usage: python fuse.py <session_directory>")
    sys.exit(1)

session_dir = Path(sys.argv[1])
keys_path = session_dir / "keystrokes.csv"
landmarks_path = session_dir / "landmarks.csv"
frame_times_path = session_dir / "frame_timestamps.csv"
metadata_path = session_dir / "metadata.json"

for p in [keys_path, landmarks_path, frame_times_path, metadata_path]:
    if not p.exists():
        print(f"ERROR: Missing required file: {p}")
        sys.exit(1)

events_csv_path = session_dir / "events.csv"
debug_dir = session_dir / "events_debug"
debug_dir.mkdir(exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================
print("Loading data...")

# Frame index -> t_mono
frame_times = {}
with open(frame_times_path) as f:
    for row in csv.DictReader(f):
        frame_times[int(row['frame_index'])] = float(row['t_mono'])

frame_indices_sorted = sorted(frame_times.keys())
frame_ts_array = np.array([frame_times[i] for i in frame_indices_sorted])

# Landmarks: frame_index -> hand -> {wrist: (x,y,z), tips: {name: (x,y,z)}}
# Each frame has up to 2 hand entries (Left, Right)
landmarks_by_frame = defaultdict(dict)

with open(landmarks_path) as f:
    for row in csv.DictReader(f):
        fi = int(row['frame_index'])
        hand = row['handedness']

        wrist = (float(row['x0']), float(row['y0']), float(row['z0']))
        tips = {}
        for tip_idx, name in FINGERTIPS.items():
            tips[name] = (
                float(row[f'x{tip_idx}']),
                float(row[f'y{tip_idx}']),
                float(row[f'z{tip_idx}']),
            )

        landmarks_by_frame[fi][hand] = {
            'wrist': wrist,
            'tips': tips,
        }

# Keystroke presses only (we don't fuse releases)
key_events = []
with open(keys_path) as f:
    for row in csv.DictReader(f):
        if row['event'] != 'press':
            continue
        key_events.append({
            't_mono': float(row['t_mono']),
            'key': row['key'],
        })

print(f"  {len(frame_times)} frames, {len(key_events)} press events, "
      f"{len(landmarks_by_frame)} frames with landmarks")


# ==========================================
# SMOOTH WRIST Z PER HAND
# ==========================================
# Build per-hand arrays of wrist Z indexed by frame order, apply moving average,
# then map back. This removes the wrist shimmer the diagnostic plot showed.

def moving_average(values, window):
    """Simple centered moving average. Edges use shorter windows."""
    n = len(values)
    out = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.mean(values[lo:hi])
    return out


smoothed_wrist_z = {'Left': {}, 'Right': {}}
for hand in ('Left', 'Right'):
    # Collect (frame_index, wrist_z) for this hand in frame order
    entries = []
    for fi in frame_indices_sorted:
        if hand in landmarks_by_frame.get(fi, {}):
            entries.append((fi, landmarks_by_frame[fi][hand]['wrist'][2]))

    if not entries:
        continue

    fis = [e[0] for e in entries]
    zs = np.array([e[1] for e in entries])
    smoothed = moving_average(zs, WRIST_SMOOTH_WINDOW)

    for fi, sz in zip(fis, smoothed):
        smoothed_wrist_z[hand][fi] = float(sz)

print(f"  Wrist Z smoothed for both hands (window={WRIST_SMOOTH_WINDOW}).")


# ==========================================
# FUSION HELPERS
# ==========================================

def nearest_frame_idx(t):
    """Return the frame index whose t_mono is closest to t."""
    pos = np.searchsorted(frame_ts_array, t)
    if pos == 0:
        return frame_indices_sorted[0]
    if pos == len(frame_ts_array):
        return frame_indices_sorted[-1]
    before = frame_ts_array[pos - 1]
    after = frame_ts_array[pos]
    return frame_indices_sorted[pos - 1 if (t - before) <= (after - t) else pos]


def window_around(center_frame_idx, before, after):
    """Return the list of frame indices in the window, clipped to available range."""
    pos = frame_indices_sorted.index(center_frame_idx)
    lo = max(0, pos - before)
    hi = min(len(frame_indices_sorted), pos + after + 1)
    return frame_indices_sorted[lo:hi]


def relative_z(fi, hand, tip_name):
    """Fingertip Z relative to the smoothed wrist Z for that hand at that frame.

    Larger values = fingertip is further from camera = closer to keyboard = pressing.
    Returns None if the hand isn't detected in this frame.
    """
    frame_data = landmarks_by_frame.get(fi, {})
    if hand not in frame_data:
        return None
    tip_z = frame_data[hand]['tips'][tip_name][2]
    wrist_z_s = smoothed_wrist_z[hand].get(fi)
    if wrist_z_s is None:
        return None
    return tip_z - wrist_z_s


def compute_velocity_for_finger(window_frames, hand, tip_name):
    """Compute mean Z-velocity for a fingertip across the window.

    Velocity here = change in relative-Z per frame. Positive = moving toward
    keyboard.

    Returns (velocity, num_valid_samples). Velocity is 0 if insufficient data.
    """
    zs = []
    for fi in window_frames:
        rz = relative_z(fi, hand, tip_name)
        if rz is not None:
            zs.append(rz)

    if len(zs) < 2:
        return 0.0, len(zs)

    # Simple velocity: (end - start) / n_samples
    # More robust than pairwise differences when some frames are missing landmarks
    velocity = (zs[-1] - zs[0]) / max(1, len(zs) - 1)
    return velocity, len(zs)


# ==========================================
# MAIN FUSION LOOP
# ==========================================
print("\nFusing keystrokes...")

events = []
stats = {
    'total': 0,
    'filtered_modifier': 0,
    'space_rule': 0,
    'fused': 0,
    'skipped_no_landmarks': 0,
}

for idx, ke in enumerate(key_events):
    stats['total'] += 1
    key = ke['key']
    t = ke['t_mono']

    event = {
        'event_idx': idx,
        't_mono': t,
        'key': key,
        'frame_index': None,
        'hand': None,
        'finger': None,
        'confidence': None,
        'velocity_ratio': None,
        'was_fused': False,
        'method': None,
    }

    # --- Rule: filtered keys ---
    if key.lower() in FILTERED_KEYS:
        event['method'] = 'filtered'
        stats['filtered_modifier'] += 1
        events.append(event)
        continue

    # --- Rule: spacebar ---
    if key.lower() in SPACE_KEYS or key == ' ':
        event['finger'] = 'thumb'
        event['hand'] = 'unknown'
        event['method'] = 'space_rule'
        nearest = nearest_frame_idx(t)
        event['frame_index'] = nearest
        stats['space_rule'] += 1
        events.append(event)
        continue

    # --- Motion fusion ---
    center_fi = nearest_frame_idx(t)
    window = window_around(center_fi, FRAMES_BEFORE, FRAMES_AFTER)

    # Compute velocity for every (hand, finger) combination
    velocities = {}
    sample_counts = {}
    for hand in ('Left', 'Right'):
        for tip_name in FINGERTIPS.values():
            v, n = compute_velocity_for_finger(window, hand, tip_name)
            velocities[(hand, tip_name)] = v
            sample_counts[(hand, tip_name)] = n

    # Filter to candidates with at least 3 valid samples in the window
    candidates = {k: v for k, v in velocities.items()
                  if sample_counts[k] >= 3 and v > 0}

    if not candidates:
        event['method'] = 'no_landmark_data'
        stats['skipped_no_landmarks'] += 1
        event['frame_index'] = center_fi
        events.append(event)
        continue

    # Rank by velocity, pick the top
    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    (top_hand, top_finger), top_v = ranked[0]

    # Confidence = ratio of top velocity to second-best (or to top if only one)
    if len(ranked) >= 2:
        _, second_v = ranked[1]
        velocity_ratio = top_v / max(second_v, 1e-9)
        # Confidence: 1.0 when ratio is large, 0.0 when ratio is 1 (tied)
        confidence = min(1.0, (velocity_ratio - 1.0) / 2.0)
    else:
        velocity_ratio = float('inf')
        confidence = 1.0

    event['frame_index'] = center_fi
    event['hand'] = top_hand
    event['finger'] = top_finger
    event['confidence'] = round(confidence, 4)
    event['velocity_ratio'] = round(velocity_ratio, 4) if velocity_ratio != float('inf') else None
    event['was_fused'] = True
    event['method'] = 'motion_fusion'
    stats['fused'] += 1
    events.append(event)

    # --- Rich debug JSON for this event ---
    debug_entry = {
        'event_idx': idx,
        't_mono': t,
        'key': key,
        'center_frame_index': center_fi,
        'window_frame_indices': window,
        'assignment': {
            'hand': top_hand,
            'finger': top_finger,
            'confidence': round(confidence, 4),
            'velocity_ratio': round(velocity_ratio, 4) if velocity_ratio != float('inf') else None,
        },
        'all_velocities': {
            f"{h}_{f}": {
                'velocity': round(velocities[(h, f)], 6),
                'samples': sample_counts[(h, f)],
            }
            for h in ('Left', 'Right') for f in FINGERTIPS.values()
        },
        'landmark_window': [],
    }
    for fi in window:
        frame_entry = {'frame_index': fi, 't_mono': frame_times.get(fi)}
        fd = landmarks_by_frame.get(fi, {})
        for hand in ('Left', 'Right'):
            if hand not in fd:
                continue
            frame_entry[hand] = {
                'wrist': fd[hand]['wrist'],
                'wrist_z_smoothed': smoothed_wrist_z[hand].get(fi),
                'tips': fd[hand]['tips'],
                'tips_relative_z': {
                    name: (tip_z := fd[hand]['tips'][name][2]) - (smoothed_wrist_z[hand].get(fi) or tip_z)
                    for name in FINGERTIPS.values()
                },
            }
        debug_entry['landmark_window'].append(frame_entry)

    # Filename: event_00042_t11.668_w.json
    safe_key = key if key.isalnum() else key.encode('unicode_escape').decode()
    debug_name = f"event_{idx:05d}_t{t:.3f}_{safe_key[:8]}.json"
    with open(debug_dir / debug_name, 'w') as f:
        json.dump(debug_entry, f, indent=2)

# ==========================================
# WRITE events.csv
# ==========================================
with open(events_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'event_idx', 't_mono', 'key', 'frame_index',
        'hand', 'finger', 'confidence', 'velocity_ratio',
        'was_fused', 'method',
    ])
    writer.writeheader()
    for e in events:
        writer.writerow(e)

# ==========================================
# SUMMARY
# ==========================================
print(f"\n{'=' * 60}")
print(f"FUSION COMPLETE")
print(f"{'=' * 60}")
print(f"Total press events: {stats['total']}")
print(f"  Motion-fused:          {stats['fused']:>4}  ({stats['fused'] / max(1, stats['total']) * 100:.1f}%)")
print(f"  Spacebar (rule):       {stats['space_rule']:>4}")
print(f"  Filtered (modifiers):  {stats['filtered_modifier']:>4}")
print(f"  No landmark data:      {stats['skipped_no_landmarks']:>4}")
print(f"")

# Finger distribution of fused events
fused_events = [e for e in events if e['was_fused']]
if fused_events:
    finger_counts = defaultdict(int)
    for e in fused_events:
        finger_counts[f"{e['hand']} {e['finger']}"] += 1

    print("Assignment distribution (motion-fused only):")
    for label, count in sorted(finger_counts.items(), key=lambda x: -x[1]):
        bar = '█' * int(count / max(finger_counts.values()) * 30)
        pct = count / len(fused_events) * 100
        print(f"  {label:20s} {count:>4} ({pct:5.1f}%)  {bar}")

    confidences = [e['confidence'] for e in fused_events if e['confidence'] is not None]
    print(f"\nConfidence:")
    print(f"  Median: {np.median(confidences):.3f}")
    print(f"  Mean:   {np.mean(confidences):.3f}")
    print(f"  <0.2 (low confidence): {sum(1 for c in confidences if c < 0.2)} events")

print(f"\nOutput:")
print(f"  events.csv:     {events_csv_path}")
print(f"  events_debug/:  {debug_dir} ({len(list(debug_dir.glob('*.json')))} files)")