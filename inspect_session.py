"""
Session inspection utility.

Generates diagnostic plots for a recorded session so you can sanity-check the
landmark data before building fusion logic, and debug specific keystrokes later.

Produces four plots in a single figure:
  1. Inter-keystroke intervals over time — reveals typing rhythm and long pauses
  2. Landmark Y-coordinate of fingertips over time — reveals motion patterns
  3. Keystroke-to-nearest-frame timing distance — reveals fusion alignment quality
  4. Palm size (wrist to middle-MCP) over time — reveals landmark stability
"""

import csv
import json
import sys
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# INPUT
# ==========================================
if len(sys.argv) != 2:
    print("Usage: python inspect_session.py <session_directory>")
    sys.exit(1)

session_dir = Path(sys.argv[1])
keys_path = session_dir / "keystrokes.csv"
landmarks_path = session_dir / "landmarks.csv"
frame_times_path = session_dir / "frame_timestamps.csv"
metadata_path = session_dir / "metadata.json"

for p in [keys_path, landmarks_path, frame_times_path, metadata_path]:
    if not p.exists():
        print(f"ERROR: Missing file: {p}")
        sys.exit(1)

# ==========================================
# LOAD DATA
# ==========================================
with open(metadata_path) as f:
    metadata = json.load(f)

# Keystrokes — just presses for rhythm analysis
key_presses = []
with open(keys_path) as f:
    for row in csv.DictReader(f):
        if row['event'] == 'press':
            key_presses.append({
                't_mono': float(row['t_mono']),
                'key': row['key'],
            })

# Frame timestamps
frame_times = {}
with open(frame_times_path) as f:
    for row in csv.DictReader(f):
        frame_times[int(row['frame_index'])] = float(row['t_mono'])

# Landmarks — organized by hand for fingertip trajectories
# We care mostly about fingertips: indices 4, 8, 12, 16, 20
FINGERTIPS = {4: 'Thumb', 8: 'Index', 12: 'Middle', 16: 'Ring', 20: 'Pinky'}

# per_hand_tips[hand]['Index'] = list of (t_mono, y) across frames
per_hand_tips = {
    'Left': {name: [] for name in FINGERTIPS.values()},
    'Right': {name: [] for name in FINGERTIPS.values()},
}
# Palm sizes: [(t_mono, hand, palm_size), ...]
palm_sizes = []

with open(landmarks_path) as f:
    for row in csv.DictReader(f):
        hand = row['handedness']
        t = float(row['t_mono'])

        # Fingertip Y positions (normalized 0-1)
        for lm_idx, name in FINGERTIPS.items():
            y = float(row[f'y{lm_idx}'])
            per_hand_tips[hand][name].append((t, y))

        # Palm size: distance from wrist (0) to middle MCP (9)
        wx, wy = float(row['x0']), float(row['y0'])
        mx, my = float(row['x9']), float(row['y9'])
        palm_size = math.hypot(mx - wx, my - wy)
        palm_sizes.append((t, hand, palm_size))

print(f"Loaded: {len(key_presses)} presses, {len(frame_times)} frames, "
      f"{len(palm_sizes)} landmark rows")

# ==========================================
# COMPUTE DIAGNOSTICS
# ==========================================
# 1. Inter-keystroke intervals
iki_times = []
iki_values = []
for i in range(1, len(key_presses)):
    dt_ms = (key_presses[i]['t_mono'] - key_presses[i - 1]['t_mono']) * 1000
    iki_times.append(key_presses[i]['t_mono'])
    iki_values.append(dt_ms)

median_iki = np.median(iki_values) if iki_values else 0
fastest_iki = min(iki_values) if iki_values else 0

# 2. Keystroke-to-nearest-frame distance (fusion alignment quality)
# For each keystroke, find the closest frame's t_mono and compute the delta
frame_ts_array = np.array(sorted(frame_times.values()))
alignment_deltas = []
for kp in key_presses:
    t = kp['t_mono']
    nearest_idx = np.searchsorted(frame_ts_array, t)
    candidates = []
    if nearest_idx > 0:
        candidates.append(abs(t - frame_ts_array[nearest_idx - 1]))
    if nearest_idx < len(frame_ts_array):
        candidates.append(abs(t - frame_ts_array[nearest_idx]))
    if candidates:
        alignment_deltas.append(min(candidates) * 1000)  # convert to ms

max_alignment = max(alignment_deltas) if alignment_deltas else 0
median_alignment = np.median(alignment_deltas) if alignment_deltas else 0

# 3. Palm size stability (for landmark sanity)
left_palms = [(t, ps) for t, h, ps in palm_sizes if h == 'Left']
right_palms = [(t, ps) for t, h, ps in palm_sizes if h == 'Right']

left_palm_vals = [ps for _, ps in left_palms]
right_palm_vals = [ps for _, ps in right_palms]

left_cv = np.std(left_palm_vals) / np.mean(left_palm_vals) if left_palm_vals else 0
right_cv = np.std(right_palm_vals) / np.mean(right_palm_vals) if right_palm_vals else 0

# ==========================================
# PRINT SUMMARY
# ==========================================
print(f"\n{'=' * 60}")
print(f"SESSION DIAGNOSTICS: {metadata['session_id']}")
print(f"{'=' * 60}")
print(f"Duration: {metadata['duration_wall_seconds']:.1f}s")
print(f"Presses: {metadata['press_count']} | Releases: {metadata['release_count']}")
print(f"")
print(f"TYPING RHYTHM")
print(f"  Median IKI: {median_iki:.0f}ms  (→ ~{60000 / median_iki / 5:.0f} WPM estimate)")
print(f"  Fastest IKI: {fastest_iki:.0f}ms")
print(f"")
print(f"FUSION ALIGNMENT")
print(f"  Median keystroke→frame delta: {median_alignment:.1f}ms")
print(f"  Max keystroke→frame delta: {max_alignment:.1f}ms")
if max_alignment > 33:
    print(f"  ⚠ Max exceeds one frame at 30fps — some keystrokes may fall "
          f"between frames")
print(f"")
print(f"LANDMARK STABILITY (coefficient of variation)")
print(f"  Left palm size CV:  {left_cv:.3f}  ({'OK' if left_cv < 0.15 else 'NOISY'})")
print(f"  Right palm size CV: {right_cv:.3f}  ({'OK' if right_cv < 0.15 else 'NOISY'})")
print(f"  (CV < 0.15 indicates a geometrically stable skeleton)")

# ==========================================
# BUILD FIGURE
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"Session Diagnostics — {metadata['session_id']}", fontsize=13)

# Plot 1: Inter-keystroke intervals
ax = axes[0, 0]
ax.scatter(iki_times, iki_values, s=15, alpha=0.6, color='steelblue')
ax.axhline(median_iki, color='red', linestyle='--', linewidth=1,
           label=f'median {median_iki:.0f}ms')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Inter-keystroke interval (ms)')
ax.set_title('Typing rhythm')
ax.set_yscale('log')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Fingertip Y trajectories over time
ax = axes[0, 1]
colors = {'Thumb': '#d62728', 'Index': '#2ca02c', 'Middle': '#ff7f0e',
          'Ring': '#9467bd', 'Pinky': '#1f77b4'}
for name in FINGERTIPS.values():
    # Only show right hand to avoid clutter; left hand is symmetric
    pts = per_hand_tips['Right'][name]
    if pts:
        ts, ys = zip(*pts)
        ax.plot(ts, ys, color=colors[name], linewidth=0.7, alpha=0.7,
                label=name)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Y-coordinate (normalized)')
ax.set_title('Right-hand fingertip Y over time')
ax.invert_yaxis()  # MediaPipe Y increases downward
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 3: Fusion alignment histogram
ax = axes[1, 0]
ax.hist(alignment_deltas, bins=30, color='teal', alpha=0.8, edgecolor='white')
ax.axvline(median_alignment, color='red', linestyle='--',
           label=f'median {median_alignment:.1f}ms')
ax.axvline(33.3, color='orange', linestyle=':',
           label='1 frame @ 30fps (33ms)')
ax.set_xlabel('Keystroke→nearest-frame distance (ms)')
ax.set_ylabel('Count')
ax.set_title('Fusion alignment quality')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Palm size stability
ax = axes[1, 1]
if left_palms:
    lt, lp = zip(*left_palms)
    ax.plot(lt, lp, color='green', linewidth=0.8, alpha=0.8, label='Left')
if right_palms:
    rt, rp = zip(*right_palms)
    ax.plot(rt, rp, color='blue', linewidth=0.8, alpha=0.8, label='Right')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Palm size (normalized, wrist→mid-MCP)')
ax.set_title('Landmark stability (constant palm size = stable skeleton)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = session_dir / "diagnostics.png"
plt.savefig(output_path, dpi=120, bbox_inches='tight')
print(f"\nFigure saved: {output_path}")