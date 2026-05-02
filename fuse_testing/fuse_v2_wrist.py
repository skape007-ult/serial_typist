"""
Phase 3 v2-wrist — Fusion via spatial position + learned keyboard layout,
using wrist-relative fingertip coordinates per hand.

Same architecture as fuse_v2.py with one substantive change:
  All fingertip and key-position math runs in wrist-relative coordinates,
  computed per hand. The hypothesis is that across-session hand drift adds
  noise to absolute-coordinate centroids, and that anchoring to the wrist
  removes that drift component while preserving the within-hand geometry that
  separates fingers.

Differences from fuse_v2.py:
  * Helper get_wrist_relative_tip() converts (x, y) to wrist-relative.
  * Pass-1 centroid aggregation stores wrist-relative tip positions.
  * Affine transforms are fit per-hand (Left, Right) instead of one global
    transform. Each hand has its own QWERTY → wrist-relative-image transform.
  * Pass-2 nearest-fingertip lookup operates entirely in wrist-relative space.

Outputs (in session_dir/):
  events_v2_wrist.csv              One row per keystroke press.
  events_v2_wrist_debug/*.json     Lean per-event debug.
  keyboard_layout_wrist.json       Learned centroids, per-hand affine, residuals.
  finger_map_wrist.json            Per-key canonical finger + distribution.

Usage:
    python fuse_v2_wrist.py sessions/20260424_121126
"""

import csv
import json
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
FRAMES_BEFORE = 5
FRAMES_AFTER = 1
WRIST_SMOOTH_WINDOW = 5
HIGH_FREQ_THRESHOLD = 3

FILTERED_KEYS = {
    'shift', 'shift_r', 'shift_l',
    'ctrl', 'ctrl_r', 'ctrl_l',
    'alt', 'alt_r', 'alt_l', 'alt_gr',
    'cmd', 'cmd_r', 'cmd_l',
    'caps_lock', 'tab', 'esc',
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
    'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
}
SPACE_KEYS = {'space', ' '}

FINGERTIPS = {4: 'thumb', 8: 'index', 12: 'middle', 16: 'ring', 20: 'pinky'}
WRIST = 0

# ==========================================
# QWERTY PRIOR
# ==========================================
QWERTY_PRIOR = {}
for i, c in enumerate('1234567890'):
    QWERTY_PRIOR[c] = (float(i), 0.0)
for i, c in enumerate('qwertyuiop'):
    QWERTY_PRIOR[c] = (i + 0.5, 1.0)
for i, c in enumerate('asdfghjkl'):
    QWERTY_PRIOR[c] = (i + 0.75, 2.0)
for i, c in enumerate('zxcvbnm'):
    QWERTY_PRIOR[c] = (i + 1.25, 3.0)
QWERTY_PRIOR.update({
    '-': (10.0, 0.0), '=': (11.0, 0.0),
    ';': (9.75, 2.0), "'": (10.75, 2.0),
    ',': (8.25, 3.0), '.': (9.25, 3.0), '/': (10.25, 3.0),
    '[': (10.5, 1.0), ']': (11.5, 1.0), '\\': (12.5, 1.0),
    '`': (-1.0, 0.0),
    'backspace': (12.5, 0.0),
    'enter': (12.75, 2.0),
    'return': (12.75, 2.0),
})

QWERTY_FINGER_MAP = {}
for c in '1qaz':        QWERTY_FINGER_MAP[c] = ('Left', 'pinky')
for c in '2wsx':        QWERTY_FINGER_MAP[c] = ('Left', 'ring')
for c in '3edc':        QWERTY_FINGER_MAP[c] = ('Left', 'middle')
for c in '45rtfgvb':    QWERTY_FINGER_MAP[c] = ('Left', 'index')
for c in '67yhnum':     QWERTY_FINGER_MAP[c] = ('Right', 'index')
for c in '8ik,':        QWERTY_FINGER_MAP[c] = ('Right', 'middle')
for c in '9ol.':        QWERTY_FINGER_MAP[c] = ('Right', 'ring')
for c in "0p;/'[]\\-=": QWERTY_FINGER_MAP[c] = ('Right', 'pinky')
QWERTY_FINGER_MAP['backspace'] = ('Right', 'ring')
QWERTY_FINGER_MAP['enter'] = ('Right', 'pinky')
QWERTY_FINGER_MAP['return'] = ('Right', 'pinky')

# ==========================================
# INPUT VALIDATION
# ==========================================
if len(sys.argv) != 2:
    print("Usage: python fuse_v2_wrist.py <session_directory>")
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

events_csv_path = session_dir / "events_v2_wrist.csv"
debug_dir = session_dir / "events_v2_wrist_debug"
debug_dir.mkdir(exist_ok=True)
layout_path = session_dir / "keyboard_layout_wrist.json"
finger_map_path = session_dir / "finger_map_wrist.json"

# ==========================================
# LOAD DATA
# ==========================================
print("Loading data (wrist-relative variant)...")

with open(metadata_path) as f:
    metadata = json.load(f)

frame_times = {}
with open(frame_times_path) as f:
    for row in csv.DictReader(f):
        frame_times[int(row['frame_index'])] = float(row['t_mono'])

frame_indices_sorted = sorted(frame_times.keys())
frame_ts_array = np.array([frame_times[i] for i in frame_indices_sorted])
frame_pos = {fi: i for i, fi in enumerate(frame_indices_sorted)}

# Same landmark structure as v2; we'll convert to wrist-relative on demand.
landmarks_by_frame = defaultdict(dict)
with open(landmarks_path) as f:
    for row in csv.DictReader(f):
        fi = int(row['frame_index'])
        hand = row['handedness']
        tips = {
            name: (float(row[f'x{idx}']), float(row[f'y{idx}']), float(row[f'z{idx}']))
            for idx, name in FINGERTIPS.items()
        }
        wrist = (float(row[f'x{WRIST}']), float(row[f'y{WRIST}']), float(row[f'z{WRIST}']))
        landmarks_by_frame[fi][hand] = {'wrist': wrist, 'tips': tips}

key_events = []
with open(keys_path) as f:
    for row in csv.DictReader(f):
        if row['event'] != 'press':
            continue
        key_events.append({'t_mono': float(row['t_mono']), 'key': row['key']})

print(f"  {len(frame_times)} frames, {len(key_events)} press events, "
      f"{len(landmarks_by_frame)} frames with landmarks")


# ==========================================
# SMOOTH WRIST Z PER HAND
# ==========================================
def moving_average(values, window):
    n = len(values)
    out = np.zeros(n)
    half = window // 2
    for i in range(n):
        out[i] = np.mean(values[max(0, i - half):min(n, i + half + 1)])
    return out


smoothed_wrist_z = {'Left': {}, 'Right': {}}
for hand in ('Left', 'Right'):
    entries = [(fi, landmarks_by_frame[fi][hand]['wrist'][2])
               for fi in frame_indices_sorted
               if hand in landmarks_by_frame.get(fi, {})]
    if not entries:
        continue
    fis = [e[0] for e in entries]
    zs = np.array([e[1] for e in entries])
    smoothed = moving_average(zs, WRIST_SMOOTH_WINDOW)
    for fi, sz in zip(fis, smoothed):
        smoothed_wrist_z[hand][fi] = float(sz)


# ==========================================
# HELPERS
# ==========================================
def nearest_frame(t):
    pos = np.searchsorted(frame_ts_array, t)
    if pos == 0:
        return frame_indices_sorted[0]
    if pos == len(frame_ts_array):
        return frame_indices_sorted[-1]
    b = frame_ts_array[pos - 1]
    a = frame_ts_array[pos]
    return frame_indices_sorted[pos - 1 if (t - b) <= (a - t) else pos]


def window_around(center_fi):
    pos = frame_pos.get(center_fi)
    if pos is None:
        return []
    lo = max(0, pos - FRAMES_BEFORE)
    hi = min(len(frame_indices_sorted), pos + FRAMES_AFTER + 1)
    return frame_indices_sorted[lo:hi]


def summed_z_velocity(window, hand):
    total = 0.0
    for name in FINGERTIPS.values():
        zs = []
        for fi in window:
            fd = landmarks_by_frame.get(fi, {}).get(hand)
            if fd is None:
                continue
            wz = smoothed_wrist_z[hand].get(fi)
            if wz is None:
                continue
            zs.append(fd['tips'][name][2] - wz)
        if len(zs) >= 2:
            total += zs[-1] - zs[0]
    return total


def select_hand(window):
    lv = summed_z_velocity(window, 'Left')
    rv = summed_z_velocity(window, 'Right')
    diag = {'left_summed_z': round(lv, 6), 'right_summed_z': round(rv, 6)}
    if lv == 0.0 and rv == 0.0:
        return None, diag
    return ('Left' if lv >= rv else 'Right'), diag


def press_frame_tips_wrist_relative(fi, hand):
    """Return dict {name: (rel_x, rel_y, z_rel)} in wrist-relative coords.

    rel_x = tip_x - wrist_x; rel_y = tip_y - wrist_y; both for the active hand
    on this frame. z_rel is fingertip z minus smoothed wrist z (kept for
    diagnostic compatibility; not used in spatial decisions).
    Returns None if landmarks for the active hand are missing on this frame.
    """
    fd = landmarks_by_frame.get(fi, {}).get(hand)
    if fd is None:
        return None
    wx, wy, _ = fd['wrist']
    wz = smoothed_wrist_z[hand].get(fi)
    if wz is None:
        wz = fd['wrist'][2]
    out = {}
    for name, (x, y, z) in fd['tips'].items():
        out[name] = (x - wx, y - wy, z - wz)
    return out


# ==========================================
# PASS 1 — rough per-event assignment (wrist-relative storage)
# ==========================================
print("\nPass 1: seeded assignment (Z-velocity for hand, QWERTY prior for finger)...")

events = []
# Centroid aggregation now keyed by (key_norm, hand) — each hand has its
# own wrist-relative coordinate frame, so we cannot pool positions across
# hands.
pass1_positions = defaultdict(list)  # (key_norm, hand) -> [(rx, ry), ...]

for idx, ke in enumerate(key_events):
    key = ke['key']
    t = ke['t_mono']

    evt = {
        'event_idx': idx, 't_mono': t, 'key': key,
        'frame_index': None,
        'hand': None, 'finger': None, 'method': None,
        'p1_finger': None, 'p1_tip_x': None, 'p1_tip_y': None,
    }

    if key.lower() in FILTERED_KEYS:
        evt['method'] = 'filtered'
        events.append(evt)
        continue

    if key in SPACE_KEYS or key.lower() in SPACE_KEYS:
        evt['hand'] = 'unknown'
        evt['finger'] = 'thumb'
        evt['method'] = 'space_rule'
        evt['frame_index'] = nearest_frame(t)
        events.append(evt)
        continue

    center_fi = nearest_frame(t)
    evt['frame_index'] = center_fi
    window = window_around(center_fi)

    hand, hand_diag = select_hand(window)
    if hand is None:
        evt['method'] = 'no_landmark_data'
        events.append(evt)
        continue

    key_norm = key.lower() if len(key) == 1 and key.isalpha() else key
    evt['_key_norm'] = key_norm

    expected = QWERTY_FINGER_MAP.get(key_norm)
    if expected is not None:
        p1_hand, p1_finger = expected
    else:
        p1_hand, p1_finger = hand, None

    tips = press_frame_tips_wrist_relative(center_fi, p1_hand)
    if tips is None:
        for fi in reversed(window):
            tips = press_frame_tips_wrist_relative(fi, p1_hand)
            if tips is not None:
                evt['frame_index'] = fi
                center_fi = fi
                break
    if tips is None:
        evt['method'] = 'no_landmark_data'
        events.append(evt)
        continue

    if p1_finger is None:
        non_thumb = {n: xyz for n, xyz in tips.items() if n != 'thumb'}
        p1_finger = max(non_thumb.items(), key=lambda kv: kv[1][2])[0]

    px, py, _ = tips[p1_finger]  # wrist-relative

    evt['hand'] = p1_hand
    evt['p1_finger'] = p1_finger
    evt['p1_tip_x'] = px
    evt['p1_tip_y'] = py
    evt['_window'] = window
    evt['_hand_diag'] = hand_diag
    evt['_z_velocity_hand'] = hand

    pass1_positions[(key_norm, p1_hand)].append((px, py))
    events.append(evt)

print(f"  Pass 1 done. {sum(1 for e in events if e.get('p1_finger'))} events rough-assigned.")

# ==========================================
# LEARN KEYBOARD LAYOUT FROM PASS 1 CENTROIDS (per hand)
# ==========================================
# learned_centroids: (key_norm, hand) -> (rx, ry)  in wrist-relative space
# key_sample_counts: (key_norm, hand) -> int
learned_centroids = {}
key_sample_counts = {}
for (key_norm, hand), positions in pass1_positions.items():
    arr = np.array(positions)
    learned_centroids[(key_norm, hand)] = (float(arr[:, 0].mean()),
                                           float(arr[:, 1].mean()))
    key_sample_counts[(key_norm, hand)] = len(positions)


# ==========================================
# FIT PER-HAND AFFINE TRANSFORMS
# ==========================================
def fit_affine(src_pts, dst_pts):
    N = len(src_pts)
    if N < 3:
        return None, None, N
    X = np.zeros((2 * N, 6))
    y = np.zeros(2 * N)
    for i, ((sx, sy), (dx, dy)) in enumerate(zip(src_pts, dst_pts)):
        X[2 * i] = [sx, sy, 1, 0, 0, 0]
        X[2 * i + 1] = [0, 0, 0, sx, sy, 1]
        y[2 * i] = dx
        y[2 * i + 1] = dy
    params, *_ = np.linalg.lstsq(X, y, rcond=None)
    M = np.array([[params[0], params[1], params[2]],
                  [params[3], params[4], params[5]]])
    predicted = np.array([
        [M[0, 0] * sx + M[0, 1] * sy + M[0, 2],
         M[1, 0] * sx + M[1, 1] * sy + M[1, 2]]
        for sx, sy in src_pts
    ])
    dists = np.linalg.norm(np.array(dst_pts) - predicted, axis=1)
    return M, float(dists.mean()), N


def apply_affine(M, qxy):
    x, y = qxy
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


per_hand_affine = {}
per_hand_residual = {}
per_hand_anchor_count = {}
per_hand_anchor_keys = {}
per_hand_kb_span = {}

for hand in ('Left', 'Right'):
    anchor_src, anchor_dst, anchor_keys = [], [], []
    for (key_norm, h), centroid in learned_centroids.items():
        if h != hand:
            continue
        if key_sample_counts[(key_norm, h)] < HIGH_FREQ_THRESHOLD:
            continue
        if key_norm not in QWERTY_PRIOR:
            continue
        anchor_src.append(QWERTY_PRIOR[key_norm])
        anchor_dst.append(centroid)
        anchor_keys.append(key_norm)

    M, residual, n_anchors = fit_affine(anchor_src, anchor_dst)

    if anchor_dst:
        kb_x_span = max(p[0] for p in anchor_dst) - min(p[0] for p in anchor_dst)
        kb_x_span = max(kb_x_span, 0.01)
    else:
        kb_x_span = 0.5

    per_hand_affine[hand] = M
    per_hand_residual[hand] = residual
    per_hand_anchor_count[hand] = n_anchors
    per_hand_anchor_keys[hand] = anchor_keys
    per_hand_kb_span[hand] = kb_x_span

    if M is not None:
        residual_frac = residual / kb_x_span
        print(f"\nAffine fit ({hand} hand): {n_anchors} anchors, "
              f"residual {residual:.4f} (~{100 * residual_frac:.1f}% of span).")
    else:
        print(f"\nAffine fit ({hand} hand): insufficient anchors ({n_anchors}).")

# Predicted positions per (key, hand)
predicted_positions = {}
for hand in ('Left', 'Right'):
    M = per_hand_affine.get(hand)
    if M is None:
        continue
    for k, qxy in QWERTY_PRIOR.items():
        predicted_positions[(k, hand)] = apply_affine(M, qxy)


def key_position(key_norm, hand):
    """Wrist-relative target position for a key, given the active hand."""
    direct = learned_centroids.get((key_norm, hand))
    if direct is not None and key_sample_counts[(key_norm, hand)] >= HIGH_FREQ_THRESHOLD:
        return direct
    pred = predicted_positions.get((key_norm, hand))
    if pred is not None:
        return pred
    if direct is not None:
        return direct
    return None


# ==========================================
# PASS 2 — refined assignment (wrist-relative space)
# ==========================================
print("\nPass 2: refined assignment in wrist-relative space...")

for evt in events:
    if evt['method'] in ('filtered', 'space_rule', 'no_landmark_data'):
        continue

    center_fi = evt['frame_index']
    hand = evt['hand']
    if hand is None or center_fi is None:
        evt['method'] = 'no_landmark_data'
        continue

    tips = press_frame_tips_wrist_relative(center_fi, hand)
    if tips is None:
        evt['method'] = 'no_landmark_data'
        continue

    key_norm = evt.get('_key_norm', evt['key'])
    target = key_position(key_norm, hand)

    if target is None:
        evt['finger'] = evt['p1_finger']
        evt['method'] = 'motion_p1_fallback'
        evt['_p2_source'] = 'p1_fallback'
        evt['_p2_target'] = None
        evt['_p2_distances'] = None
        continue

    dists = {name: math.hypot(xyz[0] - target[0], xyz[1] - target[1])
             for name, xyz in tips.items()}
    ranked = sorted(dists.items(), key=lambda kv: kv[1])
    p2_finger = ranked[0][0]

    evt['finger'] = p2_finger
    evt['method'] = 'motion_p2'
    evt['_p2_source'] = ('learned'
                         if key_sample_counts.get((key_norm, hand), 0) >= HIGH_FREQ_THRESHOLD
                         else ('predicted' if per_hand_affine.get(hand) is not None
                               else 'learned_sparse'))
    evt['_p2_target'] = target
    evt['_p2_distances'] = {k: round(v, 5) for k, v in dists.items()}

# ==========================================
# CANONICAL PER-KEY MAPPING
# ==========================================
per_key_finger_counts = defaultdict(Counter)
for evt in events:
    if evt.get('method') not in ('motion_p2', 'motion_p1_fallback'):
        continue
    if evt['hand'] is None or evt['finger'] is None:
        continue
    key_norm = evt.get('_key_norm', evt['key'])
    label = f"{evt['hand']} {evt['finger']}"
    per_key_finger_counts[key_norm][label] += 1

canonical = {}
for key_norm, counts in per_key_finger_counts.items():
    total = sum(counts.values())
    mode_label, mode_count = counts.most_common(1)[0]
    mode_hand, mode_finger = mode_label.split(' ', 1)
    canonical[key_norm] = {
        'hand': mode_hand,
        'finger': mode_finger,
        'total_presses': total,
        'distribution': dict(counts),
    }

for evt in events:
    if evt.get('method') not in ('motion_p2', 'motion_p1_fallback'):
        evt['matches_canonical'] = None
        continue
    key_norm = evt.get('_key_norm', evt['key'])
    canon = canonical.get(key_norm)
    if canon is None:
        evt['matches_canonical'] = None
    else:
        evt['matches_canonical'] = (evt['hand'] == canon['hand']
                                    and evt['finger'] == canon['finger'])

# ==========================================
# OUTPUT: events_v2_wrist.csv
# ==========================================
csv_fields = [
    'event_idx', 't_mono', 'key', 'frame_index',
    'hand', 'finger', 'method', 'matches_canonical',
    'p1_finger', 'p1_tip_x', 'p1_tip_y',
]

with open(events_csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    for evt in events:
        w.writerow({k: evt.get(k) for k in csv_fields})

# ==========================================
# OUTPUT: per-event debug JSONs
# ==========================================
for evt in events:
    if evt.get('method') not in ('motion_p2', 'motion_p1_fallback'):
        continue
    key_norm = evt.get('_key_norm', evt['key'])
    canon = canonical.get(key_norm, {})
    dbg = {
        'event_idx': evt['event_idx'],
        't_mono': evt['t_mono'],
        'key': evt['key'],
        'frame_index': evt['frame_index'],
        'coordinate_space': 'wrist_relative',
        'hand_assignment': {
            'chosen': evt['hand'],
            **(evt.get('_hand_diag') or {}),
        },
        'finger_assignment': {
            'pass1_finger': evt.get('p1_finger'),
            'pass2_finger': evt.get('finger'),
            'pass2_source': evt.get('_p2_source'),
            'learned_key_position_wrist_rel': evt.get('_p2_target'),
            'distances_from_key_wrist_rel': evt.get('_p2_distances'),
        },
        'canonical': {
            'finger': f"{canon.get('hand')} {canon.get('finger')}" if canon else None,
            'total_presses': canon.get('total_presses'),
        },
        'matches_canonical': evt.get('matches_canonical'),
    }
    safe_key = evt['key'] if (evt['key'] and evt['key'].isalnum()) \
        else evt['key'].encode('unicode_escape').decode() if evt['key'] else '_'
    fname = f"event_{evt['event_idx']:05d}_t{evt['t_mono']:.3f}_{safe_key[:8]}.json"
    with open(debug_dir / fname, 'w') as f:
        json.dump(dbg, f, indent=2)

# ==========================================
# OUTPUT: keyboard_layout_wrist.json
# ==========================================
layout_out = {
    'session_id': metadata.get('session_id'),
    'coordinate_space': 'wrist_relative',
    'high_freq_threshold': HIGH_FREQ_THRESHOLD,
    'learned_centroids': {f"{k}|{h}": list(v)
                          for (k, h), v in learned_centroids.items()},
    'sample_counts': {f"{k}|{h}": v
                      for (k, h), v in key_sample_counts.items()},
    'qwerty_prior': {k: list(v) for k, v in QWERTY_PRIOR.items()},
    'affine_transform': {
        hand: {
            'matrix_2x3': per_hand_affine[hand].tolist()
            if per_hand_affine[hand] is not None else None,
            'num_anchors': per_hand_anchor_count[hand],
            'anchor_keys': per_hand_anchor_keys[hand],
            'residual_mean_distance': per_hand_residual[hand],
            'residual_as_fraction_of_keyboard_span': (
                per_hand_residual[hand] / per_hand_kb_span[hand]
                if per_hand_residual[hand] is not None else None
            ),
        }
        for hand in ('Left', 'Right')
    },
    'predicted_positions': {f"{k}|{h}": list(v)
                            for (k, h), v in predicted_positions.items()},
}
with open(layout_path, 'w') as f:
    json.dump(layout_out, f, indent=2)

# ==========================================
# OUTPUT: finger_map_wrist.json
# ==========================================
with open(finger_map_path, 'w') as f:
    json.dump({'session_id': metadata.get('session_id'),
               'coordinate_space': 'wrist_relative',
               'canonical': canonical}, f, indent=2)

# ==========================================
# SUMMARY
# ==========================================
n_total = len(events)
n_fused = sum(1 for e in events if e.get('method') in ('motion_p2', 'motion_p1_fallback'))
n_filtered = sum(1 for e in events if e.get('method') == 'filtered')
n_space = sum(1 for e in events if e.get('method') == 'space_rule')
n_nolm = sum(1 for e in events if e.get('method') == 'no_landmark_data')

fused_events = [e for e in events if e.get('method') in ('motion_p2', 'motion_p1_fallback')]
n_matches = sum(1 for e in fused_events if e.get('matches_canonical'))
disagree_rate = (1 - n_matches / max(1, len(fused_events))) * 100

print(f"\n{'=' * 60}")
print(f"FUSION v2-WRIST COMPLETE — {metadata.get('session_id')}")
print(f"{'=' * 60}")
print(f"Coordinate space: wrist-relative (per hand)")
print(f"Total press events:         {n_total}")
print(f"  Motion-fused (pass 2):    {n_fused}")
print(f"  Spacebar (rule):          {n_space}")
print(f"  Filtered (modifiers):     {n_filtered}")
print(f"  No landmark data:         {n_nolm}")

if fused_events:
    finger_counts = Counter(f"{e['hand']} {e['finger']}" for e in fused_events
                            if e['hand'] and e['finger'])
    print(f"\nAssignment distribution:")
    max_c = max(finger_counts.values())
    for label, c in sorted(finger_counts.items(), key=lambda kv: -kv[1]):
        bar = '█' * int(c / max_c * 30)
        pct = c / len(fused_events) * 100
        print(f"  {label:20s} {c:>4} ({pct:5.1f}%)  {bar}")

print(f"\nPer-hand affine transforms:")
for hand in ('Left', 'Right'):
    M = per_hand_affine[hand]
    if M is not None:
        rf = per_hand_residual[hand] / per_hand_kb_span[hand]
        print(f"  {hand}: {per_hand_anchor_count[hand]} anchors, "
              f"residual {per_hand_residual[hand]:.4f} "
              f"(~{100 * rf:.1f}% of span)")
    else:
        print(f"  {hand}: insufficient anchors ({per_hand_anchor_count[hand]}).")

print(f"\nTyping consistency:")
print(f"  Events matching canonical:   {n_matches} / {len(fused_events)}")
print(f"  Disagreement rate:           {disagree_rate:.1f}%")

if canonical:
    inconsistent = []
    for k, info in canonical.items():
        if info['total_presses'] >= HIGH_FREQ_THRESHOLD:
            mode_count = max(info['distribution'].values())
            dom = mode_count / info['total_presses']
            inconsistent.append((k, dom, info))
    inconsistent.sort(key=lambda x: x[1])
    if inconsistent:
        print(f"\nLeast-consistent keys:")
        for k, dom, info in inconsistent[:5]:
            dist_str = ', '.join(f"{lbl}:{n}" for lbl, n in
                                 sorted(info['distribution'].items(),
                                        key=lambda kv: -kv[1]))
            print(f"  '{k}'  {info['total_presses']:>3} presses, "
                  f"mode={100 * dom:4.0f}%  →  {dist_str}")

print(f"\nOutputs:")
print(f"  events_v2_wrist.csv:          {events_csv_path}")
print(f"  events_v2_wrist_debug/:       {debug_dir} "
      f"({len(list(debug_dir.glob('*.json')))} files)")
print(f"  keyboard_layout_wrist.json:   {layout_path}")
print(f"  finger_map_wrist.json:        {finger_map_path}")