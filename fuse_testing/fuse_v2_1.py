"""
Phase 3 v2.1 — Trajectory-matching fusion.

Refinements over v2:
  * Finger selection by per-finger Y trajectory across a window, not
    single-frame spatial position.
  * Trajectory score is computed on RELATIVE Y — each fingertip's Y minus the
    hand's MCP-mean Y at the same frame — so the whole-hand rocking that all
    five fingers share is removed from the signal. The pressing finger's
    excursion is what's left after the rocking is subtracted.
  * Score is weighted by how close (in time) each finger's minimum-Y was to
    the keystroke timestamp. Pressing fingers minimise Y at the keystroke;
    neighbour-stroke motion in the window has its minimum elsewhere and is
    suppressed by the temporal weighting.
  * Sub-frame presses (no finger has a clear minimum near the keystroke)
    fall back to v2's spatial-nearest assignment, recorded with method tag.

Outputs (in session_dir/):
  events_v2_1.csv               One row per keystroke press, schema as v2.
  events_v2_1_debug/*.json      Per-event trajectory traces + assignments.
  keyboard_layout.json          Reused from v2 (overwritten with v2.1 anchors).
  finger_map.json               Canonical per-key finger map from v2.1 picks.

Usage:
    python fuse_v2_1.py sessions/<session_id>
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
FRAMES_BEFORE = 5            # for hand selection (Z-velocity sum window)
FRAMES_AFTER = 1
WRIST_SMOOTH_WINDOW = 5
HIGH_FREQ_THRESHOLD = 3

# Trajectory window — symmetric around the keystroke for finger selection
TRAJ_WINDOW_BEFORE = 3
TRAJ_WINDOW_AFTER = 3

# Temporal weighting: τ in seconds. Score includes exp(-|Δt| / τ) where Δt is
# the time between the finger's min-Y frame and the keystroke. τ = 33ms means
# a min-Y one frame away gets weight 0.37, two frames away 0.14.
TRAJ_TIMING_TAU = 0.040

# Below this excursion (normalized image units), the trajectory signal is
# considered absent and we fall back to spatial nearest-fingertip.
TRAJ_MIN_EXCURSION = 0.005

# ==========================================
# FILTERS / PRIORS (same as v2)
# ==========================================
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

# Fingertip + MCP landmark indices in MediaPipe's 21-point hand model.
FINGERTIPS = {4: 'thumb', 8: 'index', 12: 'middle', 16: 'ring', 20: 'pinky'}
WRIST = 0
MCP_INDICES = [5, 9, 13, 17]  # MCPs of index/middle/ring/pinky — anatomical
                              # reference that stays steady while fingertips move

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
})

QWERTY_FINGER_MAP = {}
for c in '1qaz':         QWERTY_FINGER_MAP[c] = ('Left',  'pinky')
for c in '2wsx':         QWERTY_FINGER_MAP[c] = ('Left',  'ring')
for c in '3edc':         QWERTY_FINGER_MAP[c] = ('Left',  'middle')
for c in '45rtfgvb':     QWERTY_FINGER_MAP[c] = ('Left',  'index')
for c in '67yhnum':      QWERTY_FINGER_MAP[c] = ('Right', 'index')
for c in '8ik,':         QWERTY_FINGER_MAP[c] = ('Right', 'middle')
for c in '9ol.':         QWERTY_FINGER_MAP[c] = ('Right', 'ring')
for c in "0p;/'[]\\-=":  QWERTY_FINGER_MAP[c] = ('Right', 'pinky')

# ==========================================
# INPUT
# ==========================================
if len(sys.argv) != 2:
    print("Usage: python fuse_v2_1.py <session_directory>")
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

events_csv_path = session_dir / "events_v2_1.csv"
debug_dir = session_dir / "events_v2_1_debug"
debug_dir.mkdir(exist_ok=True)
layout_path = session_dir / "keyboard_layout.json"
finger_map_path = session_dir / "finger_map.json"

# ==========================================
# LOAD DATA — including MCP landmarks for hand-centre reference
# ==========================================
print("Loading data...")

with open(metadata_path) as f:
    metadata = json.load(f)

frame_times = {}
with open(frame_times_path) as f:
    for row in csv.DictReader(f):
        frame_times[int(row['frame_index'])] = float(row['t_mono'])

frame_indices_sorted = sorted(frame_times.keys())
frame_ts_array = np.array([frame_times[i] for i in frame_indices_sorted])
frame_pos = {fi: i for i, fi in enumerate(frame_indices_sorted)}

# landmarks_by_frame[fi][hand] = {'wrist','tips','mcps','mcp_mean_y','mcp_mean_x'}
landmarks_by_frame = defaultdict(dict)
with open(landmarks_path) as f:
    for row in csv.DictReader(f):
        fi = int(row['frame_index'])
        hand = row['handedness']
        wrist = (float(row[f'x{WRIST}']), float(row[f'y{WRIST}']),
                 float(row[f'z{WRIST}']))
        tips = {
            name: (float(row[f'x{idx}']), float(row[f'y{idx}']),
                   float(row[f'z{idx}']))
            for idx, name in FINGERTIPS.items()
        }
        mcps = [(float(row[f'x{idx}']), float(row[f'y{idx}']),
                 float(row[f'z{idx}']))
                for idx in MCP_INDICES]
        mcp_mean_x = sum(p[0] for p in mcps) / len(mcps)
        mcp_mean_y = sum(p[1] for p in mcps) / len(mcps)
        landmarks_by_frame[fi][hand] = {
            'wrist': wrist,
            'tips': tips,
            'mcps': mcps,
            'mcp_mean_x': mcp_mean_x,
            'mcp_mean_y': mcp_mean_y,
        }

key_events = []
with open(keys_path) as f:
    for row in csv.DictReader(f):
        if row['event'] != 'press':
            continue
        key_events.append({'t_mono': float(row['t_mono']), 'key': row['key']})

print(f"  {len(frame_times)} frames, {len(key_events)} press events.")

# ==========================================
# SMOOTH WRIST Z PER HAND (for hand selection, same as v2)
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

def window_around(center_fi, before, after):
    pos = frame_pos.get(center_fi)
    if pos is None:
        return []
    lo = max(0, pos - before)
    hi = min(len(frame_indices_sorted), pos + after + 1)
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


# ==========================================
# TRAJECTORY SCORING — the v2.1 core
# ==========================================
def trajectory_press_signatures(t_keystroke, hand, window):
    """
    For each non-thumb fingertip on the active hand, compute a press signature:
      - relative Y trajectory: tip_y - mcp_mean_y across the window
      - excursion: max(rel_y) - min(rel_y)
      - time of minimum (most extended) frame
      - score = excursion * exp(-|t_min - t_keystroke| / TRAJ_TIMING_TAU)

    Thumb is excluded — its top-down landmarks are noisy enough that including
    it just creates false positives.

    Returns a dict {finger_name: {excursion, t_min, score, n_samples,
                                  rel_y_at_keystroke}}.
    Only fingers with ≥3 valid samples in the window are scored.
    """
    out = {}
    # gather (t, rel_y) per finger across the window
    for finger_name in ('index', 'middle', 'ring', 'pinky'):
        samples = []
        for fi in window:
            fd = landmarks_by_frame.get(fi, {}).get(hand)
            if fd is None:
                continue
            tip_y = fd['tips'][finger_name][1]
            ref_y = fd['mcp_mean_y']
            rel_y = tip_y - ref_y
            samples.append((frame_times[fi], rel_y, fi))
        if len(samples) < 3:
            continue
        ys = [s[1] for s in samples]
        ts = [s[0] for s in samples]
        excursion = max(ys) - min(ys)
        # We want the most-extended (= largest tip_y - mcp_y, since image-Y
        # increases downward and "extended toward keyboard" = larger Y)
        # Wait: tip_y - mcp_y is positive when tip is below MCP in image
        # coords. "Pressing forward onto a key" means tip moves to higher Y
        # (further down in image, or further forward into the keyboard area
        # depending on rotation). The press is the LOCAL MAX of rel_y.
        max_rel_idx = max(range(len(ys)), key=lambda i: ys[i])
        t_max = ts[max_rel_idx]
        dt = abs(t_max - t_keystroke)
        score = excursion * math.exp(-dt / TRAJ_TIMING_TAU)
        rel_y_at_keystroke = None
        # Find sample whose t is closest to t_keystroke
        nearest = min(range(len(samples)),
                      key=lambda i: abs(samples[i][0] - t_keystroke))
        rel_y_at_keystroke = samples[nearest][1]
        out[finger_name] = {
            'excursion': excursion,
            't_max_extension': t_max,
            'frame_max_extension': samples[max_rel_idx][2],
            'dt_from_keystroke': t_max - t_keystroke,
            'score': score,
            'n_samples': len(samples),
            'rel_y_at_keystroke': rel_y_at_keystroke,
        }
    return out


# ==========================================
# PASS 1 — trajectory-based per-event assignment
# Hand from summed Z; finger from trajectory score.  When the trajectory
# signal is below TRAJ_MIN_EXCURSION, fall back to QWERTY prior (so we still
# build a plausible centroid for the affine layout learning step).
# ==========================================
print("\nPass 1: trajectory-based finger selection...")

events = []
pass1_positions = defaultdict(list)
n_traj_picked = 0
n_qwerty_fallback = 0

for idx, ke in enumerate(key_events):
    key = ke['key']
    t = ke['t_mono']
    evt = {
        'event_idx': idx, 't_mono': t, 'key': key,
        'frame_index': None, 'hand': None, 'finger': None,
        'method': None, 'p1_finger': None, 'p1_method': None,
        'p1_tip_x': None, 'p1_tip_y': None,
        'p1_score': None, 'p1_excursion': None, 'p1_dt': None,
        'p1_signatures': None,
    }

    # Modifier filter
    if key.lower() in FILTERED_KEYS:
        evt['method'] = 'filtered'
        events.append(evt)
        continue

    # Spacebar rule
    if key in SPACE_KEYS or key.lower() in SPACE_KEYS:
        evt['hand'] = 'unknown'
        evt['finger'] = 'thumb'
        evt['method'] = 'space_rule'
        evt['frame_index'] = nearest_frame(t)
        events.append(evt)
        continue

    center_fi = nearest_frame(t)
    evt['frame_index'] = center_fi
    key_norm = key.lower() if len(key) == 1 and key.isalpha() else key
    evt['_key_norm'] = key_norm

    # Hand selection: QWERTY prior is the *measured* reliable signal here
    # (v2 hit 100% hand-accuracy on the 19 labeled events using this). The
    # Z-velocity sum is recorded as a diagnostic but NOT used to pick the
    # hand, because at fast typing speeds the resting hand can produce a
    # larger Z signal than the actively-pressing hand if the press is small.
    expected = QWERTY_FINGER_MAP.get(key_norm)
    hand_window = window_around(center_fi, FRAMES_BEFORE, FRAMES_AFTER)
    hand_diag = {
        'left_summed_z': round(summed_z_velocity(hand_window, 'Left'), 6),
        'right_summed_z': round(summed_z_velocity(hand_window, 'Right'), 6),
    }
    if expected is not None:
        hand = expected[0]
        hand_diag['source'] = 'qwerty_prior'
    else:
        # Non-prior key (backspace, enter, etc.) — fall back to Z-velocity.
        chosen, _ = select_hand(hand_window)
        hand = chosen
        hand_diag['source'] = 'z_velocity_fallback'
    if hand is None:
        evt['method'] = 'no_landmark_data'
        events.append(evt)
        continue
    evt['hand'] = hand
    evt['_hand_diag'] = hand_diag

    # Trajectory finger selection
    traj_window = window_around(center_fi, TRAJ_WINDOW_BEFORE, TRAJ_WINDOW_AFTER)
    sigs = trajectory_press_signatures(t, hand, traj_window)
    evt['p1_signatures'] = {f: {k: round(v, 6) if isinstance(v, float) else v
                                 for k, v in s.items()
                                 if k != 'rel_y_at_keystroke'}
                            for f, s in sigs.items()}

    use_trajectory = False
    if sigs:
        ranked = sorted(sigs.items(), key=lambda kv: -kv[1]['score'])
        top_finger, top_sig = ranked[0]
        if top_sig['excursion'] >= TRAJ_MIN_EXCURSION:
            use_trajectory = True
            evt['p1_finger'] = top_finger
            evt['p1_method'] = 'trajectory'
            evt['p1_score'] = round(top_sig['score'], 6)
            evt['p1_excursion'] = round(top_sig['excursion'], 6)
            evt['p1_dt'] = round(top_sig['dt_from_keystroke'], 4)
            n_traj_picked += 1

    if not use_trajectory:
        # Trajectory signal too weak — fall back to QWERTY prior so layout
        # learning still has a plausible centroid.
        expected = QWERTY_FINGER_MAP.get(key_norm)
        if expected is not None:
            evt['p1_finger'] = expected[1]
            # We honour Z-velocity's hand pick over the prior's; we only use
            # the prior's *finger*.
            evt['p1_method'] = 'qwerty_prior_fallback'
            n_qwerty_fallback += 1
        else:
            # Last resort: largest-Y fingertip on the chosen hand, excluding
            # thumb.
            fd = landmarks_by_frame.get(center_fi, {}).get(hand)
            if fd is None:
                evt['method'] = 'no_landmark_data'
                events.append(evt)
                continue
            non_thumb = {n: xyz for n, xyz in fd['tips'].items() if n != 'thumb'}
            evt['p1_finger'] = max(non_thumb.items(),
                                    key=lambda kv: kv[1][1])[0]
            evt['p1_method'] = 'largest_y_fallback'

    # Record the finger's (x, y) at the press frame for layout learning
    fd = landmarks_by_frame.get(center_fi, {}).get(hand)
    if fd is None:
        evt['method'] = 'no_landmark_data'
        events.append(evt)
        continue
    px, py, _ = fd['tips'][evt['p1_finger']]
    evt['p1_tip_x'] = px
    evt['p1_tip_y'] = py
    pass1_positions[key_norm].append((px, py))
    events.append(evt)

print(f"  Trajectory: {n_traj_picked}   QWERTY fallback: {n_qwerty_fallback}   "
      f"largest-Y fallback: {sum(1 for e in events if e.get('p1_method') == 'largest_y_fallback')}")

# ==========================================
# LEARN LAYOUT + AFFINE FIT (same as v2)
# ==========================================
learned_centroids = {}
key_sample_counts = {}
for key_norm, positions in pass1_positions.items():
    arr = np.array(positions)
    learned_centroids[key_norm] = (float(arr[:, 0].mean()),
                                    float(arr[:, 1].mean()))
    key_sample_counts[key_norm] = len(positions)

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
    pred = np.array([
        [M[0, 0] * sx + M[0, 1] * sy + M[0, 2],
         M[1, 0] * sx + M[1, 1] * sy + M[1, 2]]
        for sx, sy in src_pts
    ])
    dists = np.linalg.norm(np.array(dst_pts) - pred, axis=1)
    return M, float(dists.mean()), N

def apply_affine(M, qxy):
    x, y = qxy
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))

anchor_src, anchor_dst, anchor_keys = [], [], []
for key_norm, centroid in learned_centroids.items():
    if key_sample_counts[key_norm] >= HIGH_FREQ_THRESHOLD and key_norm in QWERTY_PRIOR:
        anchor_src.append(QWERTY_PRIOR[key_norm])
        anchor_dst.append(centroid)
        anchor_keys.append(key_norm)

M, residual, n_anchors = fit_affine(anchor_src, anchor_dst)
kb_x_span = max(0.01,
                max((p[0] for p in anchor_dst), default=0.5)
                - min((p[0] for p in anchor_dst), default=0.0))
residual_frac = residual / kb_x_span if M is not None else None

predicted_positions = {}
if M is not None:
    for k, qxy in QWERTY_PRIOR.items():
        predicted_positions[k] = apply_affine(M, qxy)

def key_position(key_norm):
    if (key_norm in learned_centroids
            and key_sample_counts[key_norm] >= HIGH_FREQ_THRESHOLD):
        return learned_centroids[key_norm]
    if key_norm in predicted_positions:
        return predicted_positions[key_norm]
    if key_norm in learned_centroids:
        return learned_centroids[key_norm]
    return None

# ==========================================
# PASS 2 — refine via spatial nearest-fingertip-to-learned-key-position.
# This becomes the v2.1 *fallback path* for events where trajectory was
# inconclusive, AND a sanity refinement on top of trajectory picks.
# ==========================================
print("\nPass 2: spatial refinement / fallback...")

for evt in events:
    if evt.get('method') in ('filtered', 'space_rule', 'no_landmark_data'):
        continue
    center_fi = evt['frame_index']
    hand = evt['hand']
    if hand is None or center_fi is None:
        evt['method'] = 'no_landmark_data'
        continue
    fd = landmarks_by_frame.get(center_fi, {}).get(hand)
    if fd is None:
        evt['method'] = 'no_landmark_data'
        continue

    key_norm = evt.get('_key_norm', evt['key'])
    target = key_position(key_norm)

    if evt['p1_method'] == 'trajectory':
        # Trust trajectory — that's the v2.1 win. Spatial-nearest is recorded
        # as a comparison signal but not used unless we find a need to
        # second-guess.
        evt['finger'] = evt['p1_finger']
        evt['method'] = 'motion_trajectory'
    else:
        # Fallback: spatial-nearest, exactly v2's pass-2 method.
        if target is None:
            evt['finger'] = evt['p1_finger']
            evt['method'] = 'motion_qwerty_fallback'
        else:
            tips_xy = {n: (xyz[0], xyz[1]) for n, xyz in fd['tips'].items()}
            dists = {n: math.hypot(xy[0] - target[0], xy[1] - target[1])
                     for n, xy in tips_xy.items()}
            evt['finger'] = min(dists.items(), key=lambda kv: kv[1])[0]
            evt['method'] = 'motion_spatial_fallback'
            evt['_p2_distances'] = {k: round(v, 5) for k, v in dists.items()}
            evt['_p2_target'] = target

# ==========================================
# CANONICAL PER-KEY MAP + DISAGREEMENT
# ==========================================
per_key_finger_counts = defaultdict(Counter)
for evt in events:
    if evt.get('method') not in ('motion_trajectory',
                                  'motion_spatial_fallback',
                                  'motion_qwerty_fallback'):
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
    if evt.get('method') not in ('motion_trajectory',
                                  'motion_spatial_fallback',
                                  'motion_qwerty_fallback'):
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
# OUTPUT — events.csv, debug JSONs, layout JSON, finger map
# ==========================================
csv_fields = ['event_idx', 't_mono', 'key', 'frame_index',
              'hand', 'finger', 'method', 'matches_canonical',
              'p1_finger', 'p1_method', 'p1_score', 'p1_excursion', 'p1_dt',
              'p1_tip_x', 'p1_tip_y']

with open(events_csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    for evt in events:
        w.writerow({k: evt.get(k) for k in csv_fields})

for evt in events:
    if evt.get('method') not in ('motion_trajectory',
                                  'motion_spatial_fallback',
                                  'motion_qwerty_fallback'):
        continue
    key_norm = evt.get('_key_norm', evt['key'])
    canon = canonical.get(key_norm, {})
    dbg = {
        'event_idx': evt['event_idx'],
        't_mono': evt['t_mono'],
        'key': evt['key'],
        'frame_index': evt['frame_index'],
        'hand_assignment': {
            'chosen': evt['hand'],
            **(evt.get('_hand_diag') or {}),
        },
        'finger_assignment': {
            'method': evt['method'],
            'pass1_method': evt['p1_method'],
            'pass1_finger': evt['p1_finger'],
            'final_finger': evt['finger'],
            'trajectory_signatures': evt.get('p1_signatures'),
            'spatial_target': evt.get('_p2_target'),
            'spatial_distances': evt.get('_p2_distances'),
        },
        'canonical': {
            'finger': f"{canon.get('hand')} {canon.get('finger')}" if canon else None,
            'total_presses': canon.get('total_presses'),
        },
        'matches_canonical': evt.get('matches_canonical'),
    }
    safe_key = (evt['key'] if evt['key'] and evt['key'].isalnum()
                else evt['key'].encode('unicode_escape').decode()
                if evt['key'] else '_')
    fname = f"event_{evt['event_idx']:05d}_t{evt['t_mono']:.3f}_{safe_key[:8]}.json"
    with open(debug_dir / fname, 'w') as f:
        json.dump(dbg, f, indent=2, default=str)

with open(layout_path, 'w') as f:
    json.dump({
        'session_id': metadata.get('session_id'),
        'fusion_version': 'v2.1',
        'high_freq_threshold': HIGH_FREQ_THRESHOLD,
        'learned_centroids': {k: list(v) for k, v in learned_centroids.items()},
        'sample_counts': key_sample_counts,
        'qwerty_prior': {k: list(v) for k, v in QWERTY_PRIOR.items()},
        'affine_transform': {
            'matrix_2x3': M.tolist() if M is not None else None,
            'num_anchors': n_anchors,
            'anchor_keys': anchor_keys,
            'residual_mean_distance': residual,
            'residual_as_fraction_of_keyboard_span': residual_frac,
        },
        'predicted_positions': {k: list(v) for k, v in predicted_positions.items()},
    }, f, indent=2)

with open(finger_map_path, 'w') as f:
    json.dump({'session_id': metadata.get('session_id'),
               'fusion_version': 'v2.1',
               'canonical': canonical}, f, indent=2)

# ==========================================
# SUMMARY
# ==========================================
n_total = len(events)
fused_events = [e for e in events
                if e.get('method') in ('motion_trajectory',
                                       'motion_spatial_fallback',
                                       'motion_qwerty_fallback')]
n_traj = sum(1 for e in fused_events if e['method'] == 'motion_trajectory')
n_spatial = sum(1 for e in fused_events if e['method'] == 'motion_spatial_fallback')
n_qfall = sum(1 for e in fused_events if e['method'] == 'motion_qwerty_fallback')
n_filtered = sum(1 for e in events if e.get('method') == 'filtered')
n_space = sum(1 for e in events if e.get('method') == 'space_rule')
n_nolm = sum(1 for e in events if e.get('method') == 'no_landmark_data')
n_match = sum(1 for e in fused_events if e.get('matches_canonical'))
disagree_rate = (1 - n_match / max(1, len(fused_events))) * 100

print(f"\n{'=' * 60}")
print(f"FUSION v2.1 COMPLETE — {metadata.get('session_id')}")
print(f"{'=' * 60}")
print(f"Total press events:                    {n_total}")
print(f"  Motion-fused:                        {len(fused_events)}")
print(f"    via trajectory matching:           {n_traj}")
print(f"    via spatial fallback (low excur.): {n_spatial}")
print(f"    via QWERTY fallback (no anchor):   {n_qfall}")
print(f"  Spacebar (rule):                     {n_space}")
print(f"  Filtered (modifiers):                {n_filtered}")
print(f"  No landmark data:                    {n_nolm}")

if fused_events:
    finger_counts = Counter(f"{e['hand']} {e['finger']}" for e in fused_events
                            if e['hand'] and e['finger'])
    print(f"\nAssignment distribution:")
    if finger_counts:
        max_c = max(finger_counts.values())
        for label, c in sorted(finger_counts.items(), key=lambda kv: -kv[1]):
            bar = '█' * int(c / max_c * 30)
            pct = c / len(fused_events) * 100
            print(f"  {label:20s} {c:>4} ({pct:5.1f}%)  {bar}")

print(f"\nAffine transform:")
if M is not None:
    print(f"  Anchor keys:           {n_anchors}")
    print(f"  Residual mean dist:    {residual:.4f}")
    print(f"  Residual as % of span: {100 * residual_frac:.1f}%  "
          f"({'OK' if residual_frac < 0.08 else 'HIGH'})")
else:
    print(f"  Not enough anchors ({n_anchors}).")

print(f"\nTyping consistency:")
print(f"  Events matching canonical: {n_match} / {len(fused_events)}")
print(f"  Disagreement rate:         {disagree_rate:.1f}%")

# Score against existing labels.csv if present
labels_path = session_dir / 'labels.csv'
if labels_path.exists():
    with open(labels_path) as f:
        labels = [r for r in csv.DictReader(f) if r['true_hand'] not in ('SKIP', '')]
    if labels:
        ev_by_idx = {int(e['event_idx']): e for e in events}
        n = len(labels)
        h_ok = sum(1 for l in labels
                   if (ev_by_idx.get(int(l['event_idx'])) or {}).get('hand')
                       == l['true_hand'])
        f_ok = sum(1 for l in labels
                   if (ev_by_idx.get(int(l['event_idx'])) or {}).get('hand')
                       == l['true_hand']
                   and (ev_by_idx.get(int(l['event_idx'])) or {}).get('finger')
                       == l['true_finger'])
        print(f"\nAccuracy vs labels.csv ({n} ground-truth events):")
        print(f"  Hand correct:  {h_ok}/{n}  ({100*h_ok/n:.1f}%)")
        print(f"  Full correct:  {f_ok}/{n}  ({100*f_ok/n:.1f}%)")

print(f"\nOutputs:")
print(f"  events_v2_1.csv:        {events_csv_path}")
print(f"  events_v2_1_debug/:     {debug_dir}")
print(f"  keyboard_layout.json:   {layout_path}")
print(f"  finger_map.json:        {finger_map_path}")
