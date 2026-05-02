"""
Ground-truth labeling tool for fusion accuracy measurement.

Samples N events from a session, displays the press-frame from
session_overlay.mp4 (skeleton already drawn), and lets you mark the correct
pressing finger with a single keystroke. At the end, compares v1, v2, and
v2-wrist (if available) assignments against your labels and prints accuracy.

Existing labels in labels.csv are preserved across runs. To re-label specific
events, delete those rows from labels.csv manually.

Key bindings while labeling (mnemonic: keyboard layout):
    LEFT HAND             RIGHT HAND
    q = pinky             y = thumb
    w = ring              u = index
    e = middle            i = middle
    r = index             o = ring
    t = thumb             p = pinky

    LEFT / RIGHT arrow (or , / .)   step backward / forward one frame
    SPACE  skip (ambiguous)
    ESC    quit early (progress is saved)
    b      go back one event

Labels persist to <session>/labels.csv and re-running will resume where you
left off. The accuracy summary always re-computes against ALL existing labels,
so you can re-run anytime after re-running fuse scripts to get fresh numbers.

Usage:
    python label.py sessions/20260424_121126
    python label.py sessions/20260424_121126 --n 100 --seed 7
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

# ==========================================
# CONFIG
# ==========================================
DEFAULT_N = 80
DEFAULT_SEED = 42
DISPLAY_SCALE = 1.6  # how much to upscale the video frame for labeling

# Keystroke → (hand, finger) for labeling input
KEY_BINDINGS = {
    ord('q'): ('Left',  'pinky'),
    ord('w'): ('Left',  'ring'),
    ord('e'): ('Left',  'middle'),
    ord('r'): ('Left',  'index'),
    ord('t'): ('Left',  'thumb'),
    ord('y'): ('Right', 'thumb'),
    ord('u'): ('Right', 'index'),
    ord('i'): ('Right', 'middle'),
    ord('o'): ('Right', 'ring'),
    ord('p'): ('Right', 'pinky'),
}
ACTION_SKIP = ord(' ')
ACTION_QUIT = 27         # ESC
ACTION_BACK = ord('b')

# Arrow key codes (OpenCV on macOS)
ARROW_LEFT  = 81
ARROW_RIGHT = 83

LABEL_FIELDS = ['event_idx', 'key', 't_mono', 'frame_index',
                'true_hand', 'true_finger',
                'v1_hand', 'v1_finger',
                'v2_hand', 'v2_finger',
                'v2w_hand', 'v2w_finger']


# ==========================================
# ARG PARSING
# ==========================================
parser = argparse.ArgumentParser(description="Ground-truth labeling tool.")
parser.add_argument('session_dir', type=Path)
parser.add_argument('--n', type=int, default=DEFAULT_N,
                    help=f"Number of events to sample (default {DEFAULT_N}).")
parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                    help=f"Random seed for sampling (default {DEFAULT_SEED}).")
args = parser.parse_args()

session_dir = args.session_dir
if not session_dir.is_dir():
    print(f"ERROR: {session_dir} is not a directory.")
    sys.exit(1)

events_v1_path = session_dir / 'events.csv'
events_v2_path = session_dir / 'events_v2.csv'
events_v2w_path = session_dir / 'events_v2_wrist.csv'
overlay_path   = session_dir / 'session_overlay.mp4'
raw_video_path = session_dir / 'session.mp4'
labels_path    = session_dir / 'labels.csv'

if not events_v2_path.exists():
    print(f"ERROR: {events_v2_path} not found. Run fuse_v2.py first.")
    sys.exit(1)

have_v1 = events_v1_path.exists()
have_v2w = events_v2w_path.exists()
if not have_v1:
    print("NOTE: events.csv (v1) not found — v1 accuracy won't be computed.")
if not have_v2w:
    print("NOTE: events_v2_wrist.csv not found — v2-wrist accuracy won't be computed.")

video_path = overlay_path if overlay_path.exists() else raw_video_path
if not video_path.exists():
    print(f"ERROR: No video found in {session_dir}.")
    sys.exit(1)
print(f"Using video: {video_path.name}")


# ==========================================
# LOAD EVENTS
# ==========================================
with open(events_v2_path) as f:
    events_v2 = list(csv.DictReader(f))
v2_by_idx = {int(e['event_idx']): e for e in events_v2}

v1_by_idx = {}
if have_v1:
    with open(events_v1_path) as f:
        for e in csv.DictReader(f):
            v1_by_idx[int(e['event_idx'])] = e

v2w_by_idx = {}
if have_v2w:
    with open(events_v2w_path) as f:
        for e in csv.DictReader(f):
            v2w_by_idx[int(e['event_idx'])] = e


# ==========================================
# SAMPLE CANDIDATES
# ==========================================
candidates = [int(e['event_idx']) for e in events_v2
              if e['method'] in ('motion_p2', 'motion_p1_fallback', 'motion_fusion')]

if not candidates:
    print("ERROR: no fused events to label.")
    sys.exit(1)

random.seed(args.seed)
n_to_sample = min(args.n, len(candidates))
sampled_idxs = sorted(random.sample(candidates, n_to_sample))
print(f"Sampled {len(sampled_idxs)} of {len(candidates)} fused events for labeling.")


# ==========================================
# LOAD EXISTING LABELS (resume + augment with new fusion versions)
# ==========================================
existing_labels = {}
if labels_path.exists():
    with open(labels_path) as f:
        for row in csv.DictReader(f):
            # Backfill missing v2w columns for old labels
            row.setdefault('v2w_hand', '')
            row.setdefault('v2w_finger', '')
            existing_labels[int(row['event_idx'])] = row
    print(f"Found {len(existing_labels)} existing labels in {labels_path.name}.")

# Refresh fusion-version columns for ALL existing labels using current
# events_v2.csv / events_v2_wrist.csv, so accuracy reflects current fusion outputs.
for eidx, row in existing_labels.items():
    v2 = v2_by_idx.get(eidx)
    if v2:
        row['v2_hand'] = v2.get('hand', '')
        row['v2_finger'] = v2.get('finger', '')
    v2w = v2w_by_idx.get(eidx)
    if v2w:
        row['v2w_hand'] = v2w.get('hand', '')
        row['v2w_finger'] = v2w.get('finger', '')
    v1 = v1_by_idx.get(eidx)
    if v1:
        row['v1_hand'] = v1.get('hand', '')
        row['v1_finger'] = v1.get('finger', '')

labels = dict(existing_labels)


def save_labels():
    """Write labels dict to labels.csv atomically."""
    tmp = labels_path.with_suffix('.csv.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for eidx in sorted(labels.keys(), key=int):
            row = {k: labels[eidx].get(k, '') for k in LABEL_FIELDS}
            w.writerow(row)
    tmp.replace(labels_path)


# Save once up front so the v2/v2w/v1 columns get refreshed even if the user
# doesn't label any new events this run.
save_labels()


# ==========================================
# RENDER THE LABELING FRAME
# ==========================================
def build_display(frame, i, total, event_key, v1_label, v2_label, v2w_label, offset):
    display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                         interpolation=cv2.INTER_CUBIC)
    h, w = display.shape[:2]

    # Dark banner at top — taller now to fit v2-wrist row
    banner_h = 120 if v2w_label is not None else 95
    cv2.rectangle(display, (0, 0), (w, banner_h), (0, 0, 0), -1)

    cv2.putText(display, f"[{i+1}/{total}]  Key: '{event_key}'",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(display, f"v1: {v1_label}    v2: {v2_label}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2,
                cv2.LINE_AA)
    if v2w_label is not None:
        cv2.putText(display, f"v2-wrist: {v2w_label}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2,
                    cv2.LINE_AA)

    # Frame offset indicator
    offset_color = (0, 220, 255) if offset != 0 else (160, 160, 160)
    cv2.putText(display, f"frame offset: {offset:+d}",
                (w - 200, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                offset_color, 1, cv2.LINE_AA)

    # Bottom legend
    cv2.rectangle(display, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.putText(display, "LEFT:  q=pinky  w=ring  e=middle  r=index  t=thumb",
                (10, h - 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200),
                1, cv2.LINE_AA)
    cv2.putText(display, "RIGHT: y=thumb  u=index  i=middle  o=ring  p=pinky",
                (10, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255),
                1, cv2.LINE_AA)
    cv2.putText(display, "← → (or , .)  scrub frames     SPACE=skip   b=back   ESC=quit",
                (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                cv2.LINE_AA)
    cv2.putText(display, "(saves progress)",
                (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1,
                cv2.LINE_AA)
    return display


def fmt_assignment(hand, finger):
    if not hand or not finger:
        return "?"
    return f"{hand} {finger}"


# ==========================================
# MAIN LABELING LOOP
# ==========================================
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"ERROR: cannot open {video_path}")
    sys.exit(1)

window_name = 'Label — scrub with arrows, press a key to assign finger'
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

quit_requested = False
i = 0
new_labels_this_run = 0

while i < len(sampled_idxs):
    eidx = sampled_idxs[i]

    # Already labeled? Skip past — we don't re-prompt for events that already
    # have ground truth, the accuracy comparison already handles them via the
    # refreshed fusion columns.
    if eidx in labels and labels[eidx].get('true_hand') and labels[eidx]['true_hand'] != '':
        i += 1
        continue

    v2 = v2_by_idx[eidx]
    v1 = v1_by_idx.get(eidx)
    v2w = v2w_by_idx.get(eidx)

    frame_idx = int(v2['frame_index']) if v2['frame_index'] else None
    if frame_idx is None:
        print(f"  Event {eidx}: no frame_index, skipping.")
        i += 1
        continue

    v2_label = fmt_assignment(v2.get('hand'), v2.get('finger'))
    v1_label = fmt_assignment(v1.get('hand') if v1 else None,
                              v1.get('finger') if v1 else None)
    v2w_label = fmt_assignment(v2w.get('hand'), v2w.get('finger')) if v2w else None

    # ── Per-event frame-scrubbing loop ────────────────────────────────────────
    display_offset = 0
    action_taken = None

    while True:
        display_fi = max(0, frame_idx + display_offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, display_fi)
        ret, frame = cap.read()
        if not ret:
            display_offset -= 1
            continue

        display = build_display(frame, i, len(sampled_idxs), v2['key'],
                                v1_label, v2_label, v2w_label, display_offset)
        cv2.imshow(window_name, display)

        k = cv2.waitKey(0) & 0xFF

        if k == ARROW_LEFT or k == ord(','):
            display_offset -= 1
            continue
        if k == ARROW_RIGHT or k == ord('.'):
            display_offset += 1
            continue

        if k == ACTION_QUIT:
            action_taken = 'quit'
            break
        if k == ACTION_SKIP:
            action_taken = 'skip'
            break
        if k == ACTION_BACK:
            action_taken = 'back'
            break
        if k in KEY_BINDINGS:
            action_taken = KEY_BINDINGS[k]
            break

    # ── Handle action ─────────────────────────────────────────────────────────
    if action_taken == 'quit':
        quit_requested = True
        break

    if action_taken == 'back':
        if i > 0:
            prev_eidx = sampled_idxs[i - 1]
            labels.pop(prev_eidx, None)
            save_labels()
            i -= 1
            print(f"  Going back to event {prev_eidx} for re-labeling.")
        else:
            print("  Already at the first event; can't go back.")
        continue

    if action_taken == 'skip':
        print(f"  [{i+1}/{len(sampled_idxs)}] event {eidx} ('{v2['key']}') skipped.")
        labels[eidx] = {
            'event_idx': eidx,
            'key': v2['key'],
            't_mono': v2['t_mono'],
            'frame_index': frame_idx,
            'true_hand': 'SKIP',
            'true_finger': 'SKIP',
            'v1_hand': v1.get('hand', '') if v1 else '',
            'v1_finger': v1.get('finger', '') if v1 else '',
            'v2_hand': v2.get('hand', ''),
            'v2_finger': v2.get('finger', ''),
            'v2w_hand': v2w.get('hand', '') if v2w else '',
            'v2w_finger': v2w.get('finger', '') if v2w else '',
        }
        save_labels()
        i += 1
        continue

    # Real label
    hand, finger = action_taken
    labels[eidx] = {
        'event_idx': eidx,
        'key': v2['key'],
        't_mono': v2['t_mono'],
        'frame_index': frame_idx,
        'true_hand': hand,
        'true_finger': finger,
        'v1_hand': v1.get('hand', '') if v1 else '',
        'v1_finger': v1.get('finger', '') if v1 else '',
        'v2_hand': v2.get('hand', ''),
        'v2_finger': v2.get('finger', ''),
        'v2w_hand': v2w.get('hand', '') if v2w else '',
        'v2w_finger': v2w.get('finger', '') if v2w else '',
    }
    save_labels()
    new_labels_this_run += 1
    v1_ok = '✓' if (v1 and v1.get('hand') == hand and v1.get('finger') == finger) else '✗'
    v2_ok = '✓' if (v2.get('hand') == hand and v2.get('finger') == finger) else '✗'
    v2w_ok_str = ''
    if v2w:
        v2w_ok = '✓' if (v2w.get('hand') == hand and v2w.get('finger') == finger) else '✗'
        v2w_ok_str = f"   v2w {v2w_ok} ({v2w_label})"
    print(f"  [{i+1}/{len(sampled_idxs)}] event {eidx} ('{v2['key']}') -> "
          f"{hand} {finger}    v1 {v1_ok} ({v1_label})   v2 {v2_ok} ({v2_label})"
          f"{v2w_ok_str}  [offset {display_offset:+d}]")
    i += 1

cap.release()
cv2.destroyAllWindows()


# ==========================================
# ACCURACY SUMMARY (always computed over all labels in labels.csv)
# ==========================================
valid = [l for l in labels.values()
         if l.get('true_hand') and l['true_hand'] not in ('', 'SKIP')]

print("\n" + "=" * 60)
if quit_requested:
    print("LABELING SESSION EXITED EARLY (progress saved)")
else:
    print("LABELING COMPLETE")
print("=" * 60)
print(f"Existing labels available:  {len(valid)}")
print(f"New labels this run:        {new_labels_this_run}")
print(f"Skipped this run:           {sum(1 for l in labels.values() if l.get('true_hand') == 'SKIP')}")

if not valid:
    print("\nNo labels to score against. Run again to label some events.")
    sys.exit(0)


def score(rows, hand_key, finger_key):
    n = len(rows)
    hand_ok = sum(1 for l in rows if l.get(hand_key) == l['true_hand'])
    full_ok = sum(1 for l in rows
                  if l.get(hand_key) == l['true_hand']
                  and l.get(finger_key) == l['true_finger'])
    return n, hand_ok, full_ok


def print_per_key_errors(rows, hand_key, finger_key, label):
    by_key = defaultdict(list)
    for l in rows:
        by_key[l['key']].append(l)
    misassigned = []
    for k, krows in by_key.items():
        wrong = sum(1 for l in krows
                    if not (l.get(hand_key) == l['true_hand']
                            and l.get(finger_key) == l['true_finger']))
        if wrong:
            misassigned.append((k, wrong, len(krows), krows))
    if misassigned:
        misassigned.sort(key=lambda x: (-x[1], -x[2]))
        print(f"\n{label} per-key errors (key, wrong/total, typical mis-assignment):")
        for k, wrong, total, krows in misassigned[:10]:
            miss_pairs = [f"{l.get(hand_key)} {l.get(finger_key)}"
                          for l in krows
                          if not (l.get(hand_key) == l['true_hand']
                                  and l.get(finger_key) == l['true_finger'])]
            common = max(set(miss_pairs), key=miss_pairs.count) if miss_pairs else ''
            truth_common = max(
                set(f"{l['true_hand']} {l['true_finger']}" for l in krows),
                key=lambda s: sum(1 for l in krows
                                  if f"{l['true_hand']} {l['true_finger']}" == s))
            print(f"  '{k}'  {wrong}/{total} wrong   truth={truth_common}   {label} often said={common}")


v1_rows = [l for l in valid if l.get('v1_hand')]
v2_rows = [l for l in valid if l.get('v2_hand')]
v2w_rows = [l for l in valid if l.get('v2w_hand')]

print(f"\n{'v1 (Z-velocity)' if v1_rows else 'v1: not available'}")
if v1_rows:
    n1, v1_hand, v1_full = score(v1_rows, 'v1_hand', 'v1_finger')
    print(f"  Hand correct:  {v1_hand}/{n1}  ({100*v1_hand/n1:.1f}%)")
    print(f"  Full correct:  {v1_full}/{n1}  ({100*v1_full/n1:.1f}%)")

print(f"\nv2 (spatial + learned layout, absolute coords)")
n, v2_hand, v2_full = score(v2_rows, 'v2_hand', 'v2_finger')
print(f"  Hand correct:  {v2_hand}/{n}  ({100*v2_hand/n:.1f}%)")
print(f"  Full correct:  {v2_full}/{n}  ({100*v2_full/n:.1f}%)")

if v2w_rows:
    print(f"\nv2-wrist (spatial + learned layout, wrist-relative coords)")
    nw, v2w_hand, v2w_full = score(v2w_rows, 'v2w_hand', 'v2w_finger')
    print(f"  Hand correct:  {v2w_hand}/{nw}  ({100*v2w_hand/nw:.1f}%)")
    print(f"  Full correct:  {v2w_full}/{nw}  ({100*v2w_full/nw:.1f}%)")

    # Direct delta vs v2 on the same events
    common_evts = set(int(l['event_idx']) for l in v2_rows) & set(int(l['event_idx']) for l in v2w_rows)
    common_v2 = [l for l in v2_rows if int(l['event_idx']) in common_evts]
    common_v2w = [l for l in v2w_rows if int(l['event_idx']) in common_evts]
    if common_v2 and common_v2w:
        _, _, v2_full_c = score(common_v2, 'v2_hand', 'v2_finger')
        _, _, v2w_full_c = score(common_v2w, 'v2w_hand', 'v2w_finger')
        delta = v2w_full_c - v2_full_c
        print(f"  Delta vs v2 on same {len(common_evts)} events: "
              f"{'+' if delta >= 0 else ''}{delta} events "
              f"({100*delta/len(common_evts):+.1f}%)")

# Per-key error breakdowns
print_per_key_errors(v2_rows, 'v2_hand', 'v2_finger', 'v2')
if v2w_rows:
    print_per_key_errors(v2w_rows, 'v2w_hand', 'v2w_finger', 'v2-wrist')

print(f"\nLabels saved to: {labels_path}")