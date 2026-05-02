"""
Phase 4 — Session analysis.

Reads a session's events_v2.csv (finger-tagged keystrokes) and keystrokes.csv
(raw timing data) and produces:

  report.md              Human-readable analysis report (the portfolio artifact).
  analysis.json          All raw numbers, structured for Phase 5 integration.
  bigram_stats.csv       Per-bigram timing statistics.
  finger_stats.csv       Per-finger workload statistics.
  key_consistency.csv    Per-key finger-assignment consistency.

The three analytical layers:

  1. Timing fundamentals: IKI distribution, press durations, WPM estimate.
  2. Bigram analysis: ranked transitions, same-finger vs cross-hand breakdown.
  3. Finger-level analysis: workload, per-key consistency, biomechanical patterns.

This is the project's actual deliverable layer. Fusion produced the dataset;
analysis surfaces the findings.

Usage:
    python analyze.py sessions/20260424_121126
"""

import csv
import json
import sys
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
SLOW_BIGRAM_PERCENTILE = 0.75   # bigrams above this percentile of session IKI flagged slow
MIN_BIGRAM_OCCURRENCES = 2      # don't flag single-occurrence bigrams as slow
CHARS_PER_WORD = 5              # standard WPM definition
INCONSISTENCY_THRESHOLD = 0.80  # if dominant finger < this fraction, flag as inconsistent
MIN_KEY_PRESSES_FOR_CONSISTENCY = 3  # need this many to talk about consistency

# Letter keys for bigram analysis. Spacebar/backspace/punctuation/modifiers
# excluded — they introduce timing artifacts that aren't motor-transition bigrams.
LETTER_KEY_PATTERN = lambda k: len(k) == 1 and k.isalpha()


# ==========================================
# INPUT
# ==========================================
if len(sys.argv) != 2:
    print("Usage: python analyze.py <session_directory>")
    sys.exit(1)

session_dir = Path(sys.argv[1])
events_path = session_dir / 'events_v2.csv'
keys_path = session_dir / 'keystrokes.csv'
metadata_path = session_dir / 'metadata.json'

for p in [events_path, keys_path, metadata_path]:
    if not p.exists():
        print(f"ERROR: Missing required file: {p}")
        sys.exit(1)

report_path = session_dir / 'report.md'
analysis_json_path = session_dir / 'analysis.json'
bigram_csv_path = session_dir / 'bigram_stats.csv'
finger_csv_path = session_dir / 'finger_stats.csv'
consistency_csv_path = session_dir / 'key_consistency.csv'


# ==========================================
# LOAD DATA
# ==========================================
with open(metadata_path) as f:
    metadata = json.load(f)

events = []
with open(events_path) as f:
    for row in csv.DictReader(f):
        events.append({
            'event_idx': int(row['event_idx']),
            't_mono': float(row['t_mono']),
            'key': row['key'],
            'hand': row.get('hand') or '',
            'finger': row.get('finger') or '',
            'method': row.get('method') or '',
        })

# Press AND release events for press-duration computation
all_keys = []
with open(keys_path) as f:
    for row in csv.DictReader(f):
        all_keys.append({
            't_mono': float(row['t_mono']),
            'event': row['event'],
            'key': row['key'],
        })


# ==========================================
# LAYER 1 — TIMING FUNDAMENTALS
# ==========================================
print("Computing timing fundamentals...")

press_events = [e for e in all_keys if e['event'] == 'press']
release_events = [e for e in all_keys if e['event'] == 'release']

# Inter-keystroke intervals (consecutive press-to-press)
ikis = []
for i in range(1, len(press_events)):
    dt_ms = (press_events[i]['t_mono'] - press_events[i-1]['t_mono']) * 1000
    ikis.append(dt_ms)

iki_median = statistics.median(ikis) if ikis else 0
iki_mean = statistics.mean(ikis) if ikis else 0
iki_stdev = statistics.stdev(ikis) if len(ikis) > 1 else 0

# Press durations: match each press with its corresponding release
# Match by key identity, in order — first press of key X matches first release of key X
press_durations = []  # ms
press_durations_by_key = defaultdict(list)
release_queue_by_key = defaultdict(list)
for r in release_events:
    release_queue_by_key[r['key']].append(r['t_mono'])

for p in press_events:
    queue = release_queue_by_key.get(p['key'], [])
    matching_release = None
    for i, rt in enumerate(queue):
        if rt > p['t_mono']:
            matching_release = rt
            queue.pop(i)
            break
    if matching_release is not None:
        dur_ms = (matching_release - p['t_mono']) * 1000
        press_durations.append(dur_ms)
        press_durations_by_key[p['key']].append(dur_ms)

press_dur_median = statistics.median(press_durations) if press_durations else 0

# ----- WPM measures (multi-dimensional, see methodology in report) -----
# Different definitions answer different questions:
#   raw_wpm    = chars/5 / wall-clock — Monkeytype-equivalent, includes pauses
#   active_wpm = chars/5 / (wall-clock - pause-time) — pauses removed
#   burst_wpm  = 60000 / (median_iki * 5) — instantaneous in-flow speed
#   peak_wpm   = 60000 / (p25_iki * 5) — sprint speed on fastest quartile
# Pauses are defined adaptively as IKI > PAUSE_IKI_MULTIPLIER × the session's
# median IKI. 5× is the default, which on a 130ms median means anything above
# ~650ms is treated as a pause.
PAUSE_IKI_MULTIPLIER = 5

if press_events:
    typing_time_seconds = press_events[-1]['t_mono'] - press_events[0]['t_mono']
    char_count = sum(1 for p in press_events if LETTER_KEY_PATTERN(p['key']))
    n_words = char_count / CHARS_PER_WORD

    # Adaptive pause detection
    pause_threshold_ms = (PAUSE_IKI_MULTIPLIER * iki_median
                          if iki_median > 0 else float('inf'))
    pause_ikis = [i for i in ikis if i > pause_threshold_ms]
    pause_count = len(pause_ikis)
    pause_total_seconds = sum(pause_ikis) / 1000
    active_typing_seconds = max(typing_time_seconds - pause_total_seconds, 0.0)

    # P25 IKI for Peak WPM (top quartile of fastest intervals)
    if len(ikis) >= 4:
        iki_p25 = statistics.quantiles(ikis, n=4)[0]
    else:
        iki_p25 = iki_median

    raw_wpm = n_words / max(typing_time_seconds / 60, 1e-6)
    active_wpm = (n_words / max(active_typing_seconds / 60, 1e-6)
                  if active_typing_seconds > 0 else raw_wpm)
    burst_wpm = 60000 / (iki_median * 5) if iki_median > 0 else 0
    peak_wpm = 60000 / (iki_p25 * 5) if iki_p25 > 0 else 0
else:
    typing_time_seconds = 0
    char_count = 0
    pause_threshold_ms = 0
    pause_count = 0
    pause_total_seconds = 0
    active_typing_seconds = 0
    iki_p25 = 0
    raw_wpm = active_wpm = burst_wpm = peak_wpm = 0

# Headline WPM: Burst is the most defensible cross-session-comparable measure
wpm_estimate = burst_wpm

print(f"  {len(press_events)} presses, {len(release_events)} releases, "
      f"{len(press_durations)} matched pairs.")
print(f"  Median IKI: {iki_median:.1f}ms  "
      f"WPM (burst/active/raw/peak): "
      f"{burst_wpm:.0f} / {active_wpm:.0f} / {raw_wpm:.0f} / {peak_wpm:.0f}")
if pause_count > 0:
    print(f"  Pauses (>{pause_threshold_ms:.0f}ms): {pause_count}, "
          f"totalling {pause_total_seconds:.1f}s of {typing_time_seconds:.1f}s wall-time")


# ==========================================
# LAYER 2 — BIGRAM ANALYSIS
# ==========================================
print("\nComputing bigram statistics...")

# Bigrams = consecutive letter-key presses.
# IKIs above the session pause threshold are excluded — those represent
# thinking gaps between bigrams, not actual motor transitions, and they
# pollute the slowest-bigram findings (a 30s pause shouldn't make 'vi'
# look like a slow bigram).
letter_presses = [p for p in press_events if LETTER_KEY_PATTERN(p['key'])]
bigrams_raw = []  # (key1, key2, iki_ms)
n_bigrams_dropped_as_pauses = 0
for i in range(1, len(letter_presses)):
    k1 = letter_presses[i-1]['key'].lower()
    k2 = letter_presses[i]['key'].lower()
    dt_ms = (letter_presses[i]['t_mono'] - letter_presses[i-1]['t_mono']) * 1000
    if pause_threshold_ms and dt_ms > pause_threshold_ms:
        n_bigrams_dropped_as_pauses += 1
        continue
    bigrams_raw.append((k1, k2, dt_ms))
if n_bigrams_dropped_as_pauses:
    print(f"  Dropped {n_bigrams_dropped_as_pauses} bigram-spanning pauses "
          f"(IKI > {pause_threshold_ms:.0f}ms)")

# Aggregate per bigram
bigram_groups = defaultdict(list)  # (k1, k2) -> [iki_ms, ...]
for k1, k2, dt in bigrams_raw:
    bigram_groups[(k1, k2)].append(dt)

bigram_stats = []
for (k1, k2), times in bigram_groups.items():
    bigram_stats.append({
        'bigram': k1 + k2,
        'k1': k1,
        'k2': k2,
        'count': len(times),
        'median_ms': statistics.median(times),
        'mean_ms': statistics.mean(times),
        'min_ms': min(times),
        'max_ms': max(times),
    })

# Sort by median IKI descending — slowest first
bigram_stats.sort(key=lambda b: -b['median_ms'])

# Compute slow-bigram threshold from session-wide IKI distribution
all_bigram_medians = [b['median_ms'] for b in bigram_stats if b['count'] >= MIN_BIGRAM_OCCURRENCES]
if all_bigram_medians:
    sorted_medians = sorted(all_bigram_medians)
    pct_idx = int(len(sorted_medians) * SLOW_BIGRAM_PERCENTILE)
    slow_threshold = sorted_medians[min(pct_idx, len(sorted_medians) - 1)]
else:
    slow_threshold = float('inf')

# Tag slow bigrams
for b in bigram_stats:
    b['is_slow'] = (b['median_ms'] >= slow_threshold
                     and b['count'] >= MIN_BIGRAM_OCCURRENCES)

# Build event lookup so we can categorize bigrams by finger assignment
event_by_t = {round(e['t_mono'], 6): e for e in events}

def find_event_for_press(press_t_mono):
    """Match a keystroke press to its event_v2 entry by timestamp."""
    key = round(press_t_mono, 6)
    return event_by_t.get(key)

# Categorize bigrams by hand/finger relationship
bigram_categories = {
    'same_finger': 0,
    'same_hand_diff_finger': 0,
    'cross_hand': 0,
    'unknown': 0,
}
bigram_category_times = defaultdict(list)

for i in range(1, len(letter_presses)):
    e1 = find_event_for_press(letter_presses[i-1]['t_mono'])
    e2 = find_event_for_press(letter_presses[i]['t_mono'])
    dt_ms = (letter_presses[i]['t_mono'] - letter_presses[i-1]['t_mono']) * 1000

    if not e1 or not e2 or not e1['hand'] or not e2['hand'] \
            or not e1['finger'] or not e2['finger']:
        bigram_categories['unknown'] += 1
        bigram_category_times['unknown'].append(dt_ms)
        continue

    if e1['hand'] != e2['hand']:
        cat = 'cross_hand'
    elif e1['finger'] == e2['finger']:
        cat = 'same_finger'
    else:
        cat = 'same_hand_diff_finger'
    bigram_categories[cat] += 1
    bigram_category_times[cat].append(dt_ms)

bigram_category_medians = {
    cat: statistics.median(times) if times else 0
    for cat, times in bigram_category_times.items()
}

print(f"  {len(bigram_stats)} unique bigrams from {len(bigrams_raw)} occurrences.")
print(f"  Slow threshold (>={int(SLOW_BIGRAM_PERCENTILE*100)}th pct): "
      f"{slow_threshold:.1f}ms  ({sum(1 for b in bigram_stats if b['is_slow'])} flagged)")


# ==========================================
# LAYER 3 — FINGER & CONSISTENCY ANALYSIS
# ==========================================
print("\nComputing finger workload and consistency...")

# Per-finger workload (only motion-fused events with valid finger labels)
fused_events = [e for e in events
                if e['method'] in ('motion_p2', 'motion_p1_fallback')
                and e['hand'] and e['finger']]

finger_counts = Counter()
finger_keys = defaultdict(Counter)  # 'Left index' -> {key: count}
for e in fused_events:
    label = f"{e['hand']} {e['finger']}"
    finger_counts[label] += 1
    finger_keys[label][e['key'].lower() if len(e['key']) == 1 else e['key']] += 1

total_fused = sum(finger_counts.values())

finger_stats = []
for label, count in sorted(finger_counts.items(), key=lambda kv: -kv[1]):
    pct = count / max(total_fused, 1) * 100
    top_keys = finger_keys[label].most_common(5)
    finger_stats.append({
        'finger': label,
        'count': count,
        'pct': pct,
        'top_keys': ', '.join(f"{k}({c})" for k, c in top_keys),
    })

# Per-key finger consistency
key_finger_dist = defaultdict(Counter)
for e in fused_events:
    k = e['key'].lower() if len(e['key']) == 1 else e['key']
    label = f"{e['hand']} {e['finger']}"
    key_finger_dist[k][label] += 1

consistency_stats = []
for k, dist in key_finger_dist.items():
    total = sum(dist.values())
    if total < MIN_KEY_PRESSES_FOR_CONSISTENCY:
        continue
    mode_label, mode_count = dist.most_common(1)[0]
    dominance = mode_count / total
    consistency_stats.append({
        'key': k,
        'total_presses': total,
        'dominant_finger': mode_label,
        'dominance': dominance,
        'is_inconsistent': dominance < INCONSISTENCY_THRESHOLD,
        'distribution': dict(dist),
    })

consistency_stats.sort(key=lambda c: c['dominance'])  # least consistent first


# ==========================================
# LAYER 4 — ERROR PATTERNS & ERGONOMIC DEVIATIONS
# ==========================================
print("\nComputing error patterns and ergonomic deviations...")

# ----- Backspace error backtracing -----
# For each backspace press, identify the bigram that caused the error and the
# bigram you intended. Walks backward through press_events to find the deleted
# character and what came before it; walks forward (skipping subsequent
# backspaces) to find what you typed instead.

backspace_errors = []  # list of {'preceding', 'wrong', 'intended', 't_mono'}
press_keys = [p['key'] for p in press_events]

for i, p in enumerate(press_events):
    if p['key'] != 'backspace':
        continue
    # Find the most recent non-backspace press before this one (the deleted character)
    j = i - 1
    while j >= 0 and press_keys[j] == 'backspace':
        j -= 1
    if j < 0:
        continue  # backspace at session start, nothing to backtrace
    deleted_char = press_keys[j]
    # The character before the deleted one — completes the error-bigram
    preceding = press_keys[j - 1] if j >= 1 else None

    # Walk forward, skipping any further backspaces, to find the intended char
    k = i + 1
    while k < len(press_keys) and press_keys[k] == 'backspace':
        k += 1
    intended = press_keys[k] if k < len(press_keys) else None

    # Skip cases where intended is itself a non-letter (modifier, etc.) — focuses
    # the analysis on character-level errors, not navigation/correction patterns
    if not (LETTER_KEY_PATTERN(deleted_char) or LETTER_KEY_PATTERN(intended or '')):
        continue

    backspace_errors.append({
        'preceding': preceding.lower() if preceding and len(preceding) == 1 else preceding,
        'wrong': deleted_char.lower() if len(deleted_char) == 1 else deleted_char,
        'intended': (intended.lower() if intended and len(intended) == 1 else intended),
        't_mono': p['t_mono'],
    })

# Aggregate error-bigram patterns
error_bigram_counts = Counter()
intended_by_error = defaultdict(Counter)
for err in backspace_errors:
    if err['preceding'] and err['wrong'] and LETTER_KEY_PATTERN(err['preceding']) \
            and LETTER_KEY_PATTERN(err['wrong']):
        error_pair = f"{err['preceding']}{err['wrong']}"
        error_bigram_counts[error_pair] += 1
        if err['intended'] and LETTER_KEY_PATTERN(err['intended']):
            intended_pair = f"{err['preceding']}{err['intended']}"
            intended_by_error[error_pair][intended_pair] += 1

top_error_bigrams = []
for pair, count in error_bigram_counts.most_common(15):
    intended_top = intended_by_error[pair].most_common(1)
    intended_str = intended_top[0][0] if intended_top else None
    top_error_bigrams.append({
        'error_bigram': pair,
        'count': count,
        'most_common_intended': intended_str,
    })

# ----- Abandonment trigger detection (tab+enter or enter+tab within 500ms) -----
ABANDONMENT_WINDOW_MS = 500
abandonment_events = []  # list of {'trigger_key', 'preceding_key', 'preceding_bigram', 't_mono'}

for i in range(len(press_events) - 1):
    k1 = press_events[i]['key']
    k2 = press_events[i + 1]['key']
    dt_ms = (press_events[i + 1]['t_mono'] - press_events[i]['t_mono']) * 1000

    is_pattern = ((k1 == 'tab' and k2 == 'enter')
                   or (k1 == 'enter' and k2 == 'tab'))
    if not is_pattern or dt_ms > ABANDONMENT_WINDOW_MS:
        continue

    # The trigger is whichever came first; what we want is the keystroke
    # that came BEFORE that trigger
    trigger_idx = i
    if trigger_idx == 0:
        continue  # no preceding keystroke

    preceding_press = press_events[trigger_idx - 1]
    preceding_key = preceding_press['key']

    # Bigram leading up to the abandonment
    if trigger_idx >= 2:
        pre_pre = press_events[trigger_idx - 2]['key']
        if LETTER_KEY_PATTERN(pre_pre) and LETTER_KEY_PATTERN(preceding_key):
            preceding_bigram = f"{pre_pre.lower()}{preceding_key.lower()}"
        else:
            preceding_bigram = None
    else:
        preceding_bigram = None

    abandonment_events.append({
        'trigger_pattern': f"{k1}+{k2}",
        'preceding_key': preceding_key,
        'preceding_bigram': preceding_bigram,
        't_mono': press_events[i]['t_mono'],
    })

# Aggregate abandonment triggers
abandonment_bigram_counts = Counter()
abandonment_key_counts = Counter()
for ab in abandonment_events:
    abandonment_key_counts[ab['preceding_key']] += 1
    if ab['preceding_bigram']:
        abandonment_bigram_counts[ab['preceding_bigram']] += 1

# ----- Ergonomic deviations (canonical fingerings vs your habits) -----
# Standard touch-typing convention. The system has been calibrated to your
# personal habits for accurate fusion; this layer surfaces where habits
# diverge from convention.
CANONICAL_FINGERINGS = {}
for c in '1qaz':       CANONICAL_FINGERINGS[c] = ('Left',  'pinky')
for c in '2wsx':       CANONICAL_FINGERINGS[c] = ('Left',  'ring')
for c in '3edc':       CANONICAL_FINGERINGS[c] = ('Left',  'middle')
for c in '45rtfgvb':   CANONICAL_FINGERINGS[c] = ('Left',  'index')
for c in '67yhnujm':   CANONICAL_FINGERINGS[c] = ('Right', 'index')
for c in '8ik,':       CANONICAL_FINGERINGS[c] = ('Right', 'middle')
for c in '9ol.':       CANONICAL_FINGERINGS[c] = ('Right', 'ring')
for c in "0p;/'[]\\-=": CANONICAL_FINGERINGS[c] = ('Right', 'pinky')
CANONICAL_FINGERINGS['backspace'] = ('Right', 'pinky')
CANONICAL_FINGERINGS['enter'] = ('Right', 'pinky')
CANONICAL_FINGERINGS['return'] = ('Right', 'pinky')
CANONICAL_FINGERINGS['tab'] = ('Left', 'pinky')

ergonomic_deviations = []
for c in consistency_stats:
    key = c['key']
    canonical = CANONICAL_FINGERINGS.get(key)
    if canonical is None:
        continue
    actual_label = c['dominant_finger']  # e.g. "Right ring"
    canonical_label = f"{canonical[0]} {canonical[1]}"
    if actual_label != canonical_label:
        ergonomic_deviations.append({
            'key': key,
            'your_dominant_finger': actual_label,
            'canonical_finger': canonical_label,
            'total_presses': c['total_presses'],
            'dominance': round(c['dominance'], 3),
        })

# Sort by usage frequency — high-usage deviations matter more
ergonomic_deviations.sort(key=lambda d: -d['total_presses'])

print(f"  Backspace errors traced:    {len(backspace_errors)}")
print(f"  Abandonment events:         {len(abandonment_events)}")
print(f"  Ergonomic deviations:       {len(ergonomic_deviations)}")


# ==========================================
# WRITE CSV ARTIFACTS
# ==========================================
print("\nWriting CSV artifacts...")

with open(bigram_csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['bigram', 'k1', 'k2', 'count',
                                       'median_ms', 'mean_ms', 'min_ms',
                                       'max_ms', 'is_slow'])
    w.writeheader()
    for b in bigram_stats:
        w.writerow(b)

with open(finger_csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['finger', 'count', 'pct', 'top_keys'])
    w.writeheader()
    for s in finger_stats:
        w.writerow({k: (round(v, 1) if k == 'pct' else v) for k, v in s.items()})

with open(consistency_csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['key', 'total_presses', 'dominant_finger',
                                       'dominance', 'is_inconsistent',
                                       'distribution'])
    w.writeheader()
    for c in consistency_stats:
        row = dict(c)
        row['dominance'] = round(row['dominance'], 3)
        row['distribution'] = json.dumps(row['distribution'])
        w.writerow(row)


# ==========================================
# WRITE JSON SIDECAR (for Phase 5)
# ==========================================
analysis_json = {
    'session_id': metadata.get('session_id'),
    'duration_seconds': metadata.get('duration_wall_seconds'),
    'timing': {
        'total_presses': len(press_events),
        'total_letter_presses': char_count,
        'typing_time_seconds': round(typing_time_seconds, 2),
        'active_typing_seconds': round(active_typing_seconds, 2),
        'pause_count': pause_count,
        'pause_total_seconds': round(pause_total_seconds, 2),
        'pause_threshold_ms': round(pause_threshold_ms, 1),
        'wpm': {
            'burst':  round(burst_wpm, 1),
            'active': round(active_wpm, 1),
            'raw':    round(raw_wpm, 1),
            'peak':   round(peak_wpm, 1),
        },
        # legacy alias kept for older Phase 5 code
        'wpm_estimate': round(burst_wpm, 1),
        'iki_median_ms': round(iki_median, 1),
        'iki_mean_ms': round(iki_mean, 1),
        'iki_stdev_ms': round(iki_stdev, 1),
        'iki_p25_ms': round(iki_p25, 1),
        'press_duration_median_ms': round(press_dur_median, 1),
    },
    'bigrams': {
        'total_unique': len(bigram_stats),
        'total_occurrences': len(bigrams_raw),
        'slow_threshold_ms': round(slow_threshold, 1),
        'slow_count': sum(1 for b in bigram_stats if b['is_slow']),
        'top_slow': [
            {k: (round(v, 1) if isinstance(v, float) else v) for k, v in b.items()}
            for b in bigram_stats[:10]
        ],
        'category_counts': bigram_categories,
        'category_medians_ms': {k: round(v, 1) for k, v in bigram_category_medians.items()},
    },
    'finger_workload': finger_stats,
    'key_consistency': [
        {**c, 'dominance': round(c['dominance'], 3)}
        for c in consistency_stats
    ],
    'error_patterns': {
        'backspace_count': len(backspace_errors),
        'top_error_bigrams': top_error_bigrams,
        'abandonment_count': len(abandonment_events),
        'abandonment_trigger_keys': dict(abandonment_key_counts.most_common(10)),
        'abandonment_trigger_bigrams': dict(abandonment_bigram_counts.most_common(10)),
    },
    'ergonomic_deviations': ergonomic_deviations,
}

with open(analysis_json_path, 'w') as f:
    json.dump(analysis_json, f, indent=2)


# ==========================================
# WRITE MARKDOWN REPORT
# ==========================================
print("\nWriting Markdown report...")

lines = []
lines.append(f"# Typing Analysis — Session `{metadata.get('session_id')}`")
lines.append("")
lines.append(f"_Recorded {metadata.get('start_time_iso', 'unknown')}, "
             f"duration {metadata.get('duration_wall_seconds', 0):.1f}s._")
lines.append("")

# --- Headline ---
lines.append("## Summary")
lines.append("")
lines.append(f"- **WPM (Burst — in-flow motor speed):** **{burst_wpm:.1f}**")
lines.append(f"- **WPM (Active — pauses excluded):** {active_wpm:.1f}")
lines.append(f"- **WPM (Raw — Monkeytype-equivalent):** {raw_wpm:.1f}")
lines.append(f"- **WPM (Peak — top-quartile sprint):** {peak_wpm:.1f}")
if pause_count > 0:
    lines.append(f"- **Pauses detected:** {pause_count} "
                 f"(total {pause_total_seconds:.1f}s of "
                 f"{typing_time_seconds:.1f}s wall-clock; "
                 f"threshold {pause_threshold_ms:.0f}ms)")
lines.append(f"- **Letter keystrokes:** {char_count}")
lines.append(f"- **Median inter-keystroke interval:** {iki_median:.1f}ms "
             f"(p25 {iki_p25:.0f}ms)")
lines.append(f"- **Median key press duration:** {press_dur_median:.1f}ms")
lines.append(f"- **Unique bigrams typed:** {len(bigram_stats)} (across {len(bigrams_raw)} occurrences)")
lines.append("")
lines.append("> **WPM definitions.** _Burst_ = `60000 / (median_iki × 5)`, "
             "the motor speed when actively typing. _Active_ removes pause time "
             f"(IKI > {PAUSE_IKI_MULTIPLIER}× session median) from the denominator. "
             "_Raw_ is the Monkeytype-equivalent over total wall-clock. _Peak_ "
             "uses the fastest 25% of intervals — sprint capability on bigram "
             "rolls. Burst is the most defensible cross-session-comparable measure.")
lines.append("")

# --- Findings: pulled-up actionable observations ---
lines.append("## Key Findings")
lines.append("")

findings_written = 0

# Finding 1: slowest bigrams with sample counts
notable_slow = [b for b in bigram_stats if b['is_slow']][:5]
if notable_slow:
    findings_written += 1
    lines.append(f"### {findings_written}. Slowest bigrams")
    lines.append("")
    lines.append(f"Your slowest transitions (median IKI in top 25% of bigrams, "
                 f"≥{MIN_BIGRAM_OCCURRENCES} occurrences):")
    lines.append("")
    lines.append("| Bigram | Median IKI (ms) | Occurrences |")
    lines.append("|--------|----------------:|------------:|")
    for b in notable_slow:
        lines.append(f"| `{b['bigram']}` | {b['median_ms']:.1f} | {b['count']} |")
    lines.append("")

# Finding 2: same-finger bigrams as a category
sf_count = bigram_categories['same_finger']
sh_count = bigram_categories['same_hand_diff_finger']
ch_count = bigram_categories['cross_hand']
if sf_count + sh_count + ch_count > 0:
    findings_written += 1
    lines.append(f"### {findings_written}. Bigram timing by finger relationship")
    lines.append("")
    lines.append("| Category | Count | Median IKI (ms) |")
    lines.append("|----------|------:|----------------:|")
    cat_order = [('same_finger', 'Same finger (same finger presses both keys)'),
                 ('same_hand_diff_finger', 'Same hand, different finger'),
                 ('cross_hand', 'Cross-hand (left then right or vice versa)')]
    for cat_key, cat_label in cat_order:
        c = bigram_categories[cat_key]
        m = bigram_category_medians.get(cat_key, 0)
        lines.append(f"| {cat_label} | {c} | {m:.1f} |")
    lines.append("")

    sf_med = bigram_category_medians.get('same_finger', 0)
    ch_med = bigram_category_medians.get('cross_hand', 0)
    if sf_med > 0 and ch_med > 0:
        ratio = sf_med / ch_med
        if ratio > 1.3:
            lines.append(f"_Same-finger transitions are {ratio:.1f}× slower than "
                         f"cross-hand transitions — this is a meaningful biomechanical "
                         f"bottleneck and the kind of pattern targeted retraining can "
                         f"address._")
            lines.append("")

# Finding 3: finger consistency
inconsistent = [c for c in consistency_stats if c['is_inconsistent']]
if inconsistent:
    findings_written += 1
    lines.append(f"### {findings_written}. Inconsistent finger usage")
    lines.append("")
    lines.append(f"Keys where you used different fingers across the session "
                 f"(dominant finger covers <{int(INCONSISTENCY_THRESHOLD*100)}% of presses):")
    lines.append("")
    lines.append("| Key | Presses | Distribution |")
    lines.append("|-----|--------:|--------------|")
    for c in inconsistent[:10]:
        dist_str = ', '.join(f"{lbl} ({n})" for lbl, n in
                              sorted(c['distribution'].items(), key=lambda kv: -kv[1]))
        lines.append(f"| `{c['key']}` | {c['total_presses']} | {dist_str} |")
    lines.append("")
    lines.append(f"_Inconsistency may indicate undertrained finger choreography. "
                 f"Picking one finger per key and drilling it builds reliable "
                 f"muscle memory._")
    lines.append("")

# --- Detailed sections ---
lines.append("## Finger Workload")
lines.append("")
lines.append("| Finger | Keystrokes | % of total | Top keys |")
lines.append("|--------|-----------:|-----------:|----------|")
for s in finger_stats:
    lines.append(f"| {s['finger']} | {s['count']} | {s['pct']:.1f}% | {s['top_keys']} |")
lines.append("")

lines.append("## All Bigrams (slowest first)")
lines.append("")
lines.append("| Bigram | Count | Median IKI (ms) | Slow? |")
lines.append("|--------|------:|----------------:|:------|")
for b in bigram_stats[:30]:  # top 30 slowest
    slow_marker = '⚠️' if b['is_slow'] else ''
    lines.append(f"| `{b['bigram']}` | {b['count']} | {b['median_ms']:.1f} | {slow_marker} |")
lines.append("")
if len(bigram_stats) > 30:
    lines.append(f"_…{len(bigram_stats) - 30} more bigrams in `bigram_stats.csv`._")
    lines.append("")

# --- Error patterns: backspace backtrace ---
if top_error_bigrams:
    lines.append("## Error Patterns (Backspace Backtrace)")
    lines.append("")
    lines.append(f"Total backspace events analyzed: **{len(backspace_errors)}**")
    lines.append("")
    lines.append("For each backspace press, the system traces backward to identify "
                 "what character was deleted, what character preceded it (the "
                 "error-bigram), and what was typed instead afterward (the intended "
                 "bigram). Bigrams that frequently trigger corrections reveal "
                 "specific motor-coordination weak spots.")
    lines.append("")
    lines.append("| Error Bigram | Count | Most Common Intended |")
    lines.append("|--------------|------:|----------------------|")
    for err in top_error_bigrams[:10]:
        intended = err['most_common_intended'] or '?'
        lines.append(f"| `{err['error_bigram']}` | {err['count']} | `{intended}` |")
    lines.append("")
    lines.append("_Patterns where you typed `xy` and corrected to `xz` "
                 "indicate that the right hand finger that should hit `z` "
                 "is being beaten to the punch by an adjacent finger reaching "
                 "for `y`. Drilling the intended bigram explicitly retrains "
                 "the finger choreography._")
    lines.append("")

# --- Abandonment triggers ---
if abandonment_events:
    lines.append("## Abandonment Triggers (Tab+Enter Restarts)")
    lines.append("")
    lines.append(f"Detected restart-pattern events (tab+enter or enter+tab "
                 f"within {ABANDONMENT_WINDOW_MS}ms): **{len(abandonment_events)}**")
    lines.append("")
    lines.append("Each abandonment is preceded by a 'tilt-point' keystroke — the "
                 "key that broke your rhythm enough to make you give up. "
                 "Bigrams that recur as tilt-points across multiple abandonments "
                 "are the highest-priority retraining targets — they don't just "
                 "slow you down, they break your focus.")
    lines.append("")
    if abandonment_bigram_counts:
        lines.append("**Top tilt-point bigrams (immediately before abandonment):**")
        lines.append("")
        lines.append("| Bigram | Times triggered restart |")
        lines.append("|--------|------------------------:|")
        for bg, count in abandonment_bigram_counts.most_common(10):
            lines.append(f"| `{bg}` | {count} |")
        lines.append("")
    if abandonment_key_counts:
        lines.append("**Top tilt-point keys (last keystroke before restart):**")
        lines.append("")
        lines.append("| Key | Times triggered restart |")
        lines.append("|-----|------------------------:|")
        for k, count in abandonment_key_counts.most_common(10):
            lines.append(f"| `{k}` | {count} |")
        lines.append("")

# --- Ergonomic deviations ---
if ergonomic_deviations:
    lines.append("## Ergonomic Deviations from Standard Touch Typing")
    lines.append("")
    lines.append("The fusion system measures *what you actually do*. This section "
                 "compares your habits against standard touch-typing convention. "
                 "Deviations are not necessarily wrong — your habits may be optimal "
                 "for you. But if you're not at your target WPM, retraining the "
                 "deviated keys to canonical fingerings is a tractable lever.")
    lines.append("")
    lines.append("| Key | Your dominant finger | Canonical finger | Presses |")
    lines.append("|-----|----------------------|------------------|--------:|")
    for d in ergonomic_deviations[:15]:
        lines.append(f"| `{d['key']}` | {d['your_dominant_finger']} "
                     f"| {d['canonical_finger']} | {d['total_presses']} |")
    lines.append("")
    high_use = [d for d in ergonomic_deviations if d['total_presses'] >= 5]
    if high_use:
        lines.append(f"_{len(high_use)} of these deviations occur on keys "
                     f"pressed 5+ times — meaning the alternate fingering is a "
                     f"settled habit, not a one-off. These are the deviations "
                     f"most likely to be costing speed._")
        lines.append("")

# --- Methodology notes ---
lines.append("## Methodology")
lines.append("")
lines.append("- Finger assignments come from the v2 fusion script "
             "(spatial-position matching against a learned keyboard layout, "
             "84.8% per-event accuracy on a 66-event ground-truth labeled subset).")
lines.append("- Bigram analysis restricted to consecutive letter-key presses; "
             "spacebar/backspace/punctuation/modifiers excluded as they introduce "
             "non-biomechanical timing artifacts.")
lines.append(f"- Slow bigram threshold: top {int((1-SLOW_BIGRAM_PERCENTILE)*100)}% "
             f"of session bigram medians, minimum {MIN_BIGRAM_OCCURRENCES} occurrences.")
lines.append(f"- Inconsistency threshold: dominant finger covers "
             f"<{int(INCONSISTENCY_THRESHOLD*100)}% of presses on a key, "
             f"minimum {MIN_KEY_PRESSES_FOR_CONSISTENCY} presses.")
lines.append("- Backspace backtrace: each backspace traced to the keystroke "
             "immediately preceding it (the deleted character), with the "
             "subsequent non-backspace keystroke recorded as the intended "
             "character. Consecutive backspaces collapsed into single correction events.")
lines.append(f"- Abandonment detection: tab+enter or enter+tab pairs within "
             f"{ABANDONMENT_WINDOW_MS}ms treated as Monkeytype restart shortcuts.")
lines.append("- Ergonomic deviations compared against a static dictionary of "
             "standard touch-typing canonical fingerings. The system models actual "
             "habits for measurement accuracy; this layer surfaces where habits "
             "diverge from convention.")
lines.append(f"- WPM reported in four flavors: Burst (60000 / median_iki / 5), "
             f"Active (chars/5 / active-time-minutes, where active-time excludes "
             f"any IKI > {PAUSE_IKI_MULTIPLIER}× the session's median IKI), "
             "Raw (chars/5 / total wall-clock minutes — Monkeytype-equivalent), "
             "and Peak (60000 / p25_iki / 5).")
lines.append("")

lines.append("## Artifacts")
lines.append("")
lines.append("- `report.md` — this file")
lines.append("- `analysis.json` — structured data for Phase 5 integration")
lines.append("- `bigram_stats.csv` — per-bigram timing details")
lines.append("- `finger_stats.csv` — per-finger workload details")
lines.append("- `key_consistency.csv` — per-key finger-usage distributions")

with open(report_path, 'w') as f:
    f.write('\n'.join(lines))


# ==========================================
# TERMINAL SUMMARY
# ==========================================
print(f"\n{'=' * 60}")
print(f"PHASE 4 ANALYSIS COMPLETE — {metadata.get('session_id')}")
print(f"{'=' * 60}")
print(f"WPM — burst {burst_wpm:.1f} | active {active_wpm:.1f} | "
      f"raw {raw_wpm:.1f} | peak {peak_wpm:.1f}")
print(f"Median IKI:          {iki_median:.1f}ms (p25 {iki_p25:.0f}ms)")
if pause_count > 0:
    print(f"Pauses (>{pause_threshold_ms:.0f}ms): {pause_count}, "
          f"{pause_total_seconds:.1f}s removed from active time")
print(f"Letter keystrokes:   {char_count}")
print(f"Unique bigrams:      {len(bigram_stats)}")
print(f"Slow bigrams (top {int((1-SLOW_BIGRAM_PERCENTILE)*100)}%): "
      f"{sum(1 for b in bigram_stats if b['is_slow'])}")
print(f"Inconsistent keys:   {sum(1 for c in consistency_stats if c['is_inconsistent'])}")
print(f"Backspace events:    {len(backspace_errors)}")
print(f"Abandonment events:  {len(abandonment_events)}")
print(f"Ergonomic deviations:{len(ergonomic_deviations)}")
print(f"")
print(f"Bigram timing by category (median IKI):")
for cat in ('same_finger', 'same_hand_diff_finger', 'cross_hand'):
    c = bigram_categories[cat]
    m = bigram_category_medians.get(cat, 0)
    print(f"  {cat:30s} {c:>4} occurrences   {m:.1f}ms")
print(f"")
print(f"Outputs:")
print(f"  {report_path}")
print(f"  {analysis_json_path}")
print(f"  {bigram_csv_path}")
print(f"  {finger_csv_path}")
print(f"  {consistency_csv_path}")