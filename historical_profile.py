"""
Build a historical bigram profile from cached Monkeytype tests.

Reads every test in monkeytype_cache/, extracts per-keystroke timing data
(keySpacing array — gaps between consecutive keystrokes in ms), and
aggregates per-bigram statistics across the full career.

The challenge: Monkeytype's API gives us keySpacing (timing) but NOT the
actual letter sequence typed. So we can't reconstruct exact bigrams from
keySpacing alone. What we CAN do: use the test's `mode`, `mode2`,
`language`, and `chartData` to build a per-test profile of speed and
errors, then surface chronic patterns at the test-level.

For TRUE bigram-level analysis we'd need the actual word list shown to the
user during each test, which the API doesn't return. So this profile is
test-level: WPM trends, accuracy trends, consistency trends, error rates
across modes/lengths.

Outputs:
    historical_profile.json    Career-wide statistics structured for cross-ref.

Usage:
    python historical_profile.py
"""

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path("monkeytype_cache")
OUTPUT_PATH = Path("historical_profile.json")

# Tests longer than 122s have keySpacing/keyDuration/chartData replaced
# with the literal string "toolong" (Monkeytype payload compression).
TOOLONG_MARKER = "toolong"


def load_all_tests():
    """Load every cached test as a dict. Returns sorted by timestamp ascending."""
    if not CACHE_DIR.exists():
        print(f"ERROR: cache dir {CACHE_DIR} not found. Run monkeytype_client.py first.")
        return []

    tests = []
    for path in CACHE_DIR.glob('*.json'):
        try:
            with open(path) as f:
                data = json.load(f)
            tests.append(data)
        except Exception as e:
            print(f"  WARN: failed to load {path}: {e}")

    tests.sort(key=lambda t: t.get('timestamp', 0))
    return tests


def has_keystroke_timing(test):
    """True if this test has resolvable per-keystroke data (not 'toolong')."""
    ks = test.get('keySpacing')
    return isinstance(ks, list) and len(ks) > 0


def extract_per_test_summary(test):
    """Pull the analytically-relevant fields out of one test."""
    return {
        'id': test.get('_id'),
        'timestamp_ms': test.get('timestamp', 0),
        'wpm': test.get('wpm'),
        'raw_wpm': test.get('rawWpm'),
        'acc': test.get('acc'),
        'consistency': test.get('consistency'),
        'mode': test.get('mode'),  # 'time', 'words', 'quote'
        'mode2': str(test.get('mode2', '')),  # '15', '30', '60', etc.
        'language': test.get('language', 'english'),
        'punctuation': test.get('punctuation', False),
        'numbers': test.get('numbers', False),
        'difficulty': test.get('difficulty', 'normal'),
        'restart_count': test.get('restartCount', 0),
        'incomplete_test_seconds': test.get('incompleteTestSeconds', 0),
        'incomplete_count': len(test.get('incompleteTests', [])),
        'test_duration': test.get('testDuration'),
        'has_timing': has_keystroke_timing(test),
        'char_correct': (test.get('charStats') or [None])[0],
        'char_incorrect': (test.get('charStats') or [None, None])[1],
    }


def aggregate_career_stats(tests):
    """Whole-career summary: WPM trend, accuracy distribution, mode breakdown."""
    summaries = [extract_per_test_summary(t) for t in tests]
    completed = [s for s in summaries if s['wpm'] is not None]

    if not completed:
        return {'error': 'no completed tests in cache'}

    # Bucket by mode/mode2 for filtered claims
    by_mode = defaultdict(list)
    for s in completed:
        key = f"{s['mode']}_{s['mode2']}"
        by_mode[key].append(s)

    mode_stats = {}
    for mode_key, group in by_mode.items():
        if len(group) < 5:
            continue  # too few samples to characterize
        wpms = [s['wpm'] for s in group]
        accs = [s['acc'] for s in group]
        mode_stats[mode_key] = {
            'count': len(group),
            'wpm_median': round(statistics.median(wpms), 2),
            'wpm_mean': round(statistics.mean(wpms), 2),
            'wpm_p90': round(sorted(wpms)[int(len(wpms) * 0.9)], 2),
            'wpm_p10': round(sorted(wpms)[int(len(wpms) * 0.1)], 2),
            'acc_median': round(statistics.median(accs), 2),
            'first_test_iso': datetime.fromtimestamp(group[0]['timestamp_ms'] / 1000).isoformat()
            if group[0]['timestamp_ms'] else None,
            'last_test_iso': datetime.fromtimestamp(group[-1]['timestamp_ms'] / 1000).isoformat()
            if group[-1]['timestamp_ms'] else None,
        }

    # Career-wide WPM trend (binned by month for chart data)
    monthly_wpm = defaultdict(list)
    for s in completed:
        if not s['timestamp_ms']:
            continue
        month_key = datetime.fromtimestamp(s['timestamp_ms'] / 1000).strftime('%Y-%m')
        monthly_wpm[month_key].append(s['wpm'])

    wpm_trend = []
    for month in sorted(monthly_wpm.keys()):
        wpms = monthly_wpm[month]
        wpm_trend.append({
            'month': month,
            'count': len(wpms),
            'median_wpm': round(statistics.median(wpms), 2),
            'mean_wpm': round(statistics.mean(wpms), 2),
        })

    # Restart / incomplete behavior — your 75% restart rate observation
    total_restarts = sum(s['restart_count'] for s in completed)
    total_incomplete = sum(s['incomplete_count'] for s in completed)
    restart_rate = total_restarts / max(len(completed), 1)
    incomplete_rate = total_incomplete / max(len(completed), 1)

    # All WPMs flat for global percentile ranking
    all_wpms = [s['wpm'] for s in completed]

    return {
        'total_tests_cached': len(summaries),
        'completed_tests': len(completed),
        'first_test_iso': datetime.fromtimestamp(completed[0]['timestamp_ms'] / 1000).isoformat()
        if completed[0]['timestamp_ms'] else None,
        'last_test_iso': datetime.fromtimestamp(completed[-1]['timestamp_ms'] / 1000).isoformat()
        if completed[-1]['timestamp_ms'] else None,
        'career_wpm_median': round(statistics.median(all_wpms), 2),
        'career_wpm_mean': round(statistics.mean(all_wpms), 2),
        'career_wpm_p90': round(sorted(all_wpms)[int(len(all_wpms) * 0.9)], 2),
        'career_wpm_p99': round(sorted(all_wpms)[int(len(all_wpms) * 0.99)], 2),
        'career_acc_median': round(statistics.median(s['acc'] for s in completed), 2),
        'restarts_per_completed': round(restart_rate, 2),
        'incompletes_per_completed': round(incomplete_rate, 2),
        'tests_with_resolvable_timing': sum(1 for s in completed if s['has_timing']),
        'mode_stats': mode_stats,
        'wpm_trend_monthly': wpm_trend,
    }


def per_test_keystroke_distribution(tests):
    """Across all tests with keySpacing, build a distribution of keystroke
    intervals. This gives us your career-wide IKI distribution that any
    single session's IKIs can be compared against."""
    all_intervals = []
    for t in tests:
        if not has_keystroke_timing(t):
            continue
        ks = t.get('keySpacing', [])
        # Filter out absurd outliers (test pauses, etc.)
        for interval in ks:
            if isinstance(interval, (int, float)) and 30 <= interval <= 2000:
                all_intervals.append(float(interval))

    if not all_intervals:
        return None

    sorted_intervals = sorted(all_intervals)
    n = len(sorted_intervals)
    return {
        'sample_count': n,
        'median_ms': round(statistics.median(sorted_intervals), 2),
        'mean_ms': round(statistics.mean(sorted_intervals), 2),
        'p25_ms': round(sorted_intervals[int(n * 0.25)], 2),
        'p75_ms': round(sorted_intervals[int(n * 0.75)], 2),
        'p90_ms': round(sorted_intervals[int(n * 0.90)], 2),
        'p99_ms': round(sorted_intervals[int(n * 0.99)], 2),
    }


def main():
    print("Loading cached tests...")
    tests = load_all_tests()
    print(f"  Loaded {len(tests)} tests from {CACHE_DIR}/\n")

    if not tests:
        return

    print("Aggregating career statistics...")
    career_stats = aggregate_career_stats(tests)

    print("Computing keystroke interval distribution...")
    iki_dist = per_test_keystroke_distribution(tests)

    profile = {
        'generated_at': datetime.now().isoformat(),
        'career_stats': career_stats,
        'keystroke_interval_distribution': iki_dist,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(profile, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"HISTORICAL PROFILE BUILT")
    print(f"{'=' * 60}")
    print(f"Tests analyzed:       {career_stats['completed_tests']}")
    print(f"Career WPM median:    {career_stats['career_wpm_median']}")
    print(f"Career WPM p90:       {career_stats['career_wpm_p90']}")
    print(f"Career accuracy:      {career_stats['career_acc_median']}%")
    print(f"Restart rate:         {career_stats['restarts_per_completed']:.2f}× per completed test")
    print(f"Incomplete rate:      {career_stats['incompletes_per_completed']:.2f}× per completed test")
    if iki_dist:
        print(f"Career median IKI:    {iki_dist['median_ms']}ms (p25={iki_dist['p25_ms']}, p75={iki_dist['p75_ms']})")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()