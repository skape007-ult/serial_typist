"""
Cross-reference a session's analysis against historical Monkeytype data.

Reads a session's analysis.json (output of analyze.py or aggregate.py) and
the historical_profile.json (output of historical_profile.py), produces a
combined report that grounds session findings in career-wide context.

The output answers questions like:
  * Is this session's WPM normal for me, or a slow day?
  * Is the median IKI consistent with my long-term distribution?
  * Are session-flagged "slow bigrams" reflective of chronic patterns or
    just session-specific noise?
  * How does the same-finger vs cross-hand timing ratio compare to my
    historical keystroke distribution?

Note: per-bigram cross-referencing is not possible without the actual word
sequence from each historical test, which the API doesn't expose. The
cross-reference layer is therefore at the session-level and IKI-distribution-
level, which is still meaningfully more grounded than session-only analysis.

Usage:
    python cross_reference.py sessions/20260424_121126
    python cross_reference.py aggregated_20260427_153012
"""

import json
import sys
from pathlib import Path

HISTORICAL_PATH = Path("historical_profile.json")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def categorize_wpm_vs_career(session_wpm, career_stats):
    """Where does this session's WPM sit relative to your career distribution?"""
    median = career_stats['career_wpm_median']
    p90 = career_stats['career_wpm_p90']
    p99 = career_stats['career_wpm_p99']

    if session_wpm >= p99:
        return 'exceptional', f'top 1% of your career'
    elif session_wpm >= p90:
        return 'excellent', f'top 10% of your career'
    elif session_wpm >= median:
        return 'above_average', f'above your career median ({median} WPM)'
    elif session_wpm >= median * 0.85:
        return 'slightly_below', f'slightly below career median ({median} WPM)'
    else:
        return 'below_average', f'meaningfully below career median ({median} WPM)'


def categorize_iki_vs_career(session_iki, iki_dist):
    """Compare session's median IKI to career-wide IKI distribution."""
    if not iki_dist:
        return None, "no career IKI data available"

    career_median = iki_dist['median_ms']
    p25 = iki_dist['p25_ms']
    p75 = iki_dist['p75_ms']

    if session_iki <= p25:
        return 'fast', f"in your fastest 25% (career p25: {p25}ms)"
    elif session_iki <= career_median:
        return 'above_median', f"faster than career median ({career_median}ms)"
    elif session_iki <= p75:
        return 'normal', f"between career median and p75"
    else:
        return 'slow', f"slower than career p75 ({p75}ms)"


def find_session_path():
    if len(sys.argv) != 2:
        print("Usage: python cross_reference.py <session_or_aggregated_dir>")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.is_dir():
        print(f"ERROR: {p} is not a directory.")
        sys.exit(1)
    return p


def main():
    session_dir = find_session_path()
    session_analysis_path = session_dir / 'analysis.json'

    if not session_analysis_path.exists():
        print(f"ERROR: {session_analysis_path} not found. Run analyze.py first.")
        sys.exit(1)
    if not HISTORICAL_PATH.exists():
        print(f"ERROR: {HISTORICAL_PATH} not found. Run historical_profile.py first.")
        sys.exit(1)

    session = load_json(session_analysis_path)
    historical = load_json(HISTORICAL_PATH)
    career = historical['career_stats']
    iki_dist = historical.get('keystroke_interval_distribution')

    # ------ Session vs career timing ------
    session_wpm = session.get('timing', {}).get('wpm_estimate') \
                  or session.get('timing', {}).get('wpm_estimate_pooled', 0)
    session_iki = session.get('timing', {}).get('iki_median_ms', 0)

    wpm_category, wpm_desc = categorize_wpm_vs_career(session_wpm, career)
    iki_category, iki_desc = categorize_iki_vs_career(session_iki, iki_dist) \
        if iki_dist else (None, "no career IKI baseline")

    # ------ Bigram category proportions ------
    bg = session.get('bigrams', {})
    cat_counts = bg.get('category_counts', {})
    cat_medians = bg.get('category_medians_ms', {})

    sf_med = cat_medians.get('same_finger', 0)
    ch_med = cat_medians.get('cross_hand', 0)
    sf_ch_ratio = sf_med / ch_med if ch_med > 0 else None

    # ------ Build the report ------
    lines = []
    lines.append(f"# Cross-Referenced Analysis — `{session_dir.name}`")
    lines.append("")
    lines.append(f"_Session findings cross-referenced against "
                 f"{career['completed_tests']} historical Monkeytype tests "
                 f"({career.get('first_test_iso', '?')[:10]} → "
                 f"{career.get('last_test_iso', '?')[:10]})._")
    lines.append("")

    # ------ Headline: how does this session compare? ------
    lines.append("## Session in Career Context")
    lines.append("")
    lines.append(f"- **Session WPM:** {session_wpm:.1f} — _{wpm_desc}_")
    lines.append(f"- **Session median IKI:** {session_iki:.1f}ms — _{iki_desc}_" if iki_dist
                 else f"- **Session median IKI:** {session_iki:.1f}ms")
    lines.append(f"- **Career median WPM:** {career['career_wpm_median']:.1f}")
    lines.append(f"- **Career p90 WPM:** {career['career_wpm_p90']:.1f}")
    lines.append("")

    # ------ Restart behavior context ------
    lines.append("## Restart and Incompletion Patterns")
    lines.append("")
    lines.append(f"- **Career restart rate:** "
                 f"{career['restarts_per_completed']:.2f}× per completed test")
    lines.append(f"- **Career incomplete rate:** "
                 f"{career['incompletes_per_completed']:.2f}× per completed test")
    lines.append("")
    if career['restarts_per_completed'] >= 1.5:
        lines.append(f"_Your career restart rate is high "
                     f"({career['restarts_per_completed']:.1f}× per completed). "
                     f"This is the behavioral signature this project was built "
                     f"to investigate — abandonment under pressure when rhythm "
                     f"breaks. Cross-referencing live session findings against "
                     f"these abandonment moments is the next analytical step._")
        lines.append("")

    # ------ Bigram timing patterns ------
    lines.append("## Biomechanical Bottleneck (Same-Finger Penalty)")
    lines.append("")
    lines.append("| Category | Count | Median IKI (ms) |")
    lines.append("|----------|------:|----------------:|")
    for cat in ('same_finger', 'same_hand_diff_finger', 'cross_hand'):
        c = cat_counts.get(cat, 0)
        m = cat_medians.get(cat, 0)
        lines.append(f"| {cat.replace('_', ' ').title()} | {c} | {m:.1f} |")
    lines.append("")

    if sf_ch_ratio:
        lines.append(f"_Same-finger transitions are **{sf_ch_ratio:.2f}× slower** "
                     f"than cross-hand transitions in this session._")
        lines.append("")
        if iki_dist and ch_med > 0:
            ch_vs_career = ch_med / iki_dist['median_ms']
            if ch_vs_career < 0.95:
                lines.append(f"_Cross-hand bigrams ({ch_med:.1f}ms) are faster "
                             f"than your career-wide median IKI "
                             f"({iki_dist['median_ms']}ms) — your hand-alternation "
                             f"speed is a strength to preserve._")
                lines.append("")
            elif ch_vs_career > 1.1:
                lines.append(f"_Cross-hand bigrams ({ch_med:.1f}ms) are slower "
                             f"than your career median IKI "
                             f"({iki_dist['median_ms']}ms) — even your fastest "
                             f"transition class is below career baseline this "
                             f"session, suggesting a slow day overall rather "
                             f"than a structural bottleneck._")
                lines.append("")

    # ------ Mode-specific context ------
    if career.get('mode_stats'):
        lines.append("## Career Performance by Mode")
        lines.append("")
        lines.append("| Mode | Tests | Median WPM | p90 WPM | Median Acc |")
        lines.append("|------|------:|-----------:|--------:|-----------:|")
        for mode_key, stats in sorted(career['mode_stats'].items(),
                                      key=lambda kv: -kv[1]['count'])[:8]:
            lines.append(f"| `{mode_key}` | {stats['count']} | "
                         f"{stats['wpm_median']} | {stats['wpm_p90']} | "
                         f"{stats['acc_median']}% |")
        lines.append("")

    # ------ Career trajectory ------
    trend = career.get('wpm_trend_monthly', [])
    if len(trend) >= 3:
        recent = trend[-3:]
        older = trend[:3] if len(trend) >= 6 else trend[:max(1, len(trend) - 3)]
        recent_med = sum(t['median_wpm'] for t in recent) / len(recent)
        older_med = sum(t['median_wpm'] for t in older) / len(older)
        delta = recent_med - older_med
        lines.append("## Career Trajectory")
        lines.append("")
        lines.append(f"- **Earliest months avg WPM:** {older_med:.1f}")
        lines.append(f"- **Most recent months avg WPM:** {recent_med:.1f}")
        lines.append(f"- **Net change:** {'+' if delta >= 0 else ''}{delta:.1f} WPM "
                     f"({'+' if delta >= 0 else ''}{100 * delta / max(older_med, 1):.1f}%)")
        lines.append("")

    # ------ Findings synthesis ------
    lines.append("## Synthesis")
    lines.append("")
    lines.append(f"This session's typing speed sits in the **{wpm_category}** "
                 f"range relative to your career distribution. ")
    if iki_dist:
        lines.append(f"Per-keystroke timing is in the **{iki_category}** range. ")
    if sf_ch_ratio and sf_ch_ratio > 1.4:
        lines.append(f"The same-finger penalty ({sf_ch_ratio:.1f}×) is a measurable "
                     f"biomechanical constraint that targeted bigram drilling "
                     f"would address. ")
    if career['restarts_per_completed'] >= 1.5:
        lines.append(f"The high career restart rate "
                     f"({career['restarts_per_completed']:.1f}× per completed test) "
                     f"suggests rhythm-break is a recurring behavioral pattern; "
                     f"identifying which biomechanical events trigger restarts "
                     f"is a natural next step.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_Inputs: `{session_analysis_path}` + `{HISTORICAL_PATH}`_")

    out_path = session_dir / 'cross_referenced_report.md'
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n{'=' * 60}")
    print(f"CROSS-REFERENCED REPORT WRITTEN")
    print(f"{'=' * 60}")
    print(f"Session WPM:       {session_wpm:.1f} ({wpm_category})")
    print(f"Session IKI:       {session_iki:.1f}ms ({iki_category if iki_category else '?'})")
    print(f"Career median WPM: {career['career_wpm_median']:.1f}")
    print(f"Career p90 WPM:    {career['career_wpm_p90']:.1f}")
    if sf_ch_ratio:
        print(f"Same-finger penalty: {sf_ch_ratio:.2f}×")
    print(f"\nOutput: {out_path}")


if __name__ == '__main__':
    main()