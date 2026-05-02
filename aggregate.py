"""
Multi-session aggregator.

Concatenates raw events_v2.csv and keystrokes.csv across sessions into a
synthetic combined session, then runs analyze.py on it. Analytical logic
lives in one place (analyze.py); aggregation is just data concatenation.

Usage:
    python aggregate.py sessions/A sessions/B
    python aggregate.py sessions/             # all session subdirs
"""

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REQUIRED = ['events_v2.csv', 'keystrokes.csv', 'metadata.json']


def find_sessions(args):
    found = []
    for arg in args:
        p = Path(arg)
        if all((p / r).exists() for r in REQUIRED):
            found.append(p)
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if all((child / r).exists() for r in REQUIRED):
                    found.append(child)
    seen, unique = set(), []
    for s in found:
        r = s.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(s)
    return unique


def concat_csv(paths, out_path, time_col='t_mono'):
    """Concatenate CSVs with the same schema. Offset t_mono per session so
    each timeline starts 60s after the previous ends — prevents cross-session
    IKIs from being computed as if continuous."""
    offset = 0.0
    header_written = False
    with open(out_path, 'w', newline='') as fout:
        writer = None
        for path in paths:
            with open(path) as fin:
                reader = csv.DictReader(fin)
                rows = list(reader)
                if not rows:
                    continue
                if not header_written:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    header_written = True
                session_max_t = max(float(r[time_col]) for r in rows)
                for r in rows:
                    r[time_col] = f"{float(r[time_col]) + offset:.6f}"
                    writer.writerow(r)
                offset += session_max_t + 60.0


sessions = find_sessions(sys.argv[1:])
if not sessions:
    print("Usage: python aggregate.py <session_dir> [session_dir ...]")
    sys.exit(1)

print(f"Aggregating {len(sessions)} session(s):")
for s in sessions:
    print(f"  {s.name}")

out_dir = Path(f"aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
out_dir.mkdir(exist_ok=True)

concat_csv([s / 'events_v2.csv' for s in sessions], out_dir / 'events_v2.csv')
concat_csv([s / 'keystrokes.csv' for s in sessions], out_dir / 'keystrokes.csv')

total_duration = sum(json.loads((s / 'metadata.json').read_text())
                     .get('duration_wall_seconds', 0) for s in sessions)
(out_dir / 'metadata.json').write_text(json.dumps({
    'session_id': out_dir.name,
    'duration_wall_seconds': total_duration,
    'sessions_included': [s.name for s in sessions],
    'session_count': len(sessions),
}, indent=2))

print(f"\nRunning analyze.py on {out_dir}...\n")
sys.exit(subprocess.run(
    [sys.executable, 'analyze.py', str(out_dir)]
).returncode)