"""
Coaching note generator.

Reads a session's analysis.json, loads all previous sessions' analyses for
historical context, and produces a conversational coaching note as
feedback.md. Optionally emails the result.

Output is structured but written like a knowledgeable peer talking — not
a technical report. Sections are conditional: skipped if the data doesn't
support a meaningful finding for that session.

Email config (read from environment, all optional):
    GMAIL_ADDRESS         — sender Gmail address
    GMAIL_APP_PASSWORD    — Gmail app password (not your account password)
    EMAIL_RECIPIENT       — recipient address (defaults to GMAIL_ADDRESS)
    TYPIST_DISABLE_EMAIL  — set to "1" to skip email even if creds are set

If credentials aren't set, the email step is silently skipped — feedback.md
is still always written.

Usage:
    python feedback.py sessions/<session_id>
    python feedback.py sessions/<session_id> --no-email
"""

import argparse
import json
import logging
import os
import re
import smtplib
import sys
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime

# Load environment variables from a project-local .env file if present.
# Looked up next to feedback.py so the daemon can find it regardless of CWD.
# override=True so .env wins over any stale env vars inherited from a
# parent daemon process whose env was baked in at launch time.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / '.env', override=True)
except ImportError:
    pass

# ==========================================
# CONFIG
# ==========================================
MEANINGFUL_CHANGE_FRACTION = 0.05    # 5% movement to surface a comparison
PATTERN_RECURRENCE_THRESHOLD = 3     # appears in 3+ sessions = real pattern
DRILL_LENGTH = 18                    # words per drill
RANDOM_SEED = None                   # set to int for reproducible drills

logger = logging.getLogger("typist.feedback")

# ==========================================
# WORDLIST (top common English words for drill generation)
# ==========================================
# Embedded so the script ships standalone. ~500 most common words covering
# roughly 70% of natural English text.
WORDLIST = """
the be to of and a in that have it for not on with he as you do at
this but his by from they we say her she or an will my one all would
there their what so up out if about who get which go me when make can
like time no just him know take people into year your good some could
them see other than then now look only come its over think also back
after use two how our work first well way even new want because any
these give day most us is are was were been being am has had does did
long without around something thought through where really another come
might few nothing while too place feel seem hand high right great such
old find want let must say head here group own life same point still
under during turn last show ask leave start play move country lot try
hold case offer different woman bad part child thing world hour begin
keep turn down again call try plan need seem follow point fact stop
become public change open eye begin again every spend grow room small
never word young mean course early run social party face read water
stand send stay write home feel near five wait love break carry death
change story write learn week kind power little house minute strong
matter young second question maybe friend able mean live tell next
consider hand area might face stand area sort live story add room
court father pull side moment far house consider find lot fire across
along behind toward believe night fall mind since send four young add
reach four ground might stand tonight watch friend sound right night
read present already though yet allow seem let grow watch level number
near family black both yes early national local within month problem
hand keep less course side state hold present black great several
white woman remain together sure understand watch white member talk
study play room important early light face become turn perhaps result
explain free without through white kind house run set side hour talk
true together bring close name leader cost real least drive stop guess
girl heart picture grow bear paper father human seven knee mile spell
dog past grew hat sing tail walk wait build fine ride low north summer
field flat near tree move wait round square pull grew hill class town
push wide happy easy done music plant black cover song fall draw drive
voice felt week carry sat hot sea ten common ride forest green dollar
fast plant felt hand sea cool fly six warm catch past iron stay color
star city stood box reach edge done deep mark front feel fact row mile
front rest correct began hour map star sat third board main dry tall
dark close summer toward fish full wood arm sky read grew toward north
lot boat top kept eight ship east lake meet mountain mountain feel ago
fly thick sand king ride heavy size stop yard inch suit oil saw glad
glad full
""".split()

WORDLIST = list(set(w for w in WORDLIST if w.isalpha() and len(w) >= 2))


# ==========================================
# DATA LOADING
# ==========================================
def load_session(session_dir):
    analysis_path = session_dir / 'analysis.json'
    if not analysis_path.exists():
        return None
    with open(analysis_path) as f:
        return json.load(f)


def load_all_sessions(sessions_root, current_session_id):
    """Load every analysis.json in the sessions directory, sorted by timestamp.
    Returns list of (session_id, analysis_dict) tuples, oldest first.
    Excludes the current session."""
    if not sessions_root.exists():
        return []
    sessions = []
    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue
        if session_dir.name == current_session_id:
            continue
        analysis = load_session(session_dir)
        if analysis is None:
            continue
        sessions.append((session_dir.name, analysis))
    return sessions


# ==========================================
# HISTORICAL ANALYSIS
# ==========================================
def compute_history(history_sessions):
    """Compute aggregate statistics across previous sessions."""
    if not history_sessions:
        return {'count': 0}

    burst_wpms = []
    same_finger_medians = []
    cross_hand_medians = []
    slow_bigrams_per_session = []  # list of sets

    for sid, analysis in history_sessions:
        timing = analysis.get('timing', {})
        wpm = timing.get('wpm', {})
        burst = wpm.get('burst', timing.get('wpm_estimate'))
        if burst:
            burst_wpms.append(burst)

        cat_medians = analysis.get('bigrams', {}).get('category_medians_ms', {})
        if cat_medians.get('same_finger'):
            same_finger_medians.append(cat_medians['same_finger'])
        if cat_medians.get('cross_hand'):
            cross_hand_medians.append(cat_medians['cross_hand'])

        slow_bigrams = {b['bigram'] for b in
                        analysis.get('bigrams', {}).get('top_slow', [])
                        if b.get('is_slow')}
        slow_bigrams_per_session.append(slow_bigrams)

    # Bigrams that appear as slow in multiple sessions
    recurring_slow_bigrams = {}
    for session_set in slow_bigrams_per_session:
        for bg in session_set:
            recurring_slow_bigrams[bg] = recurring_slow_bigrams.get(bg, 0) + 1

    return {
        'count': len(history_sessions),
        'last_burst_wpm': burst_wpms[-1] if burst_wpms else None,
        'last_same_finger_ms': same_finger_medians[-1] if same_finger_medians else None,
        'last_cross_hand_ms': cross_hand_medians[-1] if cross_hand_medians else None,
        'avg_burst_wpm': sum(burst_wpms)/len(burst_wpms) if burst_wpms else None,
        'recurring_slow_bigrams': {bg: c for bg, c in recurring_slow_bigrams.items()
                                    if c >= PATTERN_RECURRENCE_THRESHOLD},
    }


def is_meaningful_change(current, previous, fraction=MEANINGFUL_CHANGE_FRACTION):
    if previous is None or previous == 0:
        return False
    return abs(current - previous) / previous >= fraction


# ==========================================
# DRILL GENERATION
# ==========================================
def words_with_letters(letters, min_count=1):
    """Words containing all specified letters."""
    return [w for w in WORDLIST
            if all(w.count(l) >= min_count for l in letters.lower())]


def words_rich_in(letter, min_density=0.3):
    """Words with high density of a specific letter."""
    return [w for w in WORDLIST
            if w.count(letter.lower()) / max(len(w), 1) >= min_density]


def words_with_double_letter():
    """Words with any double-letter (for same-finger drilling)."""
    out = []
    for w in WORDLIST:
        for i in range(len(w) - 1):
            if w[i] == w[i+1]:
                out.append(w)
                break
    return out


def words_with_bigram(bigram):
    """Words containing the specified two-letter sequence."""
    return [w for w in WORDLIST if bigram.lower() in w.lower()]


def select_drill(words, count=DRILL_LENGTH):
    """Pick `count` words from candidates, randomized but deterministic per call."""
    if not words:
        return None
    pool = list(set(words))
    if len(pool) <= count:
        return ' '.join(pool)
    rng = random.Random(RANDOM_SEED) if RANDOM_SEED is not None else random
    return ' '.join(rng.sample(pool, count))


def drill_for_finding(finding_type, finding_data):
    """Map a finding to a word-sequence drill. Returns (drill_string, description)
    or None if no good drill is available."""
    if finding_type == 'same_finger_penalty':
        words = words_with_double_letter()
        return (select_drill(words),
                "double-letter sequences to retrain same-finger transitions")

    if finding_type == 'i_overload':
        # Words rich in 'i' — train right middle to handle them
        words = words_rich_in('i', min_density=0.25)
        return (select_drill(words),
                "i-rich words to rebuild right-middle ownership of `i`")

    if finding_type == 'bigram_specific':
        bigram = finding_data.get('bigram', '')
        words = words_with_bigram(bigram)
        if not words:
            # Fallback: words with both letters present
            words = words_with_letters(bigram)
        return (select_drill(words),
                f"words containing `{bigram}` to drill that transition")

    if finding_type == 'right_hand_finger_choreography':
        # Words exercising right-hand finger sequencing on h/j/k/l/n/m/i
        target_letters = set('hjklnmiu')
        words = [w for w in WORDLIST
                 if sum(1 for l in w.lower() if l in target_letters) >= 3]
        return (select_drill(words),
                "right-hand finger choreography (h/j/k/l/n/m/i transitions)")

    return None


# ==========================================
# COACHING NOTE COMPOSITION
# ==========================================
def compose_opening(session, history):
    timing = session['timing']
    duration_min = timing.get('typing_time_seconds', 0) / 60
    letter_count = timing.get('total_letter_presses', 0)
    burst_wpm = timing.get('wpm', {}).get('burst', timing.get('wpm_estimate', 0))
    pause_seconds = timing.get('pause_total_seconds', 0)
    pause_count = timing.get('pause_count', 0)
    total_seconds = timing.get('typing_time_seconds', 1)
    pause_fraction = pause_seconds / total_seconds if total_seconds else 0

    parts = []
    parts.append(f"That was a {duration_min:.0f}-minute session, "
                 f"{letter_count} letter keystrokes at {burst_wpm:.1f} burst WPM.")

    if pause_fraction > 0.4:
        parts.append(f"You spent {int(pause_fraction*100)}% of the wall clock "
                     f"on pauses ({pause_count} of them) — this was a "
                     f"composing session, not a transcription session. Your "
                     f"fingers were fast; your thinking took time.")
    elif pause_fraction < 0.15:
        parts.append("Almost no pause time — this was steady, sustained typing.")

    # Historical comparison on burst WPM if meaningful
    if history['count'] > 0:
        last_burst = history.get('last_burst_wpm')
        if last_burst and is_meaningful_change(burst_wpm, last_burst):
            direction = "up" if burst_wpm > last_burst else "down"
            pct = abs(burst_wpm - last_burst) / last_burst * 100
            parts.append(f"Burst WPM is {direction} from {last_burst:.1f} "
                         f"last session ({pct:.0f}%).")

    return " ".join(parts)


def compose_abandonment(session):
    err = session.get('error_patterns', {})
    abandonment_count = err.get('abandonment_count', 0)
    if abandonment_count < 3:
        return None

    trigger_keys = err.get('abandonment_trigger_keys', {})
    if not trigger_keys:
        return None

    sorted_keys = sorted(trigger_keys.items(), key=lambda kv: -kv[1])
    top_keys = sorted_keys[:3]
    top_count = top_keys[0][1]

    if top_count < 3:
        return None  # no clear concentration

    parts = []
    parts.append(f"You triggered {abandonment_count} restart sequences "
                 f"(tab+enter) tonight. The keystrokes preceding restarts "
                 f"clustered on `{top_keys[0][0]}` ({top_keys[0][1]}x)"
                 f"{', `' + top_keys[1][0] + '` (' + str(top_keys[1][1]) + 'x)' if len(top_keys) > 1 else ''}"
                 f". Those are the moments your rhythm broke. ")

    if 'space' in dict(top_keys):
        parts.append("Notably `space` is among the triggers — meaning you "
                     "tend to give up at word boundaries when the next word "
                     "feels wrong, not on specific letters. That's a planning "
                     "rhythm issue more than a motor issue.")

    return " ".join(parts)


def compose_backspace(session):
    err = session.get('error_patterns', {})
    bs_count = err.get('backspace_count', 0)
    error_bigrams = err.get('top_error_bigrams', [])

    if bs_count < 5 or not error_bigrams:
        return None

    notable = [eb for eb in error_bigrams if eb['count'] >= 2]
    if not notable:
        return None

    parts = []
    parts.append(f"You hit backspace {bs_count} times. ")

    descriptions = []
    for eb in notable[:3]:
        bg = eb['error_bigram']
        intended = eb.get('most_common_intended', '?')
        if intended and intended != bg:
            descriptions.append(f"`{bg}` corrected to `{intended}` ({eb['count']}x)")
        else:
            descriptions.append(f"`{bg}` ({eb['count']}x)")

    if descriptions:
        parts.append("The recurring error patterns: " + ", ".join(descriptions) + ".")

    return " ".join(parts)


def compose_finger_findings(session):
    deviations = session.get('ergonomic_deviations', [])
    consistency = session.get('key_consistency', [])

    parts = []

    # Surface highest-volume deviation if it's substantial
    high_vol_dev = [d for d in deviations if d['total_presses'] >= 20]
    if high_vol_dev:
        d = high_vol_dev[0]
        parts.append(f"You're pressing `{d['key']}` with "
                     f"{d['your_dominant_finger'].lower()} "
                     f"{int(d['dominance']*100)}% of the time. Standard "
                     f"touch typing puts that on {d['canonical_finger'].lower()}.")

        if d['key'] == 'i' and 'index' in d['your_dominant_finger'].lower():
            parts.append("Your right index is already covering n, h, m, y, "
                         "and u — adding `i` to that load is genuinely "
                         "expensive. Settling `i` back onto right middle "
                         "would free the index for the keys it's meant to handle.")

    # Surface most-inconsistent high-volume key
    inconsistent = [c for c in consistency
                    if c.get('is_inconsistent') and c['total_presses'] >= 30]
    if inconsistent:
        c = inconsistent[0]
        if not high_vol_dev or c['key'] != high_vol_dev[0]['key']:
            dist = c['distribution']
            top_two = sorted(dist.items(), key=lambda kv: -kv[1])[:2]
            parts.append(f"`{c['key']}` was split between two fingers tonight: "
                         f"{top_two[0][0].lower()} ({top_two[0][1]}x), "
                         f"{top_two[1][0].lower()} ({top_two[1][1]}x). "
                         f"Picking one and committing builds cleaner muscle memory.")

    return " ".join(parts) if parts else None


def compose_bigram_timing(session, history):
    bg = session.get('bigrams', {})
    cat_medians = bg.get('category_medians_ms', {})

    sf = cat_medians.get('same_finger', 0)
    ch = cat_medians.get('cross_hand', 0)
    if not (sf and ch):
        return None

    parts = []
    ratio = sf / ch if ch else 0
    if ratio >= 1.4:
        parts.append(f"Same-finger transitions ran at {sf:.0f}ms median "
                     f"vs cross-hand at {ch:.0f}ms — a {ratio:.1f}× penalty. "
                     f"That's the most fixable pattern in your typing because "
                     f"it's mechanical, not cognitive.")

    # History comparison on same-finger penalty
    if history['count'] > 0:
        last_sf = history.get('last_same_finger_ms')
        if last_sf and is_meaningful_change(sf, last_sf):
            direction = "tightened" if sf < last_sf else "regressed"
            pct = abs(sf - last_sf) / last_sf * 100
            parts.append(f"Same-finger penalty {direction} from {last_sf:.0f}ms "
                         f"last session ({pct:.0f}%).")

    # Recurring slow bigrams
    recurring = history.get('recurring_slow_bigrams', {})
    current_slow = {b['bigram'] for b in bg.get('top_slow', []) if b.get('is_slow')}
    persistent = current_slow & set(recurring.keys())
    if persistent:
        bigrams_str = ", ".join(f"`{b}`" for b in sorted(persistent))
        parts.append(f"{bigrams_str} have shown up as slow in multiple "
                     f"sessions now — these are real patterns, not noise.")

    # Specific slow bigrams from this session
    top_slow_credible = [b for b in bg.get('top_slow', [])
                         if b.get('count', 0) >= 2 and b.get('is_slow')]
    if top_slow_credible:
        top = top_slow_credible[:2]
        descriptions = [f"`{b['bigram']}` at {b['median_ms']:.0f}ms ({b['count']}x)"
                        for b in top]
        parts.append(f"Specific slow bigrams worth watching: " +
                     ", ".join(descriptions) + ".")

    return " ".join(parts) if parts else None


def compose_drills(session):
    """Decide which drills to suggest based on findings, generate the words."""
    findings = []  # list of (priority, finding_type, finding_data)

    # Find most pressing issue
    deviations = session.get('ergonomic_deviations', [])
    high_vol_dev = [d for d in deviations if d['total_presses'] >= 20]
    if high_vol_dev:
        d = high_vol_dev[0]
        if d['key'] == 'i':
            findings.append((1, 'i_overload', d))

    bg = session.get('bigrams', {})
    cat_medians = bg.get('category_medians_ms', {})
    sf = cat_medians.get('same_finger', 0)
    ch = cat_medians.get('cross_hand', 0)
    if sf and ch and sf / ch >= 1.4:
        findings.append((2, 'same_finger_penalty', None))

    # Specific slow bigrams
    top_slow = [b for b in bg.get('top_slow', [])
                if b.get('count', 0) >= 2 and b.get('is_slow')]
    if top_slow:
        # Use the slowest bigram as a drill target
        findings.append((3, 'bigram_specific', top_slow[0]))

    if not findings:
        return None

    findings.sort(key=lambda f: f[0])
    main_finding = findings[0]
    main_drill = drill_for_finding(main_finding[1], main_finding[2])

    # Drill must be a tuple AND the word-string in it must be non-None.
    # select_drill() returns None when there are no candidate words, which
    # leaves the (None, "description") tuple. Treat that as no drill.
    if main_drill is None or main_drill[0] is None:
        return None

    parts = []
    parts.append(f"### Drill for tomorrow")
    parts.append("")
    parts.append(f"Main drill — {main_drill[1]}:")
    parts.append("")
    parts.append("```")
    parts.append(main_drill[0])
    parts.append("```")

    # Bonus drill if there's a meaningfully different second finding
    if len(findings) >= 2:
        bonus_finding = findings[1]
        if bonus_finding[1] != main_finding[1]:
            bonus_drill = drill_for_finding(bonus_finding[1], bonus_finding[2])
            if bonus_drill and bonus_drill[0] is not None:
                parts.append("")
                parts.append(f"Bonus drill — {bonus_drill[1]}:")
                parts.append("")
                parts.append("```")
                parts.append(bonus_drill[0])
                parts.append("```")

    parts.append("")
    parts.append("Paste either into Monkeytype's custom-words mode and run a "
                 "60-second test. The pattern your fingers practice is the "
                 "one that gets faster.")

    return "\n".join(parts)


def compose_closing(session, history):
    """A final calibrating sentence."""
    if history['count'] == 0:
        return ("This is your first session in the dataset, so I can't tell you "
                "yet whether tonight is typical or unusual for you. Record "
                "another in the next few days and I'll start tracking trends.")

    return ("Nothing in this session was a regression. The patterns to fix "
            "are clearer than they've been because you finally have enough "
            "data to see them.")


# ==========================================
# EMAIL DELIVERY
# ==========================================
def markdown_to_html(md_text):
    """Convert the coaching-note Markdown subset to HTML.

    Handles: # h1, ## h2, ### h3, **bold**, *italic*, `inline code`,
    fenced code blocks (```), bulleted lists, paragraphs.
    Tuned to what compose_*() functions actually emit, not full CommonMark.
    """
    lines = md_text.split('\n')
    html_blocks = []
    i = 0
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            html_blocks.append('</ul>')
            in_list = False

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.strip().startswith('```'):
            close_list()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code_html = '\n'.join(code_lines)
            html_blocks.append(
                f'<pre style="background:#f4f4f4;padding:14px;border-radius:6px;'
                f'border:1px solid #e0e0e0;font-family:Menlo,Consolas,monospace;'
                f'font-size:13px;overflow-x:auto;line-height:1.5;">{code_html}</pre>'
            )
            continue

        # Headings
        if line.startswith('### '):
            close_list()
            html_blocks.append(f'<h3 style="margin-top:24px;color:#2c2c2c;">'
                                f'{line[4:].strip()}</h3>')
            i += 1
            continue
        if line.startswith('## '):
            close_list()
            html_blocks.append(f'<h2 style="margin-top:28px;color:#1a1a1a;'
                                f'border-bottom:1px solid #e0e0e0;padding-bottom:6px;">'
                                f'{line[3:].strip()}</h2>')
            i += 1
            continue
        if line.startswith('# '):
            close_list()
            html_blocks.append(f'<h1 style="color:#2c2c2c;border-bottom:2px solid '
                                f'#4a90e2;padding-bottom:10px;">'
                                f'{line[2:].strip()}</h1>')
            i += 1
            continue

        # Bullets
        m = re.match(r'^[\*\-]\s+(.+)$', line)
        if m:
            if not in_list:
                html_blocks.append('<ul>')
                in_list = True
            html_blocks.append(f'<li>{_inline_md(m.group(1))}</li>')
            i += 1
            continue

        close_list()

        # Blank line — paragraph break
        if not line.strip():
            i += 1
            continue

        # Paragraph (collect consecutive non-empty non-special lines)
        para_lines = [line]
        i += 1
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith(('#', '```', '* ', '- '))
               and not re.match(r'^[\*\-]\s+', lines[i])):
            para_lines.append(lines[i])
            i += 1
        para = ' '.join(p.strip() for p in para_lines)
        html_blocks.append(f'<p>{_inline_md(para)}</p>')

    close_list()
    return '\n'.join(html_blocks)


def _inline_md(text):
    """Apply inline transformations: **bold**, *italic*, `code`."""
    # Code spans first so their contents aren't re-processed
    text = re.sub(r'`([^`]+?)`',
                  r'<code style="background:#f4f4f4;padding:2px 6px;'
                  r'border-radius:3px;font-family:Menlo,Consolas,monospace;'
                  r'font-size:0.92em;">\1</code>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^\*]+?)\*(?!\*)', r'<em>\1</em>', text)
    return text


def send_email(feedback_md, session_id, force_disable=False):
    """Send feedback.md via Gmail SMTP. Returns True on success.

    Silent skip (returns False) if any of:
      - --no-email flag was passed (force_disable=True)
      - TYPIST_DISABLE_EMAIL=1 in environment
      - GMAIL_ADDRESS or GMAIL_APP_PASSWORD missing
    """
    if force_disable:
        return False
    if os.environ.get('TYPIST_DISABLE_EMAIL') == '1':
        print("Email skipped (TYPIST_DISABLE_EMAIL=1).")
        return False

    sender = os.environ.get('GMAIL_ADDRESS')
    password = os.environ.get('GMAIL_APP_PASSWORD')
    recipient = os.environ.get('EMAIL_RECIPIENT', sender)

    if not sender or not password:
        print("Email skipped (GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set).")
        return False

    # Build a friendly subject from the session ID
    try:
        dt = datetime.strptime(session_id, '%Y%m%d_%H%M%S')
        subject = f"Typing session recap — {dt.strftime('%b %d, %I:%M %p').replace(' 0', ' ')}"
    except ValueError:
        subject = f"Typing session recap — {session_id}"

    html_body = f"""<html>
<body style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
             max-width:720px;margin:auto;padding:24px;color:#1a1a1a;
             line-height:1.65;font-size:15px;">
{markdown_to_html(feedback_md)}
<hr style="border:none;border-top:1px solid #e0e0e0;margin:32px 0 16px 0;">
<p style="color:#888;font-size:11px;text-align:center;margin:0;">
Generated by serial_typist — session {session_id}
</p>
</body>
</html>"""

    recipients = [r.strip() for r in recipient.split(',') if r.strip()]

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)

            for r in recipients:
                msg = MIMEMultipart('alternative')
                msg['Subject'] = subject
                msg['From'] = sender
                msg['To'] = r
                msg.attach(MIMEText(feedback_md, 'plain'))
                msg.attach(MIMEText(html_body, 'html'))
                server.sendmail(sender, [r], msg.as_string())

        print(f"Email sent to {len(recipients)} recipient(s).")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        logger.exception("Email send failed")
        return False


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Generate coaching note for a session.")
    parser.add_argument('session_dir', type=Path)
    parser.add_argument('--no-email', action='store_true',
                        help='Skip the email step even if Gmail credentials are set.')
    args = parser.parse_args()

    session_dir = args.session_dir
    if not session_dir.is_dir():
        print(f"ERROR: {session_dir} is not a directory.")
        sys.exit(1)

    session = load_session(session_dir)
    if session is None:
        print(f"ERROR: no analysis.json in {session_dir}")
        sys.exit(1)

    sessions_root = session_dir.parent
    history_sessions = load_all_sessions(sessions_root, session_dir.name)
    history = compute_history(history_sessions)

    # Timestamp from session_id
    sid = session.get('session_id', session_dir.name)
    try:
        dt = datetime.strptime(sid, '%Y%m%d_%H%M%S')
        date_str = dt.strftime('%B %d, %I:%M %p').replace(' 0', ' ')
    except ValueError:
        date_str = sid

    # Compose the note
    lines = []
    lines.append(f"# Session Recap — {date_str}")
    lines.append("")

    opening = compose_opening(session, history)
    if opening:
        lines.append(opening)
        lines.append("")

    abandonment = compose_abandonment(session)
    if abandonment:
        lines.append(abandonment)
        lines.append("")

    backspace = compose_backspace(session)
    if backspace:
        lines.append(backspace)
        lines.append("")

    finger = compose_finger_findings(session)
    if finger:
        lines.append(finger)
        lines.append("")

    bigram = compose_bigram_timing(session, history)
    if bigram:
        lines.append(bigram)
        lines.append("")

    drills = compose_drills(session)
    if drills:
        lines.append(drills)
        lines.append("")

    closing = compose_closing(session, history)
    lines.append(closing)
    lines.append("")

    feedback_md = '\n'.join(lines)

    out_path = session_dir / 'feedback.md'
    out_path.write_text(feedback_md)

    print(f"Coaching note written to: {out_path}")
    print(f"History context: {history['count']} previous session(s)")

    # Optional email delivery
    send_email(feedback_md, sid, force_disable=args.no_email)


if __name__ == '__main__':
    main()