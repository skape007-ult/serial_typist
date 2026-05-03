**Final goal:** a multimodal typing analysis system that fuses camera-based hand tracking with keystroke telemetry to identify finger-level patterns in the user's own typing — including which fingers hit which keys, which bigrams cost time biomechanically, and where motion efficiency could be improved.

*A log of forks in the road and why certain paths were chosen over others.* 

*Format: each entry is dated, names the fork, states the choice, and the reason.*   
---

## **Phase 1 — Hardware & Video Pipeline**

### **Pivoted from Camo Studio to IP Webcam**

**Why:** macOS on Apple Silicon aggressively sandboxes virtual cameras. Camo Studio worked fine over USB-C for apps with Apple's certification (Zoom, browsers after recent updates), but a Python process running inside a `.venv` could not enumerate the virtual camera through AVFoundation. OpenCV's `cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)` returned `isOpened() == True` on index 0 only — the built-in webcam. The diagnostic loop scanning indices 0–9 confirmed Camo was invisible to Python.

The fix was to stop trying to use a "virtual camera" abstraction entirely. IP Webcam turns the phone into a local HTTP video server. OpenCV reads the stream by URL, which macOS treats as a network resource, not a camera. Sandboxing bypassed completely.

### **Chose direct hotspot bridge over USB tethering over Wi-Fi router**

**Why:** Three attempts, three different failure modes.

1. **Wi-Fi through the home router:** latency too high for high-WPM capture. Video was visibly lagging keystroke events by \~40ms, which at 30 FPS corresponds to a full frame of drift — enough to misassign fingers on fast bigrams.  
2. **USB tethering (the "wired IP stream" idea):** macOS doesn't natively support Android's RNDIS tethering protocol. Apple stripped it years ago. Toggling USB Tethering on the phone while Wi-Fi was off resulted in the Mac seeing no network interface at all. Confirmed by typing the tethered IP into Safari — page didn't load. Would have required installing ADB and manual port-forwarding, which is a rabbit hole that kills the sprint.  
3. **Direct hotspot bridge (what is being currently used):** 

*Benefits of said approach:*

* The phone becomes the access point, to which the mac connects directly  
* Latency dropped to near-imperceptible.   
* Internet access is limited to the hotspot provided by the phone from the router, which slows down the internet speed, but not enough to make a difference. 

### 

### **Kept video resolution at 1080p instead of dropping to 720p**

Why: At first it seemed obvious to reduce resolution — after cropping to the keyboard region, we threw away most of the pixels anyway. But cropping is digital zoom, not resizing. The phone's field of view doesn't change with resolution; it just gets pixelated. At 720p after the same physical crop, MediaPipe sees blurry, low-pixel-density hands and starts conflating adjacent fingers on fast bigrams.

Caveat worth remembering: this reasoning is principle-based, not empirically tested. If a future session demands higher FPS and 1080p can't sustain 60 FPS on this specific phone, revisit by recording identical sessions at both resolutions and diffing the MediaPipe confidence scores.

### **Decoupled video capture from analysis (record-first, analyze-later)**

Why: The original plan was real-time MediaPipe processing during the typing session. Three problems with that:

1. Real-time inference competes with the keystroke listener for CPU. Any lag in the video pipeline corrupts the timestamp precision of keystrokes.  
2. A bug in the finger-assignment heuristic discovered three weeks in would force re-recording every session. With saved video, I can iterate on analysis code against a fixed dataset.  
3. Real-time visualization at 30 FPS constrains model complexity. Offline analysis has no frame budget — can use model\_complexity=1 or higher for better accuracy.  
   

Tradeoff: no live demo during capture. Mitigated by the playback visualizer in Phase 3 — recording a session, running analysis, and rendering an overlay is a stronger demo than a live feed anyway.

### **Auto-trigger recording on first keystroke instead of manual sync markers (F12 x3)**

**Why:** Machine-driven sync is always more precise than human-driven. The first keystroke and the frame timestamp derive from the same `time.monotonic()` clock, so alignment is exact by construction — no sync-marker detection needed during analysis.

Hidden subtlety: if recording starts *at* the first keystroke, the finger motion leading up to that keystroke is missing from the video. Solved with a 2-second rolling buffer that's continuously filling before the first press. When recording goes hot, the buffer is flushed first, then live frames follow. Same trick as GoPro pre-event recording. Mirrored at session end with a 2-second post-roll for finger-lift data on the last keystroke.

(Done by implementing a ring buffer using a double-ended queue to always store the last two seconds before the first keypress)

### **Transform (crop \+ rotate) applied before writing, not during analysis**

**Why:** Storing raw video would preserve flexibility — you could re-crop later if the rig changes. But the rig is physically fixed to the cupboard, the crop coordinates are calibrated, and every downstream script would otherwise have to redo the transform on every frame. Baking the transform into the saved MP4 keeps the analysis pipeline simpler.

Metadata records the exact transform parameters so future-me can tell which sessions were recorded with which calibration. Non-obvious but essential when the rig inevitably shifts.

---

## **Phase 2 — Time & Synchronization**

### **Used `time.monotonic()` as the canonical session clock**

**Why:** Wall-clock time (`time.time()`) is not monotonic. NTP sync, manual clock adjustments, and DST transitions can make it jump forward or backward. Monotonic time is guaranteed to only move forward and cannot be changed by the OS. For relative timing within a session, that property is non-negotiable.

Wall-clock time is still logged as a secondary column, purely for human debugging though, 

### **Sidecar `frame_timestamps.csv` instead of trusting MP4's internal timing**

**Why:** `cv2.VideoWriter` records video assuming a constant frame rate (30 FPS in this case). Actual frame arrival is jittery — network hiccups over the hotspot cause frames to arrive late or in bursts. If the analysis script computed frame timing as `frame_index / 30`, keystroke alignment would be wrong by up to hundreds of milliseconds in bad moments.

The sidecar records the `t_mono` value at the exact moment each frame was captured (before the buffer, before disk write). Analysis uses the sidecar for alignment, ignoring the MP4's internal clock.

### **Triple-threaded architecture: video, keyboard, inactivity monitor**

**Why:** These three concerns have fundamentally different timing characteristics and cannot share a loop without compromising one of them.

* Video reads frames as fast as the network delivers. Blocking on anything else causes drops.  
* Keyboard listener must respond with sub-millisecond precision to keypresses. Any I/O in the event handler poisons the timestamp.  
* Inactivity monitor runs on a slow (0.5s) poll. Putting this check inside either of the above would add useless work to their hot paths.

Shared state goes through a single `threading.Lock`. Lock held only for dict read/write, never during I/O. Standard pattern, keeps the concurrency boring.

### **Inactivity timeout AND manual Esc (either-or), not just one**

**Why:** Auto-stop prevents zombie sessions where I walked away mid-recording and forgot to end it. Manual stop prevents the timeout from killing a session during a legitimate typing pause. Both were cheap to implement and together cover the termination failure modes.

---

## **Phase 3 — MediaPipe API Choice**

### **Switched from Tasks API (HandLandmarker / GestureRecognizer) to legacy `mp.solutions.hands`**

**Why:** Started with the Tasks API because it's what Google actively maintains and it handles multi-hand detection with a clean async callback pattern. Worked great for the real-time visualizer during the exploration phase.

For the telemetry pipeline, the Tasks API became a liability. Async callbacks introduce race conditions between frame processing and CSV writes. Timestamp monotonicity is required by the Tasks API (`Timestamp must be strictly increasing`) and Apple Silicon's M3 was fast enough to occasionally generate duplicate millisecond timestamps, crashing the recognizer.

`mp.solutions.hands` is synchronous. Call `hands.process(frame)`, get results immediately on the same thread, write to CSV, move on. No race conditions, no timestamp panic, no shared global state between threads. The "legacy" label is misleading — for deterministic batch processing it's better architecture.

*Async:* running a given task and while you wait for it to finish, you load other tasks to do. 

*Sync:* running a given task and waiting until that specific task is entirely finished. 

### **Model complexity differs by phase**

**Why:** Real-time capture needs throughput. `model_complexity=0` gave 30+ FPS on the M3, at slightly reduced landmark accuracy. Hard to see the accuracy difference by eye, but it's there.

Offline analysis has no frame budget. `model_complexity=1` is the default for good reason — meaningfully better landmark precision, and processing 500+ frames in under a minute is fine when you only run it once per session.

The two scripts use different values for the same MediaPipe API deliberately. If you're rereading this and wondering why, that's why.

**Placed the L/R handedness swap in the extractor, not the capture script**:

MediaPipe assumes a selfie-view camera and mirrors its L/R labels accordingly. Our top-down rig has a different geometry, so labels are inverted relative to the physical hands. The fix could live in three places: 

* (a) the capture script as a pre-processing step,   
* (b) the extractor where labels first get written to CSV, or   
* (c) the overlay renderer as a display-time correction. 

Chose (b) because the CSV is the canonical dataset — everything downstream (overlay renderer, fusion script, eventual ML training) reads from it. If the fix lives in the extractor, no downstream script needs to know the mirror quirk exists. The canonical data matches physical reality, and consumers of that data don't carry knowledge about a camera artifact in their codebases. Fix at the capture layer would work but couples a data concern to a networking/IO script. Fix at the overlay layer means every future consumer of the data has to remember to apply the same correction — a subtle bug waiting to happen.

## **Phase 4 \- Fusion (Finger-to-Key Assignment)**

### **Built the fusion pipeline as an iterative arc (v1 → v2 → v2.1) rather than a single attempt**

**Why:** The fusion problem — given keystroke timing and hand-landmark trajectories, identify which finger pressed which key — has no obvious right answer. Multiple plausible heuristics exist, each with different failure modes. Building one version and shipping it would have meant guessing which heuristic was correct; building several and measuring them produced a real arc of falsified hypotheses and informed decisions.

The progression:

- v1: pure Z-velocity. The "intellectually honest" version that uses no layout priors. Falsified by ground-truth labeling at \~6% full accuracy.  
- v2: Z-velocity for hand selection \+ spatial nearest-fingertip-to-learned-key for finger selection, with QWERTY prior as the seed for learned positions. Achieves 84% on resolvable events.  
- v2.1: trajectory-matching across a 7-frame window, hypothesizing that press motion would have a clearer signature than single-frame position. Falsified by ground-truth comparison at 58% full accuracy.

Keeping all three scripts in the repo. The negative results are portfolio artifacts, not noise to delete.

### **Hand selection from QWERTY prior, not from Z-velocity**

**Why:** v1 and the initial v2.1 both tried Z-velocity for hand selection, reasoning that the descending hand would have larger summed Z-velocity than the lifting hand. Empirically this works on isolated keystrokes, but in fast typing the "non-pressing" hand is often executing a larger Z motion than the pressing hand — preparing for the next key, transitioning between rows, recoiling from the previous press. The Z-velocity signal of "this hand is descending right now" doesn't correspond to "this hand pressed the current key."

The QWERTY finger map gives 100% hand accuracy on the labeled set. It works because for letter keys, the hand assignment is structurally fixed — F is always pressed by the left hand, J always by the right, regardless of typing style. We don't need to recover hand from kinematic data; the keymap tells us.

Trade-off: this only works for keys whose canonical hand is unambiguous in QWERTY (letters, most punctuation). Spacebar (rule-assigned to "thumb, unknown hand") and modifiers (filtered) are handled separately.

### **Used spatial nearest-fingertip-to-learned-key instead of motion-based heuristic for finger selection**

**Why:** Z-velocity carries enough signal to distinguish hands but not fingers within a hand. The five fingertips of one hand all move together during a press because the entire hand descends as a unit; their per-finger Z-velocities differ by \~3×10⁻⁴, which is at MediaPipe's per-frame noise floor.

X-position spread is roughly 100× larger than Z-velocity spread. Each fingertip occupies a distinct X-lane on the keyboard with \~0.05–0.10 normalized-units of separation, against a noise floor of \~0.001. Using spatial proximity to a learned key position is a vastly stronger signal than any motion-derived feature we tested.

The "learned key position" is computed by aggregating fingertip positions across all presses of each key and fitting an affine transform from QWERTY canonical coordinates to image coordinates. Residual error 5.6% of keyboard span, well under the 8% threshold for trustworthy rare-key inference.

### **Adopted ground-truth labeling as the primary evaluation metric**

**Why:** The original development workflow used indirect signals to judge fusion quality — distribution shape, confidence scores, internal consistency. These are useful diagnostics but not measurements. v1's plausible-looking distribution turned out to mask 6% accuracy. v2's distribution looked similar in shape but achieved 84%. Without ground truth we couldn't have told them apart.

Built `label.py` as a small utility that plays the overlay video at each keystroke moment and records the visually-correct finger. 30 events, \~15 minutes of work, produced the validation set that retroactively justified every fusion-design decision and falsified v2.1.

The portfolio cost-benefit is decisive: 15 minutes of labeling unlocks the ability to say "v2 is 84% accurate" rather than "v2 looks plausible." The labeled set also serves as a held-out eval set for any future iteration.

### **Did not pursue v3 (iterative bootstrap of learned layout)**

**Why:** v3 was originally motivated by the concern that v2's QWERTY-prior seeding would bias the learned key positions, particularly for users whose actual finger mappings deviated from touch-typing convention. The hypothesized fix was iterative re-learning: pass 1 with QWERTY, learn centroids, pass 2 with learned centroids, re-learn, until convergence.

After labeling, the data falsified the motivation. The failure mode in v2 was not QWERTY bias — hand selection from the QWERTY prior achieved 100% accuracy. The remaining errors (3 of 19 events, all adjacent-finger same-hand confusions) are at the spatial-resolution limit of the rig, not the algorithmic limit. Bootstrap iteration cannot increase MediaPipe's landmark precision.

Documenting the un-built v3 is itself the decision: we measured, learned the motivation was wrong, and chose not to build. That's a research-taste choice in the same arc as the v2.1 experiment that we did build.

### **Accepted v2's 84% per-event accuracy as the production ceiling for this rig**

**Why:** Per-event accuracy is bounded by the spatial signal-to-noise ratio of the capture rig. With an 820×600 crop covering a \~30cm keyboard (\~27 pixels/cm), a home-row press produces \~5mm of vertical key travel ≈ 13 pixels of fingertip Y-descent. MediaPipe's landmark jitter at this resolution is empirically several pixels (visible in the palm-stability plot from inspect\_session.py). The signal-to-noise ratio at the press moment is roughly 2:1 in the best case and often closer to 1:1 for fast, micro-flex home-row presses.

Algorithms cannot extract signals smaller than their input's noise floor. v2.1's failure (trajectory matching at 58%) confirms this empirically: motion-based features carry less signal than spatial-position features for this rig's resolution and this user's typing style.

The remaining path to \>84% per-event accuracy is hardware (higher fps, higher spatial resolution) or richer sensor data (per-key force sensors, external cameras at oblique angles for depth disambiguation). Neither is in scope for the project's timeline.

The 84% ceiling is acceptable because Phase 4's analysis operates on *aggregated* statistics (per-key finger distributions, per-bigram speeds, per-finger workload) where per-event noise averages out across many samples. A claim like "your H key is pressed by Right index 87% of the time" is robust to the underlying 16% per-event noise as long as that noise is not systematically biased — which the labeling confirms it is not (errors are adjacent-finger, not cross-hand or cross-finger-class).

### **Concluded the v2 architecture as production at 84.8% per-event accuracy**

**Why:** Two independent attempts to push past v2's ceiling (v2.1 trajectory matching, v2-wrist coordinate correction) both regressed accuracy with identifiable mechanisms. The remaining errors in v2 are uniformly adjacent-finger same-hand confusions concentrated on specific keys (backspace, h, m, i) — exactly the failure mode predicted by the spatial signal-to-noise floor analysis.

The strategic decision: stop tuning fusion. The project's deliverable is biomechanical pattern identification through aggregated analysis, where per-event noise at this level averages out across many samples. A claim like "your H key is pressed by Right index 87% of the time and Right middle 13%" is robust to v2's residual 16% per-event noise as long as errors are not systematically biased — which the labeled set confirms they are not.

Moving to Phase 4\. v1, v2, v2.1, and v2-wrist all preserved in the repo as the experimental record.

### **Phase 5 — Monkeytype API integration: built and abandoned.**

Decision: drop the Monkeytype historical bigram analysis after the API research revealed it can't support per-bigram career analysis. Worth capturing the *reasoning chain* not just the conclusion: API exposes keySpacing (timing) but not the actual word sequences typed in each test, so per-bigram alignment isn't possible. Quote-mode tests would be alignable but you've never used quote mode. Words-mode tests can yield position-based timing patterns but not character-level. Decided the analytical content didn't justify the engineering. Pivoted away from Monkeytype entirely. Built career-baseline scripts (monkeytype\_client.py, historical\_profile.py, cross\_reference.py) that were never wired into the production pipeline.

This is a *good* falsification to log because it demonstrates calibrated reasoning about what data can and can't support what claims.

### **Phase 6 — Continuous capture daemon: shipped.**

Decision: build a long-running daemon that captures keystrokes via pynput, manages session boundaries, and triggers analysis automatically. Architecture choices that should be logged:

The state-machine design (IDLE → ACTIVE → ENDING → IDLE) and why three states instead of two.

The CLI-via-command-file IPC pattern instead of sockets/RPC. Reasoning: simpler, robust to daemon restarts, debuggable by `cat`\-ing the file.

The choice to spawn `capture_p1.py` as a subprocess rather than reimplementing video capture inside the daemon. Reasoning: tested code path, isolated failure modes, accept losing pre-roll buffer as the cost.

The pivot from auto-detection to manual-only sessions. Initially designed with auto-detection (5 min idle \= session end). You pushed back, wanting explicit control. Documented this is the privacy/safety reasoning — manual sessions mean opt-in keystroke logging.

Threshold settings for auto-analysis: 200 letter keystrokes minimum, 60% letter fraction. Reasoning: filters out short non-typing-practice activity from the analysis pipeline.

### **Phase 6.5 — launchd integration: shipped.**

Decision: launchd plist with `RunAtLoad: false, KeepAlive: false` after initial pass had auto-start enabled. The pivot reasoning matters here. Originally enabled auto-start because the ambient-tool framing called for it. You pushed back saying you wanted to choose when sessions begin. Manual loading via `typist begin` is more aligned with privacy-conscious use.

The Accessibility-permission-doesn't-transfer-from-terminal-to-launchd quirk is worth a one-line note for future-you. macOS treats launchd-spawned processes as separate trust subjects.

### **Phase 7 — Rules engine integrated into analyze.py: shipped.**

Decision: instead of a separate rules engine module, add three new analytical layers directly to analyze.py. Reasoning: rules are simple enough they don't warrant a separate module, and integrated they appear in the same report rather than a disjoint one. The three layers:

Backspace error backtracing — for each backspace, identify the deleted character, the preceding character (forming the error-bigram), and the subsequent intended character. Output: ranked error-bigram patterns.

Abandonment trigger detection — tab+enter sequences within 500ms identified as Monkeytype restart attempts; preceding keystroke logged as the "tilt point."

Ergonomic deviations — comparison of dominant-finger-per-key against canonical touch-typing fingerings. Output: deviations flagged with framing that "deviation isn't necessarily wrong, but it's a tractable lever."

The ergonomic-deviation framing is the architecturally interesting part. Capture: the system needs to model your actual habits (for fusion accuracy) AND surface where habits diverge from optimal (for coaching). Solving this requires separating "describing what you do" from "describing what's optimal" — the static canonical-fingerings dictionary lives in the rules layer, the measured-habits live in the consistency analysis, and the deviation surfaces only at the report level.

### **Phase 8 — Coaching note generator: shipped.**

Decision: build a Python templating script that generates a conversational coaching note (`feedback.md`) alongside the technical report, with the explicit voice goal of "knowledgeable peer talking, not technical report." Voice was iterated by hand-drafting an example, agreeing on tone, then writing the generator to reproduce that tone for arbitrary sessions.

Architectural choice worth logging: rules-based templating chosen over LLM-generated prose. Reasoning: predictability over naturalness, no API key dependency, no rate limits, no hallucinations. LLM-based version explicitly deferred as "experimental Phase 8" that may or may not ship.

Specific design decisions in the templating:

Conditional sections — each compose function returns None if the data doesn't support a meaningful finding, and the main flow concatenates only non-None outputs.

Historical comparison thresholds — 5% movement on metrics, 3+ session recurrence on patterns. Below these thresholds the comparison is suppressed to avoid filling the note with non-findings.

Drill word generation — embedded \~350-word common-English wordlist, mapped to specific finding types via dedicated word-selection functions. Format: main drill always, bonus drill only if a meaningfully different second finding exists.

### **Phase 8b — Email delivery: shipped.**

Decision: integrate Gmail SMTP delivery into feedback.py rather than a separate email module. Reasoning: tightly coupled to feedback generation, splitting adds an import dependency for no architectural benefit.

Specific decisions worth logging:

Markdown→HTML conversion written from scratch rather than using a library. Reasoning: minimal dependency surface, tuned to the specific Markdown subset feedback.py actually emits (handles fenced code blocks for drill words, which most generic converters handle awkwardly).

python-dotenv for .env loading, plist EnvironmentVariables for daemon-spawned runs. Belt-and-suspenders covering both invocation paths.

Silent skip on missing credentials — feedback.md is always written, email is opportunistic. Three opt-out mechanisms: \--no-email flag, TYPIST\_DISABLE\_EMAIL=1, missing credentials.

### **Phase 9 — Daemon auto-pipeline integration: shipped.**

Decision: wire feedback.py into the daemon's trigger\_analysis\_pipeline so coaching notes generate and email automatically after every analyzed session. Closes the loop from "type → email arrives" without manual intervention.

The pipeline now runs: keystroke capture → video capture (if enabled) → landmark extraction → fusion → analyze → feedback → email. End-to-end automated. The two-command workflow (`typist begin --record` / `typist end`) is the user-facing surface.

**A meta-decision worth logging:**

The repeated pattern across this development arc has been *resisting feature bloat*. Specific instances: pushing back on the 600-line aggregator (rewritten to 70), declining to build the LLM-feedback layer in the MVP, dropping Monkeytype after the API research, choosing not to extend the wordlist beyond what fits in a single file. The discipline to ship the smallest version that works and resist additions is itself a research-relevant decision.

---

## **Failure Log**

### **The Apple Silicon AVFoundation wall**

**Error:** `OpenCV: out device of bound (0-0): 2` for every index except 0\. **False assumption:** That any virtual camera visible to macOS would be visible to OpenCV via a camera index. Wrong — Apple's AVFoundation sandbox gates camera enumeration, and the sandbox blocks uncertified processes from seeing virtual devices entirely. **Paradigm shift:** Stop treating the phone as a camera. Treat it as a network video source. HTTP streams bypass AVFoundation completely.

### **The RNDIS hole in macOS**

**Error:** Browser test of the tethered IP returned "This site can't be reached." The phone reported USB Tethering active; Mac showed no new network interface. **False assumption:** That USB Tethering is a standard cross-platform feature. Wrong — Android uses RNDIS, which Apple removed from macOS support years ago. Windows and Linux see the tethered phone as a wired router immediately. Mac sees a charging cable. **Paradigm shift:** Abandon wired tethering on macOS entirely. Use the phone's hotspot as a direct wireless bridge, accept that you lose internet during capture.

### **The "Google Meet doesn't see my camera either" data point**

**Error:** Camo Studio appeared in Camo's own UI but not in Google Meet's device list. **What it meant:** The issue wasn't specific to Python. Any sandboxed application (including Chrome) couldn't see the Camo virtual camera on this particular macOS version. This should have been the cue to pivot hours earlier — if the system itself can't see the camera, no amount of Python code is going to fix it. **Lesson:** When debugging a software integration, test the cheapest non-code check first. "Does this camera appear in any other app?" would have saved hours.

### **The silent NumPy slice bug**

**Error:** No error. Video recorded successfully, but the output dimensions were 570×560 instead of the expected 810×560. **False assumption:** That slicing a NumPy array beyond its bounds would raise an error. Wrong — NumPy silently clips to the array's actual size. A frame that was supposed to be cropped `[150:960, 700:1260]` from a 1080p source was instead cropped against a 720p source (because IP Webcam's resolution setting changed) and silently returned a 570-row slice instead of 810\. **Paradigm shift:** Validate frame dimensions explicitly after the transform, not before. The capture script probe logs the raw dimensions to confirm 1920x1080, but the `transform_frame` function needs to either assert the expected output size or the capture script should fail loudly when the IP Webcam resolution changes. **Lesson:** Silent correctness bugs are worse than loud errors. Add an assertion after the first transformed frame that the output dimensions match expectations. A one-line check that would have caught this before 18 seconds of garbage video got recorded.

### **The open-close-flush pattern on every keystroke**

**Error:** None, but the pattern was wrong. The first draft of the keystroke logger opened the CSV, wrote a row, and closed it on every press and release event. **Why it was wrong:** At 120 WPM, that's \~20 filesystem syscalls per second. Not a functional bug, but wasteful, and it would have made long sessions scale poorly. **Fix:** Open the file handle for the session's lifetime, flush every 2 seconds. Standard logging pattern. **Lesson:** Even when something "works," treat the code as a proposal, not a solution. If the I/O pattern is weird, it will bite you later when the session length grows or the event rate increases.

### **The invisible wrist hallucination** 

**Error:** None — all quality metrics looked pristine. Tracking rate 100%, handedness confidence 0.98 average, 697/697 frames with both hands detected. 

**False assumption:** That aggregate tracking quality metrics meaningfully indicate landmark correctness. Wrong — MediaPipe's Solutions API doesn't expose per-landmark confidence, only per-hand detection confidence. It's entirely possible for the model to confidently report hand presence while inventing positions for occluded landmarks. 

**What was actually happening:** The capture crop (CROP\_Y\_END=960) was clipping the forearms out of frame. When the wrist landmark was outside the visible region, MediaPipe didn't report "wrist missing" — it hallucinated a wrist position based on its internal hand model. The fingertips stayed accurate (they were visible); everything from the MCP joints back was made up. Only caught by visually scrubbing the overlay video frame-by-frame. 

**Paradigm shift:** Tracking rate is necessary but not sufficient for landmark quality. Visual verification of the skeleton overlay is non-negotiable before trusting landmark data for downstream analysis. 

**Fix:** Widened CROP\_Y\_END from 960 to 1080 (full frame height). Gave MediaPipe another 120px of vertical room. Re-recorded session; ran a geometric sanity check (palm size \= wrist-to-middle-MCP distance) and confirmed zero outliers \>60% from session median. Hallucination gone. 

**Lesson:** Build a visual verification loop into every phase, not just the fusion phase. The metrics that look cleanest are the ones most likely to hide structural bugs.

### **v1's Z-velocity heuristic was at the noise floor**

**Error:** v1 ran end-to-end, produced events.csv with 113 fused events at median confidence 0.110. Distribution looked superficially plausible (no single finger over 20% of assignments). Actual accuracy on the labeled set: 6% full, 33% hand. **False assumption:** That Z-velocity carries enough information to distinguish both hands and fingers within a hand. It does carry enough for hand (left summed-Z and right summed-Z have opposite signs during a press — descending hand vs. lifting hand). It does not carry enough for finger, because all five fingertips of the descending hand move together as a unit. **Paradigm shift:** Don't ask one signal to do two jobs. Use Z for hand, spatial position for finger. The split fixed the architecture. **Lesson:** A heuristic's outputs can look reasonable in shape while being wrong in content. The shape (distribution histograms, confidence scores) was checked in the inspect utility and looked fine. Only labeling caught the fact that the answers were near-random.

### **The "no fingertip on the key" skip rate was a temporal alignment issue, not an ambiguity issue**

**Error:** Initial labeling had a 37% skip rate. First interpretation: the overlay frames were visually ambiguous and labels couldn't be established with confidence. **False assumption:** That skip-rate corresponded to "image quality." The user clarified that skips happened because no fingertip was visibly near the pressed key in the chosen frame — meaning the fusion script was looking at the wrong frame relative to the actual press moment. **Paradigm shift:** Single-frame matching at the keystroke timestamp is the wrong analysis unit for a continuous motion. The right unit is a window that captures the trajectory of the press. This motivated v2.1 (trajectory matching across a 7-frame window). **Lesson:** When users skip events in a labeling task, *ask why they skipped*. "Skip" is overloaded — it can mean ambiguous image, missing data, or "the question doesn't make sense given what I'm seeing." All three have different implications.

### **v2.1 (trajectory matching) was strictly worse than v2 (single-frame spatial)**

**Error:** v2.1 achieved 58% full accuracy vs. v2's 84%. 8 new errors introduced, 3 of v2's errors fixed. Net regression of 5 events on a 19-event set. **False assumption:** That the pressing finger has a larger Y-excursion in the window around a keystroke than non-pressing fingers, because the press is the obvious motion in the scene. **What was actually happening:** For fast home-row typing, the pressing finger barely moves (\~5mm of key travel ≈ 13 pixels). Meanwhile, non-pressing fingers exhibit larger natural oscillation amplitudes due to hand-wide preparation motion and anatomical coupling. Event 88 ('f', truth Left index): the ring finger had 3.6× the index's Y-excursion in the press window, with the same temporal alignment to the keystroke. Trajectory matching confidently picked the wrong finger. **Paradigm shift:** "The press is the loudest motion" is true for slow, deliberate typing where each press is a discrete reach-and-strike. It is false for fast, home-row-based typing where presses are micro-flexes embedded in continuous hand motion. Different typing styles need different fusion approaches; the one we built doesn't generalize across styles. **Lesson:** Reverted v2.1 and kept v2. Documented the experiment as a measured negative result. The experiment was not wasted — it produced the specific characterization of "press signal is quieter than resting oscillation," which is the sharpest argument for the spatial-resolution ceiling.

### **v2-wrist (wrist-relative coordinates) regressed to 80.3% from v2's 84.8%**

**Error:** Same evaluation set as v2.1 (66 ground-truth labeled events). v2-wrist achieved 80.3% full accuracy vs v2's 84.8% — a 4.5 percentage-point regression. Hand accuracy stayed at 97% for both, confirming the regression was entirely in finger selection. **False assumption:** That hand drift across the session was adding noise to v2's learned key positions in absolute image coordinates, and that subtracting the wrist landmark per frame would remove that drift, leaving cleaner within-hand finger geometry. **What was actually happening:** The wrist landmark itself is noisy at the per-frame level. Subtracting a noisy reference point from each fingertip adds noise rather than removing it. The right-hand affine fit residual jumped from v2's 5.6% (single transform on absolute coords) to 15.1% in wrist-relative space — a quantitative confirmation that wrist-relative coordinates were *less* well-behaved than absolute. The diagnostic plot from inspect\_session.py had hinted at this with a palm-size CV of 0.12 — geometric stability was decent but not perfect, and "not perfect" turned out to be enough to dominate the small per-finger spatial differences we were trying to measure. **Paradigm shift:** Reference-frame correction only helps when the reference frame is more stable than the noise it's correcting for. Here the wrist's per-frame jitter (\~4-5 pixels) is comparable to the inter-finger spatial separation we're trying to resolve. Wrist-relative adds noise instead of removing it. **Lesson:** Reverted v2-wrist and kept v2 as production at 84.8%. The combination of v2.1's failure (motion-based methods can't extract a sub-noise-floor signal) and v2-wrist's failure (coordinate-frame correction adds noise when the reference is itself noisy) bounds v2's accuracy ceiling from two independent directions. Both failures are documented as portfolio artifacts: hypothesis → experiment → falsification → informed decision.

---

## **Open Questions / Deferred Decisions** 

### **Higher-fps capture (deferred from earlier; still deferred)**

**Why deferred:** v2.1's failure showed that the per-event accuracy ceiling on the current rig is data-limited at the spatial resolution level, not purely the temporal one. Higher fps would help narrow the time window per keystroke (cleaner trajectory signals per press) but would not increase the per-frame pixel resolution. The actually-effective hardware change would be higher *spatial* resolution (4K capture or a closer camera mount), which has its own complications (network throughput, MediaPipe inference cost). Neither is in scope for the current project, but if a v3 of this project were ever built, this is the first lever to pull.

### **Custom-trained classifier for finger assignment (the "Stage 7" ML step)**

**Status:** Now justified, but not now built. The v2.1 negative result sharpened the case for ML: the remaining errors in v2 are all adjacent-finger same-hand confusions, where the spatial signal is at the noise floor. A small classifier trained on labeled events with features beyond raw position (relative finger geometry, hand pose, temporal context) could plausibly push past 84%. But this requires more labeled data than 19 events. Decision: not in scope for the MVP.  
