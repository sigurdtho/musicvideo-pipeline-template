# HUD style: Launch Window (layer 2)

Status: **decided 2026-10-13 (r906b, Plan Q A: the user said yes to on-screen HUD text).** It is the type spec for PLAN v4 §6.2 and is meant to be adopted into `style_guide.md` when that file is drafted. `style_guide.md` is still the unapproved template, and its "On-screen text: none / Noto Serif / no terminal HUD" lines are template defaults that this file replaces. Still generation waits for the style guide either way.

## 0. The one rule

**Every letter and digit on screen is drawn by code on layer 2, never by a model.** That covers the HUD, the slams, the counter, the end card and any hull lettering (§8). Image and video prompts carry no text (CLAUDE.md rule 5), and the negative line keeps `text, letters, lettering, logo, wordmark, numbers, watermark`. Any glyph a model draws anyway is a QA fail: a Klein local edit to blank, or a crop.

Layers: **1** = the plate (still or Wan clip, graded). **1b** = plate-attached composites (hull lettering, §8), which move with the plate and are graded into it. **2** = the screen-space HUD in this file, rendered by headless Chrome (`pipeline/overlay.py`, 1920×1080 at 24 fps, alpha PNG/WebM) and composited last, after the grade, so the type stays clean.

## 1. Fonts (all OFL, already in `slopkit/templates/fonts`)

| Role | Font | Weight / setting | Used by |
| --- | --- | --- | --- |
| Slam | **Anton** Regular | All caps, tracking +2 %, line height 0.9 | `slam`, thumbnail words |
| Data | **JetBrains Mono** | 500; tabular by nature, so digits never jitter; all caps | `counter`, `telemetry`, `label`, `phase` |
| Place / prose | **Inter** (variable) | 600, all caps, tracking +18 % for `anchor`; 400 sentence case for the end card | `anchor`, end card |

- Minus sign is U+2212 (`T−10`), never a hyphen. The separator is the middle dot `·` with a thin space on each side.
- No other fonts. Orbitron, faux-terminal green, glitch fonts and karaoke stay out (they read as a stock "sci-fi HUD", not mission ops).

## 2. Colour

From PLAN §6.1, three colours plus white:

| Token | Hex | Use |
| --- | --- | --- |
| `ink` | `#F2F4F6` | All primary type |
| `tint` | `#9DB4CC` | Secondary type (units, small labels), orbit lines for Mars-side elements, brackets |
| `accent` | `#FF5A1F` | **One** element at a time: the live counter digit at T−10…T−1, the orbit arc, the "WE GO"/"ARE" swap word. It counts toward the ≤ 2 orange elements per frame (plume and robot shoulders are the others) |
| `night` | `#05070A` | Halo and scrim only |

- **Legibility:** every text element gets a soft halo of `0 0 14px rgba(5,7,10,0.55)`, with no hard drop shadows. Before compositing, the overlay QA measures the plate's mean luminance under each text box. If the WCAG contrast ratio against `ink` is below 4.5:1 (bright steam, floodlights, sunlit hull), a local radial scrim of `night` at ≤ 35 % goes in behind that element only. No full-frame darkening.
- No gradients in the letters, no glow on body type. The only bloom is on the `slam` impact frame (§4).

## 3. Sizes and safe areas

**16:9 master (1920×1080).** Title-safe is the inner 90 % (96 px left/right, 54 px top/bottom). Nothing crosses it.

| Kind | Position | Size (cap height at 1080p) |
| --- | --- | --- |
| `slam` | Centre or lower-centre third, per shot (masked shots: wherever the matte sits) | 150–210 px; a 2-word slam fills ≤ 70 % of the width |
| `counter` | Top-right, right-aligned to x = 1824, baseline y = 120 | Digits 46 px, unit 22 px in `tint` |
| `anchor` | Bottom-left, x = 96, baseline y = 1026 | 20 px, tracking +18 % |
| `label` | Next to its bracket, 12 px off the bracket corner | 18 px |
| `telemetry` | Bottom-right block, right-aligned to x = 1824 | 22 px, 2 lines max |
| `phase` | Under the counter | 22 px |
| `orbit` / `grid` | Full frame inside title-safe | Lines **2 px** (v3 said 1 px, which vanishes on phones); planet dots 10 / 8 px |
| End card | Centred column, max width 1200 px | Inter 400, 30 px; line height 1.4 |

**9:16 Shorts (1080×1920).** Avoid the platform UI: keep the top 240 px clear (status and search), the bottom 420 px (title, captions, music chip) and the right 150 px (like/comment/share rail). The text-safe box is therefore x 60–930, y 240–1500.
- `counter` moves to top centre (baseline y = 330).
- `anchor` sits just above the safe-box floor (y = 1480).
- `slam` stays centred, at ≤ 80 % of the safe width.
- Sizes scale ×1.25 from 1080p (the type has to survive a phone at arm's length).

**Thumbnail (1280×720):** at most 2 words in Anton, white, cap height ≥ 18 % of the frame height, kept out of the bottom-right 20 % (the duration stamp).

## 4. Timing and animation

Everything is on the take's grid: 128.0 BPM, beat = 0.469 s ≈ 11.25 frames at 24 fps, bar *n* starts at 0.02 + 1.875 (*n* − 1) s. Event times are taken from the stage-2 beat grid and quantised to the nearest frame, never typed in by hand.

| Kind | In | Hold | Out |
| --- | --- | --- | --- |
| `slam` | **Lands on the impact frame** (the downbeat frame itself, no pre-roll): 2 frames at 106 % scale → 100 %, plus a 1-frame bloom on the letters | Until the next line cue, ≥ 1.0 s | Hard cut on a beat, or masked out behind the subject |
| `slam` word swap (HERE IT COMES → HERE WE GO → HERE WE ARE) | The old word drops 40 px and fades over 3 frames, ending **on** the downbeat; the new word lands on the downbeat frame in `accent`, then settles to `ink` over 1 beat | — | — |
| `counter` tick | Hard digit swap on the beat, no tween (a mission clock doesn't roll) | — | — |
| `counter` unit change (MONTHS → DAYS → SECONDS) | The new unit types in at 1 character per frame on the sung word | — | — |
| `label` | The bracket draws in over 6 frames (corners first, then edges); the text types at 30 cps after it | ≥ 1.5 s | Reverse draw, 4 frames |
| `anchor` | Fades in over 8 frames at the start of a section | The whole section | Cut |
| `telemetry` | Numbers tick every frame (ALT/VEL follow a code-computed ascent curve, not random digits) | — | Cut |
| `orbit` arcs | Draw at constant angular speed; the transfer arc draws over exactly 2 bars (row 14) | — | — |
| `grid` | Cells flip on beats | — | — |

- **Reading time:** anything that is meant to be read stays ≥ 1.0 s and ≤ 3 words per second. Counter ticks are exempt, since they are read as motion.
- **Flash limit (code-enforced, PLAN §6.4):** ≤ 3 full-screen flashes in any 1 s, ≥ 0.25 s apart. Slam blooms count as flashes when they brighten more than 25 % of the frame.
- **Masked slams** (rows 13, 25, 34) sit *behind* vapour or the robot's head. The matte is computed on the CPU (luminance key for steam, plus a hand-tuned shape for the head on a still). If the matte reads poorly, the slam goes in front at 85 % opacity. No GPU segmentation in the overlay pass (rule 2: one GPU job at a time).
- **Bridge:** the counter is the only text (rows 30–33). **Rows 41–42:** only the new counter ticks; the film ends on one stark image.

## 5. Text by shot (rows from PLAN §9 / `pipeline/shot_plan.py`)

| Row | Shot id | Text |
| --- | --- | --- |
| 1 | `in_orbit` | `slam` EVERY 26 MONTHS (0:00.5–0:03.0) |
| 2 | `in_pad` | `anchor` EARTH · PAD |
| 3 | `in_lights` / `in_frost` / `in_eye` | none |
| 4 | `in_l0` | `counter` T−26 MONTHS appears |
| 5 | `in_kick` | `counter` T−26 MONTHS |
| 6 | `in_l1` | `orbit` arc in `accent` on "window" |
| 7 | `v1_stack` | `counter` T−20 MONTHS; `label` STACK |
| 8 | `v1_built` | `label` PAYLOAD 01; `counter` T−14 MONTHS |
| 9 | `v1_lookup` | `counter` T−7 MONTHS |
| 10 | `v1_redlight` | `label` MARS (bracket on the reflection); `counter` T−3 MONTHS |
| 11 | `p1_catch` / `p1_flare` / `p1_rain` | `counter` MONTHS → DAYS → SECONDS on the words; `label` CATCH; `telemetry` TESTS 7 |
| 12 | `p1_consoles` / `p1_dunes` / `p1_orbit` | `counter` WINDOW OPENS IN 00:00:08… per beat; `phase` 44° |
| 13 | `c1_comes` | `slam` HERE IT COMES (masked behind the vapour); `counter` WINDOW OPEN |
| 14 | `c1_arc` | `label` DEPARTURE ARC |
| 15–16 | `c1_beach`, `c1_salute` | none |
| 17 | `d1_*` | `counter` T−3 DAYS |
| 18 | `d1_grid`, `d1_dawn` | `counter` racing → T−1 DAY |
| 19 | `v2_tower` | `anchor` EARTH · PAD |
| 20 | `v2_seeds` | `label` SEEDS |
| 21 | `v2_photo` | none |
| 22 | `v2_arm` | `counter` T−00:30:00 |
| 23 | `p2_orbit`, `p2_frost` | `phase` 44°; `counter` units swap; T−10 on the beat at 1:32.9 |
| 24 | `p2_hold` | `counter` T−9 … T−5 on the beats (live digit in `accent`) |
| 25 | `p2_steam` | `slam` T−4 … T−1, one per beat, masked behind the steam |
| 26 | `c2_liftoff` | `slam` HERE IT COMES → HERE WE GO (swap on the drop downbeat) |
| 27 | `c2_world` | none |
| 28 | `c2_climb` | `telemetry` ALT / VEL |
| 29 | `c2_ground`, `c2_limb` | `counter` DAY 1; `anchor` TRANSIT |
| 30 | `br_quiet` | `counter` DAY 87 |
| 31 | `br_seed` | none |
| 32 | `br_blue` | `counter` DAY 214 |
| 33 | `br_foot` | `anchor` MARS |
| 34 | `p3_lookup` | `label` EARTH (bracket); `slam` LOOK UP, masked behind the head |
| 35 | `p3_point` | the bracket only |
| 36 | `c3_sprout` | `slam` HERE WE GO → HERE WE ARE ("GO" swaps out) |
| 37 | `c3_dusk` | none |
| 38 | `c3_next` | `anchor` EARTH · PAD |
| 39 | `c3_fan` | `counter` DAY 1 / T−26 MONTHS, both sides |
| 40 | `c3_grid` | none (flash-limited grid) |
| 41 | `out_two` | `counter` a new T−26 MONTHS starts ticking on "months" |
| 42 | `out_orbit` | `counter` ticking until the fade |
| — | end card (2:37.2 →, ~12 s) | Inter 400: "AI-made, locally on one GPU. The rocket is SpaceX's Starship, shown as a fan tribute; the robot is an original design. Not affiliated with or endorsed by SpaceX or any space agency. The launch window is real: every ~26 months." |

**Vocabulary rules:**
- No years and no dates anywhere (PLAN §0).
- HUD labels stay generic mission-ops words (STACK, CATCH, PAYLOAD 01). The vehicle's product names may appear only on the end card and in post copy. The HUD is the story's voice, and a product name in it makes a frame read as an ad without telling the target crowd anything new.
- No person's name, ever.

## 6. Build and QA

- Each event is one JSON entry (kind, text, t_in, t_out, position, mask) rendered by `pipeline/overlay.py` with a new `hud.html` template (to be written; `box.html` is the notice-box template from another film). Each rendered sequence gets a manifest entry (stage 9, kind `overlay`).
- QA per overlay pass: (1) every event lands within ±1 frame of its grid time; (2) the contrast check from §2; (3) the flash-limit check; (4) a 9:16 render checked against the Shorts safe box; (5) a spell-check of every string against this file.

## 7. Shorts and the X cut

Same events, re-timed to the cut's own timeline (`t − cut_start`, plus the edit offset after S1's bar-12 → bar-21 splice). Every event must lie wholly on one side of a splice: no counter tick or slam may straddle the edit. The post caption is platform text, not burned in.

## 8. Hull lettering and livery (layer 1b, decided 2026-10-13)

Real SpaceX livery is allowed (PLAN v4). Models garble lettering, so:

1. **The default is to omit it:** prompts ask for the hull's bare steel and black tiles, and the negative line keeps the lettering words. A blank hull is still an accurate view from most angles and in most lighting.
2. **Composite only on held stills** (still + code push), never on Wan clips: a moving hull needs tracking that a 13-day schedule can't QA. The mark is warped onto the hull cylinder by code (measured axis and radius), multiplied by the plate's luminance, matched in blur and grain, and placed where dated photos of the same vehicle version put it.
3. **The source must be the real mark** (an official vector). A wordmark retyped in a look-alike font reads as fake to exactly this audience. With no clean official vector, the answer is omit. The same goes for the flag (stripe and star counts are a garble risk for models, but not for vector code).
4. **Elon likeness:** never, in any layer.

## Addendum 2026-10-13 (the user): slopcore density + synced lyrics overlay (full or partial, at minimum all choruses/hooks) using alignment timings.
