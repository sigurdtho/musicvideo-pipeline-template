# Example: scripts from one finished film ("Launch Window")

These files are copied from a real project and are **project-specific examples**, not drop-in modules. They import
the core `pipeline` package and use that film's constants (128 BPM bar grid, shot ids, asset ids such as
`song:take:r3:14`, a rocket/launch-tower subject). Read them as patterns; adapt constants before use.

| File | Pattern it shows |
| --- | --- |
| `pipeline/song_rounds.py` | Stage-1 song gate on top of the core score: `BAR = 0.9`, kick check from Drop 1, 6-10 s fade/ring-out ending (no hard stop), length 150-170 s, GPU lock via `$GPU_LOCK` |
| `pipeline/song_regen.py` | Regenerate takes with an instrumental outro tag; ending measurement (fade / ring-out / short release); copies listen MP3s to `$REVIEW_DIR` |
| `pipeline/stage1.py` | That film's stage 1 (core version is `pipeline/stage1.py`) |
| `pipeline/hud.py` | HUD overlay layer: spec from shot list + beat grid + WhisperX words, checks (flash limit, beat alignment, overflow), alpha PNG render in headless Chrome |
| `slopkit/templates/lw_hud.html`, `lw_endcard.html` | The HTML/JS the HUD and end card are rendered from (every glyph is code) |
| `pipeline/sync_check.py` | Layer-2 sync QA: every HUD event on a beat / word / cut; lyric highlight equals word start; pixel test in Chrome |
| `pipeline/starship3d.py` | Code-guide renders: a numpy ray caster builds the subject with exact counts and proportions; Klein img2img then photorealises the guide (guide, never plate) |
| `pipeline/klein_edit.py` | One-off Flux.2 Klein edits of a single image with manifest entries |
| `pipeline/clipqa.py` | Deep clip QA: contact sheets of >= 16 frames per clip, optional zoom box |
| `pipeline/final1080.py` | Final 1080p: SeedVR2 7B upscale of Wan clips + RIFE x3, code layers re-rendered at 1080p, two-pass loudnorm to -14 LUFS / -1 dBTP |
| `pipeline/gpu.py` | Later version of the core GPU guard |
| `pipeline/tools/depth_matte.py` | Depth Anything V2 mattes for compositing |

Not copied (too project-specific): `rocket_round.py`, `rocket_plates.py`, `r906c.py`, `song_oomph.py`, `short1.py`.
Some docstrings still mention them; paths such as `video/shotlist.json` and `audio/timing/*.json` are that film's
generated data, which is not included.
