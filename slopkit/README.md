# slopkit: slopcore / motion-graphics mode for the music-video pipeline

CPU-side tools for anabology-style videos: the edit, the UI and the typography carry the pace, and the GPU only
renders paintings and short clips. Run everything from the project root with the shared pipeline env (a read-only symlink, nothing installed into it):

```bash
PY=envs/pipeline/bin/python      # numpy, opencv, librosa, scipy, playwright
cd ~/Claude/__SLUG__ && $PY -m slopkit.smoke --workers 8           # CPU smoke tests + 10 s sample strip
```

Chrome is `/usr/bin/google-chrome-stable`, launched with `--disable-gpu`. Fonts (OFL) are in `templates/fonts`.

| Need (master.md "Pipeline extensions") | Module | Where it plugs in |
| --- | --- | --- |
| Word-by-word kinetic lyrics + noun flashes | `kinetic.py` + `templates/kinetic.html` | after `stage2 align`: `kinetic words audio/timing/lyrics.json` → Stage 9 render |
| Lyrics-vs-take mismatch gate | `lyriccheck.py` | after `stage2 align` (free ASR in `audio/timing/asr/vocals.json`); FAIL lines → repaint or a logged text change; `kinetic words --lyriccheck` drops unsung words |
| Kick/snare + beat grid | `beatgrid.py` | after `stage2 stems`: `beatgrid detect audio/stems/drums.flac audio/timing/hits.json --beats audio/timing/beats.json` |
| Punch-ins + flash frames (≤ 3/s) | `fxplan.py` | `fxplan plan hits.json sections.json words.json fx.json`; Stage 10 applies `punch_scale`/`zoom` and `apply_flash` per frame |
| Cut grid, 2-3 cuts per clip | `cutgrid.py` | replaces the one-shot-per-beat plan: `cutgrid plan spec.json plan.json --beats ... --shotlist video/shotlist_cuts.json`; entries carry `reuse_clip` + `clip_range` (stage10 `i2v_window` already honours `clip_range`) |
| Infinite zoom + match-cut morphs | `zoomchain.py`, `wf_slop.py` | `plan` writes Klein-outpaint / Wan FLF2V graphs (GPU, not run); `compose` (CPU) builds the zoom; `morph-preview` = draft morph |
| Texture passes (after upscale) | `texture.py` | Stage 10: MythTexture on graded myth frames (zoom-locked cracks), UITexture on the UI PNGs; preview uses `PREVIEW_STRENGTH` |
| UI layer (terminal, HUD, bbox paths, slider, bars, loss curve, meter, chat, loadbar, toast) | `ui.py` + `templates/ui.html` | one `ui_timeline.json` → one PNG per frame; `ui check` = overflow gate |
| Terminal zoom-out | `zoomout.py` + `templates/desktop.html` | Stage 10 comp shot: painting clip → window on a desktop |
| Text never overflows | `htmlrender.py` (`SLOP.overflowAt`) | every render: text ink outside its `[data-box]`, a box off-frame, or text that can't fit at the minimum size raises `TextOverflow`; readable text < 36 px is a warning |

Layer order in the edit: myth video → MythTexture → punch-in → flash (white/invert/data/glitch) → UI (UITexture) → kinetic
lyrics (crisp) → noun cards (inside the kinetic layer). Flash budget is shared: noun cards and beat flashes pass one cap.

GPU-only parts (not run yet): Klein outpaint levels, Wan FLF2V morphs, the Wan micro-motion per zoom level, and the
real song (ACE-Step) feeding beatgrid/kinetic/lyriccheck. `zoomchain run` refuses to start without `--gpu-ok`.
