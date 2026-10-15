# Music-video pipeline template

A local, mostly-automated pipeline that turns a lyric idea into a finished music video: song, stills, clips,
overlays, edit and 1080p master. It is a *template*: copy it per film with `new_project.sh`, then let a headless
coding agent work through the stages while a human approves a few gates from their phone.

> Status: draft extracted from a working personal setup. Code is as used in production for several films, but it
> assumes the hardware and tools below and has no test suite. Treat it as a reference implementation.

## Architecture

```
 Orchestrator (Grok Bot)  ──writes job briefs──▶  runner/run_job.sh  ──▶  Claude Opus, headless (`claude -p`, `--continue`)
        ▲                                              │ GPU lock, done-file loop,                │ runs python -m pipeline.*
        │ reports / questions                          │ clean stop on session limit              ▼
        │                                                                   Local GPU (RTX 4090, 24 GB)
 Phone review ◀── synced folder ($REVIEW_DIR, e.g. Dropbox) ◀── MP3 / MP4 / gate.md        ComfyUI: Flux.2 Klein, Wan 2.2, SeedVR2, RIFE
                                                                                            ACE-Step 1.5 API server (song)
```

- **Orchestrator:** a chat agent (Grok Bot) plans, writes a task brief per job (`runner/example_task.md`), starts the
  runner, reads reports and relays gate questions to the human.
- **Workers:** Claude Opus in headless mode does the actual stage work in the project folder, following `CLAUDE.md`
  (agent rules) and `master.md` (single source of truth for lyrics, characters, style, specs).
- **GPU box:** one RTX 4090. ComfyUI runs as a user service for Flux.2 Klein (stills/edits), Wan 2.2 I2V (clips),
  SeedVR2 7B (upscale) and RIFE (interpolation); ACE-Step 1.5 runs as its own API server for songs and repaints.
  Audio tools (BS-RoFormer stems, WhisperX alignment, beat tracking, aesthetics scoring, TTS) live in their own venvs
  under a shared `envs/` tree.
- **Review:** gate packages (top song takes, 480p preview under 30 MiB) are copied to a synced folder and listened to /
  watched on a phone. `qc/serve_review.py` and `index.html` give a local decision hub as an alternative.
- **Bookkeeping:** every asset gets an entry in `manifest.json` (`pipeline/manifest.py`); stages are idempotent and
  resumable; decisions go to `DECISIONS.md`; each stage writes `qc/stage-N.md`.

## Repository layout

| Path | What |
| --- | --- |
| `pipeline/` | Core stages 0-10 and shared modules (ComfyUI client, workflow builders, GPU guard, jobs, gates, manifest) |
| `pipeline/tools/` | Scripts run inside tool venvs (stems, alignment, beats, aesthetics, TTS, SyncNet) |
| `slopkit/` | Code-made visual layers: kinetic type, UI widgets, zoom chains, textures, HTML render in headless Chrome |
| `overlays/` | HTML templates for on-screen boxes, dialogue and karaoke lyrics |
| `workflows/smoke/` | ComfyUI API-format smoke workflows used by stage 0 |
| `qc/` | Review server, coherence and lyrics briefs, decision files (empty) |
| `docs/qa/` | QA checklists and style specs |
| `runner/` | Headless job runner + example brief |
| `examples/launch-window/` | Project-specific scripts from one finished film (HUD layer, song gate, code-guide renders, 1080p finish) |
| `CLAUDE.md`, `master.md`, `style_guide.*`, `DECISIONS.md`, `manifest.json`, `index.html` | Per-film templates (`__SLUG__` is replaced by `new_project.sh`) |

## Setup

1. Linux, NVIDIA GPU with 24 GB (developed on an RTX 4090), `ffmpeg`, Python 3.11+, `uv`, headless Chrome, `curl`,
   `flock`, systemd user units.
2. Install ComfyUI and the ACE-Step 1.5 API server yourself; set `COMFY_DIR` and `COMFY_URL`.
3. `cp .env.example .env` and fill the placeholders (`PROJECTS_DIR`, `ENVS_DIR`, `COMFY_DIR`, `REVIEW_DIR`, `GPU_LOCK`).
   `config.example.yaml` documents the same knobs plus gate thresholds.
4. Build tool environments once: `python -m pipeline.setup_tools` (writes `envs/<tool>/.installed.json`).
5. Fetch models from official repos only: `python -m pipeline.fetch_models` (pinned in `pipeline/models.py`).
   No weights are included here.
6. Fonts (SIL OFL) are not vendored; download the files listed in `*/fonts/SOURCE.txt` from google/fonts.
7. New film: `./new_project.sh my-film`, edit `master.md`, then run stage 0 (`python -m pipeline.stage0 smoke all`).
8. Install the Claude Code CLI and log in. Start a job: `runner/run_job.sh "$PROJECTS_DIR/my-film" task.md qc/.done`
   (ideally as a `systemd-run --user` unit). The runner uses `--dangerously-skip-permissions`; only run it on a
   dedicated machine/user.

## Stages

| # | Stage | Module |
| --- | --- | --- |
| 0 | Preflight: smoke-test every workflow, log time and VRAM peak; lyrics brief and research | `stage0.py`, `stage0_report.py` |
| 1 | Song: 16 ACE-Step takes, per-take checks, ranking, repaint weak sections, **song gate** | `stage1.py`, `gate.py`, `acestep.py` |
| 2 | Stems, lyric timing, beat grid, song edit list, shot list | `stage2.py`, `shot_plan.py` |
| 3 | Voices and sound effects | `stage3.py` |
| 4 | Mix and master (pedalboard + numpy), -14 LUFS | `stage4.py`, `master.py` |
| 5 | Looks: candidates, review, pick, multi-view character sheet | `stage5.py`, `characters.py` |
| 6 | Stills: Flux.2 Klein 9B candidates per shot, review, pick (after `style_guide.md` approval) | `stage6.py` |
| 7 | Clips: Wan 2.2 I2V 720p per shot, RIFE x3 | `stage7.py` |
| 8 | Lip-sync (off by default) | `stage8.py` |
| 9 | Overlays: code-rendered text/HUD layers timed to the song | `stage9.py`, `overlay.py`, `slopkit/` |
| 10 | Edit and finish, SeedVR2 upscale, **final-cut gate** | `stage10.py`, `final1080.py`, `post.py` |

Gates (set in `CLAUDE.md`): `song_take=on`, `looks=off`, `final_cut=on`. `autopilot.py` keeps the GPU busy with
flagged provisional picks while a gate waits.

## QA philosophy

- **Measure, don't glance.** Song takes are scored automatically: lyric recall from WhisperX on the vocal stem
  (weighted 0.45), weakest-section recall (0.20), instrumental breaks (0.08), aesthetics CE/PQ (0.25), tempo (0.02).
  The example song gate (`examples/launch-window/pipeline/song_rounds.py`) adds a 0.9 bar, a kick check from the first
  drop, a 6-10 s fade/ring-out ending and a length window. Thresholds are not changed mid-job.
- **Counts and physics by construction.** Anything with a correct count or physical rule is built in code (e.g. the
  ray-cast guide renders in `starship3d.py`, HUD glyphs in HTML) and photorealised afterwards, or checked by counting on
  several frames. Contact sheets of >= 16 frames per clip (`clipqa.py`) replace single-frame checks.
- **Fixed retry ladder.** Three new seeds, then rewrite the prompt, then degrade the shot (CLAUDE.md rule 4).
- **No text in generated pixels.** On-screen text is always an overlay layer.
- **Human gates only where taste matters** (song take, final cut); everything else is reviewed by the agent with
  written `qc/stage-N.md` reports. Checklists: `docs/qa/draft-check.md`, `docs/qa/HUD_STYLE.md`,
  `docs/qa/style_guide.template.md` (the docs/qa files are filled-in examples from one film), `qc/coherence.md`, `qc/lyrics_brief.md`.
- **Originality.** No real person's name, voice or likeness; models only from official repos, pinned.

## Costs

Everything runs locally; there are no per-call API costs in the pipeline (`budget_cap: 0`, rule 9: never spend money
without approval). The original author reports about **5-25 NOK of electricity per project** on an RTX 4090. The
agent subscriptions (orchestrator and Claude) are separate and not included in that figure.

## Not included

Song/video outputs and other media, model weights, fonts, API keys, `.env`, personal paths and data. See the
"What was left out" notes in `examples/launch-window/README.md` for project scripts that were not copied.

## Author

SigurdJorsalfar. Follow on X: https://x.com/SigurdJorsalfar

## License

MIT, see `LICENSE`.

This repository is not accepting contributions.
