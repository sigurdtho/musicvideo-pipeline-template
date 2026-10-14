# __SLUG__: agent rules

master.md is the source of truth for lyrics, characters, style and specs.
style_guide.md is the numbered gate for pictures. Still generation waits until its status is approved.

## Config

- project: __SLUG__
- style: cinematic
- gates: song_take=on, looks=off, final_cut=on
- budget_cap: 0

## Layout

pipeline/ stages, manifest.json, DECISIONS.md, qc/stage-N.md, out/final/.
The decision hub is index.html. Gallery and listen pages, when this film has them, stay linked from that hub.
Shared tool environments are the symlink envs/ (do not copy or rebuild them).
ComfyUI is the shared user unit comfyui. ACE-Step is the shared user unit acestep.
Job units and ComfyUI output folders use the prefix __SLUG__.

## Rules

1. Never change lyrics, characters or style without a DECISIONS.md entry.
2. One GPU job at a time. Leave a running render alone.
3. Every file gets a manifest entry. Stages are idempotent and resumable.
4. Retry a rejected asset 3 times with new seeds, then rewrite its prompt, then degrade the shot.
5. No text in image or video prompts. On-screen text is an overlay, and this film's style guide says whether any text is allowed.
6. Each tool stays in its own environment. Do not install into the shared tree.
7. Models and nodes only from official repos. Pin commits. Prefer .safetensors.
8. Keep designs original. No real person's name, voice or recording. No book covers or screen stills.
9. Never spend money without approval.
10. Write qc/stage-N.md after each stage.
11. Approve style_guide.md before generating stills.
12. After the final cut is approved, update lessons.md, the music-video-pipeline skill, and the _template.
