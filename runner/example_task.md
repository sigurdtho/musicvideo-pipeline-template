# Example job brief (stage 1 song rounds)

Project: `$PROJECT_DIR`. Read CLAUDE.md and master.md first.

1. Generate a round of song takes: `envs/pipeline/bin/python -m pipeline.stage1 takes`.
2. Score them: `python -m pipeline.stage1 analyze`. Do not change the scoring or the gate bar.
3. Prepare the gate: `python -m pipeline.gate song prepare 3`, copy the MP3s to `$REVIEW_DIR/<slug>/listen/`.
4. Write `qc/stage-1.md`, then create the done marker named by the runner.
