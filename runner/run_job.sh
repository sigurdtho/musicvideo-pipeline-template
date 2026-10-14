#!/bin/bash
# Headless Claude job runner (generalised from the original per-job runner scripts).
#   runner/run_job.sh <project_dir> <task.md> <done_marker> [max_runs]
# 1. Busy-waits for the GPU: other job units active, VRAM above a threshold, or ComfyUI queue busy.
# 2. Takes the machine-wide GPU lock (flock on fd 9), keeps it for the whole job.
# 3. Runs `claude -p` once, then `claude --continue -p` until <done_marker> exists (max_runs rounds).
# 4. Never exits while ComfyUI still has queued work; stops cleanly on Claude's session limit.
# Config via env (see .env.example): CLAUDE_BIN, CLAUDE_MODEL, GPU_LOCK, COMFY_URL, VRAM_BUSY_MB, JOB_UNIT_GLOBS, SELF_UNIT.
set -u
P=${1:?project dir}; TASK=${2:?task file}; M=${3:?done marker}; MAX=${4:-20}
CLAUDE_BIN=${CLAUDE_BIN:-claude}; CLAUDE_MODEL=${CLAUDE_MODEL:-opus}
CL="$CLAUDE_BIN --model $CLAUDE_MODEL --dangerously-skip-permissions"   # headless: run only on a dedicated machine/user
LOCK=${GPU_LOCK:-$HOME/.cache/mvp/gpu.lock}; COMFY_URL=${COMFY_URL:-http://127.0.0.1:8188}
VRAM_BUSY_MB=${VRAM_BUSY_MB:-16000}; JOB_UNIT_GLOBS=${JOB_UNIT_GLOBS:-mvp-*}; SELF_UNIT=${SELF_UNIT:-__none__}
cd "$P" || exit 2; mkdir -p "$(dirname "$LOCK")"
FIRST="Do the task in $TASK. You already hold the shared GPU lock (fd 9 of the runner). Do not change the song gate. Create $M only when everything is done."
MSG="Continue the task in $TASK from where you stopped until it is fully done and $M exists. You already hold the shared GPU lock (fd 9 of the runner). Do not change the song gate. Wait for running/queued generation with pipeline.jobs wait or the ComfyUI queue; do not exit early."
qbusy(){ q=$(curl -s -m5 "$COMFY_URL/queue"); [ -z "$q" ] && return 1
 echo "$q" | python3 -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d["queue_running"] or d["queue_pending"] else 1)'; }
busy(){ [ -n "$(systemctl --user list-units --state=active --no-legend $JOB_UNIT_GLOBS | grep -v -e "$SELF_UNIT")" ] && return 0
 [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt "$VRAM_BUSY_MB" ] && return 0
 qbusy; }
exec 9>>"$LOCK"
while true; do
  while busy; do echo "$(date +%T) GPU busy, waiting"; sleep 60; done
  if flock -n 9; then busy && { flock -u 9; continue; }; break; fi
  echo "$(date +%T) gpu.lock held, waiting"; sleep 60
done
echo "$(date +%T) GPU lock taken"
i=0; first=1
while [ ! -f "$M" ] && [ $i -lt "$MAX" ]; do i=$((i+1)); echo "$(date +%T) run #$i"
  if [ $first = 1 ]; then out=$($CL -p "$FIRST" 2>&1); first=0; else out=$($CL --continue -p "$MSG" 2>&1); fi; echo "$out"
  while qbusy; do echo "$(date +%T) generation still queued, waiting"; sleep 60; done
  if echo "$out" | grep -q "hit your session limit"; then echo "$(date +%T) session limit, clean stop"; break; fi
  [ -f "$M" ] || sleep 30
done
while qbusy; do echo "$(date +%T) generation still queued, waiting before exit"; sleep 60; done
[ -f "$M" ]; rc=$?
flock -u 9; echo "$(date +%T) lock released rc=$rc"; exit $rc
