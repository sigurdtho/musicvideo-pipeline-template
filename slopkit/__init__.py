"""slopkit: slopcore / motion-graphics extensions for the music-video pipeline (CPU-side tools).

Modules (each has a CLI; run from the project root with the shared pipeline env's python):
  htmlrender  headless-Chrome PNG-sequence renderer with an automatic text-overflow check (CPU, --disable-gpu)
  ui          UI-layer timeline helpers (terminal, HUD, bbox paths, slider, bars, loss curve, meter, chat, loadbar)
  kinetic     word-by-word kinetic lyrics from WhisperX word times (+ noun flash cards)
  beatgrid    kick/snare onset detection + beat grid JSON (librosa; beat_this beats when present)
  fxplan      punch-ins and flash frames from the beat grid (<= 3 full-screen flashes per second)
  cutgrid     cut-grid shot planner: one rendered clip feeds 2-3 cuts (non-overlapping windows)
  texture     global texture passes: myth (grain, craquelure, canvas, varnish) and UI (CRT, scanlines, CA, bloom)
  zoomchain   infinite-zoom chain (Klein outpaint graphs + CPU compositor) and match-cut morphs (Wan FLF2V graphs,
              CPU optical-flow morph preview)
  zoomout     terminal zoom-out composite (painting -> window on a desktop)
  lyriccheck  WhisperX transcript vs intended lyrics: word-level mismatch report (gate for on-screen lyrics)
  comp        alpha compositing + ffmpeg frame IO
  smoke       CPU smoke tests + the 10 s sample strip (qc/tools_smoke/)
No module queues GPU work except zoomchain.run, which refuses to start without --gpu-ok.
"""
from pathlib import Path

PKG = Path(__file__).resolve().parent
TEMPLATES = PKG / "templates"
W, H, FPS = 1920, 1080, 24
