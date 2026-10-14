"""One entry per picture. Fill SHOTS from master.md after the style guide is approved.

Each shot: id, anchor, kind, chars, shot, motion, and optional boxes, montage, run_on.
anchor is ("line", section, line_index) or ("spoken", line_id).
"""
from pipeline.shot_overrides import OVERRIDES

SHOTS = []
FILL = {}

for _shot in SHOTS:
    extra = OVERRIDES.get(_shot["id"])
    if extra:
        _shot.update({k: v for k, v in extra.items() if k != "why"})
