"""Write review briefs for subagents (stills, clips).

  python -m pipeline.review_briefs stills <batch_name> <shot_id> [...]
  python -m pipeline.review_briefs clips <batch_name> <shot_id> [...]
"""
import json
import sys

from pipeline import characters, manifest
from pipeline.manifest import ROOT

SHORT = {}  # character id -> the few traits a reviewer must see

HEADER = """# Review brief: {kind}

You are reviewing generated {what} for a music video. For each item, view the labelled grid.
Judge the picture, not the prompt's excuses.

Criteria (priority order):
{criteria}

Return ONLY a JSON object mapping each item id to
{{"pick": "<label or null>", "reason": "<one sentence>", "fix": "<what to change if pick is null, else empty>"}}.
"""

STILL_CRITERIA = """1. Right characters with their key traits, and nobody duplicated.
2. The SHOT is depicted: setting, action, framing and props as written.
3. No text, letters, numbers, logos or watermarks.
4. Clean anatomy: hands, faces, limbs.
5. The frame will animate: the subject is not cut awkwardly at the edge."""

CLIP_CRITERIA = """1. No morphing or melting, and no sudden identity change.
2. The MOTION described happens, and the camera move is smooth.
3. No text or letters appear in any frame.
4. No flicker, freeze, or abrupt jump.
5. Treat motion-stat flags as hints and judge from the frames."""


def stills(batch: str, ids: list[str]) -> str:
    sl = {s["id"]: s for s in json.load(open(ROOT / "video" / "shotlist.json"))["shots"]}
    out = [HEADER.format(kind=f"stills ({batch})", what="still images", criteria=STILL_CRITERIA)]
    for sid in ids:
        sh = sl[sid]
        traits = "; ".join(f"{c}: {SHORT.get(c, characters.CHARACTERS.get(c, {}).get('line', ''))}"
                           for c in sh.get("chars", [])) or "no characters"
        out.append(f"\n## {sid}\n- grid: {ROOT}/qc/stage6/{sid}.jpg\n- SHOT: {sh.get('shot')}\n- characters: {traits}")
    dest = ROOT / "qc" / "stage6" / f"{batch}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n")
    return str(dest)


def clips(batch: str, ids: list[str]) -> str:
    sl = {s["id"]: s for s in json.load(open(ROOT / "video" / "shotlist.json"))["shots"]}
    out = [HEADER.format(kind=f"clips ({batch})", what="video clips as contact sheets", criteria=CLIP_CRITERIA)]
    for sid in ids:
        sh = sl[sid]
        found = [a for a in manifest.query(stage="7", kind="clip") if a["meta"]["shot"] == sid and a["status"] == "done"]
        for a in found:
            out.append(
                f"\n## {sid} (attempt {a['meta']['attempt']})\n"
                f"- sheet: {ROOT}/qc/stage7/{sid}_a{a['meta']['attempt']}.jpg\n"
                f"- MOTION: {a['prompt']}\n"
                f"- return pick = \"{a['meta']['attempt']}\" to accept, null to reject"
            )
    dest = ROOT / "qc" / "stage7" / f"{batch}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n")
    return str(dest)


def main(argv):
    if argv[:1] == ["stills"] and len(argv) >= 3:
        print(stills(argv[1], argv[2:]))
    elif argv[:1] == ["clips"] and len(argv) >= 3:
        print(clips(argv[1], argv[2:]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
