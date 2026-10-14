"""Characters and style, parsed from master.md (## Characters, STYLE line).

A still prompt is STYLE + SHOT + CHARACTER. LOOKS lists the ids that get a turnaround sheet.
"""
import re

from pipeline.manifest import ROOT


def _md() -> str:
    path = ROOT / "master.md"
    return path.read_text() if path.exists() else ""


def _style(md: str) -> str:
    m = re.search(r"^STYLE(?: \([^)]*\))?: (.+)$", md, re.M)
    return m.group(1).strip() if m else ""


def _negative(md: str) -> str:
    m = re.search(r"^NEGATIVE(?: \([^)]*\))?: (.+)$", md, re.M)
    return m.group(1).strip() if m else ""


def _characters(md: str) -> dict:
    if "## Characters" not in md:
        return {}
    body = md.split("## Characters", 1)[1].split("\n## ", 1)[0]
    out = {}
    for part in re.split(r"\n### ", body)[1:]:
        heading, _, rest = part.partition("\n")
        key = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")[:40] or "lead"
        m = re.search(r"^CHARACTER: (.+)$", rest, re.M)
        line = m.group(1).strip() if m else ""
        out[key] = {"heading": heading.strip(), "line": line, "full": line}
    return out


_MD = _md()
STYLE = _style(_MD)
STYLE_ANIME = ""
NEGATIVE = _negative(_MD)
CHARACTERS = _characters(_MD)
LOOKS = []  # character ids that get candidates and a multi-view sheet
