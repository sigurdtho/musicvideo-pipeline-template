"""Song inputs from master.md. Defaults keep the module importable before the master is filled in."""
import re

from pipeline.manifest import ROOT


def _md() -> str:
    path = ROOT / "master.md"
    return path.read_text() if path.exists() else ""


def _bpm(md: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*BPM", md, re.I)
    return float(m.group(1)) if m else 120.0


def _key(md: str) -> str:
    m = re.search(r"\b([A-G](?:#|b)? (?:minor|major))\b", md)
    return m.group(1) if m else "C minor"


def _duration(md: str) -> float:
    m = re.search(r"DURATION\s*=\s*(\d+(?:\.\d+)?)", md)
    return float(m.group(1)) if m else 180.0


_MD = _md()
BPM = _bpm(_MD)
KEYSCALE = _key(_MD)
DURATION = _duration(_MD)
GENRE_STYLE = "fill the genre line in master.md"
EXTRA = ""
CAPTION = GENRE_STYLE


def generator_lyrics() -> str:
    """The ```text block under '### Generator lyrics' in master.md."""
    md = _md()
    if "### Generator lyrics" not in md:
        return "[Verse 1]\n\n"
    sec = md.split("### Generator lyrics", 1)[1]
    m = re.search(r"```text\n(.*?)```", sec, re.S)
    return (m.group(1).strip() + "\n") if m else "[Verse 1]\n\n"
