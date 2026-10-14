"""Word-by-word kinetic lyrics (templates/kinetic.html), timed from WhisperX forced-alignment word times.

Input: the pipeline's audio/timing/lyrics.json (Stage 2 `align`: one entry per lyric line with word_times
[{word, start, end}]). Words inside (parentheses) are backing vocals -> ghost words. Hook words get the gold accent;
nouns are reported so slopkit.fxplan can plan noun flash cards (and cap them with every other flash).

  python -m slopkit.kinetic words <lyrics.json> <out_words.json> [--lyriccheck report.json]
  python -m slopkit.kinetic render <words.json> <fx.json|-> <outdir> <t0> <t1> [--workers 4]
"""
import json
import re
import sys

from slopkit import htmlrender

HOOKS = set()  # words that take the accent. Fill from this song.
NOUNS = set()  # words that may become noun cards. Fill from this song.
ZONES = {"Chorus": "center", "Drop": "center", "Intro": "lower", "Verse": "lower", "Pre-Chorus": "lower",
         "Bridge": "lower", "Outro": "lower", "Coda": "lower"}
FADE = ("Outro", "Coda", "Intro")
_CLEAN = re.compile(r"[^\w'\u2019\-]+")


def norm(w: str) -> str:
    return _CLEAN.sub("", w.lower().replace("\u2019", "'")).strip("'")


def display(w: str) -> str:
    return re.sub(r"^[\"(\u201c]+|[\")\u201d,.;:!?\u2014\u2013-]+$", "", w.strip()).strip() or w


def zone_for(section: str) -> str:
    for k, z in sorted(ZONES.items(), key=lambda kv: -len(kv[0])):  # longest first: "Pre-Chorus" before "Chorus"
        if section.startswith(k) or k in section:
            return z
    return "lower"


def words_from_lines(lines: list[dict]) -> list[dict]:
    """lines: [{section, text, word_times: [{word, start, end}]}] -> kinetic words."""
    out = []
    for li, l in enumerate(lines):
        ghost = False
        for w in l.get("word_times", []):
            raw = w["word"]
            if raw.startswith("("):
                ghost = True
            if w.get("start") is not None:
                n = norm(raw)
                out.append({"w": display(raw), "s": round(w["start"], 3), "e": round(w.get("end") or w["start"] + 0.3, 3),
                            "zone": zone_for(l["section"]), "hook": n in HOOKS and not ghost, "ghost": ghost,
                            "noun": n in NOUNS and not ghost, "fade": l["section"].startswith(FADE),
                            "section": l["section"], "line": li})
            if raw.endswith(")") or raw.rstrip(",.!?").endswith(")"):
                ghost = False
    return out


def even_word_times(text: str, start: float, end: float) -> list[dict]:
    """Synthetic word times (smoke tests / drafts before alignment): words spread evenly over [start, end]."""
    ws = text.split()
    d = (end - start) / max(1, len(ws))
    return [{"word": w, "start": start + i * d, "end": start + (i + 0.8) * d} for i, w in enumerate(ws)]


def drop_unsung(words: list[dict], report: dict) -> list[dict]:
    """Remove words the lyric check could not find in the take (never show words that aren't sung).
    Matched per line by normalized word (report from slopkit.lyriccheck over the same lyrics.json lines)."""
    from collections import Counter
    from slopkit.lyriccheck import norm_words
    bad = Counter((m["line"], m["word"]) for m in report.get("unsung", []))
    out = []
    for w in words:
        hit = next(((w["line"], x) for x in norm_words(w["w"]) if bad[(w["line"], x)] > 0), None)
        if hit:
            bad[hit] -= 1
            continue
        out.append(w)
    return out


def render(words: list[dict], cards: list[dict], outdir, t0, t1, fps=24, workers=1) -> dict:
    spec = {"fps": fps, "hold_max": 0.8, "words": words, "cards": cards}
    return htmlrender.render_sequence("kinetic.html", spec, outdir, t0, t1, fps=fps, workers=workers)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "words":
        ws = words_from_lines(json.load(open(a[1])))
        if "--lyriccheck" in a:
            ws = drop_unsung(ws, json.load(open(a[a.index("--lyriccheck") + 1])))
        json.dump(ws, open(a[2], "w"), indent=1)
        print(json.dumps({"words": len(ws), "nouns": sum(w["noun"] for w in ws), "ghost": sum(w["ghost"] for w in ws)}))
    else:
        ws = json.load(open(a[1]))
        cards = [] if a[2] == "-" else [f for f in json.load(open(a[2]))["flashes"] if f["kind"] == "noun"]
        cards = [{"t": c["t"], "frames": c["frames"], "text": c["text"]} for c in cards]
        wk = int(a[a.index("--workers") + 1]) if "--workers" in a else 1
        print(json.dumps(render(ws, cards, a[3], float(a[4]), float(a[5]), workers=wk)))
