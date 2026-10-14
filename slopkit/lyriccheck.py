"""Lyrics-vs-take check: does the picked take sing exactly the words we will put on screen?

The pipeline already has lyrics.match (per-section recall, Stage 1 ranking) and Stage 2's forced alignment; forced
alignment always "finds" the given text, so it can't catch a substituted or skipped word. This compares the FREE
WhisperX transcript of the vocal stem (audio/timing/asr/vocals.json from `stage2 align`) with the intended lyric lines
word by word and lists every missing, substituted and extra word with its time. Near-misses (spelling/homophone
ratio >= 0.75) are warnings; everything else fails the line. A failed line must be
listened to: repaint it, or (with the user's OK, logged in DECISIONS.md) change the on-screen text to what is sung.
slopkit.kinetic drops words listed under "unsung".

  python -m slopkit.lyriccheck <asr_words.json> <lyrics.json|master.md> <out.json> [--md out.md]
"""
import json
import re
import sys
from difflib import SequenceMatcher

_WORD = re.compile(r"[a-z0-9']+")
SOUNDALIKE = {"ok": "okay"}  # add this song's homophones here


def norm_words(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower().replace("\u2019", "'").replace("\u2018", "'")):
        w = w.strip("'")
        out.append(SOUNDALIKE.get(w, w))
    return [w for w in out if w]


def lines_from_master(md_path) -> list[dict]:
    md = open(md_path, encoding="utf-8").read()
    block = re.search(r"### Generator lyrics.*?```text\n(.*?)```", md, re.S).group(1)
    out, sec = [], None
    for ln in block.splitlines():
        m = re.match(r"\s*\[(.+?)\]\s*$", ln)
        if m:
            sec = m.group(1)
        elif ln.strip():
            out.append({"section": sec, "text": ln.strip()})
    return out


def check(asr_words: list[dict], lines: list[dict], near=0.75) -> dict:
    ref, owner, ridx = [], [], []
    for li, l in enumerate(lines):
        ws = norm_words(l["text"])
        ref += ws
        owner += [li] * len(ws)
        ridx += list(range(len(ws)))
    hyp, times = [], []
    for w in asr_words:
        for t in norm_words(w.get("word", "")):
            hyp.append(t)
            times.append(w.get("start"))
    sm = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    per = [{"line": i, "section": l.get("section"), "text": l["text"], "missing": [], "substituted": [], "near": [], "extra": []}
           for i, l in enumerate(lines)]
    unsung = []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "insert":
            li = owner[min(a0, len(owner) - 1)] if owner else 0
            per[li]["extra"] += [{"word": hyp[j], "t": times[j]} for j in range(b0, b1)]
            continue
        for k, i in enumerate(range(a0, a1)):
            j = b0 + k if op == "replace" and b0 + k < b1 else None
            item = {"word": ref[i], "ref_index": ridx[i], "t": times[j] if j is not None else None}
            if j is not None:
                item["sung"] = hyp[j]
                if SequenceMatcher(a=ref[i], b=hyp[j]).ratio() >= near:
                    per[owner[i]]["near"].append(item)
                    continue
                per[owner[i]]["substituted"].append(item)
            else:
                per[owner[i]]["missing"].append(item)
            unsung.append({"line": owner[i], "ref_index": ridx[i], "word": ref[i]})
        if op == "replace" and b1 - b0 > a1 - a0:
            per[owner[a1 - 1]]["extra"] += [{"word": hyp[j], "t": times[j]} for j in range(b0 + (a1 - a0), b1)]
    for p in per:
        p["status"] = "FAIL" if p["missing"] or p["substituted"] else ("WARN" if p["near"] or p["extra"] else "PASS")
    n_ok = sum(p["status"] != "FAIL" for p in per)
    return {"status": "PASS" if n_ok == len(per) else "FAIL", "lines": len(per), "lines_ok": n_ok,
            "word_recall": round(sum(1 for o in sm.get_opcodes() if o[0] == "equal" for _ in range(o[2] - o[1])) / max(1, len(ref)), 3),
            "unsung": unsung, "per_line": per}


def to_md(rep: dict) -> str:
    out = [f"# Lyric check: {rep['status']} ({rep['lines_ok']}/{rep['lines']} lines ok, word recall {rep['word_recall']})", ""]
    for p in rep["per_line"]:
        if p["status"] == "PASS":
            continue
        bits = [f"missing {', '.join(m['word'] for m in p['missing'])}" if p["missing"] else "",
                f"sung {', '.join(s['sung'] + ' for ' + s['word'] for s in p['substituted'])}" if p["substituted"] else "",
                f"near {', '.join(s['sung'] + '~' + s['word'] for s in p['near'])}" if p["near"] else "",
                f"extra {', '.join(e['word'] for e in p['extra'])}" if p["extra"] else ""]
        out.append(f"- **{p['status']}** [{p['section']}] {p['text']}: " + "; ".join(b for b in bits if b))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    a = sys.argv[1:]
    asr = json.load(open(a[0]))
    asr = asr["words"] if isinstance(asr, dict) else asr
    lines = lines_from_master(a[1]) if a[1].endswith(".md") else json.load(open(a[1]))
    rep = check(asr, lines)
    json.dump(rep, open(a[2], "w"), indent=1)
    if "--md" in a:
        open(a[a.index("--md") + 1], "w").write(to_md(rep))
    print(json.dumps({k: rep[k] for k in ("status", "lines", "lines_ok", "word_recall")}))
    sys.exit(0 if rep["status"] == "PASS" else 1)
