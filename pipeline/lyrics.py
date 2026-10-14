"""Lyrics structure and transcript matching for the song checks (Stage 1) and timing (Stage 2).

Sections come from master.md's generator lyrics; matching aligns transcript words (with times) to the
lyric word sequence, giving per-section recall and each section's sung time span.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from pipeline import song_spec

_WORD = re.compile(r"[a-z0-9']+")
# sung homophones / spellings that whisper writes differently
_NORMAL = {"ok": "okay", "o.k.": "okay"}  # add this song's homophones here


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n: int) -> str:
    """Integer to words (0-9999), so whisper's digits match sung numbers ('30' -> 'thirty', '2,000' -> 'two thousand')."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:  # "a 100" is sung "a hundred": drop the leading "one"
        head = "hundred" if n // 100 == 1 else _ONES[n // 100] + " hundred"
        return head + ("" if n % 100 == 0 else " " + _num_words(n % 100))
    return _ONES[n // 1000] + " thousand" + ("" if n % 1000 == 0 else " " + _num_words(n % 1000))


def norm_words(text: str) -> list[str]:
    out = []
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = re.sub(r"\b(\d{1,4})\b", lambda m: _num_words(int(m.group(1))), text)
    for w in _WORD.findall(text.lower().replace("’", "'").replace("‘", "'")):
        w = w.strip("'")
        if w.endswith("'"):
            w = w[:-1]
        out.append(_NORMAL.get(w, w))
    return [w for w in out if w]


@dataclass
class Section:
    idx: int
    tag: str                 # e.g. "Verse 1", "Chorus", "Break"
    lines: list[str] = field(default_factory=list)

    @property
    def words(self) -> list[str]:
        return [w for l in self.lines for w in norm_words(l)]

    @property
    def sung(self) -> bool:
        return bool(self.words)


def sections(lyrics: str | None = None) -> list[Section]:
    lyrics = lyrics or song_spec.generator_lyrics()
    out: list[Section] = []
    for line in lyrics.splitlines():
        m = re.match(r"\s*\[(.+?)\]\s*$", line)
        if m:
            out.append(Section(len(out), m.group(1)))
        elif line.strip() and out:
            out[-1].lines.append(line.strip())
    # disambiguate repeated tags: Chorus 1, Chorus 2, ...
    totals: dict[str, int] = {}
    for s in out:
        totals[s.tag] = totals.get(s.tag, 0) + 1
    seen: dict[str, int] = {}
    for s in out:
        base = s.tag
        seen[base] = seen.get(base, 0) + 1
        if totals[base] > 1:
            s.tag = f"{base} {seen[base]}"
    return out


def match(transcript_words: list[dict], secs: list[Section] | None = None) -> dict:
    """transcript_words: [{"word", "start", "end"}...] (WhisperX). Returns per-section recall and time spans."""
    secs = secs or sections()
    ref, owner = [], []
    for s in secs:
        for w in s.words:
            ref.append(w)
            owner.append(s.idx)
    hyp, times = [], []
    for w in transcript_words:
        for t in norm_words(w.get("word", "")):
            hyp.append(t)
            times.append((w.get("start"), w.get("end")))
    sm = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    hit = [None] * len(ref)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            hit[a + k] = b + k
    per = []
    for s in secs:
        ids = [i for i, o in enumerate(owner) if o == s.idx]
        if not ids:
            per.append({"section": s.tag, "words": 0, "recall": None, "start": None, "end": None})
            continue
        got = [hit[i] for i in ids if hit[i] is not None]
        ts = [times[j] for j in got if times[j][0] is not None]
        per.append({"section": s.tag, "words": len(ids), "matched": len(got), "recall": round(len(got) / len(ids), 3),
                    "start": round(min(t[0] for t in ts), 2) if ts else None,
                    "end": round(max(t[1] for t in ts), 2) if ts else None})
    total = sum(1 for h in hit if h is not None)
    return {"recall": round(total / max(1, len(ref)), 3), "sections": per,
            "extra_words": max(0, len(hyp) - total)}
