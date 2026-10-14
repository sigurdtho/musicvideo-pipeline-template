"""Layer 2 for Launch Window (qc/HUD_STYLE.md + the reviewer's 2026-10-13 18:57 slopcore addendum): build and render.

  python -m pipeline.hud spec                 overlays/hud_spec.json from the shot list, beat grid and aligned words
  python -m pipeline.hud check                flash limit, event/beat alignment, overflow (Chrome) -> qc/hud_check.md
  python -m pipeline.hud stills <t> [t ...]   one PNG per time over the draft frame (review) -> qc/stage9/hud_<t>.png
  python -m pipeline.hud render [--workers N] the full-length alpha PNG sequence -> overlays/render/hud/ (stage 9 asset);
                                            masked slams (steam / the robot's head) go to hud_mask_<i>/ and are keyed in stage10
  python -m pipeline.hud endcard              the 12 s end card after the song -> overlays/render/endcard/
Every glyph is code (slopkit/templates/lw_hud.html). Times come from video/shotlist.json, the 128.0 BPM bar grid
(bar n = 0.02 + 1.875 (n - 1) s) and the WhisperX word times in audio/timing/lyrics.json, quantised to frames.
"""
import json
import sys
from pathlib import Path

import numpy as np

from pipeline import manifest
from pipeline.manifest import ROOT

FPS = 24
BEAT0, BEAT = 0.02, 60.0 / 128.0
TEMPLATE = ROOT / "slopkit" / "templates" / "lw_hud.html"
SPEC = ROOT / "overlays" / "hud_spec.json"
OUT = ROOT / "overlays" / "render" / "hud"
ENDCARD = ROOT / "overlays" / "render" / "endcard"
ENDCARD_T = ROOT / "slopkit" / "templates" / "lw_endcard.html"
ENDCARD_S = 12.0
M = "−"  # minus sign
DOT = " · "

KEYS = {"months", "twenty-six", "window", "window's", "open", "opens", "steel", "look", "up", "red", "light", "seconds",
        "seeds", "photo", "world's", "ground", "home", "seed", "blue", "first", "around", "us"}
HOOK_LINES = {("Chorus 1", 0), ("Chorus 1", 2), ("Chorus 2", 0), ("Chorus 2", 2), ("Chorus 3", 0), ("Chorus 3", 2)}
# bracket boxes (x, y, w, h) on the picked plates; refined after the stills review
LABEL_BOX = {"v1_stack": (770, 170, 150, 300), "v1_built": (1020, 70, 640, 880), "v1_redlight": (740, 386, 80, 80),
             "p1_catch": (1110, 270, 140, 450), "v2_seeds": (700, 410, 470, 330), "p3_lookup": (1245, 95, 90, 90),
             "p3_point": (850, 440, 95, 95)}


def fr(t):
    return round(t * FPS) / FPS


def bar(n):
    return BEAT0 + 1.875 * (n - 1)


def beats_between(t0, t1):
    k0 = int(np.ceil((t0 - BEAT0) / BEAT - 1e-6))
    out = []
    while BEAT0 + k0 * BEAT < t1 - 1e-6:
        out.append(BEAT0 + k0 * BEAT)
        k0 += 1
    return out


def shots():
    return {s["id"]: s for s in json.load(open(ROOT / "video" / "shotlist.json"))["shots"]}


def lines():
    return json.load(open(ROOT / "audio" / "timing" / "lyrics.json"))


def word_time(section, line, word):
    l = next(x for x in lines() if x["section"] == section and x["line"] == line)
    for w in l["word_times"]:
        if w["word"].lower().strip(",.:") == word:
            return w["start"]
    return l["start"]


def picked_still(sid):
    a = next((a for a in manifest.query(stage="6", kind="still") if a["meta"].get("shot") == sid and a["status"] == "picked"), None)
    return (ROOT / a["path"]).as_uri() if a else None


def levels(n_frames):
    """8 log-spaced band energies per video frame from the master (0..1, per-band 2-98 % range)."""
    import soundfile as sf
    x, sr = sf.read(str(ROOT / "out" / "master" / "launch-window_master.wav"), always_2d=True)
    x = x.mean(axis=1)
    hop = sr // FPS
    win = 4096
    edges = np.geomspace(40, 12000, 9)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    out = []
    w = np.hanning(win)
    for k in range(n_frames):
        c = k * hop
        seg = x[max(0, c - win // 2): c + win // 2]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        mag = np.abs(np.fft.rfft(seg * w))
        out.append([np.log10(1e-6 + mag[(freqs >= edges[i]) & (freqs < edges[i + 1])].mean()) for i in range(8)])
    a = np.array(out)
    lo, hi = np.percentile(a, 2, axis=0), np.percentile(a, 98, axis=0)
    a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    return np.round(a, 2).tolist()


def build() -> dict:
    sh = shots()
    S = lambda i: sh[i]["start"]
    E = lambda i: sh[i]["end"]
    dur = json.load(open(ROOT / "video" / "shotlist.json"))["duration"]
    ev, code, flashes = [], [], []

    # ---- code pictures (layer 1 replacements, drawn here)
    code += [
        dict(type="orbit", mode="intro", t_in=0.0, t_out=E("in_orbit")),
        dict(type="orbit", mode="cycle", t_in=S("in_l0"), t_out=E("in_l0")),
        dict(type="orbit", mode="lock", t_in=S("p1_orbit"), t_out=E("p1_orbit")),
        dict(type="orbit", mode="arc", t_in=S("c1_arc"), t_out=E("c1_arc"), arc_s=2 * 1.875),
        dict(type="orbit", mode="race", t_in=S("p2_orbit"), t_out=E("p2_orbit")),
        dict(type="orbit", mode="depart", t_in=S("c2_ground"), t_out=E("c2_ground")),
        dict(type="orbit", mode="fan", t_in=S("c3_fan"), t_out=E("c3_fan"), arcs=7),
        dict(type="orbit", mode="drift", t_in=S("out_orbit"), t_out=dur),
    ]
    grid1 = [u for u in (picked_still(x) for x in ("in_pad", "in_lights", "in_frost", "in_l1", "v1_stack", "p1_catch",
                                                     "p1_dunes", "c1_comes", "c1_beach", "d1_arms", "d1_crowd", "d1_sea",
                                                     "p1_rain", "v1_redlight")) if u]
    grid2 = [u for u in (picked_still(x) for x in ("in_pad", "c3_sprout", "c3_dusk", "v1_redlight", "br_blue", "c3_next")) if u]
    if grid1:
        code.append(dict(type="grid", t_in=S("d1_grid"), t_out=E("d1_grid"), sizes=[3, 4], beats_per_size=6, images=grid1))
    if grid2:
        code.append(dict(type="grid", t_in=S("c3_grid"), t_out=E("c3_grid"), sizes=[2], beats_per_size=8, images=grid2))

    # ---- slams (the device + the hook lines, all on the sung words)
    def slam(words, t_in, t_out, **kw):
        ev.append(dict(kind="slam", words=words, t_in=fr(t_in), t_out=fr(t_out), **kw))
    slam(["EVERY", "26", "MONTHS"], 0.5, 3.0, size=180)
    line_end = {(l["section"], l["line"]): l["end"] for l in lines()}
    slam(["HERE", "IT", "COMES"], S("c1_comes"), line_end[("Chorus 1", 0)] + 0.35, size=210, mask="steam")
    slam(["HERE", "IT", "COMES"], word_time("Chorus 1", 2, "here"), line_end[("Chorus 1", 2)] + 0.35, size=210)
    # bar 52: T-4 .. T-1 on the last four beats, masked behind the steam (r906c: a real luminance matte, stage10)
    for k, tb in enumerate(beats_between(bar(52), bar(53))[:4]):
        slam([f"T{M}{4 - k}"], tb, tb + BEAT, size=220, accent_idx=0, mask="steam", y=500)  # sinks into the steam bank
    slam(["HERE", "IT", "COMES"], S("c2_liftoff") - BEAT / 2, line_end[("Chorus 2", 0)] + 0.35, size=210,
         swap=dict(t=fr(S("c2_liftoff")), idx=[1, 2], words=["HERE", "WE", "GO"], new_idx=2))
    slam(["HERE", "WE", "GO"], word_time("Chorus 2", 2, "here"), line_end[("Chorus 2", 2)] + 0.35, size=210)
    slam(["LOOK", "UP"], word_time("Pre-Chorus 3", 0, "look"), S("p3_point"), size=200, y=300, mask="depth")
    slam(["HERE", "WE", "GO"], S("c3_sprout") - BEAT / 2, line_end[("Chorus 3", 0)] + 0.35, size=210,
         swap=dict(t=fr(S("c3_sprout")), idx=[2], words=["HERE", "WE", "ARE"], new_idx=2))
    slam(["HERE", "WE", "ARE"], word_time("Chorus 3", 2, "here"), min(line_end[("Chorus 3", 2)] + 0.35, S("c3_fan")), size=210)
    # r906c: masked slams live on their own layer, so one must end where the next slam begins (T-1 vs the swap slam)
    sl = sorted([e for e in ev if e["kind"] == "slam"], key=lambda e: e["t_in"])
    for a_, b_ in zip(sl, sl[1:]):
        a_["t_out"] = min(a_["t_out"], b_["t_in"])
    for t in (S("c1_comes"), S("c2_liftoff"), S("c3_sprout")):
        flashes.append(dict(t=fr(t), frames=3, a=0.75))

    # ---- counter
    def counter(t_in, t_out, text, unit="", ticks=None, live=False, **kw):
        ev.append(dict(kind="counter", t_in=fr(t_in), t_out=fr(t_out), text=text, unit=unit, ticks=ticks or [], live=live, **kw))
    counter(S("in_l0"), S("v1_stack"), f"T{M}26", "MONTHS")
    counter(S("v1_stack"), S("p1_catch"), f"T{M}20", "MONTHS",
            [dict(t=fr(S(i)), text=f"T{M}{n}") for i, n in (("v1_built", 14), ("v1_lookup", 7), ("v1_redlight", 3))])
    t_m, t_d, t_s = (word_time("Pre-Chorus 1", 0, w) for w in ("months", "days", "seconds"))
    counter(S("p1_catch"), S("p1_consoles"), f"T{M}1", "MONTH", [dict(t=fr(t_d), text=f"T{M}12", unit="DAYS"),
                                                                   dict(t=fr(t_s), text=f"T{M}90", unit="SECONDS")],
            type_unit=True, type_from=fr(t_s))
    bt = beats_between(S("p1_consoles"), S("c1_comes"))
    counter(S("p1_consoles"), S("c1_comes"), "00:00:0" + str(min(9, len(bt) - 1)), "WINDOW OPENS IN",
            [dict(t=fr(b), text=f"00:00:0{max(0, len(bt) - 1 - i)}") for i, b in enumerate(bt)])
    counter(S("c1_comes"), S("d1_flap_up"), "WINDOW OPEN", "", live=True)
    counter(S("d1_flap_up"), S("d1_grid"), f"T{M}3", "DAYS", [dict(t=fr(S("d1_sea")), text=f"T{M}2")])
    bt = beats_between(S("d1_grid"), S("d1_dawn"))
    counter(S("d1_grid"), S("d1_dawn"), f"T{M}47:00:00", "", [dict(t=fr(b), text=f"T{M}{47 - 2 * i:02d}:00:00") for i, b in enumerate(bt)])
    counter(S("d1_dawn"), S("v2_arm"), f"T{M}1", "DAY", [dict(t=fr(S("v2_seeds")), text=f"T{M}12:00:00", unit=""),
                                                          dict(t=fr(S("v2_photo")), text=f"T{M}06:00:00", unit="")])
    bt = beats_between(S("v2_arm"), S("p2_orbit"))
    counter(S("v2_arm"), S("p2_orbit"), f"T{M}00:30:00", "", [dict(t=fr(b), text=f"T{M}00:29:{59 - i:02d}") for i, b in enumerate(bt)])
    tm2, ts2 = word_time("Pre-Chorus 2", 0, "months"), word_time("Pre-Chorus 2", 0, "seconds")
    t10 = min(beats_between(90.0, 95.0), key=lambda x: abs(x - 92.9))  # r906c: snapped to the beat (HUD_STYLE: T-10 at 1:32.9)
    counter(S("p2_orbit"), t10, f"T{M}26", "MONTHS", [dict(t=fr((tm2 + ts2) / 2), text=f"T{M}26", unit="DAYS"),
                                                     dict(t=fr(ts2), text=f"T{M}26", unit="SECONDS")])
    bt = beats_between(t10, bar(52))
    counter(t10, bar(52), f"T{M}10", "", [dict(t=fr(b), text=f"T{M}{10 - i}") for i, b in enumerate(bt) if 10 - i >= 5], live=True)
    bt = beats_between(S("c2_liftoff"), S("c2_ground"))
    counter(S("c2_liftoff"), S("c2_ground"), "T+00:00:00", "", [dict(t=fr(b), text=f"T+00:00:{int((b - S('c2_liftoff')) // 1):02d}") for b in bt])
    counter(S("c2_ground"), S("br_quiet"), "DAY 1", "")
    counter(S("br_quiet"), S("br_blue"), "DAY 87", "")
    counter(S("br_blue"), S("p3_lookup"), "DAY 214", "")
    counter(S("c3_fan"), S("c3_grid"), f"DAY 1{DOT}T{M}26", "MONTHS")
    t_mo = word_time("Outro", 0, "months")
    bt = beats_between(t_mo, dur)
    counter(t_mo, dur, f"T{M}26", "MONTHS", [dict(t=fr(b), text=f"T{M}790:23:59:{59 - i:02d}", unit="") for i, b in enumerate(bt[1:])])

    # ---- phase, anchor, telemetry, labels
    for i in ("p1_orbit", "p2_orbit"):
        ev.append(dict(kind="phase", t_in=fr(S(i)), t_out=fr(E(i)), text=f"PHASE 44.0°{DOT}LOCK"))
    for a, b_, txt in ((S("in_pad"), S("in_l0"), f"EARTH{DOT}PAD"), (S("v2_tower"), S("p2_orbit"), f"EARTH{DOT}PAD"),
                       (S("c2_ground"), S("br_quiet"), "TRANSIT"), (S("br_foot"), S("p3_lookup"), "MARS"),
                       (S("c3_next"), S("c3_fan"), f"EARTH{DOT}PAD")):
        ev.append(dict(kind="anchor", t_in=fr(a), t_out=fr(b_), text=txt, fade_in=8 / FPS))
    ev.append(dict(kind="telemetry", mode="tests", t_in=fr(S("p1_catch")), t_out=fr(S("p1_consoles")), step=(S("p1_consoles") - S("p1_catch")) / 7))
    ev.append(dict(kind="telemetry", mode="alt", t_in=fr(S("c2_climb")), t_out=fr(S("c2_ground")), t0=S("c2_liftoff")))
    for sid, txt in (("v1_stack", "STACK"), ("v1_built", "PAYLOAD 01"), ("v1_redlight", "MARS"), ("p1_catch", "CATCH"),
                     ("v2_seeds", "SEEDS"), ("p3_lookup", "EARTH")):
        bx = LABEL_BOX[sid]
        ev.append(dict(kind="label", t_in=fr(S(sid)) + 1 / FPS, t_out=fr(E(sid) - 0.05), text=txt, box=list(bx),
                       side="left" if bx[0] + bx[2] > 1500 else "right"))
    ev.append(dict(kind="label", t_in=fr(S("p3_point")), t_out=fr(E("p3_point")), text="", box=list(LABEL_BOX["p3_point"])))
    # departure arc label at the arc's midpoint (same geometry as the template)
    ev.append(dict(kind="label", t_in=fr(S("c1_arc") + 1.875), t_out=fr(E("c1_arc")), text="DEPARTURE ARC",
                   box=[960 - 300 - 45, 540 - 330 - 45, 90, 90], side="left"))

    # r906c: row 6, the orange arc from the red point to the ship's nose on "window" (plate pixels of the in_l1 pick,
    # 16:9-cropped; it follows the shot's code push)
    t_win = word_time("Intro", 1, "window")
    ip = sh["in_l1"].get("post") or {}
    ev.append(dict(kind="arc", t_in=fr(t_win), t_out=fr(E("in_l1")), p0=[930, 146], p1=[915, 429], bend=0.18, draw_s=0.5,
                   shot_t0=S("in_l1"), shot_t1=E("in_l1"), push=float(ip.get("push", 0.0)), focus=ip.get("focus", [0.5, 0.5])))

    op = sh["out_two"].get("post") or {}
    ev.append(dict(kind="beacon", t_in=fr(S("out_two")), t_out=fr(E("out_two")), p0=[1180, 300],
                   p1=[1205, 470], shot_t0=S("out_two"), shot_t1=E("out_two"), push=float(op.get("push", 0.0)),
                   focus=op.get("focus", [0.5, 0.5])))

    # ---- slopcore furniture, data rail, stickers, flashes
    furniture = [[S("in_pad"), S("br_quiet")], [S("c3_sprout"), S("out_two")]]
    for a, b_ in ((S("v1_stack"), S("c1_comes")), (S("v2_tower"), S("p2_steam"))):
        ev.append(dict(kind="rail", t_in=fr(a), t_out=fr(b_)))
    slots = [(150, 210, -5), (1430, 230, 4), (170, 690, 3), (1400, 700, -4)]
    STK = {"c1": ["WINDOW OPEN", "PROP LOAD", "GO", "LOOK UP", "26 MO", "TOWER CLEAR", "44°", "HERE IT COMES"],
           "d1": ["TESTS 7", "CATCH ✓", "GO", "T−3", "STACK", "GO FOR LOAD", "WINDOW OPEN", "LOOK UP", "26 MO", "GO"],
           "c2": ["LIFTOFF", "TOWER CLEAR", "GO", "MAX-Q", "ALL ENGINES", "HERE WE GO", "LOOK UP", "GO"],
           "c3": ["MARS", "DAY 214", "SEEDS ✓", "HERE WE ARE", "NEXT WINDOW", "26 MO", "LOOK UP", "HOME"]}
    styles = ["a", "b", "c"]

    avoid = {"c3_dusk": {1: (1400, 760, 4)}}  # r906c: slot 1 covered the robot's face

    def stickers(key, t0, t1, every):
        bt = beats_between(t0, t1)
        for i, b in enumerate(bt[::every]):
            x, y, r = slots[i % 4]
            cur = next((k for k, v in sh.items() if v["start"] <= b < v["end"]), None)
            if cur in avoid and i % 4 in avoid[cur]:
                x, y, r = avoid[cur][i % 4]
            txt = STK[key][i % len(STK[key])]
            ev.append(dict(kind="sticker", t_in=fr(b), t_out=fr(min(t1, b + every * BEAT)), text=txt, x=x, y=y, rot=r,
                           style=styles[i % 3]))
    stickers("c1", bar(25) + 4 * BEAT, S("d1_flap_up"), 4)
    stickers("d1", S("d1_flap_up"), S("d1_grid"), 2)
    stickers("c2", S("c2_liftoff") + 4 * BEAT, S("br_quiet"), 4)
    stickers("c3", S("c3_sprout") + 4 * BEAT, S("c3_fan"), 4)
    for n in range(33, 41):  # the instrumental drop: a soft flash on every downbeat
        if n not in (33,):
            flashes.append(dict(t=fr(bar(n)), frames=1, a=0.3))
    for b in beats_between(S("c3_grid"), E("c3_grid"))[::2]:
        flashes.append(dict(t=fr(b), frames=1, a=0.25))

    # ---- lyrics, word-timed (hook lines are carried by the slams)
    lyr = []
    L = lines()
    for i, l in enumerate(L):
        if (l["section"], l["line"]) in HOOK_LINES:
            continue
        nxt = L[i + 1]["start"] if i + 1 < len(L) else dur
        style = "hook" if l["section"].startswith("Chorus") else "verse"
        words = []
        for w in l["word_times"]:
            raw = w["word"].strip()
            disp = raw.rstrip(",.:")
            key = disp.lower() in KEYS
            words.append(dict(w=disp, s=round(w["start"], 3), e=round(w["end"], 3), key=key))
        lyr.append(dict(t_in=fr(l["start"] - 0.18), t_out=fr(min(nxt - 0.06, l["end"] + 0.7)), style=style, words=words,
                        section=l["section"]))
    n = int(round(dur * FPS))
    marks = [0.02 + 1.875 * (b - 1) for b in (8, 13, 21, 25, 33, 41, 49, 53, 61, 69, 73, 81)]
    return dict(fps=FPS, dur=dur, beat0=BEAT0, beat=BEAT, code=code, events=ev, lyrics=lyr, flashes=flashes,
                furniture=furniture, marks=marks, levels=levels(n))


def split_masked(spec):
    """The main spec without the masked slams, and one slam-only spec per masked slam (t0, t1, matte mode, spec)."""
    main = dict(spec, events=[e for e in spec["events"] if not e.get("mask")])
    subs = []
    for e in spec["events"]:
        if e.get("mask"):
            t0, t1 = fr(e["t_in"]), fr(e["t_out"])
            subs.append((t0, t1, e["mask"], dict(spec, code=[], events=[e], lyrics=[], flashes=[], furniture=[])))
    return main, subs


def flash_check(spec) -> list:
    ts = sorted(f["t"] for f in spec["flashes"])
    bad = []
    for i, t in enumerate(ts):
        win = [x for x in ts if t <= x < t + 1.0]
        if len(win) > 3:
            bad.append(f"{len(win)} flashes in 1 s from {t:.2f}")
        if i and t - ts[i - 1] < 0.25 - 1e-6:
            bad.append(f"flashes {ts[i - 1]:.2f} and {t:.2f} under 0.25 s apart")
    return bad


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "spec":
        spec = build()
        SPEC.parent.mkdir(parents=True, exist_ok=True)
        SPEC.write_text(json.dumps(spec))
        print(f"spec: {len(spec['events'])} events, {len(spec['lyrics'])} lyric lines, {len(spec['code'])} code pictures, "
              f"flash issues: {flash_check(spec)}")
    elif cmd == "check":
        from slopkit import htmlrender
        spec = json.loads(SPEC.read_text())
        ts = [k / 4 for k in range(int(spec["dur"] * 4))]
        rep = htmlrender.check(TEMPLATE, spec, ts)
        md = ["# Layer 2 check", "", f"- flash limit: {flash_check(spec) or 'ok'}", f"- overflow: {rep['overflow'][:10] or 'none'}",
              f"- small text (warn): {len(rep['small_text'])} elements (HUD data is 18-22 px by spec)", ""]
        (ROOT / "qc" / "hud_check.md").write_text("\n".join(md))
        print("\n".join(md))
    elif cmd == "stills":
        from slopkit import htmlrender
        spec = json.loads(SPEC.read_text())
        out = ROOT / "qc" / "stage9"
        out.mkdir(parents=True, exist_ok=True)
        for t in argv[1:]:
            htmlrender.render_still(TEMPLATE, spec, out / f"hud_{float(t):07.2f}.png", float(t))
            print(out / f"hud_{float(t):07.2f}.png")
    elif cmd == "render":
        from slopkit import htmlrender
        wk = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 12
        spec = json.loads(SPEC.read_text())
        main_spec, masked = split_masked(spec)
        for i, (t0, t1, mode, sub) in enumerate(masked):  # r906c: masked slams on their own layer, keyed in stage10
            out = OUT.parent / f"hud_mask_{i}"
            r = htmlrender.render_sequence(TEMPLATE, sub, out, t0, t1, fps=FPS, workers=min(wk, 4))
            manifest.upsert(f"overlay:hud_mask:{i}", stage="9", kind="overlay", path=str(out.relative_to(ROOT)), prompt=None,
                            seed=None, model="headless Chrome", workflow="pipeline/hud.py + slopkit/templates/lw_hud.html",
                            score=None, status="done", meta={"t_in": t0, "t_out": t1, "frames_dir": str(out.relative_to(ROOT)),
                                                             "frames": r["frames"], "matte": mode, "order": 0})
            print(f"masked slam layer {i}: {t0:.2f}-{t1:.2f} {mode}")
        spec = main_spec
        res = htmlrender.render_sequence(TEMPLATE, spec, OUT, 0.0, spec["dur"], fps=FPS, workers=wk)
        manifest.upsert("overlay:hud", stage="9", kind="overlay", path=str(OUT.relative_to(ROOT)), prompt=None, seed=None,
                        model="headless Chrome", workflow="pipeline/hud.py + slopkit/templates/lw_hud.html", score=None,
                        status="done", meta={"t_in": 0.0, "frames_dir": str(OUT.relative_to(ROOT)), "frames": res["frames"],
                                             "render_seconds": res["seconds"]})
        print(json.dumps({k: v for k, v in res.items() if k != "small_text"}))
    elif cmd == "endcard":  # r906c: the end card (HUD_STYLE 5), ENDCARD_S after the song, over silence
        from slopkit import htmlrender
        dur = json.load(open(ROOT / "video" / "shotlist.json"))["duration"]
        k0 = len(beats_between(word_time("Outro", 0, "months"), dur)) - 1
        spec = dict(fps=FPS, dur=ENDCARD_S, beat=BEAT, k0=k0)
        res = htmlrender.render_sequence(ENDCARD_T, spec, ENDCARD, 0.0, ENDCARD_S, fps=FPS, workers=4)
        manifest.upsert("overlay:endcard", stage="9", kind="endcard", path=str(ENDCARD.relative_to(ROOT)), prompt=None,
                        seed=None, model="headless Chrome", workflow="pipeline/hud.py endcard + slopkit/templates/lw_endcard.html",
                        score=None, status="done", meta={"seconds": ENDCARD_S, "frames": res["frames"],
                                                         "overflow": res.get("overflow")})
        print(json.dumps({k: v for k, v in res.items() if k != "small_text"}))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
