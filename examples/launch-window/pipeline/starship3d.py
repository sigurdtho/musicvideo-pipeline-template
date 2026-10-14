"""Code-built Starship + launch tower (r906b): a numpy ray caster, so every count and proportion is exact by construction.

  python -m pipeline.starship3d render <preset> <out.png> [W H]    one guide render
  python -m pipeline.starship3d presets                            list presets

The renders are guides, never plates: Klein img2img photorealises them (pipeline.rocket_round, family "guide").
Version lock (DECISIONS.md 2026-10-13 r906b): the Flight 5 configuration, the first booster catch: Super Heavy with
four grid fins at the top and the vented hot-staging ring, the Block 1 ship (4 flaps, tiles on the windward half), the
Pad 1 tower with chopsticks and the ship quick-disconnect arm, and the launch mount table on six legs.
Units: metres, z up, the stack's axis at x = y = 0, the tower at -x, the ship's tiled (windward) side faces -y.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EPS = 1e-3
# ---- dimensions (Flight 5 stack: ~121 m; booster 71 m, ring 1.8 m, ship 50 m; 9 m diameter)
R = 4.5
Z_BOOST0, Z_BOOST1 = 19.0, 90.0
Z_RING1 = 91.8
Z_SHIP1 = 141.8
Z_NOSE0 = 124.0
TOWER_X, TOWER_W, TOWER_H = -21.0, 11.0, 140.0
WINDWARD = np.array([0.0, -1.0, 0.0])

MAT = {"sky": 0, "ground": 1, "booster": 2, "ship": 3, "tower": 4, "ring": 5, "fin": 6, "olm": 7, "arm": 8, "flap": 9,
       "deflector": 10}


def _norm(v):
    return v / np.linalg.norm(v)


def nose_r(z):
    """Ogive nose radius at height z (Z_NOSE0..Z_SHIP1); a tangent ogive with a small rounded tip."""
    L = Z_SHIP1 - Z_NOSE0
    rho = (R * R + L * L) / (2 * R)
    x = np.clip(z - Z_NOSE0, 0, L)
    return np.maximum(np.sqrt(np.maximum(rho * rho - x * x, 0)) + R - rho, 0.25)


class Scene:
    def __init__(self):
        self.prims = []  # dicts: type, params, mat, group

    def box(self, center, axes, half, mat, group="misc", quad=None):
        axes = np.array([_norm(np.asarray(a, float)) for a in axes]).T  # columns = local axes
        self.prims.append(dict(type="box", c=np.asarray(center, float), R=axes, h=np.asarray(half, float), mat=mat,
                               group=group, quad=quad))

    def beam(self, p0, p1, w, mat, group, up=(0, 0, 1)):
        """A square-section member from p0 to p1."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        a = p1 - p0
        L = np.linalg.norm(a)
        a /= L
        u = np.asarray(up, float)
        if abs(np.dot(u, a)) > 0.9:
            u = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        b = _norm(np.cross(a, u))
        c = np.cross(b, a)
        self.box((p0 + p1) / 2, (a, b, c), (L / 2, w / 2, w / 2), mat, group)

    def cyl(self, r0, r1, z0, z1, mat, group, cx=0.0, cy=0.0):
        self.prims.append(dict(type="cone", r0=r0, r1=r1, z0=z0, z1=z1, cx=cx, cy=cy, mat=mat, group=group))


# ------------------------------------------------------------------ the vehicle, tower and pad
def build(cfg: dict) -> Scene:
    s = Scene()
    lift = cfg.get("ship_lift", 0.0)          # v1_stack: the ship held above the booster by the chopsticks
    ship = cfg.get("ship", True)
    booster = cfg.get("booster", True)
    bz = cfg.get("booster_lift", 0.0)         # p1_catch: the booster held in the arms above the mount
    ship_dz = lift if booster else cfg.get("ship_z", 0.0)
    if cfg.get("stack_lift"):
        bz = bz + cfg["stack_lift"]
        ship_dz = ship_dz + cfg["stack_lift"]

    if booster:
        s.cyl(R, R, Z_BOOST0 + bz, Z_BOOST1 + bz, MAT["booster"], "stack")
        s.cyl(R, R, Z_BOOST1 + bz, Z_RING1 + bz, MAT["ring"], "stack")
        # raceway (conduit cover) along the booster, on the +y side
        s.box((0, R + 0.25, (Z_BOOST0 + Z_BOOST1) / 2 + bz), ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
              (0.45, 0.3, (Z_BOOST1 - Z_BOOST0 - 6) / 2), MAT["booster"], "stack")
        # four grid fins near the top, sticking out radially, panel plane horizontal
        for az in (45, 135, 225, 315):
            a = math.radians(az)
            rd = np.array([math.cos(a), math.sin(a), 0])
            tg = np.array([-math.sin(a), math.cos(a), 0])
            s.box(rd * (R + 2.0) + np.array([0, 0, 87.4 + bz]), (rd, tg, (0, 0, 1)), (2.0, 2.4, 0.42), MAT["fin"], "stack")
            s.box(rd * (R + 0.35) + np.array([0, 0, 87.6 + bz]), (rd, tg, (0, 0, 1)), (0.4, 0.8, 1.0), MAT["fin"], "stack")
    if ship:
        z0 = Z_RING1 + ship_dz
        s.cyl(R, R, z0, Z_NOSE0 + ship_dz, MAT["ship"], "stack")
        zs = np.linspace(Z_NOSE0, Z_SHIP1, 14)
        rs = nose_r(zs)
        for i in range(len(zs) - 1):
            s.cyl(float(rs[i]), float(rs[i + 1]), float(zs[i]) + ship_dz, float(zs[i + 1]) + ship_dz, MAT["ship"], "stack")
        # raceway on the leeward side
        s.box((0, R + 0.18, (z0 + Z_NOSE0 + ship_dz) / 2), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0.35, 0.2, (Z_NOSE0 - Z_RING1 - 3) / 2),
              MAT["ship"], "stack")
        # flaps: plates hinged on the hull at +-72 deg from windward (they stick out sideways, a little toward the
        # tiled side). aft_deflect / fwd_deflect (deg) swing them about their hinge toward windward (+) or leeward (-).
        for side in (-1, 1):
            ang = math.radians(-90 + side * 72)
            hinge_dir = np.array([math.cos(ang), math.sin(ang), 0])
            for which, defl in (("aft", cfg.get("aft_deflect", 0.0)), ("fwd", cfg.get("fwd_deflect", 0.0))):
                a2 = ang - side * math.radians(defl)
                rd = np.array([math.cos(a2), math.sin(a2), 0])
                tg = np.array([-math.sin(a2), math.cos(a2), 0])
                if which == "aft":
                    P0 = hinge_dir * (R - 0.05) + np.array([0, 0, z0])
                    q = [(0.0, 0.6), (0.0, 11.5), (3.8, 8.3), (3.8, 1.2)]
                else:
                    zr0, zr1 = Z_NOSE0 + 1.0, Z_NOSE0 + 10.5
                    rr0, rr1 = float(nose_r(zr0)), float(nose_r(zr1))
                    rm = (rr0 + rr1) / 2 - 0.2
                    P0 = hinge_dir * rm + np.array([0, 0, Z_RING1 + ship_dz])
                    q = [(rr0 - rm, zr0 - Z_RING1), (rr1 - rm, zr1 - Z_RING1), (rr1 - rm + 3.4, zr1 - Z_RING1 - 3.2),
                         (rr0 - rm + 3.8, zr0 - Z_RING1 + 0.6)]
                us, vs = [x[0] for x in q], [x[1] for x in q]
                uc, vc = (min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2
                hu, hv = (max(us) - min(us)) / 2, (max(vs) - min(vs)) / 2
                s.box(P0 + rd * uc + np.array([0, 0, vc]), (rd, tg, (0, 0, 1)), (hu, 0.15, hv), MAT["flap"], "stack",
                      quad=(q, P0, rd))

    if cfg.get("pad", True):
        # launch mount: a table on six legs, a deflector plate below
        s.box((0, 0, 18.5), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (9.5, 9.5, 1.6), MAT["olm"], "pad")
        for k in range(6):
            a = math.radians(30 + 60 * k)
            s.box((8.2 * math.cos(a), 8.2 * math.sin(a), 8.5), ((math.cos(a), math.sin(a), 0), (-math.sin(a), math.cos(a), 0),
                  (0, 0, 1)), (1.4, 1.4, 8.5), MAT["olm"], "pad")
        s.box((0, 0, 0.6), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (6.0, 6.0, 0.6), MAT["deflector"], "pad")
        if cfg.get("detail", True):  # r906c detail pass: the clamp ring, 20 hold-down clamps, leg bracing, booster QD
            s.cyl(5.6, 5.6, 20.1, 21.0, MAT["olm"], "pad")
            for k in range(20):
                a = math.radians(9 + 18 * k)
                rd = np.array([math.cos(a), math.sin(a), 0])
                s.box(rd * 5.3 + np.array([0, 0, 21.6]), (rd, (-math.sin(a), math.cos(a), 0), (0, 0, 1)), (0.55, 0.35, 0.7),
                      MAT["ring"], "pad")
            for k in range(6):
                a0, a1 = math.radians(30 + 60 * k), math.radians(90 + 60 * k)
                p0 = np.array([8.2 * math.cos(a0), 8.2 * math.sin(a0), 0])
                p1 = np.array([8.2 * math.cos(a1), 8.2 * math.sin(a1), 0])
                s.beam(p0 + (0, 0, 9.0), p1 + (0, 0, 9.0), 0.8, MAT["olm"], "pad")
                s.beam(p0 + (0, 0, 2.0), p1 + (0, 0, 15.5), 0.6, MAT["olm"], "pad")
            s.box((-10.2, 0, 22.2), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1.6, 2.6, 2.2), MAT["arm"], "pad")  # booster QD
    if cfg.get("tower", True):
        _tower(s, cfg)
    return s


def _tower(s: Scene, cfg: dict):
    cx, w, H = TOWER_X, TOWER_W, TOWER_H
    hw = w / 2
    corners = [(cx - hw, -hw), (cx + hw, -hw), (cx + hw, hw), (cx - hw, hw)]
    for (x, y) in corners:
        s.box((x, y, H / 2), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0.75, 0.75, H / 2), MAT["tower"], "tower")
    step = cfg.get("tower_step", 6.0)
    levels = np.arange(0, H + 0.1, step)
    for i, z in enumerate(levels):
        for k in range(4):
            (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
            s.beam((x0, y0, z), (x1, y1, z), 0.7, MAT["tower"], "tower")
            if i + 1 < len(levels):
                z1 = levels[i + 1]
                s.beam((x0, y0, z), (x1, y1, z1), 0.45, MAT["tower"], "tower")
                s.beam((x1, y1, z), (x0, y0, z1), 0.45, MAT["tower"], "tower")
    if cfg.get("detail", True):  # r906c: K-bracing at mid-module and a secondary column on each face
        for i, z in enumerate(levels[:-1]):
            zm = z + step / 2
            for k in range(4):
                (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                s.beam((x0, y0, zm), (x1, y1, zm), 0.35, MAT["tower"], "tower")
        for k in range(4):
            (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
            s.box(((x0 + x1) / 2, (y0 + y1) / 2, H / 2), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0.4, 0.4, H / 2), MAT["tower"], "tower")
        # carriage rails on the stack-side face
        for y in (-hw + 1.5, hw - 1.5):
            s.box((cx + hw + 0.5, y, H / 2), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0.35, 0.35, H / 2 - 2), MAT["arm"], "tower")
    for z in np.arange(12, H, 24.0):  # service platforms
        s.box((cx, 0, z), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (hw + 0.4, hw + 0.4, 0.35), MAT["tower"], "tower")
    # top deck and lightning mast
    s.box((cx, 0, H + 1.2), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (hw + 0.6, hw + 0.6, 1.2), MAT["tower"], "tower")
    s.box((cx, 0, H + 9), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0.3, 0.3, 7.5), MAT["tower"], "tower")
    # concrete base
    s.box((cx, 0, 3), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (hw + 2, hw + 2, 3), MAT["olm"], "tower")

    # chopsticks: carriage around the tower + two truss arms straddling the stack
    za = cfg.get("arms_z", 128.0)
    opn = math.radians(cfg.get("arms_open", 18.0))
    s.box((cx, 0, za), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (hw + 1.2, hw + 1.2, 3.2), MAT["arm"], "arms")
    for side in (-1, 1):
        pivot = np.array([cx + hw + 0.5, side * (hw - 0.8), za])
        L = 36.0
        d = np.array([math.cos(opn * side), math.sin(opn * side), 0.0])
        top0, top1 = pivot + np.array([0, 0, 1.7]), pivot + d * L + np.array([0, 0, 1.2])
        bot0, bot1 = pivot - np.array([0, 0, 2.2]), pivot + d * (L - 2) - np.array([0, 0, 0.6])
        s.beam(top0, top1, 1.3, MAT["arm"], "arms")
        s.beam(bot0, bot1, 1.0, MAT["arm"], "arms")
        # a second top chord on the inner side makes the arm a box truss
        off = np.array([-d[1], d[0], 0]) * (-side) * 1.6
        s.beam(top0 + off, top1 + off, 0.9, MAT["arm"], "arms")
        if cfg.get("detail", True):  # r906c: deeper box truss, the catch rail on top, a second bottom chord, the hinge
            s.beam(top0 + np.array([0, 0, 0.9]) + off / 2, top1 + np.array([0, 0, 0.9]) + off / 2, 1.6, MAT["arm"], "arms")
            s.beam(bot0 + off, bot1 + off, 0.8, MAT["arm"], "arms")
            s.box(pivot + np.array([0, 0, -0.2]), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1.4, 1.4, 3.4), MAT["arm"], "arms")
            for k in range(1, 9):
                f = k / 9
                s.beam(bot0 + (bot1 - bot0) * f, bot0 + off + (bot1 - bot0) * f, 0.35, MAT["arm"], "arms")
        n = 9
        for k in range(n):
            f0, f1 = k / n, (k + 1) / n
            a0 = top0 + (top1 - top0) * f0
            b1 = bot0 + (bot1 - bot0) * f1
            s.beam(a0, b1, 0.5, MAT["arm"], "arms")
            s.beam(top0 + (top1 - top0) * f1, b1, 0.4, MAT["arm"], "arms")
    if cfg.get("ship", True) and cfg.get("booster", True) and cfg.get("qd_arm", True):
        zq = Z_RING1 + 4.0 + cfg.get("ship_lift", 0.0)
        x0, x1 = cx + hw, -R - 0.2
        s.beam((x0, -1.2, zq + 1.2), (x1, -1.2, zq + 1.2), 0.8, MAT["arm"], "arms")
        s.beam((x0, 1.2, zq + 1.2), (x1, 1.2, zq + 1.2), 0.8, MAT["arm"], "arms")
        s.beam((x0, 0, zq - 1.0), (x1, 0, zq - 1.0), 0.8, MAT["arm"], "arms")
        for k in range(5):
            xa = x0 + (x1 - x0) * k / 5
            xb = x0 + (x1 - x0) * (k + 1) / 5
            s.beam((xa, 0, zq - 1.0), (xb, 1.2, zq + 1.2), 0.35, MAT["arm"], "arms")
            s.beam((xa, 0, zq - 1.0), (xb, -1.2, zq + 1.2), 0.35, MAT["arm"], "arms")
        if cfg.get("detail", True):  # r906c: the QD hood that clamps onto the ship's aft section, and a deeper arm
            s.box((-R - 1.0, 0, zq + 0.4), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1.6, 3.2, 3.6), MAT["arm"], "arms")
            s.beam((x0, -1.2, zq - 1.0), (x1 - 1.0, -1.2, zq - 1.0), 0.6, MAT["arm"], "arms")
            s.beam((x0, 1.2, zq - 1.0), (x1 - 1.0, 1.2, zq - 1.0), 0.6, MAT["arm"], "arms")


# ------------------------------------------------------------------ intersection
def _hit_box(o, d, p):
    Rm, c, h = p["R"], p["c"], p["h"]
    ol = (o - c) @ Rm
    dl = d @ Rm
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(dl) < 1e-9, 1e-9, dl)
        t1 = (-h - ol) * inv
        t2 = (h - ol) * inv
    tmn = np.minimum(t1, t2)
    tmx = np.maximum(t1, t2)
    tnear = tmn.max(axis=1)
    tfar = tmx.min(axis=1)
    ok = (tfar >= np.maximum(tnear, EPS)) & (tnear > EPS)
    ax = tmn.argmax(axis=1)
    nl = np.zeros_like(ol)
    idx = np.arange(len(o))
    nl[idx, ax] = -np.sign(dl[idx, ax])
    n = nl @ Rm.T
    if p.get("quad") is not None and ok.any():
        quad, P0, ax0 = p["quad"]
        hp = o + d * tnear[:, None]
        rr = (hp - P0) @ ax0
        vv = hp[:, 2] - P0[2]
        ok &= _in_quad(rr, vv, quad)
    return np.where(ok, tnear, np.inf), n


def _in_quad(u, v, quad):
    inside = np.ones_like(u, bool)
    q = list(quad)
    sgn = None
    # convex polygon test, works for either winding
    cross = []
    for i in range(len(q)):
        (x0, y0), (x1, y1) = q[i], q[(i + 1) % len(q)]
        cross.append((x1 - x0) * (v - y0) - (y1 - y0) * (u - x0))
    cross = np.array(cross)
    inside = (cross >= 0).all(axis=0) | (cross <= 0).all(axis=0)
    return inside


def _hit_cone(o, d, p):
    r0, r1, z0, z1, cx, cy = p["r0"], p["r1"], p["z0"], p["z1"], p["cx"], p["cy"]
    k = (r1 - r0) / (z1 - z0)
    ox, oy, oz = o[:, 0] - cx, o[:, 1] - cy, o[:, 2]
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    A = r0 + k * (oz - z0)
    B = k * dz
    a = dx * dx + dy * dy - B * B
    b = 2 * (ox * dx + oy * dy - A * B)
    c = ox * ox + oy * oy - A * A
    disc = b * b - 4 * a * c
    good = disc >= 0
    sq = np.sqrt(np.where(good, disc, 0))
    a_s = np.where(np.abs(a) < 1e-9, 1e-9, a)
    tbest = np.full(len(o), np.inf)
    for t in ((-b - sq) / (2 * a_s), (-b + sq) / (2 * a_s)):
        z = oz + t * dz
        q = A + B * t
        ok = good & (t > EPS) & (z >= z0) & (z <= z1) & (q > 0) & (t < tbest)
        tbest = np.where(ok, t, tbest)
    hp = o + d * np.where(np.isfinite(tbest), tbest, 0)[:, None]
    q = r0 + k * (hp[:, 2] - z0)
    n = np.stack([hp[:, 0] - cx, hp[:, 1] - cy, -q * k], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return tbest, n


def _aabb(p):
    if p["type"] == "cone":
        r = max(p["r0"], p["r1"])
        return np.array([p["cx"] - r, p["cy"] - r, p["z0"]]), np.array([p["cx"] + r, p["cy"] + r, p["z1"]])
    ext = np.abs(p["R"]) @ p["h"]
    return p["c"] - ext, p["c"] + ext


def _ray_aabb(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    t1, t2 = (lo - o) * inv, (hi - o) * inv
    tn = np.minimum(t1, t2).max(axis=1)
    tf = np.maximum(t1, t2).min(axis=1)
    return tf >= np.maximum(tn, 0)


def trace(scene: Scene, o, d):
    n_r = len(d)
    T = np.full(n_r, np.inf)
    N = np.zeros((n_r, 3))
    M = np.zeros(n_r, int)
    # ground plane
    tg = np.where(d[:, 2] < -1e-6, -o[:, 2] / np.where(d[:, 2] < -1e-6, d[:, 2], -1), np.inf)
    if not getattr(scene, "ground", True):
        tg[:] = np.inf
    T = np.where(tg > 0, tg, np.inf)
    N[:] = (0, 0, 1)
    M[np.isfinite(T)] = MAT["ground"]
    groups = {}
    for p in scene.prims:
        groups.setdefault(p["group"], []).append(p)
    for g, prims in groups.items():
        los, his = zip(*[_aabb(p) for p in prims])
        lo, hi = np.min(los, axis=0) - 0.5, np.max(his, axis=0) + 0.5
        sel = np.nonzero(_ray_aabb(o, d, lo, hi))[0]
        if len(sel) == 0:
            continue
        os_, ds_ = o[sel], d[sel]
        for p in prims:
            plo, phi = _aabb(p)
            sub = np.nonzero(_ray_aabb(os_, ds_, plo - 0.05, phi + 0.05))[0]
            if len(sub) == 0:
                continue
            idx = sel[sub]
            t, n = (_hit_cone if p["type"] == "cone" else _hit_box)(o[idx], d[idx], p)
            better = t < T[idx]
            if better.any():
                ii = idx[better]
                T[ii], N[ii], M[ii] = t[better], n[better], p["mat"]
    return T, N, M


# ------------------------------------------------------------------ shading
def _sky(dirs, cfg):
    z = np.clip(dirs[:, 2], -1, 1)
    top = np.array(cfg.get("sky_top", (0.003, 0.006, 0.013)))
    hor = np.array(cfg.get("sky_hor", (0.035, 0.055, 0.085)))
    gnd = np.array(cfg.get("env_ground", (0.03, 0.03, 0.035)))
    f = np.clip(z, 0, 1)[:, None] ** 0.45
    sky = hor * (1 - f) + top * f
    return np.where((z < 0)[:, None], gnd, sky)


def shade(scene, o, d, T, N, M, cfg):
    col = np.zeros((len(d), 3))
    hit = np.isfinite(T)
    P = o + d * np.where(hit, T, 0)[:, None]
    lights = cfg.get("lights", [((60, -90, 25), 9000.0), ((-70, -80, 25), 7000.0), ((40, 80, 25), 5000.0),
                                ((-60, 60, 25), 4000.0)])
    lc = np.array(cfg.get("light_color", (0.82, 0.90, 1.0)))
    amb = np.array(cfg.get("ambient", (0.012, 0.018, 0.028)))
    diff = np.zeros((len(d), 3))
    spec = np.zeros((len(d), 3))
    V = -d
    for pos, power in lights:
        L = np.asarray(pos, float) - P
        dist = np.linalg.norm(L, axis=1, keepdims=True) + 1e-6
        L /= dist
        att = power / (dist[:, 0] ** 2 + 400.0)
        ndl = np.clip((N * L).sum(1), 0, None)
        diff += (ndl * att)[:, None] * lc
        Hh = L + V
        Hh /= np.linalg.norm(Hh, axis=1, keepdims=True) + 1e-9
        spec += ((np.clip((N * Hh).sum(1), 0, None) ** 40) * att * 1.4)[:, None] * lc
    if cfg.get("sun"):
        sd, sp, sc = cfg["sun"]
        L = _norm(np.asarray(sd, float))
        ndl = np.clip(N @ L, 0, None)
        diff += (ndl * sp)[:, None] * np.asarray(sc)
        Hh = L[None] + V
        Hh /= np.linalg.norm(Hh, axis=1, keepdims=True) + 1e-9
        spec += ((np.clip((N * Hh).sum(1), 0, None) ** 40) * sp * 1.4)[:, None] * np.asarray(sc)
    refl = d - 2 * (d * N).sum(1, keepdims=True) * N
    env = _sky(refl, cfg)
    # material params: albedo, metal (env share), spec
    alb = {MAT["ground"]: 0.14, MAT["booster"]: 0.55, MAT["ship"]: 0.55, MAT["tower"]: 0.10, MAT["ring"]: 0.12,
           MAT["fin"]: 0.16, MAT["olm"]: 0.26, MAT["arm"]: 0.11, MAT["flap"]: 0.5, MAT["deflector"]: 0.2}
    met = {MAT["booster"]: 0.85, MAT["ship"]: 0.85, MAT["flap"]: 0.8, MAT["fin"]: 0.4, MAT["ring"]: 0.3}
    for m, a in alb.items():
        sel = hit & (M == m)
        if not sel.any():
            continue
        a_arr = np.full(sel.sum(), a)
        mt = met.get(m, 0.0)
        mt_arr = np.full(sel.sum(), mt)
        Ps, Ns = P[sel], N[sel]
        if m in (MAT["ship"], MAT["flap"]):
            # tiles on the windward half (and the windward face of each flap): black hexagonal tiles
            wind = (Ns[:, :2] @ WINDWARD[:2]) > -0.02 if m == MAT["ship"] else (Ns[:, :2] @ WINDWARD[:2]) > 0.1
            th = np.arctan2(Ps[:, 1], Ps[:, 0])
            cell = np.floor(th * 300).astype(np.int64) * 7919 + np.floor(Ps[:, 2] * 3.3).astype(np.int64) * 104729
            tile = 0.028 + 0.014 * ((cell * 2654435761) % 1000 / 1000.0)
            a_arr = np.where(wind, tile, a_arr)
            mt_arr = np.where(wind, 0.08, mt_arr)
        if m in (MAT["booster"], MAT["ship"]):
            weld = (np.abs(((Ps[:, 2] + 0.3) % 1.83) - 0.9) > 0.86)  # ring welds every ~1.8 m
            a_arr = np.where(weld, a_arr * 0.75, a_arr)
        if m == MAT["ring"]:
            th = np.arctan2(Ps[:, 1], Ps[:, 0])
            vent = (np.sin(th * 18) > 0.2) & (np.abs(Ps[:, 2] - (Ps[:, 2].min() + Ps[:, 2].max()) / 2) < 0.7)
            a_arr = np.where(vent, 0.015, 0.25)
        if m == MAT["fin"]:
            g = (np.sin(Ps[:, 0] * 9) > 0.6) | (np.sin(Ps[:, 1] * 9) > 0.6)
            a_arr = np.where(g, 0.05, 0.2)
        c = (diff[sel] + amb) * a_arr[:, None] * (1 - mt_arr[:, None]) + env[sel] * mt_arr[:, None] * 0.9 \
            + (diff[sel] * 0.25 * a_arr[:, None] * mt_arr[:, None]) + spec[sel] * (0.15 + mt_arr[:, None] * 0.85)
        if m == MAT["ground"] and cfg.get("ground_tint"):
            c = c * np.asarray(cfg["ground_tint"])
        col[sel] = c
    col[~hit] = _sky(d[~hit], cfg)
    # haze
    hz = np.array(cfg.get("haze", (0.035, 0.05, 0.075)))
    dens = cfg.get("haze_density", 1 / 1600.0)
    f = 1 - np.exp(-np.where(hit, T, 3000) * dens)
    col = col * (1 - f[:, None]) + hz * f[:, None]
    return col


# ------------------------------------------------------------------ camera
ENVS = {
    "night": {"sky_dark": (0.32, 0.75)},
    "dusk": dict(sky_top=(0.03, 0.04, 0.09), sky_hor=(0.42, 0.26, 0.20), env_ground=(0.05, 0.04, 0.04),
                 ambient=(0.05, 0.05, 0.07), haze=(0.20, 0.15, 0.15), sun=((1, -0.4, 0.05), 0.25, (1.0, 0.6, 0.4)),
                 lights=[]),
    "bluehour": dict(sky_top=(0.02, 0.04, 0.10), sky_hor=(0.12, 0.20, 0.36), env_ground=(0.03, 0.04, 0.06),
                     ambient=(0.03, 0.05, 0.09), haze=(0.08, 0.12, 0.20), haze_density=1 / 6000.0),
    "dawn": dict(sky_top=(0.10, 0.16, 0.30), sky_hor=(0.75, 0.62, 0.55), env_ground=(0.10, 0.10, 0.11),
                 ambient=(0.10, 0.11, 0.14), haze=(0.35, 0.33, 0.35), haze_density=1 / 5000.0, sun=((-1, -0.6, 0.12), 0.9, (1.0, 0.85, 0.7)),
                 lights=[]),
    "space": dict(sky_top=(0, 0, 0), sky_hor=(0, 0, 0), env_ground=(0.02, 0.04, 0.09), ambient=(0.01, 0.012, 0.02),
                  haze=(0, 0, 0), haze_density=0.0, sun=((1, -0.3, 0.2), 1.6, (1.0, 0.97, 0.92)), lights=[]),
    "mars": dict(sky_top=(0.05, 0.08, 0.16), sky_hor=(0.30, 0.40, 0.55), env_ground=(0.18, 0.08, 0.05),
                 ambient=(0.05, 0.05, 0.07), haze=(0.20, 0.22, 0.30), haze_density=1 / 3000.0, ground_tint=(1.6, 0.75, 0.5),
                 sun=((-1, 0.2, 0.06), 0.6, (0.7, 0.8, 1.0)), lights=[]),
}


def render(cfg: dict, W=768, H=1344, ss=2):
    cfg = {**ENVS[cfg.get("env", "night")], **cfg}
    scene = build(cfg)
    scene.ground = cfg.get("ground", True)
    cam = np.asarray(cfg["cam"], float)
    tgt = np.asarray(cfg["target"], float)
    fov = math.radians(cfg.get("fov", 30.0))
    fwd = _norm(tgt - cam)
    right = _norm(np.cross(fwd, (0, 0, 1)))
    up = np.cross(right, fwd)
    w, h = W * ss, H * ss
    th = math.tan(fov / 2)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (2 * (xs + 0.5) / w - 1) * th * w / h
    v = (1 - 2 * (ys + 0.5) / h) * th
    d = fwd[None, None] + u[..., None] * right[None, None] + v[..., None] * up[None, None]
    d = (d / np.linalg.norm(d, axis=2, keepdims=True)).reshape(-1, 3)
    o = np.broadcast_to(cam, d.shape).copy()
    T, N, M = trace(scene, o, d)
    col = shade(scene, o, d, T, N, M, cfg)
    img = col.reshape(h, w, 3)
    # floodlight glows (lights in view)
    for pos, power in cfg.get("glow", []):
        pv = np.asarray(pos, float) - cam
        z = pv @ fwd
        if z <= 0:
            continue
        px = ((pv @ right) / z / (th * w / h) + 1) * w / 2
        py = (1 - (pv @ up) / z / th) * h / 2
        rr = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
        img += (power * np.exp(-rr / (0.012 * h)))[..., None] * np.array([0.85, 0.92, 1.0])
    img = img / (1 + img)  # soft tone map
    img = np.clip(img ** (1 / 2.2) * 1.15, 0, 1)
    if cfg.get("sky_dark"):  # r906c: the tone map lifted the night sky to a flat grey-blue that Klein copied
        k0, k1 = cfg["sky_dark"]  # brightness factor at the zenith side of the frame / at the horizon
        sky = (M.reshape(h, w) == 0)
        grad = k0 + (k1 - k0) * np.clip(ys / h, 0, 1) ** 1.5
        img = np.where(sky[..., None], img * grad[..., None], img)
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    mask = Image.fromarray(((M.reshape(h, w) != 0) & (M.reshape(h, w) != MAT["ground"])).astype(np.uint8) * 255).resize((W, H))
    return im, mask


GLOW = [((60, -90, 25), 1.2), ((-70, -80, 25), 1.0), ((140, -20, 22), 0.8), ((-130, 30, 22), 0.7)]

PRESETS = {
    # the look sheet: full stack, three-quarter side view from the leeward/steel side, slightly below
    # shot guides (16:9). The robot and people are added later by a Klein edit, never drawn here.
    "in_pad": dict(cam=(70, 190, 1.7), target=(-6, 0, 64), fov=52, glow=GLOW),
    "in_lights": dict(cam=(-40, 26, 2.0), target=(-21, 0, 105), fov=62, glow=[]),
    "in_frost": dict(cam=(15, 13, 104), target=(0, -1, 107), fov=30, glow=[]),
    "in_l1": dict(cam=(4, 48, 1.5), target=(-4, 0, 105), fov=72, glow=[]),
    "v1_stack": dict(cam=(70, 120, 10), target=(-6, 0, 98), fov=46, ship_lift=14.0, arms_z=134.0, arms_open=0.0,
                     qd_arm=False, glow=GLOW),
    "p1_catch": dict(env="dusk", cam=(110, 210, 4), target=(-6, 0, 80), fov=46, ship=False, booster_lift=22.0,
                     arms_z=108.6, arms_open=0.0, glow=[]),
    "c1_comes": dict(cam=(60, 175, 1.6), target=(-6, 0, 62), fov=54, glow=GLOW),
    "c1_salute": dict(cam=(45, 140, 6), target=(-4, 0, 70), fov=58, glow=GLOW),
    "d1_flap_up": dict(cam=(16, -10, 74), target=(6, -2, 97), fov=34, aft_deflect=-45.0, glow=[]),
    "d1_flap_dn": dict(cam=(16, -10, 74), target=(6, -2, 97), fov=34, aft_deflect=10.0, glow=[]),
    "d1_arms": dict(cam=(-2, 40, 105), target=(-4, 0, 130), fov=55, glow=[]),
    "d1_sea": dict(env="dawn", cam=(900, 2600, 3), target=(-5, 0, 50), fov=12, glow=[]),
    "d1_dawn": dict(env="dawn", cam=(120, 330, 3), target=(-6, 0, 62), fov=40, glow=[]),
    "v2_tower": dict(cam=(-2, 40, 1.4), target=(-8, 0, 100), fov=76, glow=[]),
    "p2_frost": dict(cam=(7, 10, 60), target=(0, 0, 62), fov=40, glow=[]),
    "c2_liftoff": dict(cam=(140, 380, 3), target=(-6, 0, 80), fov=42, stack_lift=25.0, qd_arm=False, arms_z=110.0,
                       arms_open=35.0, glow=GLOW),
    "c2_climb": dict(cam=(200, 900, 2400), target=(0, 0, 2470), fov=12, tower=False, pad=False, stack_lift=2400.0,
                     ground=False, haze_density=0.0, glow=[]),
    "c2_limb": dict(env="space", cam=(60, 120, 115), target=(0, 0, 118), fov=40, booster=False, tower=False, pad=False,
                    ground=False, glow=[]),
    "br_blue": dict(env="mars", cam=(700, 1500, 4), target=(0, 0, 60), fov=16, booster=False, tower=False, pad=False,
                    ship_z=-Z_RING1 + 2.0, glow=[]),
    "c3_dusk": dict(env="mars", cam=(500, 900, 3), target=(0, 0, 40), fov=24, booster=False, tower=False, pad=False,
                    ship_z=-Z_RING1 + 2.0, glow=[]),
    "p1_dunes": dict(env="bluehour", cam=(1300, 2600, 6), target=(-5, 0, 55), fov=9, glow=[]),
    "c1_beach": dict(env="bluehour", cam=(-900, 2700, 4), target=(-5, 0, 55), fov=10, glow=[]),
    "d1_crowd": dict(cam=(35, 120, 1.8), target=(-6, 0, 28), fov=52, glow=GLOW),
    "v2_arm": dict(cam=(-9, 34, 99.5), target=(-9, 0, 96.5), fov=50, glow=[]),
    "p2_steam": dict(cam=(45, 95, 3), target=(-10, 0, 16), fov=52, glow=GLOW),
    "c3_next": dict(cam=(80, 160, 2.0), target=(-6, 0, 85), fov=56, ship_lift=14.0, arms_z=134.0, arms_open=0.0,
                    qd_arm=False, glow=GLOW),
    "look_34": dict(cam=(150, 260, 22), target=(-8, 0, 72), fov=34, glow=GLOW),
    "look_tiles": dict(cam=(120, -270, 20), target=(-8, 0, 72), fov=34, glow=GLOW),
    "look_side": dict(cam=(10, 300, 30), target=(-8, 0, 72), fov=32, glow=GLOW),
}


def main(argv):
    if argv[:1] == ["presets"]:
        print(json.dumps(PRESETS, indent=1))
        return 0
    if argv[:1] == ["render"]:
        cfg = PRESETS.get(argv[1]) or json.loads(Path(argv[1]).read_text())
        W, H = (int(argv[3]), int(argv[4])) if len(argv) > 4 else (768, 1344)
        im, mask = render(cfg, W, H)
        im.save(argv[2])
        mask.save(str(argv[2]).replace(".png", "_mask.png"))
        print(argv[2])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
