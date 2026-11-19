"""Shape sliders of the remade body: vertex morphs (one per slider) written into corps.pmx, and
the tool that applies them to a PMX — the bare body or a body dressed by wear.py: each vertex of a
garment follows the closest skin point.

  python3 -I sliders.py --body <rebuilt body.pmx> --model <pmx to deform> --values chest=1,hips=-0.5 \
      --output <pmx>

A slider changes a volume, not a proportion: limb length and shoulder width belong to the
skeleton (one template per skeleton). Values from -1 to 1; the displacements add up.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "atlas"), os.path.join(HERE, "..", "head")):
    if d not in sys.path:
        sys.path.insert(0, d)

import geom as G              # noqa: E402

CM = 8.0
MIR = np.array([-1.0, 1.0, 1.0])
PREFIX = "shape_"   # prefix of the shape morphs written in the remade bodies (format 2, revision 2)


def bell(x, center, width):
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def ramp(x, a, b):
    """0 before a, 1 after b, smooth in between."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def around_axe(P, a, b):
    """Coordinate along the axis a -> b (cm from a), radial vector, length of the axis."""
    d = b - a
    L = np.linalg.norm(d)
    d = d / L
    s = (P - a) @ d
    return s, (P - a) - np.outer(s, d), L


def morphs(g, R):
    """Displacement of each vertex (cm) for each slider at +1."""
    P, sym = g["P"], g["sym"]
    o = R.os
    n = len(P)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    # what belongs to the trunk and legs: not the arm beyond the shoulder
    sb, rb, Lb = around_axe(P * [1, 1, 1], o["左腕"], o["左ひじ"])
    sbd, rbd, _ = around_axe(P * MIR, o["左腕"], o["左ひじ"])
    arm_g = (x > 0) & (sb > 2.0) & (np.linalg.norm(rb, axis=1) < 9.0) & (y > 80)
    arm_d = (x < 0) & (sbd > 2.0) & (np.linalg.norm(rbd, axis=1) < 9.0) & (y > 80)
    main = (np.abs(x) > 30) & (y > 80)
    trunk = ~(arm_g | arm_d | main)
    zc = float(o["上半身"][2]) + 3.0                          # trunk axis, front to back
    rad = np.stack([x, np.zeros(n), z - zc], 1)             # horizontal vector from the trunk axis
    M = {}

    # waist: the waist circumference, around its narrowest height
    y_t = float(o["上半身1"][1]) + 4.0
    M["size"] = rad * (0.12 * bell(y, y_t, 7.0) * trunk)[:, None]

    # hips: the width of the pelvis and of the top of the thighs
    y_h = float(o["下半身"][1]) - 6.0
    M["hips"] = np.stack([x * 0.11, np.zeros(n), (z - zc) * 0.04], 1) * (bell(y, y_h, 8.0) * trunk)[:, None]

    # buttocks: backwards, around the buttock bones
    c = 0.5 * (o["左尻1"] + o["左尻2"])
    d = np.minimum(np.linalg.norm(P - c, axis=1), np.linalg.norm(P - c * MIR, axis=1))
    w = bell(d, 0.0, 7.5) * ramp(z, zc, zc + 5.0) * trunk
    M["buttocks"] = np.stack([x * 0.03, np.zeros(n), np.full(n, 2.2)], 1) * w[:, None]

    # chest: each breast grows from its base
    b = o["左胸上2"]
    base = np.array([b[0] + 1.0, b[1] - 4.5, b[2] + 3.5])
    D = np.zeros((n, 3))
    for s_ in (1.0, -1.0):
        c = base * [s_, 1, 1]
        v = P - c
        w = bell(np.linalg.norm(v, axis=1), 0.0, 6.5) * ramp(-(z - zc), 4.0, 9.0) * trunk * ramp(x * s_, 0.3, 3.5)
        D += (v * 0.22 + np.array([0.0, -0.3, -1.2])) * w[:, None]
    M["chest"] = D

    # thighs: the thigh circumference, from the crotch to the knee
    D = np.zeros((n, 3))
    for s_ in (1.0, -1.0):
        s, r, L = around_axe(P * [s_, 1, 1], o["左足"], o["左ひざ"])
        w = ramp(s, 3.0, 12.0) * (1 - ramp(s, L - 8.0, L + 2.0)) * (x * s_ > 0) * trunk
        # on the crotch side, the thigh grows less: the two thighs do not run into each other
        inside = 1.0 - 0.6 * ramp(-(r[:, 0]), 0.0, 4.0)
        D += r * [s_, 1, 1] * (0.10 * w * inside)[:, None]
    M["thighs"] = D

    # arms: the circumference of the upper arm and forearm
    D = np.zeros((n, 3))
    for s_, mask in ((1.0, arm_g), (-1.0, arm_d)):
        Q = P * [s_, 1, 1]
        s1, r1, L1 = around_axe(Q, o["左腕"], o["左ひじ"])
        s2, r2, L2 = around_axe(Q, o["左ひじ"], o["左手首"])
        top = s2 < 0
        r = np.where(top[:, None], r1, r2)
        w = np.where(top, ramp(s1, 4.0, 10.0), 1 - ramp(s2, L2 - 6.0, L2 - 1.0)) * mask
        D += r * [s_, 1, 1] * (0.10 * w)[:, None]
    M["arm"] = D

    # shoulders: the volume of the shoulder (shoulder width itself belongs to the skeleton)
    D = np.zeros((n, 3))
    for s_ in (1.0, -1.0):
        c = (o["左腕"] + [1.0, -1.0, 0.0]) * [s_, 1, 1]
        v = P - c
        w = bell(np.linalg.norm(v, axis=1), 0.0, 5.0) * (x * s_ > 0) * ~main
        D += v * (0.13 * w)[:, None]
    M["shoulders"] = D

    out = {}
    for name, D in M.items():
        D = 0.5 * (D + D[sym] * MIR)                 # exact symmetry
        D[np.linalg.norm(D, axis=1) < 0.005] = 0.0
        out[PREFIX + name] = D
    return out


# ---------------------------------------------------------------------------------------- application
def follow(Q, V, D, k=4, block=128):
    """Displacement of each point of Q: that of the k closest skin vertices, weighted by the inverse
    square of the distance (a skin vertex itself gets exactly its own). Also returns the distance."""
    disp = np.zeros_like(Q)
    dist = np.zeros(len(Q))
    v2 = (V * V).sum(1)
    for a in range(0, len(Q), block):
        q = Q[a:a + block]
        d2 = np.maximum((q * q).sum(1)[:, None] - 2 * q @ V.T + v2[None], 0.0)
        i = np.argpartition(d2, k, axis=1)[:, :k]
        dk = np.take_along_axis(d2, i, 1)
        w = 1.0 / (dk + 1e-6)
        w /= w.sum(1, keepdims=True)
        disp[a:a + block] = (D[i] * w[:, :, None]).sum(1)
        dist[a:a + block] = np.sqrt(dk.min(1))
    return disp, dist


def apply(body, model, values, reach_cm=8.0, log=print):
    """body: remade PMX (its shape morphs, PREFIX); model: PMX to deform (dict read by pmx.read), modified in
    place. Skin vertices follow their morph; the others follow the closest skin point, less and
    less beyond `reach_cm`."""
    import pmx as PMX
    t = PMX.to_arrays(body)
    skin = [m["name"] for m in body["materials"]].index("肌")
    D = np.zeros((len(t["pos"]), 3))
    known = {m["name"][len(PREFIX):]: m for m in body["morphs"] if m["name"].startswith(PREFIX)}
    for name, v in values.items():
        if name not in known:
            raise SystemExit(f"unknown slider: {name} (known: {', '.join(known)})")
        for i, d in known[name]["offsets"]:
            D[i] += v * np.array(d) * CM
    Fp = t["faces"][t["mat_face"] == skin]
    ids = np.unique(Fp)
    V, Ds = t["pos"][ids] * CM, D[ids]
    s = model["vertices"]
    Q = np.array(s["pos"]) * CM
    disp, dist = follow(Q, V, Ds)
    disp *= (1 - ramp(dist, reach_cm, 2 * reach_cm))[:, None]
    Q2 = Q + disp
    s["pos"] = [tuple(float(x) for x in p / CM) for p in Q2]
    # normals: recomputed per material, vertices welded
    tm = PMX.to_arrays(model)
    nor = np.array(s["nor"])
    for i in range(len(model["materials"])):
        Fm = tm["faces"][tm["mat_face"] == i]
        if not len(Fm):
            continue
        ids = np.unique(Fm)
        r2, i2 = G.weld(Q2[ids], 1e-3)
        loc = -np.ones(len(Q2), int)
        loc[ids] = i2
        Fl = loc[Fm]
        Fl = Fl[(Fl[:, 0] != Fl[:, 1]) & (Fl[:, 1] != Fl[:, 2]) & (Fl[:, 0] != Fl[:, 2])]
        nor[ids] = G.normals_vertices(Q2[ids][r2], Fl)[i2]
    s["nor"] = [tuple(float(x) for x in v) for v in nor]
    log(f"sliders {values}: {int((np.linalg.norm(disp, axis=1) > 0.01).sum())} vertices moved, "
            f"up to {np.linalg.norm(disp, axis=1).max() * 10:.1f} mm")
    return model


def main():
    import pmx as PMX
    import pmx_write as pmx_write
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", dest="body", required=True)
    ap.add_argument("--model", dest="model", help="PMX to deform (default: the body itself)")
    ap.add_argument("--values", dest="values", required=True, help="chest=1,hips=-0.5")
    ap.add_argument("--output", dest="output", required=True)
    a = ap.parse_args()
    values = {k: float(v) for k, v in (kv.split("=") for kv in a.values.split(","))}
    body = PMX.read(a.body)
    model = PMX.read(a.model or a.body)
    apply(body, model, values)
    pmx_write.write(model, a.output)


if __name__ == "__main__":
    main()
