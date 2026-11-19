"""Shape repair: where the transfer broke the shape of a part, it is rebuilt from the original.

The transfer places each vertex relative to the body; a few vertices land in the wrong place
(attached to the arm instead of the chest, to the other leg…) and pull out spikes, blades and
tears. The original shape (same vertices, same order) says what the part should be: a vertex
most of whose edges no longer have the expected length is faulty; it is put back from its sound
neighbours, copying the original shape at the scale and in the orientation it has taken around
it. Vertices that coincided in the original and were separated by the transfer are sewn back.

Then the whole part takes back its shape, in the body's frame: close to the skin (under `PRES`),
the transfer is right, the part must hug this body; far from the skin (over `LOIN`: skirt hem,
coat flap, bow), nothing holds it to the body and the original shape is right, at the part's
scale. In between, each vertex gives way more or less. The solver looks for the positions where
each edge is the original one, rotated like its neighbourhood ("as rigid as possible" shape),
held to their transferred place all the more firmly as they are close to the skin.

Independent check: the skin must not show through in front of the part (review.skin_outside). Each
part takes the boldest repair that respects it: whole shape loosely held, then firmly held, then
the faulty vertices alone, otherwise it stays as it is.

    python3 repair.py measure [id ...]        review score before and after, without writing anything
    python3 repair.py apply [id ...]      rewrites piece.glb in the cache (the old one kept as part.before-repair.glb)
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import glb as GLB  # noqa: E402
import outfit as CB  # noqa: E402
import review as RV  # noqa: E402
import skin as PE  # noqa: E402

LONG = 2.0        # an edge at least LONG times longer than expected…
COURT = 0.4       # … or shrunk to less than COURT times its length…
GAP = 0.008     # … and off by at least this much, is wrong
PART = 0.5        # share of wrong edges from which a vertex is faulty
SEAM = 0.002   # vertices that coincided in the original, separated by more than this: sewn back
UNSEWN = 0.01    # separated by more than this: their place is no longer known, they are faulty
AROUND = 6        # rings of neighbours of a faulty zone whose transferred place is barely trusted
LOOPS = 2       # sound neighbours taken around a faulty zone to recover scale and orientation
NEAR = 0.015      # height above the skin up to which the part follows the transfer
FAR = 0.05       # height from which it only follows its original shape
OUTFITS = (3.0, 30.0)   # from loosest to firmest: the part takes the first one that does not make the skin show through
SKIN_TOLERATED = 2.0     # cm² of extra skin in front of the part that a repair is allowed to cost…
SKIN_GAIN = 0.05       # … or this per score point gained: a shattered part put back in shape is worth a little skin
SKIN_PART = 0.002      # … or this share of the part's area, if larger
OUTFIT_FAR = 0.02 # what is left of it far from the skin: the part does not drift
ROUND = 8        # shape repairs, each followed by lifting back above the skin
STEPS = (4, 8)   # rounds whose state is offered: rounds do not always converge, the last is not always the best
ARROW = 0.015    # a face between outside vertices only dips under the skin by its sag; deeper, it is not the cause
ON_SKIN = 0.002  # minimum height above the skin
PERIMETERS = 8         # rotation / position alternations
PASSES = 4        # successive repairs of the same part, as long as its score drops
POSE_MIN = 0.6    # share of a foot's vertices that must agree with the rigid placement of a shoe
POSE_SKIN = 20.0  # cm² of skin in front that a rigid placement is allowed to cost (the edge of the shaft on the ankle)
ZONE_MAX = 0.5    # beyond this share of the part, it is no longer a repair: the part must be transferred again


def _similarity(A, B):
    """Matrix M (3x3) such that (A - centre A) @ M ≈ B - centre B: rotation and scale."""
    a, b = A - A.mean(0), B - B.mean(0)
    U, s, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    D = np.diag([1.0, 1.0, d])
    samp = (s * np.diag(D)).sum() / max((a * a).sum(), 1e-12)
    return samp * (U @ D @ Vt)


def resume(S, Z, F, shape=True, outfit=None, faces=False, steps=False):
    """Returns (corrected Z, summary). S: original shape, Z: transferred part, same vertices; F: faces."""
    w = RV._weld(S)
    n = int(w.max()) + 1
    nb = np.bincount(w, minlength=n)
    Sw, Zw = np.zeros((n, 3)), np.zeros((n, 3))
    np.add.at(Sw, w, S)
    np.add.at(Zw, w, Z)
    Sw /= nb[:, None]
    Zw /= nb[:, None]
    gap = np.zeros(n)
    np.maximum.at(gap, w, np.linalg.norm(Z - Zw[w], axis=1))
    Fw = w[F]
    E = np.unique(np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1), axis=0)
    E = E[E[:, 0] != E[:, 1]]
    assessment = {"vertices": n, "fautifs": 0, "recousus": 0, "zones": 0, "laissees": 0}
    if not len(E):
        return Z.copy(), assessment
    ls = np.linalg.norm(Sw[E[:, 0]] - Sw[E[:, 1]], axis=1)
    r0 = float(np.median(np.linalg.norm(Zw[E[:, 0]] - Zw[E[:, 1]], axis=1) / np.maximum(ls, 1e-9)))

    def false(X, alive):
        lz = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
        f = (((lz > LONG * r0 * ls) & (lz - r0 * ls > GAP)) | ((lz < COURT * r0 * ls) & (r0 * ls - lz > GAP)))
        return f & alive[E[:, 0]] & alive[E[:, 1]]

    free = gap > UNSEWN
    for _ in range(12):
        alive = ~free
        f = false(Zw, alive)
        if not f.any():
            break
        deg, bad = np.zeros(n), np.zeros(n)
        ev = alive[E[:, 0]] & alive[E[:, 1]]
        np.add.at(deg, E[ev].ravel(), 1)
        np.add.at(bad, E[f].ravel(), 1)
        part = bad / np.maximum(deg, 1)
        worst = np.maximum(part[E[f, 0]], part[E[f, 1]])
        new = np.zeros(n, bool)
        for k in (0, 1):      # of each wrong edge, the most faulty end
            v = E[f, k]
            new[v[(part[v] >= worst - 1e-9) & (part[v] >= PART)]] = True
        if not new.any():     # wrong edges with no clearly faulty vertex: the worst end of each
            a, b = E[f, 0], E[f, 1]
            new[np.where(part[a] >= part[b], a, b)] = True
        free |= new
    X = Zw.copy()
    if free.any() and free.mean() <= ZONE_MAX:
        # zones: components of faulty vertices
        label = np.where(free, np.arange(n), -1)
        el = E[free[E[:, 0]] & free[E[:, 1]]]
        for _ in range(200):
            m = np.maximum(label[el[:, 0]], label[el[:, 1]])
            before = label.copy()
            np.maximum.at(label, el[:, 0], m)
            np.maximum.at(label, el[:, 1], m)
            if (label == before).all():
                break
        neigh = [[] for _ in range(n)]
        for a, b in E:
            neigh[a].append(b)
            neigh[b].append(a)
        for z in np.unique(label[free]):
            in_ = np.where(label == z)[0]
            border, front = set(), set(in_.tolist())
            for _ in range(LOOPS):
                suite = {v for u in front for v in neigh[u] if not free[v] and v not in border}
                border |= suite
                front = suite
            border = np.array(sorted(border), np.int64)
            if len(border) < 3:
                assessment["laissees"] += 1
                continue
            M = _similarity(Sw[border], Zw[border])
            en = np.zeros(n, bool)
            en[in_] = True
            ez = E[en[E[:, 0]] | en[E[:, 1]]]
            ez = ez[(en[ez] | ~free[ez]).all(1)]      # towards the zone or towards sound vertices, not towards another zone
            target = (Sw[ez[:, 0]] - Sw[ez[:, 1]]) @ M  # expected X[a] - X[b]
            X[in_] = Zw[border].mean(0) + (Sw[in_] - Sw[border].mean(0)) @ M
            for _ in range(300):
                vert, cpt = np.zeros((n, 3)), np.zeros(n)
                np.add.at(vert, ez[:, 0], X[ez[:, 1]] + target)
                np.add.at(vert, ez[:, 1], X[ez[:, 0]] - target)
                np.add.at(cpt, ez.ravel(), 1)
                fresh = vert[in_] / np.maximum(cpt[in_], 1)[:, None]
                finished = np.abs(fresh - X[in_]).max() < 1e-5
                X[in_] = fresh
                if finished:
                    break
            assessment["zones"] += 1
            assessment["fautifs"] += int(len(in_))
    elif free.any():
        assessment["laissees"] = -1      # too much of the part is wrong
        free[:] = False
    assessment["recousus"] = int((~free & (gap > SEAM)).sum())
    if not shape:
        out = Z.copy()
        sewn = free | (gap > SEAM)
        out[sewn[w]] = X[w][sewn[w]]
        return out, assessment
    # the whole part takes back its shape, held to the transfer according to its height above the skin
    OUTFIT = outfit or OUTFITS[0]
    h = CB.lines(Zw, np.zeros((n, 3)))[1]
    lam = np.where(np.isfinite(h), OUTFIT_FAR + (OUTFIT - OUTFIT_FAR) * np.clip((FAR - h) / (FAR - NEAR), 0, 1), OUTFIT_FAR)
    lam[free] = 0.0
    # around a faulty zone, the vertices were dragged along by it: their transferred place is no
    # better, barely less wrong. They are held less and less firmly as they get closer to it.
    near = free.astype(float)
    for k in range(1, AROUND + 1):
        vu = near > 0
        suite = np.zeros(n, bool)
        suite[E[vu[E[:, 0]], 1]] = True
        suite[E[vu[E[:, 1]], 0]] = True
        near[suite & ~vu] = 1.0 - k / (AROUND + 1.0)
    lam = lam * (1.0 - near) ** 2 + OUTFIT_FAR * (1.0 - (1.0 - near) ** 2) * (near > 0) * (~free)
    a, b = E[:, 0], E[:, 1]
    dS = (Sw[a] - Sw[b]) * r0
    deg = np.bincount(E.ravel(), minlength=n).astype(float)

    def total(idx, V):
        return np.stack([np.bincount(idx, V[:, k], minlength=n) for k in range(3)], 1)

    def A(Y):      # (Laplacian + hold) · Y
        d = Y[a] - Y[b]
        return total(a, d) - total(b, d) + lam[:, None] * Y

    targetZ = Zw.copy()
    Fu = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 2] != Fw[:, 0])]
    def rendered(X):
        return X[w], dict(assessment, shift_mm=round(float(np.percentile(np.linalg.norm(X - Zw, axis=1), 95) * 1000), 1))

    done = []      # with steps: the state at each round of STEPS, and the last one
    for round_ in range(ROUND):
      if round_ in STEPS:
        done.append(rendered(X))
      if round_:     # the repaired shape does not go under the skin: what went down there comes back up, and is held there
        nl, h2 = CB.lines(X, np.zeros((n, 3)))
        missing = np.where(np.isfinite(h2), ON_SKIN - h2, 0.0)
        # between outside vertices, a large face dips under a curved skin (a breast under a top
        # with large faces): the midpoint of an edge or the centre of a face that is under lifts its vertices
        for G in ((E, Fu) if faces else ()):
            hm = CB.lines(X[G].mean(1), np.zeros((len(G), 3)))[1]
            mm = np.where(np.isfinite(hm) & (hm > -ARROW), ON_SKIN - hm, 0.0)
            for k in range(G.shape[1]):
                np.maximum.at(missing, G[:, k], mm)
        bottom = (missing > 0) & np.isfinite(h2)
        if not bottom.any():
            break
        targetZ[bottom] = X[bottom] + missing[bottom, None] * nl[bottom]
        lam[bottom] = np.maximum(lam[bottom], OUTFIT)
      for _ in range(PERIMETERS):
          # rotation of each vertex: the one that brings its original edges onto its current edges
          dX = X[a] - X[b]
          cov = np.einsum("ei,ej->eij", dS, dX)
          Cv = np.zeros((n, 3, 3))
          for i in range(3):
              for j in range(3):
                  Cv[:, i, j] = np.bincount(a, cov[:, i, j], minlength=n) + np.bincount(b, cov[:, i, j], minlength=n)
          U, _, Vt = np.linalg.svd(Cv + 1e-12 * np.eye(3))
          sg = np.sign(np.linalg.det(U @ Vt))
          U[:, :, 2] *= sg[:, None]
          Rot = U @ Vt                                    # dS @ Rot ≈ dX
          target = 0.5 * (np.einsum("ei,eij->ej", dS, Rot[a]) + np.einsum("ei,eij->ej", dS, Rot[b]))
          sec = total(a, target) - total(b, target) + lam[:, None] * targetZ
          r = sec - A(X)                                  # conjugate gradient
          p, rr = r.copy(), (r * r).sum()
          for _ in range(200):
              if rr < 1e-14:
                  break
              Ap = A(p)
              al = rr / max((p * Ap).sum(), 1e-30)
              X = X + al * p
              r = r - al * Ap
              rn = (r * r).sum()
              p, rr = r + (rn / rr) * p, rn
    done.append(rendered(X))
    return done if steps else done[-1]


def normals(S, Z0, N0, Z, F):
    """Normals of the repaired part: the previous ones where nothing moved (2 mm), otherwise recomputed
    (smooth, over vertices that coincided in the original)."""
    w = RV._weld(S)
    n = int(w.max()) + 1
    nf = np.cross(Z[F[:, 1]] - Z[F[:, 0]], Z[F[:, 2]] - Z[F[:, 0]])
    accessory = np.zeros((n, 3))
    for k in range(3):
        np.add.at(accessory, w[F[:, k]], nf)
    N = accessory[w]
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
    guard = (np.linalg.norm(Z - Z0, axis=1) < 0.002) | (np.linalg.norm(N, axis=1) < 0.5)
    N[guard] = N0[guard]
    return N


def skin_cm2(p, Z, hidden=None):
    """Area of skin that shows in front of the part alone on the body (cm²): the repair must not open it up."""
    c = CB.resolve([CB.Group(Z, p.nor, p.F, p.meta, p.id, p.flags)])[0]
    return round(sum(d["area_cm2"] for d in RV.skin_outside(c, None if hidden is None else {p.id: hidden})), 1)


def place(S, Z):
    """A shoe is a rigid object: each foot takes back its original shape, at the place, scale and
    orientation the transfer gave it (similarity fitted on the vertices that agree, the others set
    aside). Returns None if a foot does not agree to POSE_MIN."""
    out = Z.copy()
    for side in (S[:, 0] < 0, S[:, 0] >= 0):
        if side.sum() < 10:
            continue
        A, B = S[side], Z[side]
        ok = np.ones(len(A), bool)
        for _ in range(8):
            M = _similarity(A[ok], B[ok])
            P = (A - A[ok].mean(0)) @ M + B[ok].mean(0)
            r = np.linalg.norm(P - B, axis=1)
            ok2 = r <= max(2.5 * np.median(r), 0.003)
            if ok2.sum() < 10 or (ok2 == ok).all():
                break
            ok = ok2
        if ok.mean() < POSE_MIN:
            return None
        out[side] = P
    return out


def _write_positions(path, Z, N):
    """Replaces positions and normals of the first mesh of a GLB (same number of vertices)."""
    import struct
    doc, b = GLB.read(path)
    at = doc["meshes"][0]["primitives"][0]["attributes"]
    accessory = doc["accessors"][at["POSITION"]]
    assert accessory["count"] == len(Z)
    accessory["min"], accessory["max"] = [float(x) for x in Z.min(0)], [float(x) for x in Z.max(0)]
    b = bytearray(b)
    o = doc["bufferViews"][accessory["bufferView"]].get("byteOffset", 0)
    b[o:o + len(Z) * 12] = Z.astype("<f4").tobytes()
    o = doc["bufferViews"][doc["accessors"][at["NORMAL"]]["bufferView"]].get("byteOffset", 0)
    b[o:o + len(N) * 12] = N.astype("<f4").tobytes()
    j = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    j += b" " * (-len(j) % 4)
    b += b"\0" * (-len(b) % 4)
    with open(path + ".tmp", "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 28 + len(j) + len(b)) + struct.pack("<I4s", len(j), b"JSON") + j
                 + struct.pack("<I4s", len(b), b"BIN\0") + bytes(b))
    os.replace(path + ".tmp", path)


def main():
    mode, ids = sys.argv[1], sys.argv[2:] or RV.ids_of_cache()
    tot = [0.0, 0.0, 0.0, 0.0]
    for i in ids:
        # a repair sometimes uncovers another (a faulty vertex hidden by its neighbour): start
        # again as long as the score drops. When measuring, a single pass: nothing is written.
        for passed in range(PASSES if mode == "apply" else 1):
            r = _one(i, mode, passed)
            if r is None:
                break
            if passed == 0:
                tot[0] += r[0]
                tot[2] += r[2]
                tot[1] += r[1]
                tot[3] += r[3]
            else:
                tot[1] += r[1] - r[0]
                tot[3] += r[3] - r[2]
            if not r[1] < r[0]:
                break
    print("total: score %.0f -> %.0f; skin in front %.0f -> %.0f cm²" % tuple(tot))


def _one(i, mode, passed):
    """One pass on a part: (score before, after, skin in front before, after), None if there is nothing to do."""
    p = CB.part_name(i)
    S = RV.origin(p.meta)
    if S is None or len(S) != len(p.pos):
        return None
    # the hidden skin goes with the shape: measured with the transfer's (wide: everything within
    # 5 cm under the fabric), a repair that pushes the fabric under the skin goes unseen, and the skin
    # shows through when skin.py renders it (Zhao's top: 394 cm² on the breasts)
    garment = p.meta.get("class") == "garment"
    hidden_of = lambda Zx: PE.de(CB.Group(Zx, p.nor, p.F, p.meta, p.id, p.flags)) if garment else None

    def note(Zx, hidden):
        """(score, cm² of skin in front): the review score, shape and skin showing through together. Judged
        on shape alone, a repair that tucked 18 cm² of breast under a sweatshirt at equal shape score
        was set aside, and a part with no shape defect was never repaired."""
        # with the normals the written part will have: the skin showing through is judged against them
        nx = p.nor if Zx is p.pos else normals(S, p.pos, p.nor, Zx, p.F)
        c = CB.resolve([CB.Group(Zx, nx, p.F, p.meta, p.id, p.flags)])[0]
        dp = RV.skin_outside(c, None if hidden is None else {p.id: hidden})
        return RV.note(RV.shapes(S, Zx, p.F)["defaults"] + dp), round(sum(d["area_cm2"] for d in dp), 1)

    before, pa = note(p.pos, hidden_of(p.pos))
    if not before:
        return None
    T = p.pos[p.F]
    tol = max(SKIN_TOLERATED, SKIN_PART * 1e4 * float(np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1).sum()) / 2)
    # whole shape: at each hold, with and without lifting the faces, halfway and at the end of the
    # rounds. No setting works everywhere (lifting the faces saves Koleda's top, eight rounds shatter
    # Zhao's): the best score that the skin guard accepts is kept
    whole = []
    for attempt in [{"outfit": t, "faces": f} for t in OUTFITS for f in (False, True)]:
        for Zr, assessment in resume(S, p.pos, p.F, steps=True, **attempt):
            hidden = hidden_of(Zr)
            after, pb = note(Zr, hidden)
            if not after < before:
                continue
            if pb <= pa + max(tol, SKIN_GAIN * (before - after)):
                assessment["resumption"] = "shape, hold %g%s" % (attempt["outfit"], ", faces" if attempt["faces"] else "")
                whole.append((after, len(whole), Zr, assessment, pb, hidden))
    hidden = None
    for attempt in (["block"] if p.meta.get("type") in PE.FOOTWEAR else []) + ["whole", {"shape": False}, None]:
        if attempt == "whole":
            if not whole:
                continue
            after, _, Zr, assessment, pb, hidden = min(whole, key=lambda e: e[:2])
            break
        if attempt is None:      # no repair beats the part as it is
            Zr, assessment, after, pb, hidden = p.pos, {"resumption": "none"}, before, pa, None
            break
        if attempt == "block":
            Zr = place(S, p.pos)
            if Zr is None:
                continue
            hidden = PE.de(CB.Group(Zr, p.nor, p.F, p.meta, p.id, p.flags))   # the hidden skin goes with the placement
            after, pb = note(Zr, hidden)
            if after < before and pb <= pa + max(tol, POSE_SKIN):
                assessment = {"resumption": "shoe placed rigidly", "shift_mm": round(float(np.median(np.linalg.norm(Zr - p.pos, axis=1))) * 1000, 1)}
                break
            hidden = None
            continue
        Zr, assessment = resume(S, p.pos, p.F, **attempt)
        hidden = hidden_of(Zr)
        after, pb = note(Zr, hidden)
        if not after < before:
            hidden = None
            continue
        if pb <= pa + max(tol, SKIN_GAIN * (before - after)):
            assessment["resumption"] = "faulty vertices only"
            break
        hidden = None
    if passed == 0 or after < before:
        print("%-40s score %7.1f -> %7.1f   skin in front %5.1f -> %5.1f cm²   %s" % (i, before, after, pa, pb, json.dumps(assessment)), flush=True)
    if mode == "apply" and after < before:
        d = os.path.join(C.CACHE, "parts", i)
        if not os.path.exists(os.path.join(d, "part.before-repair.glb")):
            shutil.copy2(os.path.join(d, "part.glb"), os.path.join(d, "part.before-repair.glb"))
        _write_positions(os.path.join(d, "part.glb"), Zr, normals(S, p.pos, p.nor, Zr, p.F))
        if hidden is not None:
            C.write_bytes(os.path.join(d, "hidden_skin.bin"), hidden.tobytes())
            m = C.read_json(os.path.join(d, "meta.json"))
            m["hidden_skin"] = int(len(hidden))
            C.write_json(os.path.join(d, "meta.json"), m)
    return before, after, pa, pb


if __name__ == "__main__":
    main()
