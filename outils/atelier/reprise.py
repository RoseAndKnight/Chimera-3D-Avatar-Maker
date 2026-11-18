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

Independent check: the skin must not show through in front of the part (revue.peau_dehors). Each
part takes the boldest repair that respects it: whole shape loosely held, then firmly held, then
the faulty vertices alone, otherwise it stays as it is.

    python3 reprise.py mesurer [id ...]        review score before and after, without writing anything
    python3 reprise.py appliquer [id ...]      rewrites piece.glb in the cache (the old one kept as piece.avant-reprise.glb)
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun as C  # noqa: E402
import glb as GLB  # noqa: E402
import combinaison as CB  # noqa: E402
import revue as RV  # noqa: E402
import peau as PE  # noqa: E402

LONG = 2.0        # an edge at least LONG times longer than expected…
COURT = 0.4       # … or shrunk to less than COURT times its length…
ECART = 0.008     # … and off by at least this much, is wrong
PART = 0.5        # share of wrong edges from which a vertex is faulty
COUTURE = 0.002   # vertices that coincided in the original, separated by more than this: sewn back
DECOUSU = 0.01    # separated by more than this: their place is no longer known, they are faulty
AUTOUR = 6        # rings of neighbours of a faulty zone whose transferred place is barely trusted
ANNEAUX = 2       # sound neighbours taken around a faulty zone to recover scale and orientation
PRES = 0.015      # height above the skin up to which the part follows the transfer
LOIN = 0.05       # height from which it only follows its original shape
TENUES = (3.0, 30.0)   # from loosest to firmest: the part takes the first one that does not make the skin show through
PEAU_TOLEREE = 2.0     # cm² of extra skin in front of the part that a repair is allowed to cost…
PEAU_GAIN = 0.05       # … or this per score point gained: a shattered part put back in shape is worth a little skin
PEAU_PART = 0.002      # … or this share of the part's area, if larger
TENUE_LOIN = 0.02 # what is left of it far from the skin: the part does not drift
RONDES = 8        # shape repairs, each followed by lifting back above the skin
ETAPES = (4, 8)   # rounds whose state is offered: rounds do not always converge, the last is not always the best
FLECHE = 0.015    # a face between outside vertices only dips under the skin by its sag; deeper, it is not the cause
SUR_PEAU = 0.002  # minimum height above the skin
TOURS = 8         # rotation / position alternations
PASSES = 4        # successive repairs of the same part, as long as its score drops
POSE_MIN = 0.6    # share of a foot's vertices that must agree with the rigid placement of a shoe
POSE_PEAU = 20.0  # cm² of skin in front that a rigid placement is allowed to cost (the edge of the shaft on the ankle)
ZONE_MAX = 0.5    # beyond this share of the part, it is no longer a repair: the part must be transferred again


def _similitude(A, B):
    """Matrix M (3x3) such that (A - centre A) @ M ≈ B - centre B: rotation and scale."""
    a, b = A - A.mean(0), B - B.mean(0)
    U, s, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    D = np.diag([1.0, 1.0, d])
    ech = (s * np.diag(D)).sum() / max((a * a).sum(), 1e-12)
    return ech * (U @ D @ Vt)


def reprendre(S, Z, F, forme=True, tenue=None, faces=False, etapes=False):
    """Returns (corrected Z, summary). S: original shape, Z: transferred part, same vertices; F: faces."""
    w = RV._souder(S)
    n = int(w.max()) + 1
    nb = np.bincount(w, minlength=n)
    Sw, Zw = np.zeros((n, 3)), np.zeros((n, 3))
    np.add.at(Sw, w, S)
    np.add.at(Zw, w, Z)
    Sw /= nb[:, None]
    Zw /= nb[:, None]
    ecart = np.zeros(n)
    np.maximum.at(ecart, w, np.linalg.norm(Z - Zw[w], axis=1))
    Fw = w[F]
    E = np.unique(np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1), axis=0)
    E = E[E[:, 0] != E[:, 1]]
    bilan = {"sommets": n, "fautifs": 0, "recousus": 0, "zones": 0, "laissees": 0}
    if not len(E):
        return Z.copy(), bilan
    ls = np.linalg.norm(Sw[E[:, 0]] - Sw[E[:, 1]], axis=1)
    r0 = float(np.median(np.linalg.norm(Zw[E[:, 0]] - Zw[E[:, 1]], axis=1) / np.maximum(ls, 1e-9)))

    def fausses(X, vivant):
        lz = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
        f = (((lz > LONG * r0 * ls) & (lz - r0 * ls > ECART)) | ((lz < COURT * r0 * ls) & (r0 * ls - lz > ECART)))
        return f & vivant[E[:, 0]] & vivant[E[:, 1]]

    libre = ecart > DECOUSU
    for _ in range(12):
        vivant = ~libre
        f = fausses(Zw, vivant)
        if not f.any():
            break
        deg, mal = np.zeros(n), np.zeros(n)
        ev = vivant[E[:, 0]] & vivant[E[:, 1]]
        np.add.at(deg, E[ev].ravel(), 1)
        np.add.at(mal, E[f].ravel(), 1)
        part = mal / np.maximum(deg, 1)
        pire = np.maximum(part[E[f, 0]], part[E[f, 1]])
        nouveau = np.zeros(n, bool)
        for k in (0, 1):      # of each wrong edge, the most faulty end
            v = E[f, k]
            nouveau[v[(part[v] >= pire - 1e-9) & (part[v] >= PART)]] = True
        if not nouveau.any():     # wrong edges with no clearly faulty vertex: the worst end of each
            a, b = E[f, 0], E[f, 1]
            nouveau[np.where(part[a] >= part[b], a, b)] = True
        libre |= nouveau
    X = Zw.copy()
    if libre.any() and libre.mean() <= ZONE_MAX:
        # zones: components of faulty vertices
        etiq = np.where(libre, np.arange(n), -1)
        el = E[libre[E[:, 0]] & libre[E[:, 1]]]
        for _ in range(200):
            m = np.maximum(etiq[el[:, 0]], etiq[el[:, 1]])
            avant = etiq.copy()
            np.maximum.at(etiq, el[:, 0], m)
            np.maximum.at(etiq, el[:, 1], m)
            if (etiq == avant).all():
                break
        vois = [[] for _ in range(n)]
        for a, b in E:
            vois[a].append(b)
            vois[b].append(a)
        for z in np.unique(etiq[libre]):
            dans = np.where(etiq == z)[0]
            bord, front = set(), set(dans.tolist())
            for _ in range(ANNEAUX):
                suite = {v for u in front for v in vois[u] if not libre[v] and v not in bord}
                bord |= suite
                front = suite
            bord = np.array(sorted(bord), np.int64)
            if len(bord) < 3:
                bilan["laissees"] += 1
                continue
            M = _similitude(Sw[bord], Zw[bord])
            en = np.zeros(n, bool)
            en[dans] = True
            ez = E[en[E[:, 0]] | en[E[:, 1]]]
            ez = ez[(en[ez] | ~libre[ez]).all(1)]      # towards the zone or towards sound vertices, not towards another zone
            cible = (Sw[ez[:, 0]] - Sw[ez[:, 1]]) @ M  # expected X[a] - X[b]
            X[dans] = Zw[bord].mean(0) + (Sw[dans] - Sw[bord].mean(0)) @ M
            for _ in range(300):
                som, cpt = np.zeros((n, 3)), np.zeros(n)
                np.add.at(som, ez[:, 0], X[ez[:, 1]] + cible)
                np.add.at(som, ez[:, 1], X[ez[:, 0]] - cible)
                np.add.at(cpt, ez.ravel(), 1)
                neuf = som[dans] / np.maximum(cpt[dans], 1)[:, None]
                fini = np.abs(neuf - X[dans]).max() < 1e-5
                X[dans] = neuf
                if fini:
                    break
            bilan["zones"] += 1
            bilan["fautifs"] += int(len(dans))
    elif libre.any():
        bilan["laissees"] = -1      # too much of the part is wrong
        libre[:] = False
    bilan["recousus"] = int((~libre & (ecart > COUTURE)).sum())
    if not forme:
        out = Z.copy()
        cousu = libre | (ecart > COUTURE)
        out[cousu[w]] = X[w][cousu[w]]
        return out, bilan
    # the whole part takes back its shape, held to the transfer according to its height above the skin
    TENUE = tenue or TENUES[0]
    h = CB.lignes(Zw, np.zeros((n, 3)))[1]
    lam = np.where(np.isfinite(h), TENUE_LOIN + (TENUE - TENUE_LOIN) * np.clip((LOIN - h) / (LOIN - PRES), 0, 1), TENUE_LOIN)
    lam[libre] = 0.0
    # around a faulty zone, the vertices were dragged along by it: their transferred place is no
    # better, barely less wrong. They are held less and less firmly as they get closer to it.
    pres = libre.astype(float)
    for k in range(1, AUTOUR + 1):
        vu = pres > 0
        suite = np.zeros(n, bool)
        suite[E[vu[E[:, 0]], 1]] = True
        suite[E[vu[E[:, 1]], 0]] = True
        pres[suite & ~vu] = 1.0 - k / (AUTOUR + 1.0)
    lam = lam * (1.0 - pres) ** 2 + TENUE_LOIN * (1.0 - (1.0 - pres) ** 2) * (pres > 0) * (~libre)
    a, b = E[:, 0], E[:, 1]
    dS = (Sw[a] - Sw[b]) * r0
    deg = np.bincount(E.ravel(), minlength=n).astype(float)

    def somme(idx, V):
        return np.stack([np.bincount(idx, V[:, k], minlength=n) for k in range(3)], 1)

    def A(Y):      # (Laplacian + hold) · Y
        d = Y[a] - Y[b]
        return somme(a, d) - somme(b, d) + lam[:, None] * Y

    cibleZ = Zw.copy()
    Fu = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 2] != Fw[:, 0])]
    def rendu(X):
        return X[w], dict(bilan, deplacement_mm=round(float(np.percentile(np.linalg.norm(X - Zw, axis=1), 95) * 1000), 1))

    faits = []      # with etapes: the state at each round of ETAPES, and the last one
    for ronde in range(RONDES):
      if ronde in ETAPES:
        faits.append(rendu(X))
      if ronde:     # the repaired shape does not go under the skin: what went down there comes back up, and is held there
        nl, h2 = CB.lignes(X, np.zeros((n, 3)))
        manque = np.where(np.isfinite(h2), SUR_PEAU - h2, 0.0)
        # between outside vertices, a large face dips under a curved skin (a breast under a top
        # with large faces): the midpoint of an edge or the centre of a face that is under lifts its vertices
        for G in ((E, Fu) if faces else ()):
            hm = CB.lignes(X[G].mean(1), np.zeros((len(G), 3)))[1]
            mm = np.where(np.isfinite(hm) & (hm > -FLECHE), SUR_PEAU - hm, 0.0)
            for k in range(G.shape[1]):
                np.maximum.at(manque, G[:, k], mm)
        bas = (manque > 0) & np.isfinite(h2)
        if not bas.any():
            break
        cibleZ[bas] = X[bas] + manque[bas, None] * nl[bas]
        lam[bas] = np.maximum(lam[bas], TENUE)
      for _ in range(TOURS):
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
          cible = 0.5 * (np.einsum("ei,eij->ej", dS, Rot[a]) + np.einsum("ei,eij->ej", dS, Rot[b]))
          sec = somme(a, cible) - somme(b, cible) + lam[:, None] * cibleZ
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
    faits.append(rendu(X))
    return faits if etapes else faits[-1]


def normales(S, Z0, N0, Z, F):
    """Normals of the repaired part: the previous ones where nothing moved (2 mm), otherwise recomputed
    (smooth, over vertices that coincided in the original)."""
    w = RV._souder(S)
    n = int(w.max()) + 1
    nf = np.cross(Z[F[:, 1]] - Z[F[:, 0]], Z[F[:, 2]] - Z[F[:, 0]])
    acc = np.zeros((n, 3))
    for k in range(3):
        np.add.at(acc, w[F[:, k]], nf)
    N = acc[w]
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
    garde = (np.linalg.norm(Z - Z0, axis=1) < 0.002) | (np.linalg.norm(N, axis=1) < 0.5)
    N[garde] = N0[garde]
    return N


def peau_cm2(p, Z, cachee=None):
    """Area of skin that shows in front of the part alone on the body (cm²): the repair must not open it up."""
    c = CB.resoudre([CB.Groupe(Z, p.nor, p.F, p.meta, p.id, p.drapeaux)])[0]
    return round(sum(d["aire_cm2"] for d in RV.peau_dehors(c, None if cachee is None else {p.id: cachee})), 1)


def poser(S, Z):
    """A shoe is a rigid object: each foot takes back its original shape, at the place, scale and
    orientation the transfer gave it (similarity fitted on the vertices that agree, the others set
    aside). Returns None if a foot does not agree to POSE_MIN."""
    out = Z.copy()
    for cote in (S[:, 0] < 0, S[:, 0] >= 0):
        if cote.sum() < 10:
            continue
        A, B = S[cote], Z[cote]
        ok = np.ones(len(A), bool)
        for _ in range(8):
            M = _similitude(A[ok], B[ok])
            P = (A - A[ok].mean(0)) @ M + B[ok].mean(0)
            r = np.linalg.norm(P - B, axis=1)
            ok2 = r <= max(2.5 * np.median(r), 0.003)
            if ok2.sum() < 10 or (ok2 == ok).all():
                break
            ok = ok2
        if ok.mean() < POSE_MIN:
            return None
        out[cote] = P
    return out


def _ecrire_positions(chemin, Z, N):
    """Replaces positions and normals of the first mesh of a GLB (same number of vertices)."""
    import struct
    doc, b = GLB.lire(chemin)
    at = doc["meshes"][0]["primitives"][0]["attributes"]
    acc = doc["accessors"][at["POSITION"]]
    assert acc["count"] == len(Z)
    acc["min"], acc["max"] = [float(x) for x in Z.min(0)], [float(x) for x in Z.max(0)]
    b = bytearray(b)
    o = doc["bufferViews"][acc["bufferView"]].get("byteOffset", 0)
    b[o:o + len(Z) * 12] = Z.astype("<f4").tobytes()
    o = doc["bufferViews"][doc["accessors"][at["NORMAL"]]["bufferView"]].get("byteOffset", 0)
    b[o:o + len(N) * 12] = N.astype("<f4").tobytes()
    j = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    j += b" " * (-len(j) % 4)
    b += b"\0" * (-len(b) % 4)
    with open(chemin + ".tmp", "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 28 + len(j) + len(b)) + struct.pack("<I4s", len(j), b"JSON") + j
                 + struct.pack("<I4s", len(b), b"BIN\0") + bytes(b))
    os.replace(chemin + ".tmp", chemin)


def main():
    mode, ids = sys.argv[1], sys.argv[2:] or RV.ids_du_cache()
    tot = [0.0, 0.0, 0.0, 0.0]
    for i in ids:
        # a repair sometimes uncovers another (a faulty vertex hidden by its neighbour): start
        # again as long as the score drops. When measuring, a single pass: nothing is written.
        for passe in range(PASSES if mode == "appliquer" else 1):
            r = _une(i, mode, passe)
            if r is None:
                break
            if passe == 0:
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


def _une(i, mode, passe):
    """One pass on a part: (score before, after, skin in front before, after), None if there is nothing to do."""
    p = CB.piece(i)
    S = RV.origine(p.meta)
    if S is None or len(S) != len(p.pos):
        return None
    # the hidden skin goes with the shape: measured with the transfer's (wide: everything within
    # 5 cm under the fabric), a repair that pushes the fabric under the skin goes unseen, and the skin
    # shows through when peau.py renders it (Zhao's top: 394 cm² on the breasts)
    vetement = p.meta.get("classe") == "vetement"
    cachee_de = lambda Zx: PE.de(CB.Groupe(Zx, p.nor, p.F, p.meta, p.id, p.drapeaux)) if vetement else None

    def noter(Zx, cachee):
        """(score, cm² of skin in front): the review score, shape and skin showing through together. Judged
        on shape alone, a repair that tucked 18 cm² of breast under a sweatshirt at equal shape score
        was set aside, and a part with no shape defect was never repaired."""
        # with the normals the written part will have: the skin showing through is judged against them
        nx = p.nor if Zx is p.pos else normales(S, p.pos, p.nor, Zx, p.F)
        c = CB.resoudre([CB.Groupe(Zx, nx, p.F, p.meta, p.id, p.drapeaux)])[0]
        dp = RV.peau_dehors(c, None if cachee is None else {p.id: cachee})
        return RV.note(RV.formes(S, Zx, p.F)["defauts"] + dp), round(sum(d["aire_cm2"] for d in dp), 1)

    avant, pa = noter(p.pos, cachee_de(p.pos))
    if not avant:
        return None
    T = p.pos[p.F]
    tol = max(PEAU_TOLEREE, PEAU_PART * 1e4 * float(np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1).sum()) / 2)
    # whole shape: at each hold, with and without lifting the faces, halfway and at the end of the
    # rounds. No setting works everywhere (lifting the faces saves Koleda's top, eight rounds shatter
    # Zhao's): the best score that the skin guard accepts is kept
    entieres = []
    for essai in [{"tenue": t, "faces": f} for t in TENUES for f in (False, True)]:
        for Zr, bilan in reprendre(S, p.pos, p.F, etapes=True, **essai):
            cachee = cachee_de(Zr)
            apres, pb = noter(Zr, cachee)
            if not apres < avant:
                continue
            if pb <= pa + max(tol, PEAU_GAIN * (avant - apres)):
                bilan["reprise"] = "shape, hold %g%s" % (essai["tenue"], ", faces" if essai["faces"] else "")
                entieres.append((apres, len(entieres), Zr, bilan, pb, cachee))
    cachee = None
    for essai in (["bloc"] if p.meta.get("type") in PE.CHAUSSANTS else []) + ["entiere", {"forme": False}, None]:
        if essai == "entiere":
            if not entieres:
                continue
            apres, _, Zr, bilan, pb, cachee = min(entieres, key=lambda e: e[:2])
            break
        if essai is None:      # no repair beats the part as it is
            Zr, bilan, apres, pb, cachee = p.pos, {"reprise": "none"}, avant, pa, None
            break
        if essai == "bloc":
            Zr = poser(S, p.pos)
            if Zr is None:
                continue
            cachee = PE.de(CB.Groupe(Zr, p.nor, p.F, p.meta, p.id, p.drapeaux))   # the hidden skin goes with the placement
            apres, pb = noter(Zr, cachee)
            if apres < avant and pb <= pa + max(tol, POSE_PEAU):
                bilan = {"reprise": "shoe placed rigidly", "deplacement_mm": round(float(np.median(np.linalg.norm(Zr - p.pos, axis=1))) * 1000, 1)}
                break
            cachee = None
            continue
        Zr, bilan = reprendre(S, p.pos, p.F, **essai)
        cachee = cachee_de(Zr)
        apres, pb = noter(Zr, cachee)
        if not apres < avant:
            cachee = None
            continue
        if pb <= pa + max(tol, PEAU_GAIN * (avant - apres)):
            bilan["reprise"] = "faulty vertices only"
            break
        cachee = None
    if passe == 0 or apres < avant:
        print("%-40s score %7.1f -> %7.1f   skin in front %5.1f -> %5.1f cm²   %s" % (i, avant, apres, pa, pb, json.dumps(bilan)), flush=True)
    if mode == "appliquer" and apres < avant:
        d = os.path.join(C.CACHE, "pieces", i)
        if not os.path.exists(os.path.join(d, "piece.avant-reprise.glb")):
            shutil.copy2(os.path.join(d, "piece.glb"), os.path.join(d, "piece.avant-reprise.glb"))
        _ecrire_positions(os.path.join(d, "piece.glb"), Zr, normales(S, p.pos, p.nor, Zr, p.F))
        if cachee is not None:
            C.ecrire_octets(os.path.join(d, "peau_cachee.bin"), cachee.tobytes())
            m = C.lire_json(os.path.join(d, "meta.json"))
            m["peau_cachee"] = int(len(cachee))
            C.ecrire_json(os.path.join(d, "meta.json"), m)
    return avant, apres, pa, pb


if __name__ == "__main__":
    main()
