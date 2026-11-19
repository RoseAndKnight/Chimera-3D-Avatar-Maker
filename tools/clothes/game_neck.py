"""Body skin cut as on a game model: a neck under the face, no scalp.

Why: on a ZZZ model, the body skin stops at the top of the neck, hidden under the face and the hair
cap. The outfit's body comes from the skin-completion research tool (not published), whose skin is closed by a scalp sphere; it
poked out around the grafted face and through the hair. That sphere goes entirely (skin piece lying
wholly above the neck), and so does the body skin above the top of the donor head's neck, never lower
than `UNDER_FACE_MM` under the face.

A single join, at the face edge, with no seam on the neck: the neck stays the body's, in one piece,
with its weights. The head brings only its ears (`island_of_neck`), which follow the head alone, and its
scalp skin (`skull_of_game`), which closes the head under the hair; the bottom of that scalp slides
under the body's neck (`_place_nape`), with the neck's weights where it touches it. A seam between
the head's neck and the body's would open when the head turns.

The body's neck is never touched. It is the skin vertices that the head (bone 頭 and its children)
carries at less than `NECK_WEIGHT`: neither they nor any face touching one of them move, go away or
lose their outline. The cut only takes from the body's head. a workshop check proves it on
what the page draws, vertex by vertex, face by face and pixel by pixel.
"""
import numpy as np

MM = 1.0 / 80.0
SECTORS = 36
ZONE_MM = 80.0      # only skin above the bottom of the face minus this margin is considered
RADIUS_MM = 140.0    # and within this distance of the neck axis (not the shoulders)
WITHOUT_NECK_MM = 30.0  # donor head without a neck: cut at this height above the bottom of the face
WITHOUT_TRAIT_MM = 10.0  # the body's head within this distance of the face draws no outline
UNDER_FACE_MM = 12.0  # the body rises at least to this height below the bottom of the face, direction by direction
HEAD_ALONE = 0.98   # minimum weight share on the head bone (and its children) for a taken ear vertex
NECK_WEIGHT = 0.5     # a vertex the head carries at less than this belongs to the neck: left untouched
SEEDS_MM = 2.0      # the original neck is measured over its surface, at least one point every this much
FLUSH_MM = 3.0   # body head sticking out of the face by less than this is duplicate skin
TUCKED_MM = 0.5     # it goes back under the face by this much
NAPE_MM = 15.0     # height of the head's lower scalp strip laid on the body's neck
SMOOTH_MM = 6.0      # the strip follows the edge vertices within this much
NEAR_MM = 20.0    # a lower scalp edge is laid only if the body skin passes within this much
COVERS_MM = 8.0   # above the laid edge, the body keeps this much skin under the scalp
IN_FACE_MM = 20.0  # scalp skin up to this much under the face is a hidden duplicate


def _sector(p, c):
    a = np.arctan2(p[:, 2] - c[1], p[:, 0] - c[0])
    return np.floor((a + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS


def surface(pos, faces):
    """Points on the surface of the triangles `faces`, at most `SEMIS_MM` apart, vertices included.

    A donor head's neck has only a few vertex rings: measured at its vertices alone, most cells
    (5 mm slice × 10° sector) stayed empty or saw a single vertex, and the target radius wobbled
    from one body ring to the next (wrinkle under Nicole's jaw)."""
    pos = np.asarray(pos, float)
    faces = np.asarray(faces, int).reshape(-1, 3)
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    side = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], axis=0)
    n = np.maximum(1, np.ceil(side / (SEEDS_MM * MM)).astype(int))
    points = [pos[np.unique(faces)]]
    for k in np.unique(n):
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1))
        u, v = (i[i + j <= k] / k)[None, :, None], (j[i + j <= k] / k)[None, :, None]
        m = n == k
        points.append((a[m, None] * (1 - u - v) + b[m, None] * u + c[m, None] * v).reshape(-1, 3))
    return np.concatenate(points)


def _components(pos, F):
    """Connected-component label of each face (vertices welded by position)."""
    if not len(F):
        return np.zeros(0, int)
    _, rep = np.unique(np.round(pos, 5), axis=0, return_inverse=True)
    rep = rep.ravel()
    per = np.arange(rep.max() + 1)

    def root(a):
        while per[a] != a:
            per[a] = per[per[a]]
            a = per[a]
        return a
    for a, b, c in rep[F].tolist():
        ra = root(a)
        for x in (b, c):
            rx = root(x)
            if rx != ra:
                per[rx] = ra
    return np.array([root(a) for a in rep[F[:, 0]].tolist()])


def _islands(F):
    """Label of each face: piece connected through vertex indices (a UV island)."""
    if not len(F):
        return np.zeros(0, int)
    per = np.arange(F.max() + 1)

    def root(a):
        while per[a] != a:
            per[a] = per[per[a]]
            a = per[a]
        return a
    for a, b, c in F.tolist():
        ra = root(a)
        for x in (b, c):
            rx = root(x)
            if rx != ra:
                per[rx] = ra
    return np.array([root(a) for a in F[:, 0].tolist()])


def os_of_head(os_pmx, i_head):
    """Indices of the head bone and all its descendants."""
    taken = {i_head}
    for _ in range(len(os_pmx)):
        n = len(taken)
        taken |= {i for i, o in enumerate(os_pmx) if o["parent"] in taken}
        if len(taken) == n:
            break
    return np.array(sorted(taken))




def island_of_neck(pos, faces, mat_face, i_skin, i_face, axis, os_s=None, weight=None, head=None, uv=None, img=None):
    """Donor head skin faces to graft along with the face (the ears): in the first candidate
    material surrounding the head, the pieces welded to the face and the isolated ears, without the
    black cap faces, and only above the bottom of the face, where all their vertices follow the head
    alone (`HEAD_ALONE`): nothing there is stitched to the body's neck.

    Only UV islands taken whole remain. Survey of 09/11 on the 16 cached heads: the ears are always
    whole islands; the rest taken was, on each side, a corner of scalp skin behind the jaw, cut into
    a sawtooth by the weight and height filters (22 faces of 15 to 25 mm on Maelys). Its points hung
    below the ear, through the hair."""
    pos = np.asarray(pos, float)
    taken = np.zeros(len(faces), bool)
    fv = faces[np.isin(mat_face, i_face)]
    if not len(fv):
        return taken
    vv = np.unique(fv)
    y_bottom = pos[vv, 1].min()
    y0 = y_bottom - ZONE_MM * MM
    c = np.asarray(axis, float)
    in_ = (pos[:, 1] > y0) & (np.hypot(pos[:, 0] - c[0], pos[:, 2] - c[1]) < RADIUS_MM * MM)
    key_v = {tuple(x) for x in np.round(pos[vv] / 1e-4).astype(np.int64)}
    touched_v = np.array([tuple(x) in key_v for x in np.round(pos / 1e-4).astype(np.int64)])
    part = (np.asarray(weight, float) * np.isin(os_s, head)).sum(1) if os_s is not None else np.ones(len(pos))
    candidates = [int(i_skin)] if isinstance(i_skin, (int, np.integer)) else [int(x) for x in i_skin]
    for im in candidates:
        fp = np.where(mat_face == im)[0]
        if not len(fp):
            continue
        F = faces[fp]
        comp = _components(pos, F)
        out_of = np.unique(comp[~in_[F].all(1)])
        guard = ~np.isin(comp, out_of)
        if not guard.any():
            # one-piece body (e.g. Pulchra): cut above the choker
            top = in_[F].all(1) & (pos[F, 1] > y_bottom - 30.0 * MM).all(1)
            if not top.any():
                continue
            comp_h = _components(pos, F[top])
            tc = np.unique(comp_h[touched_v[F[top]].any(1)])
            guard = np.zeros(len(fp), bool)
            guard[top] = np.isin(comp_h, tc)
        else:
            ear = ((np.abs(pos[F, 0] - c[0]) > 35.0 * MM).any(1)
                       & (pos[F, 1] > y_bottom).all(1) & (part[F] >= HEAD_ALONE).all(1))
            good = np.unique(comp[guard & (touched_v[F].any(1) | ear)])
            guard &= np.isin(comp, good)
            # black inner cap under the hair at the back of the skull (e.g. Miyabi)
            cap = ((part[F] >= HEAD_ALONE).all(1) & (pos[F, 2] > c[1] + 22.0 * MM).all(1)
                       & (np.abs(pos[F, 0] - c[0]) < 42.0 * MM).all(1) & ~touched_v[F].any(1))
            guard &= ~cap
        guard &= (part[F] >= HEAD_ALONE).all(1) & (pos[F, 1] > y_bottom).all(1)
        if img is not None and uv is not None and guard.any():
            h, w = img.shape[:2]
            u = np.mod(np.asarray(uv, float)[F].mean(1), 1.0)
            px = img[(u[:, 1] * h).astype(int).clip(0, h - 1), (u[:, 0] * w).astype(int).clip(0, w - 1), :3]
            black = (px.max(1) < 0.28) & ((px.max(1) - px.min(1)) < 0.08)
            guard &= ~black
        if guard.any():
            island = _islands(F)
            cuts = np.unique(island[~guard])
            taken[fp[guard & ~np.isin(island, cuts)]] = True
            break
    return taken


def _under_chin(pos, Fv, q):
    """Points of `q` that a triangle of `Fv` covers vertically, lying entirely above them."""
    a, b, c = (pos[Fv[:, k]][:, [0, 2]] for k in range(3))
    y_min = pos[Fv, 1].min(1)
    v0, v1 = c - a, b - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den[np.abs(den) < 1e-18] = 1e-18
    out = np.zeros(len(q), bool)
    for k, p in enumerate(q):
        v2 = p[[0, 2]] - a
        d02, d12 = (v0 * v2).sum(1), (v1 * v2).sum(1)
        u, v = (d11 * d02 - d01 * d12) / den, (d00 * d12 - d01 * d02) / den
        out[k] = ((u >= 0) & (v >= 0) & (u + v <= 1) & (y_min > p[1])).any()
    return out


def skull_of_game(pos, nor, faces, mat_face, i_skin, i_face, axis, os_s, weight, head):
    """Scalp skin faces of the donor head (back, top, sides behind the jaw): in the first candidate
    material that has any, those whose vertices all follow the head at `NECK_WEIGHT` or more, except
    under the chin and under the face (up to `IN_FACE_MM` inside: a hidden duplicate).

    The body has lost its scalp sphere (`cut`) and `island_of_neck` only takes the ears: the head
    stayed open at the nape, and under the hair the back side of the face showed (survey of 10/11 on
    Pulchra and Grace, wearing their own hair). A game model closes its head with this very skin, to
    which its hair is fitted; its bottom is then laid on the body's neck (`_place_nape`)."""
    pos, nor = np.asarray(pos, float), np.asarray(nor, float)
    taken = np.zeros(len(faces), bool)
    Fv = faces[np.isin(mat_face, i_face)]
    if not len(Fv):
        return taken
    y0 = pos[np.unique(Fv), 1].min() - ZONE_MM * MM
    c = np.asarray(axis, float)
    in_ = (pos[:, 1] > y0) & (np.hypot(pos[:, 0] - c[0], pos[:, 2] - c[1]) < RADIUS_MM * MM)
    part = (np.asarray(weight, float) * np.isin(os_s, head)).sum(1)
    candidates = [int(i_skin)] if isinstance(i_skin, (int, np.integer)) else [int(x) for x in i_skin]
    for im in candidates:
        fp = np.where(mat_face == im)[0]
        F = faces[fp]
        sel = (part[F] >= NECK_WEIGHT).all(1) & in_[F].all(1)
        if not sel.any():
            continue
        k = np.where(sel)[0]
        sel[k[_under_chin(pos, Fv, pos[F[k]].mean(1))]] = False
        k = np.where(sel)[0]
        d, _ = _under_face(pos, nor, Fv, pos[F[k]].mean(1))
        sel[k[(d < 0) & (d > -IN_FACE_MM * MM)]] = False
        taken[fp[sel]] = True
        break
    return taken


def _smooth_sectors(val, seen):
    k = np.arange(SECTORS)
    out = np.interp(k, np.concatenate([k[seen] - SECTORS, k[seen], k[seen] + SECTORS]), np.tile(val[seen], 3))
    kernel = np.hanning(7)[1:-1]
    return np.convolve(np.concatenate([out[-2:], out, out[:2]]), kernel / kernel.sum(), mode="valid")



def _skull_of_body(pos, faces, mat_face, i_skin, y0):
    """The scalp sphere of a skin-completed body: any body skin piece lying entirely above y0."""
    fp_idx = np.where(mat_face == i_skin)[0]
    is_skull = np.zeros(len(faces), bool)
    if len(fp_idx):
        comp_p = _components(pos, faces[fp_idx])
        keep_c = np.unique(comp_p[~(pos[faces[fp_idx], 1] > y0).all(1)])
        is_skull[fp_idx[~np.isin(comp_p, keep_c)]] = True
    return is_skull


def cut(pos, faces, mat_face, i_skin, i_face, neck_head, axis, frozen=None):
    """Body skin faces to remove (mask over `faces`) and summary.

    pos: vertices (n, 3); neck_head: neck vertices of the donor head, placed (m, 3);
    axe: (x, z) of the neck axis (bone 頭); frozen: vertices never cut or moved (the neck and its faces).
    """
    pos = np.asarray(pos, float)
    fv = faces[np.isin(mat_face, i_face)]
    y_bottom = pos[np.unique(fv), 1].min()
    y0 = y_bottom - ZONE_MM * MM
    c = np.asarray(axis, float)
    is_skull = _skull_of_body(pos, faces, mat_face, i_skin, y0)
    neck = np.asarray(neck_head, float).reshape(-1, 3)
    neck = neck[(neck[:, 1] > y0) & (np.hypot(neck[:, 0] - c[0], neck[:, 2] - c[1]) < RADIUS_MM * MM)]
    top = np.full(SECTORS, -np.inf)
    if len(neck):
        np.maximum.at(top, _sector(neck, c), neck[:, 1])
    seen = np.isfinite(top)
    if seen.sum() < SECTORS // 3:
        top[:] = y_bottom + WITHOUT_NECK_MM * MM
        source = "default"
    else:
        top = _smooth_sectors(top, seen)
        source = "head neck"
    # where the donor head's neck stopped under its chin, the cut edge showed as a V under the jaw:
    # the body always rises under the face
    qv = pos[np.unique(fv)]
    bottom_v = np.full(SECTORS, np.inf)
    np.minimum.at(bottom_v, _sector(qv, c), qv[:, 1] + UNDER_FACE_MM * MM)
    top = np.where(np.isfinite(bottom_v), np.maximum(top, bottom_v), top)
    fp = (mat_face == i_skin) & ~is_skull
    vp = np.unique(faces[fp])
    q = pos[vp]
    near = (q[:, 1] > y0) & (np.hypot(q[:, 0] - c[0], q[:, 2] - c[1]) < RADIUS_MM * MM)
    lim = np.full(len(pos), np.inf)
    lim[vp[near]] = _height(top, q[near], c)
    if frozen is not None:
        lim[frozen] = np.inf
    too = pos[:, 1] > lim
    # a straddling face stays, its upper vertices are brought down to the cut height: clean edge
    remove = is_skull | (fp & too[faces].all(1))
    border = np.unique(faces[fp & ~remove][too[faces[fp & ~remove]]])
    assessment = {"faces_removed": int(remove.sum()), "faces_skull": int(is_skull.sum()),
             "vertices_border": int(len(border)), "cutting": source,
             "top_of_neck_mm": [round(float((h - y_bottom) / MM), 1) for h in (top.min(), np.median(top), top.max())]}
    return remove, border, lim, assessment


def _on_cut(pos, Fp, border, lim):
    """Where each vertex of `border` goes down to: the surface point at the cut height, on the most vertical
    edge linking it to a neighbour below the cut.

    Moved straight down, a nape vertex (which slants) ended up to 3 mm under the surface, next to the
    ring below: steps in the side silhouette. Copies of a vertex (UV seams) go down together; with no
    edge crossing the cut, it goes straight down.

    Without a hairstyle, the side outline stays dashed at the nape: the shell has nothing behind an open
    edge. That edge is under the hair, as on a game model (survey of 10/11 on four short cuts,
    Maelys, Anby, Corin, Seth: hidden)."""
    pos = np.asarray(pos, float)
    _, rep = np.unique(np.round(pos, 5), axis=0, return_inverse=True)
    rep = rep.ravel()
    too = pos[:, 1] > lim
    E = np.concatenate([Fp[:, [0, 1]], Fp[:, [1, 2]], Fp[:, [2, 0]]])
    E = np.concatenate([E, E[:, ::-1]])
    E = E[too[E[:, 0]] & (pos[E[:, 1], 1] <= lim[E[:, 0]])]
    a, b, h = pos[E[:, 0]], pos[E[:, 1]], lim[E[:, 0]]
    drop = np.maximum(a[:, 1] - b[:, 1], 1e-12)
    pt = a + ((a[:, 1] - h) / drop)[:, None] * (b - a)
    slope = np.hypot(b[:, 0] - a[:, 0], b[:, 2] - a[:, 2]) / drop
    order = np.lexsort((slope, rep[E[:, 0]]))
    r_e = rep[E[order, 0]]
    first = np.r_[True, r_e[1:] != r_e[:-1]]
    chosen = dict(zip(r_e[first].tolist(), pt[order][first]))
    return [chosen.get(int(rep[v]), np.array([pos[v, 0], lim[v], pos[v, 2]])) for v in border.tolist()]


def _height(top, q, c):
    """Cut height interpolated between sectors, at the azimuth of each point."""
    a = (np.arctan2(q[:, 2] - c[1], q[:, 0] - c[0]) + np.pi) / (2 * np.pi) * SECTORS - 0.5
    k = np.arange(SECTORS)
    return np.interp(a % SECTORS, np.concatenate([k - SECTORS, k, k + SECTORS]), np.tile(top, 3))



def _under_face(pos, nor, Fv, q):
    """Signed distance (positive outside) from each point of `q` to the skin `Fv`, and the outward normal at the foot;
    nan when no triangle receives it vertically (at the face edge, the sign means nothing)."""
    a, b, c = pos[Fv[:, 0]], pos[Fv[:, 1]], pos[Fv[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    n *= np.sign((nor[Fv].sum(1) * n).sum(1) + 1e-12)[:, None]
    e0, e1 = b - a, c - a
    d00, d01, d11 = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
    dist = np.full(len(q), np.nan)
    nn = np.zeros((len(q), 3))
    for k, p in enumerate(q):
        h = ((p - a) * n).sum(1)
        w = p - h[:, None] * n - a
        d20, d21 = (w * e0).sum(1), (w * e1).sum(1)
        v, u = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        ok = (v >= 0) & (u >= 0) & (u + v <= 1)
        if ok.any():
            j = np.where(ok)[0][np.abs(h[ok]).argmin()]
            dist[k], nn[k] = h[j], n[j]
    return dist, nn


def _tuck(pos, nor, faces, fm, i_skin, i_vis, frozen):
    """The kept body head that lies flush with the face goes under it.

    The cut keeps the body up to `UNDER_FACE_MM` under the bottom of the face: otherwise, a gap under the face.
    In that strip, the body skin stayed at the back of the head 0.4-1.3 mm in front of the face, the two
    skins crossing (survey of 10/11 on the sixteen heads: 2 to 16 faces each). Returns (positions, summary);
    the frozen neck does not move. The faces moved under remain: on Nicole, 36 of the 53 are still visible
    from outside between the face edge and the skin (rays, hair on, 10/11)."""
    Fv = faces[np.isin(fm, i_vis)]
    vp = np.unique(faces[fm == i_skin])
    vp = vp[~frozen[vp]]
    pv = pos[np.unique(Fv)]
    lo, hi = pv.min(0) - FLUSH_MM * MM, pv.max(0) + FLUSH_MM * MM
    vp = vp[((pos[vp] > lo) & (pos[vp] < hi)).all(1)]
    d, n = _under_face(pos, nor, Fv, pos[vp])
    pos = pos.copy()
    moves = (d > -TUCKED_MM * MM) & (d < FLUSH_MM * MM)
    pos[vp[moves]] -= ((d[moves] + TUCKED_MM * MM)[:, None]) * n[moves]
    return pos, {"vertices_rentres": int(moves.sum())}


def _unify_normals(s, faces, fm, i_face):
    """A single normal on both sides of the face's internal seam (颜 / 颜_A)."""
    if len(i_face) < 2:
        return
    pos = np.asarray(s["pos"], float)
    nor = np.asarray(s["nor"], float)
    v0 = np.unique(faces[fm == i_face[0]])
    v1 = np.unique(faces[fm == i_face[1]])
    the_key = lambda v: [tuple(x) for x in np.round(pos[v] / 1e-4).astype(np.int64)]
    m1 = {k: j for k, j in zip(the_key(v1), v1)}
    for a, k in zip(v0.tolist(), the_key(v0)):
        if k in m1:
            b = m1[k]
            nm = nor[a] + nor[b]
            nm /= max(float(np.linalg.norm(nm)), 1e-12)
            s["nor"][a] = s["nor"][b] = tuple(float(x) for x in nm)


def _welded(pos):
    """Shared label for the copies of a same vertex (UV and material seams)."""
    _, rep = np.unique(np.round(np.asarray(pos, float) / 1e-4).astype(np.int64), axis=0, return_inverse=True)
    return rep.ravel()


def _edges(rep, F):
    return np.sort(rep[np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])], axis=1)


def _skin_near(pos, Fb, Fi):
    """Faces of `Fb` that have a vertex near the bounding box of faces `Fi`."""
    vi = np.unique(Fi)
    margin = (NEAR_MM + NAPE_MM) * MM
    lo, hi = pos[vi].min(0) - margin, pos[vi].max(0) + margin
    return Fb[((pos[Fb] > lo) & (pos[Fb] < hi)).all(2).any(1)]


def _point_triangle(p, a, b, c):
    """Point of each triangle (a, b, c) closest to p (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac = b - a, c - a
    d1, d2 = (ab * (p - a)).sum(1), (ac * (p - a)).sum(1)
    d3, d4 = (ab * (p - b)).sum(1), (ac * (p - b)).sum(1)
    d5, d6 = (ab * (p - c)).sum(1), (ac * (p - c)).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        r = a + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]
        for ok, x in (((va <= 0) & (d4 >= d3) & (d5 >= d6), b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]),
                      ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * (d2 / (d2 - d6))[:, None]),
                      ((d6 >= 0) & (d5 <= d6), c),
                      ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * (d1 / (d1 - d3))[:, None]),
                      ((d3 >= 0) & (d4 <= d3), b),
                      ((d1 <= 0) & (d2 <= 0), a)):
            r = np.where(ok[:, None], x, r)
    return np.where(np.isfinite(r), r, a)


def _distance_to_body(pos, nor, Fb, q):
    """Signed distance (positive outside) from each point of `q` to the closest point of the skin `Fb`,
    outward normal of the triangle carrying it, and that point. The vertical projection of `_under_face`
    sometimes picked a triangle on the other side of the neck (60 mm inside, survey of 10/11 on Grace, Anby, ZhuYuan)."""
    a, b, c = pos[Fb[:, 0]], pos[Fb[:, 1]], pos[Fb[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    n *= np.sign((nor[Fb].sum(1) * n).sum(1) + 1e-12)[:, None]
    d, nn, pp = np.zeros(len(q)), np.zeros((len(q), 3)), np.zeros((len(q), 3))
    for k, p in enumerate(q):
        r = _point_triangle(p, a, b, c)
        j = int(np.argmin(((r - p) ** 2).sum(1)))
        pp[k], nn[k] = r[j], n[j]
        d[k] = np.linalg.norm(p - r[j]) * (1.0 if float((p - r[j]) @ n[j]) >= 0 else -1.0)
    return d, nn, pp


def _border_of_nape(pos, nor, faces, fm, skins, Fb):
    """Free edge of the head skins `skins` taken together (without the edges of another part): its lower
    welded vertices within `NEAR_MM` of the body skin `Fb`, the others, the label `rep` and one vertex
    per label."""
    rep = _welded(pos)
    m = rep.max() + 1
    first = np.zeros(m, int)
    first[rep[::-1]] = np.arange(len(rep))[::-1]
    Fi = faces[np.isin(fm, skins)]
    u, n = np.unique(_edges(rep, Fi), axis=0, return_counts=True)
    b = u[n == 1]
    Fo = faces[~np.isin(fm, skins)]
    if len(Fo):
        Eo = _edges(rep, Fo)
        b = b[~np.isin(b[:, 0] * m + b[:, 1], Eo[:, 0] * m + Eo[:, 1])]
    vb = np.unique(b)
    if not len(vb) or not len(Fb):
        return vb[:0], vb, rep, first
    # lower edge (under the jaw, sides, nape): the taken skin is not below the edge; the hairline edge,
    # with skin below it (pointing down by at least 0.5 on four heads, 10/11), also passes near the
    # body at the nape (16 mm on Grace) and does not move
    A = rep[np.concatenate([Fi[:, [0, 1]], Fi[:, [1, 2]], Fi[:, [2, 0]]])]
    t3 = rep[np.concatenate([Fi[:, 2], Fi[:, 0], Fi[:, 1]])]
    As = np.sort(A, axis=1)
    o = np.argsort(As[:, 0] * m + As[:, 1])
    k = o[np.searchsorted((As[:, 0] * m + As[:, 1])[o], b[:, 0] * m + b[:, 1])]
    P = pos[first]
    e = P[b[:, 1]] - P[b[:, 0]]
    e /= np.maximum(np.linalg.norm(e, axis=1), 1e-12)[:, None]
    inside = P[t3[k]] - 0.5 * (P[b[:, 0]] + P[b[:, 1]])
    inside -= (inside * e).sum(1)[:, None] * e
    inside /= np.maximum(np.linalg.norm(inside, axis=1), 1e-12)[:, None]
    bottom = np.isin(vb, b[inside[:, 1] > -0.4])
    d, _, _ = _distance_to_body(pos, nor, Fb, P[vb])
    nearest = bottom & (np.abs(d) < NEAR_MM * MM)
    return vb[nearest], vb[~nearest], rep, first


def _mix(s, v, j, w):
    """Weights of vertex v pulled towards those of vertex j, by the share w (four bones at most)."""
    accessory = {}
    for src, f in ((v, 1.0 - w), (j, w)):
        for b, x in zip(s["bones"][src], s["weights"][src]):
            if b >= 0 and x > 0:
                accessory[int(b)] = accessory.get(int(b), 0.0) + f * float(x)
    taken = sorted(accessory.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(x for _, x in taken)
    os_, p = [b for b, _ in taken], [x / tot for _, x in taken]
    s["sdef"][v] = None
    if len(os_) == 1:
        s["type"][v], s["bones"][v], s["weights"][v] = 0, (os_[0], -1, -1, -1), (1.0, 0.0, 0.0, 0.0)
    elif len(os_) == 2:
        s["type"][v], s["bones"][v], s["weights"][v] = 1, (os_[0], os_[1], -1, -1), (p[0], 1.0 - p[0], 0.0, 0.0)
    else:
        os_ += [os_[0]] * (4 - len(os_))
        p += [0.0] * (4 - len(p))
        s["type"][v], s["bones"][v], s["weights"][v] = 2, tuple(os_), tuple(p)


def _place_nape(s, faces, fm, i_skin, i_island, i_face):
    """The bottom of the head's scalp slides under the body's neck, which does not move.

    The taken scalp (`skull_of_game`) stops where the head carries it at `NECK_WEIGHT`; the donor head's
    neck was thicker than the body's: at that height, this edge stayed 3 to 9 mm outside (survey of
    10/11 on six heads), open under the hair. Each vertex of that edge goes `TUCKED_MM` under the body
    skin; scalp vertices within `NAPE_MM` of the edge (along the skin) smoothly follow the mean
    displacement of the neighbouring edge vertices (within `SMOOTH_MM`), with the body's normal and
    weights (they turn with the neck) and no outline; at the top of the strip, nothing changes.
    Measured each on its own, a high vertex under the hair, with no body neck facing it, was pulled
    towards the shoulder (up to 45 mm on Miyabi, 10/11). The bottom of the face (`i_visage`) is taken
    along with the scalp: where the head's neck is cut, its edge stayed up to 10 mm outside the body's
    neck and the back side of the face showed underneath (Jane, Caesar, Anby, 10/11). Vertices
    stitched to another part stay put."""
    pos = np.asarray(s["pos"], float)
    nor = np.asarray(s["nor"], float)
    skins = [i_island] + list(i_face)
    if not (fm == i_island).any():
        return {"nape_border": 0}
    Fi = faces[np.isin(fm, skins)]
    Fb = _skin_near(pos, faces[fm == i_skin], Fi)
    seeds, far, rep, first = _border_of_nape(pos, nor, faces, fm, skins, Fb)
    assessment = {"nape_border": int(len(seeds)), "nape_border_far": int(len(far))}
    if not len(seeds):
        return assessment
    P = pos[first]
    E = np.unique(_edges(rep, Fi), axis=0)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    g = np.full(len(P), np.inf)
    g[seeds] = 0.0
    for _ in range(len(P)):
        g2 = g.copy()
        np.minimum.at(g2, E[:, 0], g[E[:, 1]] + L)
        np.minimum.at(g2, E[:, 1], g[E[:, 0]] + L)
        if np.array_equal(g2, g):
            break
        g = g2
    sewn = np.zeros(len(P), bool)
    sewn[rep[np.unique(faces[~np.isin(fm, skins + [i_skin])])]] = True
    vi = np.unique(Fi)
    r = np.unique(rep[vi])
    r = r[(g[r] < NAPE_MM * MM) & ~sewn[r]]
    x = 1.0 - g[r] / (NAPE_MM * MM)
    w = x * x * (3.0 - 2.0 * x)
    d, n, pp = _distance_to_body(pos, nor, Fb, P[seeds])
    disp_g = pp - TUCKED_MM * MM * n - P[seeds]
    K = np.exp(-(np.linalg.norm(P[r][:, None] - P[seeds][None], axis=2) / (SMOOTH_MM * MM)) ** 2)
    disp = w[:, None] * (K @ disp_g) / np.maximum(K.sum(1), 1e-12)[:, None]
    is_g = np.isin(r, seeds)
    disp[is_g] = disp_g[np.searchsorted(seeds, r[is_g])]
    vb = np.unique(Fb)
    target = P[r] + disp
    j = vb[np.argmin(((target[:, None] - pos[vb][None]) ** 2).sum(-1), axis=1)]
    a_island = np.zeros(len(pos), bool)
    a_island[vi] = True
    for v in np.where(a_island & np.isin(rep, r))[0].tolist():
        k = int(np.searchsorted(r, rep[v]))
        s["pos"][v] = tuple(float(c) for c in pos[v] + disp[k])
        nv = (1.0 - w[k]) * nor[v] + w[k] * nor[j[k]]
        s["nor"][v] = tuple(float(c) for c in nv / max(float(np.linalg.norm(nv)), 1e-12))
        _mix(s, v, int(j[k]), float(w[k]))
        s["contour"][v] = float(s["contour"][v]) * (1.0 - float(w[k]))
    assessment.update({"nape_vertices": int((a_island & np.isin(rep, r)).sum()),
                  "nape_shift_mm": round(float(np.linalg.norm(disp, axis=1).max() / MM), 1),
                  "nape_gap_border_mm": [round(float(x / MM), 1) for x in np.percentile(d, [0, 50, 100])]})
    return assessment


def apply(s, faces, mat_face, materials, i_skin, i_face, neck_head, axis, i_island=None, os_head=None):
    """Remove the body's scalp without touching its neck; returns (kept faces, summary) and updates
    `materials`.

    i_ile: material of the ears and scalp brought by the head; the body goes under it as under the
    face, and the bottom of the scalp under the body's neck (`_place_nape`).
    os_head: the head bone and its children (`os_of_head`); without them, nothing counts as neck."""
    pos0 = np.asarray(s["pos"], float)
    fp = faces[mat_face == i_skin]
    frozen = np.zeros(len(pos0), bool)
    if os_head is not None:
        o, w = np.asarray(s["bones"]), np.asarray(s["weights"], float)
        neck = np.zeros(len(pos0), bool)
        neck[np.unique(fp)] = True
        neck &= (w * np.isin(o, os_head)).sum(1) < NECK_WEIGHT * w.sum(1)
        frozen[np.unique(fp[neck[fp].any(1)])] = True
        # copies of a frozen vertex (UV seams) are frozen too: `_tuck` pushed only one of them and the
        # skin split at the neck edge, 0.6 to 3.5 mm slits on nine heads out of ten (survey of 10/11)
        vp = np.unique(fp)
        rep = _welded(pos0)
        frozen[vp] |= np.isin(rep[vp], rep[frozen])
    if i_island is not None:
        # above the edge of the head scalp that will rest there, the body keeps `COVERS_MM` of skin underneath
        y0 = pos0[np.unique(faces[np.isin(mat_face, i_face)]), 1].min() - ZONE_MM * MM
        Fb0 = faces[(mat_face == i_skin) & ~_skull_of_body(pos0, faces, mat_face, i_skin, y0)]
        Fb0 = _skin_near(pos0, Fb0, faces[mat_face == i_island])
        seeds, _, _, first = _border_of_nape(pos0, np.asarray(s["nor"], float), faces, mat_face, [i_island], Fb0)
        if len(seeds):
            neck_head = np.vstack([np.asarray(neck_head, float).reshape(-1, 3),
                                  pos0[first[seeds]] + np.array([0.0, COVERS_MM * MM, 0.0])])
    remove, border, lim, assessment = cut(pos0, faces, mat_face, i_skin, i_face, neck_head, axis, frozen)
    assessment["vertices_figes"] = int(frozen.sum())
    guard = ~remove
    for v, p in zip(border.tolist(), _on_cut(pos0, faces[guard & (mat_face == i_skin)], border, lim)):
        s["pos"][v] = tuple(float(x) for x in p)
    faces, fm = faces[guard], mat_face[guard]
    if i_island is not None:
        assessment.update(_place_nape(s, faces, fm, i_skin, i_island, list(i_face)))
    i_vis_all = list(i_face) + ([i_island] if i_island is not None else [])
    pos1, b_r = _tuck(np.asarray(s["pos"], float), np.asarray(s["nor"], float), faces, fm, i_skin,
                                  i_vis_all, frozen)
    for v in np.where((pos1 != np.asarray(s["pos"], float)).any(1))[0].tolist():
        s["pos"][v] = tuple(float(x) for x in pos1[v])
    assessment.update(b_r)
    for i, m in enumerate(materials):
        m["n_indices"] = int(3 * (fm == i).sum())
    _unify_normals(s, faces, fm, list(i_face))
    pos = np.asarray(s["pos"], float)
    vv = np.unique(faces[np.isin(fm, i_vis_all)])
    vp = np.unique(faces[fm == i_skin])
    lo, hi = pos[vv].min(0) - WITHOUT_TRAIT_MM * MM, pos[vv].max(0) + WITHOUT_TRAIT_MM * MM
    vp = vp[((pos[vp] > lo) & (pos[vp] < hi)).all(1)]
    d = np.array([np.sqrt(((pos[vv] - pos[i]) ** 2).sum(1)).min() for i in vp.tolist()]) if len(vp) else np.zeros(0)
    without = vp[(d < WITHOUT_TRAIT_MM * MM) & ~frozen[vp]]
    for i in without.tolist():
        s["contour"][i] = 0.0
    assessment["without_trait"] = int(len(without))
    return faces, assessment
