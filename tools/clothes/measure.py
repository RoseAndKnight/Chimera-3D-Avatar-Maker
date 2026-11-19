"""Measures how ZZZ garments (official PMX files) are built, modelled on
tools/hair/measure.py. Accessories excluded (pockets, buttons, straps, buckles...).

    bin/blender-queue -b --factory-startup \
        --python tools/clothes/measure.py -- [--output DIR] [--only id1,id2] [--no-poses]
    ... -- --pmx CANDIDATE.pmx --mats A,B --type top|pull|jacket|... [--id NAME]
        [--layers X] --output DIR: measures a manufactured garment and prints the checks.

Writes DIR/<id>.json per garment (with "checks": guardrails and contracts,
see check), DIR/ensemble.json and DIR/tableau.md.
Default DIR: <data>/clothes (paths.data in the settings), otherwise $TMPDIR/clothes.

Blender is used for the BVHs (mathutils) and for reading textures (bpy). The PMX files stay
read-only. Frame and units: see model.py (metres; outputs are in mm).

Each measurement block states what it changes for manufacturing (docstring "-> manufacturing").
"""
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as MD  # noqa: E402
from model import P, PMX  # noqa: E402
import bench as BENCH  # noqa: E402  (tools/atlas, put on the path by model)

_spec = importlib.util.spec_from_file_location("measure_hair", os.path.join(HERE, "..", "hair", "measure.py"))
MC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MC)
r3, q, write_json, log = MC.r3, MC.q, MC.write_json, MC.log
segment_rigid, dist_segments, collide, ignores = MC.segment_rigid, MC.dist_segments, MC.collide, MC.ignores

try:
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
except ImportError:  # outside Blender: nothing geometric
    bpy = None

# ====================================================================== the garments studied
# Chosen in the active game (description tools/games/<game>/game.json, "studies.clothes"): one garment per requested
# type, on different characters when possible, contrasting constructions (single sheet, closed shell,
# double-sided, garment = body), then real sweaters; the description keeps the reason for each.
CLOTHES = MD.games.active().value("studies.clothes", [])   # a game with no study: no garment chosen

RADIUS_UNDER = 0.15     # m: beyond this, "nothing underneath"
CROSSES_MAX = 0.010     # m: an outgoing ray hitting the back of a surface closer than this = clipping
SLIDE_MAX = 0.005       # m: sliding along the surface beyond which the plane of the tracked face no longer holds
CROSSES_THRESHOLD = 0.0005  # m: depth below which a tracked pair is counted as clipping in a pose
CLASSES_GAP = [(0.0, 0.001, "glue_lt1mm"), (0.001, 0.003, "1_3mm"), (0.003, 0.010, "3_10mm"),
                 (0.010, 0.030, "10_30mm"), (0.030, RADIUS_UNDER, "30mm_plus")]
BODY_REG = ("trunk_top", "chest", "belly", "pelvis", "buttocks", "neck", "head", "arm",
             "forearm", "hand", "finger", "thigh", "leg", "foot")
JOINTS = [("shoulder", "左腕", "左肩"), ("elbow", "左ひじ", "左腕"), ("hip", "左足D", "下半身"),
                 ("knee", "左ひざD", "左足D")]
RIG_DYN = (1, 2)


def mm(x):
    return None if x is None else r3(np.asarray(x) * 1000, 1)


def qmm(v, ps=(10, 50, 90)):
    v = np.asarray(v, float)
    return q(v * 1000, ps, 2)


def part(m, weight=None):
    m = np.asarray(m, bool)
    if not m.size:
        return None
    if weight is None:
        return r3(m.mean())
    return r3((np.asarray(weight) * m).sum() / max(np.asarray(weight).sum(), 1e-20))


# ====================================================================== BVH

def bvh(pos, F):
    return BVHTree.FromPolygons([Vector(p) for p in pos], [tuple(int(x) for x in f) for f in F], all_triangles=True)


def rays(tree, O, D, dmax):
    """First hit: (face index or -1, distance, normal of the face hit)."""
    n = len(O)
    idx, dist, nor = np.full(n, -1), np.full(n, np.inf), np.zeros((n, 3))
    for i in range(n):
        h = tree.ray_cast(Vector(O[i]), Vector(D[i]), dmax)
        if h[0] is not None:
            idx[i], dist[i], nor[i] = h[2], h[3], h[1]
    return idx, dist, nor


def barycenters(p, A, B, C):
    v0, v1, v2 = B - A, C - A, p - A
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], 1)


# ====================================================================== context of a garment

class Garment:
    def __init__(self, M, spec):
        self.M, self.spec = M, spec
        self.sel, self.kept, self.discarded = MD.isolate(M, spec)
        cal = np.zeros(len(M.F), bool)
        if spec.get("texture_layers"):
            sc = dict(spec, mats=spec["texture_layers"])
            cal, _, _ = MD.isolate(M, sc)
        self.texture_layer = cal
        self.fg = np.where(self.sel)[0]
        self.Fg = M.F[self.fg]
        self.Fgs = M.Fs[self.fg]
        self.vs = np.unique(self.Fgs)                 # welded vertices of the garment
        self.vr = np.unique(self.Fg)                  # raw vertices
        loc = np.full(len(M.pos_s), -1)
        loc[self.vs] = np.arange(len(self.vs))
        self.loc = loc
        self.Fl = loc[self.Fgs]                       # faces as local welded indices
        self.P = M.pos_s[self.vs]
        self.nf = M.nf[self.fg]
        self.area = M.area[self.fg]
        # geometric normals per welded vertex (area-weighted)
        ng = np.zeros_like(self.P)
        for k in range(3):
            np.add.at(ng, self.Fl[:, k], self.nf * self.area[:, None])
        self.ng = ng / np.maximum(np.linalg.norm(ng, axis=1), 1e-20)[:, None]
        # body bones and "outward" direction: from the nearest body bone
        body = np.array([(M.region_os[b] in BODY_REG or M.family_os[b] == "root") for b in range(len(M.names_os))])
        self.os_body = body
        seg = [(b, M.heads[b], M.tails[b]) for b in np.where(body)[0]
               if np.linalg.norm(M.tails[b] - M.heads[b]) > 0.01 and M.region_os[b] not in ("finger", "head")]
        best = np.full(len(self.P), np.inf)
        self.outside = np.zeros_like(self.P)
        self.os_near = np.full(len(self.P), -1)
        for b, a, c in seg:
            ab = c - a
            t = np.clip(((self.P - a) @ ab) / (ab @ ab), 0, 1)
            pr = a + t[:, None] * ab
            d = np.linalg.norm(self.P - pr, axis=1)
            m = d < best
            best[m], self.outside[m], self.os_near[m] = d[m], (self.P - pr)[m], b
        self.outside /= np.maximum(np.linalg.norm(self.outside, axis=1), 1e-20)[:, None]
        self.sheet_ext = (self.ng * self.outside).sum(1) > 0
        # "underneath": everything else, except empty, head (hair), accessories, overlay layers and the
        # other parts made of the same materials (pockets placed on top)
        roles = np.array([M.role_material(i) for i in range(len(M.mats))])
        same = np.isin(M.mf, [i for i, n in enumerate(M.names_mat) if n in spec["mats"] + spec.get("texture_layers", [])])
        if spec.get("uv_boxes"):  # material carrying the whole outfit: only the cut-out garment counts
            same &= MD.zone_uv(M, spec)
        self.fd = np.where(np.isin(roles[M.mf], ["skin", "outfit"]) & ~self.sel & ~same)[0]
        self.skin = np.isin(roles[M.mf], ["skin"])
        # layers lying over the garment (jacket over a t-shirt): removed from the underneath.
        # Vote per welded part: a +n ray that sees its back within 30 mm says "over",
        # a -n ray that sees its front says "under".
        self.layers_over = []
        if len(self.fd):
            pcs = M.parts(np.isin(np.arange(len(M.F)), self.fd))
            part_of = np.full(len(M.F), -1)
            for k, g in enumerate(pcs):
                part_of[g] = k
            arb0 = bvh(M.pos, M.F[self.fd])
            ext = self.sheet_ext
            iu, _, nu = rays(arb0, self.P[ext] + self.ng[ext] * 1e-5, self.ng[ext], 0.03)
            idn, _, nd_ = rays(arb0, self.P[ext] + self.ng[ext] * 1e-4, -self.ng[ext], 0.03)
            over = np.zeros(len(pcs))
            under = np.zeros(len(pcs))
            ngx = self.ng[ext]
            for i in np.where((iu >= 0) & ((nu * ngx).sum(1) > 0))[0]:
                over[part_of[self.fd[iu[i]]]] += 1
            for i in np.where((idn >= 0) & ((nd_ * ngx).sum(1) > 0))[0]:
                under[part_of[self.fd[idn[i]]]] += 1
            top = np.where((over > under) & (over >= 5))[0]
            for k in top:
                self.layers_over.append({"materials": sorted({M.names_mat[i] for i in np.unique(M.mf[pcs[k]])}),
                                            "faces": int(len(pcs[k])), "votes_over": int(over[k]),
                                            "votes_under": int(under[k])})
            if len(top):
                removes = np.concatenate([pcs[k] for k in top])
                self.fd = np.setdiff1d(self.fd, removes)
        # region of each welded vertex
        dom = M.dom[M.sv[self.vs]]
        self.region = np.array([M.region_ancestor(int(b)) if b >= 0 else "without" for b in dom])
        self.tree = bvh(self.P, self.Fl)
        if len(self.fd):
            self.tree_d = bvh(M.pos, M.F[self.fd])
        else:
            self.tree_d = None
        # welded edges of the garment
        E, c, corner = MD.edges_faces(self.Fl)
        self.E, self.Ec, self.corner = E, c, corner
        adj = [[] for _ in range(len(self.P))]
        for a, b in E:
            adj[a].append(b)
            adj[b].append(a)
        self.adj = adj

    def rings(self, sources, n):
        """Distance in edges (BFS) from the source vertices, up to n."""
        d = np.full(len(self.P), 99)
        d[sources] = 0
        front = list(sources)
        for k in range(1, n + 1):
            nv = []
            for a in front:
                for b in self.adj[a]:
                    if d[b] > k:
                        d[b] = k
                        nv.append(b)
            front = nv
        return d


# ====================================================================== A. mesh

def measure_mesh(V):
    """-> manufacturing: kind of sheet to model (single, thick, shell), edge finish,
    face budget and density relative to the skin, folds to model or to paint, normals,
    outline."""
    M = V.M
    E, c = V.E, V.Ec
    border = c == 1
    # thickness: -n ray from each face, hit on the garment with an opposite normal
    cen = V.P[V.Fl].mean(1)
    n = V.nf
    O = cen - n * 1e-5
    idx, dist, nor = rays(V.tree, O, -n, 0.02)
    pair = (idx >= 0) & ((nor * n).sum(1) < -0.5)
    # vertices near an edge (3 rings)
    vb = np.unique(E[border])
    an = V.rings(vb, 3) if len(vb) else np.full(len(V.P), 99)
    f_near_border = (an[V.Fl] <= 3).any(1)
    # parts
    pcs = [d["faces"] for d in V.kept]
    closed_ones = sum(1 for d in V.kept if d["closed"])
    # edge lengths: garment, skin, underneath
    L = np.linalg.norm(V.P[E[:, 0]] - V.P[E[:, 1]], axis=1)

    def lengths(fi):
        if not len(fi):
            return None
        Ed, _, _ = MD.edges_faces(M.Fs[fi])
        return np.linalg.norm(M.pos_s[Ed[:, 0]] - M.pos_s[Ed[:, 1]], axis=1)
    Lp = lengths(np.where(M.skin_faces)[0]) if hasattr(M, "skin_faces") else lengths(np.where(V.skin)[0])
    Ld = lengths(V.fd)
    area = V.area.sum()
    # geometric folds: offset from a smoothed surface (Taubin, edges fixed), along the normal
    smooth = taubin(V, 30)
    dev = ((V.P - smooth) * V.ng).sum(1)
    regions = {}
    for r in np.unique(V.region):
        m = V.region == r
        if m.sum() >= 30:
            regions[r] = {"vertices": int(m.sum()), "abs_p90_mm": r3(np.percentile(np.abs(dev[m]), 90) * 1000, 2),
                          "part_gt2mm": r3((np.abs(dev[m]) > 0.002).mean())}
    # interior dihedral angles
    f2 = np.where(c == 2)[0]
    # faces of each interior edge
    fa = np.full((len(E), 2), -1)
    cnt = np.zeros(len(E), int)
    for k in range(3):
        for fi, e in enumerate(V.corner[:, k]):
            if cnt[e] < 2:
                fa[e, cnt[e]] = fi
            cnt[e] += 1
    ok = (c == 2)
    dih = np.degrees(np.arccos(np.clip((V.nf[fa[ok, 0]] * V.nf[fa[ok, 1]]).sum(1), -1, 1)))
    # stored normals against geometric and smoothed normals
    nr = M.nor[V.vr]
    nr = nr / np.maximum(np.linalg.norm(nr, axis=1), 1e-20)[:, None]
    ls = V.loc[M.s[V.vr]]
    ang_geo = np.degrees(np.arccos(np.clip((nr * V.ng[ls]).sum(1), -1, 1)))
    nl = normals_vertices(smooth, V.Fl)
    ang_smooth = np.degrees(np.arccos(np.clip((nr * nl[ls]).sum(1), -1, 1)))
    # split normals: a welded vertex carries several stored normals
    avg = np.zeros_like(V.P)
    np.add.at(avg, ls, nr)
    avg /= np.maximum(np.linalg.norm(avg, axis=1), 1e-20)[:, None]
    gap = np.degrees(np.arccos(np.clip((nr * avg[ls]).sum(1), -1, 1)))
    cutting = np.zeros(len(V.P), bool)
    np.logical_or.at(cutting, ls, gap > 5)
    uvs = M.uv[V.vr]
    uv_avg = np.zeros((len(V.P), 2))
    np.add.at(uv_avg, ls, uvs)
    nb = np.bincount(ls, minlength=len(V.P))
    uv_avg /= np.maximum(nb, 1)[:, None]
    seam_uv = np.zeros(len(V.P), bool)
    np.logical_or.at(seam_uv, ls, np.abs(uvs - uv_avg[ls]).max(1) > 1e-5)
    # outline
    mats = sorted({int(i) for i in np.unique(M.mf[V.fg])})
    cont = M.outline[V.vr]
    skin_mats = [i for i in range(len(M.mats)) if M.role_material(i) == "skin"]
    reg_raw = V.region[ls]
    cont_reg = {r: r3(np.median(cont[reg_raw == r])) for r in np.unique(reg_raw) if (reg_raw == r).sum() >= 20}
    return {
        "faces": int(len(V.fg)), "vertices_welded": int(len(V.P)), "vertices_raw": int(len(V.vr)),
        "faces_model": int(len(M.F)), "part_faces_model": r3(len(V.fg) / len(M.F)),
        "parts": len(pcs), "faces_per_part": pcs[:12], "parts_closed": closed_ones,
        "border_edges": int(border.sum()), "edges_non_manifold": int((c > 2).sum()),
        "part_border_edges": r3(border.mean()),
        "double_face_flag": [bool(M.mats[i]["flags"] & 1) for i in mats],
        "thickness": {"part_faces_in_vis_a_vis": part(pair, V.area),
                      "distance_mm": qmm(dist[pair]) if pair.any() else None,
                      "part_near_border": part(pair[f_near_border], V.area[f_near_border]) if f_near_border.any() else None,
                      "part_far_border": part(pair[~f_near_border], V.area[~f_near_border]) if (~f_near_border).any() else None},
        "sheet_outer_part": part(V.sheet_ext),
        "density": {"edge_mm": qmm(L), "edge_skin_mm": qmm(Lp) if Lp is not None and len(Lp) else None,
                    "edge_under_mm": qmm(Ld) if Ld is not None and len(Ld) else None,
                    "report_edge_skin": r3(np.median(L) / np.median(Lp)) if Lp is not None and len(Lp) else None,
                    "report_edge_under": r3(np.median(L) / np.median(Ld)) if Ld is not None and len(Ld) else None,
                    "area_dm2": r3(area * 100, 2), "faces_per_dm2": r3(len(V.fg) / (area * 100), 1)},
        "folds_geometric": {"gap_smooth_abs_mm": q(np.abs(dev) * 1000, (50, 90, 99), 2),
                              "max_mm": r3(np.abs(dev).max() * 1000, 1),
                              "part_gt2mm": r3((np.abs(dev) > 0.002).mean()),
                              "part_gt5mm": r3((np.abs(dev) > 0.005).mean()),
                              "per_region": regions,
                              "dihedral_deg": q(dih, (50, 90, 99), 1),
                              "part_dihedral_gt30": r3((dih > 30).mean()), "part_dihedral_gt60": r3((dih > 60).mean())},
        "normals": {"angle_geo_deg": q(ang_geo, (50, 90, 99), 1),
                     "angle_smooth_deg": q(ang_smooth, (50, 90, 99), 1),
                     "plus_near_of_smooth_part": r3((ang_smooth < ang_geo - 2).mean()),
                     "vertices_cuts_part": r3(cutting.mean()),
                     "cuts_on_seam_uv_part": r3(seam_uv[cutting].mean()) if cutting.any() else None,
                     "cuts_outside_seam_uv": int((cutting & ~seam_uv).sum())},
        "contour": {"materials": [{"name": M.names_mat[i], "trace": bool(M.mats[i]["flags"] & 0x10),
                                   "size": r3(M.mats[i]["outline_size"]),
                                   "color": r3(M.mats[i].get("outline_color")) if M.mats[i].get("outline_color") is not None else None}
                                  for i in mats],
                    "size_skin": sorted({r3(M.mats[i]["outline_size"]) for i in skin_mats}),
                    "vertex": q(cont, (1, 10, 50, 90)), "part_vertex_not_1": r3((np.abs(cont - 1) > 1e-3).mean()),
                    "part_vertex_0": r3((cont < 1e-3).mean()), "per_region_med": cont_reg},
    }


def normals_vertices(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    out = np.zeros_like(P)
    for k in range(3):
        np.add.at(out, F[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1), 1e-20)[:, None]


def taubin(V, it, lam=0.5, mu=-0.53):
    """Taubin smoothing (no shrinkage) of the welded vertices, edges fixed."""
    P = V.P.copy()
    E = V.E
    border = np.zeros(len(P), bool)
    border[V.E[V.Ec == 1].ravel()] = True
    deg = np.zeros(len(P))
    np.add.at(deg, E[:, 0], 1)
    np.add.at(deg, E[:, 1], 1)
    deg = np.maximum(deg, 1)
    for k in range(it * 2):
        s = np.zeros_like(P)
        np.add.at(s, E[:, 0], P[E[:, 1]])
        np.add.at(s, E[:, 1], P[E[:, 0]])
        lap = s / deg[:, None] - P
        f = lam if k % 2 == 0 else mu
        P[~border] += f * lap[~border]
    return P


# ====================================================================== B. fit

def measure_adjustment(V, poses=True):
    """-> manufacturing: offset to give the fabric per zone, keep or remove the hidden skin,
    edges to watch (pinching) and poses where skinning causes clipping."""
    M = V.M
    res = {}
    if V.tree_d is None:
        return {"under": "no surface underneath"}
    # inward offset (-n ray from the outer sheet)
    ext = V.sheet_ext
    O = V.P - V.ng * (-1e-4)          # start 0.1 mm outside
    idx, dist, nor = rays(V.tree_d, O, -V.ng, RADIUS_UNDER)
    face_ok = (idx >= 0) & ((nor * V.ng).sum(1) > 0)      # surface underneath seen from the front
    d = np.where(face_ok, dist - 1e-4, np.inf)
    class_ = {}
    for lo, hi, name in CLASSES_GAP:
        class_[name] = part(ext & (d >= lo) & (d < hi)) if ext.any() else None
    class_["nothing_under"] = part(ext & ~np.isfinite(d))
    # materials hit underneath
    fd = V.fd
    mat_under = {}
    for i in np.where(ext & np.isfinite(d))[0]:
        n = M.names_mat[M.mf[fd[idx[i]]]]
        mat_under[n] = mat_under.get(n, 0) + 1
    tot = max(sum(mat_under.values()), 1)
    per_region = {}
    for r in np.unique(V.region):
        m = (V.region == r) & ext
        if m.sum() < 20:
            continue
        dd = d[m]
        per_region[r] = {"vertices": int(m.sum()), "nothing_under": r3((~np.isfinite(dd)).mean()),
                         "gap_mm": qmm(dd[np.isfinite(dd)]) if np.isfinite(dd).any() else None}
    res["gap"] = {"classes_part": class_, "gap_mm": qmm(d[ext & np.isfinite(d)]) if (ext & np.isfinite(d)).any() else None,
                    "per_region": per_region,
                    "materials_under": {k: r3(v / tot) for k, v in sorted(mat_under.items(), key=lambda x: -x[1])[:6]}}
    # clipping at rest: a +n ray hits the back of a surface underneath within TRAVERSE_MAX
    O2 = V.P + V.ng * 1e-5
    idx2, dist2, nor2 = rays(V.tree_d, O2, V.ng, CROSSES_MAX)
    cross = (idx2 >= 0) & ((nor2 * V.ng).sum(1) > 0)
    tm = {}
    for i in np.where(cross)[0]:
        n = M.names_mat[M.mf[fd[idx2[i]]]]
        tm[n] = tm.get(n, 0) + 1
    res["rest_crossing"] = {"fraction": part(cross), "depth_mm": qmm(dist2[cross]) if cross.any() else None,
                              "per_material": tm,
                              "regions": {r: int((cross & (V.region == r)).sum()) for r in np.unique(V.region[cross])}}
    # skin and underneath hidden by the garment: +n ray from the faces underneath
    Fd = M.F[fd]
    cen = M.pos[Fd].mean(1)
    nd = M.nf[fd]
    idx3, dist3, nor3 = rays(V.tree, cen + nd * 1e-5, nd, 0.05)
    occluded = idx3 >= 0
    ad = M.area[fd]
    per_mat = {}
    for i in np.where(occluded)[0]:
        n = M.names_mat[M.mf[fd[i]]]
        per_mat[n] = per_mat.get(n, 0) + ad[i]
    pe = V.skin[fd]
    res["cache"] = {"area_hidden_on_area_garment": r3(ad[occluded].sum() / V.area.sum()),
                    "area_hidden_skin_on_area_garment": r3(ad[occluded & pe].sum() / V.area.sum()),
                    "gap_hidden_skin_mm": qmm(dist3[occluded & pe]) if (occluded & pe).any() else None,
                    "per_material_dm2": {k: r3(v * 100, 3) for k, v in sorted(per_mat.items(), key=lambda x: -x[1])[:6]}}
    # edges: free or joined, pinching
    res["borders"] = borders(V, d)
    # poses
    if poses:
        res["poses"] = measure_poses(V, idx, d, cross)
    return res


def borders(V, d):
    """Edge loops of the garment: joined to another surface (vertices shared after
    welding) or free; offset at the hem against the offset 3 to 5 rings further in (pinching)."""
    M = V.M
    E, c = V.E, V.Ec
    eb = E[c == 1]
    if not len(eb):
        return {"loops": 0}
    # joined: the welded edge exists in a face outside the garment (not empty). Copies of the
    # garment's faces ("+" overlay, flipped copy) do not count: they border the whole garment.
    others = np.where(~V.sel & ~M.faces_role(["empty"]))[0]
    copies = set(map(tuple, np.sort(M.Fs[V.sel], 1).tolist()))
    others = others[[tuple(x) not in copies for x in np.sort(M.Fs[others], 1).tolist()]]
    Ea, _, _ = MD.edges_faces(M.Fs[others])
    ga = V.vs[eb]
    ga = np.sort(ga, 1)
    key_a = set(map(tuple, Ea.tolist()))
    join = np.array([tuple(x) in key_a for x in ga.tolist()])
    # role of the other face
    lab = MD.components(len(V.P), eb)
    loops = []
    an_all = None
    for l in np.unique(lab[eb[:, 0]]):
        m = lab[eb[:, 0]] == l
        vb = np.unique(eb[m])
        if len(vb) < 4:
            continue
        an = V.rings(vb, 5)
        loop = (an >= 3) & (an <= 5)
        g0 = d[vb][np.isfinite(d[vb])]
        g3 = d[loop][np.isfinite(d[loop])]
        long_ = np.linalg.norm(V.P[eb[m, 0]] - V.P[eb[m, 1]], axis=1).sum()
        reg, nreg = np.unique(V.region[vb], return_counts=True)
        e0 = float(np.median(g0)) if len(g0) >= 3 else None
        e3 = float(np.median(g3)) if len(g3) >= 3 else None
        loops.append({"edges": int(m.sum()), "length_mm": r3(long_ * 1000, 1),
                        "region": str(reg[np.argmax(nreg)]), "welded_part": r3(join[m].mean()),
                        "nothing_under_part": r3((~np.isfinite(d[vb])).mean()),
                        "gap_hem_mm": mm(e0), "gap_loop3_5_mm": mm(e3),
                        "pince": bool(join[m].mean() <= 0.5 and e0 is not None and e3 is not None
                                      and e0 < 0.5 * e3 and e0 < 0.003)})
    loops.sort(key=lambda b: -b["length_mm"])
    return {"loops": len(loops), "welded": sum(b["welded_part"] > 0.5 for b in loops),
            "free": sum(b["welded_part"] <= 0.5 for b in loops),
            "pinched": sum(b["pince"] for b in loops), "detail": loops[:12]}


def measure_poses(V, idx, d, cross_rest):
    """Bench of 22 poses (tools/atlas/bench.py), linear skinning (SDEF read as BDEF2).
    1) tracking: each outer vertex and the surface underneath it sees at rest (barycentric
       point); clipping if the signed offset drops below -0.5 mm.
    2) new clipping: 10 mm +n ray towards the back of a deformed surface underneath,
       minus those already present at rest."""
    M = V.M
    ok = np.isfinite(d) & V.sheet_ext
    if not ok.any():
        return {"pairs": 0, "note": "nothing underneath: no pair to track"}
    fd = V.fd
    fi = fd[idx[ok]]
    Fq = M.F[fi]
    rep = M.sv[V.vs[ok]]                         # representative raw vertex
    P0 = M.pos[rep]
    bc = barycenters(P0 - V.ng[ok] * d[ok][:, None], M.pos[Fq[:, 0]], M.pos[Fq[:, 1]], M.pos[Fq[:, 2]])
    s0 = d[ok]
    rep_all = M.sv[V.vs]
    # vertices carried (even slightly) by the garment's own bones (physics): physics moves them
    # in game, skinning alone says nothing about them
    Wv, Iv = M.weight[rep], M.idx[rep]
    free = (np.where((Iv >= 0) & ~V.os_body[np.maximum(Iv, 0)], Wv, 0).sum(1) > 0.05)
    res = {"pairs": int(ok.sum()), "pairs_body_alone": int((~free).sum()), "pairs_own_bones_part": r3(free.mean()),
           "skinning": "LBS, SDEF ignored, physics bones left at rest", "per_pose": {}}
    worst = []
    for name in BENCH.BENCH:
        try:
            rot = BENCH.pose_bench(name, M.sq)
        except Exception as e:  # missing bone
            res["per_pose"][name] = {"error": str(e)}
            continue
        Pd, _, _ = P.deform(M.pos, M.idx, M.weight, M.sq, rotations=rot)
        A, B, C = Pd[Fq[:, 0]], Pd[Fq[:, 1]], Pd[Fq[:, 2]]
        qd = bc[:, :1] * A + bc[:, 1:2] * B + bc[:, 2:] * C
        nq = np.cross(B - A, C - A)
        nq /= np.maximum(np.linalg.norm(nq, axis=1), 1e-20)[:, None]
        s1 = ((Pd[rep] - qd) * nq).sum(1)
        tr = s1 < -CROSSES_THRESHOLD
        arb = bvh(Pd, M.F[fd])
        # the plane of the tracked face only holds if the vertex stayed above it: a garment
        # sliding around a curved surface (a belt turning with the chest) goes below that plane
        # without clipping anything. Once it has slid, clipping is confirmed against the
        # deformed surface itself: under its nearest face, and along its axis.
        slide = np.linalg.norm((Pd[rep] - qd) - s1[:, None] * nq, axis=1) > SLIDE_MAX
        for j in np.where(tr & slide)[0]:
            loc, nf, _, dist = arb.find_nearest(Vector(Pd[rep[j]]))
            sd = float((Vector(Pd[rep[j]]) - loc).dot(nf)) if loc is not None else 0.0
            if sd < -CROSSES_THRESHOLD and -sd >= 0.7 * dist:
                s1[j] = sd
            else:
                tr[j] = False
        # new clipping (all surfaces underneath)
        Pg = Pd[rep_all]
        ngp = normals_vertices(Pg, V.Fl)
        i2, d2, n2 = rays(arb, Pg + ngp * 1e-5, ngp, CROSSES_MAX)
        tn = (i2 >= 0) & ((n2 * ngp).sum(1) > 0) & ~cross_rest
        reg = V.region[ok][tr]
        u, n_ = np.unique(reg, return_counts=True)
        reg2 = V.region[tn]
        u2, n2_ = np.unique(reg2, return_counts=True)
        res["per_pose"][name] = {"track_crosses_part": r3(tr.mean(), 4),
                                # share of the whole part, not of the tracked points only: a skirt with physics
                                # has only a handful of points held by the body, and 5 out of 14 got it rejected
                                "track_body_alone_part": r3(tr[~free].sum() / max(len(rep_all), 1), 4) if (~free).any() else None,
                                "track_depth_mm": qmm(-s1[tr], (50, 90, 100)) if tr.any() else None,
                                "track_regions": {str(a): int(b) for a, b in zip(u, n_)},
                                "new_part": r3(tn.mean(), 4),
                                "new_regions": {str(a): int(b) for a, b in zip(u2, n2_)}}
        worst.append((tr.mean() + tn.mean(), name))
    worst.sort(reverse=True)
    vals = [v for v in res["per_pose"].values() if "track_crosses_part" in v]
    res["track_max_part"] = r3(max(v["track_crosses_part"] for v in vals), 4) if vals else None
    res["new_max_part"] = r3(max(v["new_part"] for v in vals), 4) if vals else None
    cs = [v["track_body_alone_part"] for v in vals if v.get("track_body_alone_part") is not None]
    res["track_body_alone_max_part"] = r3(max(cs), 4) if cs else None
    res["poses_without_crossing"] = sum(1 for v in vals if v["track_crosses_part"] == 0 and v["new_part"] == 0)
    res["poses_measured"] = len(vals)
    res["worst"] = [n for _, n in worst[:4]]
    return res


# ====================================================================== C. UV and texture

def measure_uv(V):
    """-> manufacturing: UV cutting (where to place seams), texel density to aim for, shared
    symmetry, what is painted (folds, shadows, pattern) and auxiliary textures."""
    M = V.M
    F = V.Fg
    uv = M.uv
    # islands (raw vertices, unwelded)
    Er = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]])
    lab = MD.components(len(M.pos), Er)
    il = lab[F[:, 0]]
    u, cnt = np.unique(il, return_counts=True)
    # seams: interior welded edge whose two faces do not share the same raw edge
    Ebr = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), 2)   # (f,3,2) raw
    ebr = Ebr.reshape(-1, 2)
    key_s = V.corner.reshape(-1)         # welded edge of each corner (f*3); coin is (f,3) -> order [01,12,20]
    # V.coin comes from edges_faces(Fl), which stacks [01],[12],[20]: same order as Ebr
    seam = np.zeros(len(V.E), bool)
    uvd = np.zeros(len(V.E), bool)
    first = np.full((len(V.E), 2), -1)
    for k in range(len(ebr)):
        e = key_s[k]
        if first[e, 0] < 0:
            first[e] = ebr[k]
        elif (first[e] != ebr[k]).any():
            seam[e] = True
            a, b = first[e]
            c_, d_ = ebr[k]
            # same position, different UVs?
            pa, pc = M.s[a], M.s[c_]
            u1 = (uv[a], uv[b]) if pa == pc else (uv[b], uv[a])
            if np.abs(np.array(u1) - np.array([uv[c_], uv[d_]])).max() > 1e-5:
                uvd[e] = True
    cs = seam & uvd
    Lc = np.linalg.norm(V.P[V.E[:, 0]] - V.P[V.E[:, 1]], axis=1)
    # position of the seams around the nearest body bone
    mid = (V.P[V.E[:, 0]] + V.P[V.E[:, 1]]) / 2
    sectors = {}
    for e in np.where(cs)[0]:
        b = V.os_near[V.E[e, 0]]
        if b < 0:
            continue
        a, c_ = M.heads[b], M.tails[b]
        ax = (c_ - a) / max(np.linalg.norm(c_ - a), 1e-9)
        r = mid[e] - a
        r = r - (r @ ax) * ax
        av = np.array([0, -1.0, 0]) - (np.array([0, -1.0, 0]) @ ax) * ax
        av /= max(np.linalg.norm(av), 1e-9)
        ang = np.degrees(np.arccos(np.clip(r @ av / max(np.linalg.norm(r), 1e-9), -1, 1)))
        family = family_region(M.region_os[b])
        sec = "before" if ang < 30 else "front_side" if ang < 75 else "side" if ang < 105 else "back_side" if ang < 150 else "back"
        if family == "arm" and np.linalg.norm(r) > 0:
            sec = "under" if r[2] < -0.5 * np.linalg.norm(r) else "outer" if r[2] > 0.5 * np.linalg.norm(r) else sec
        sectors.setdefault(family, {}).setdefault(sec, 0.0)
        sectors[family][sec] += Lc[e]
    for family in sectors:
        t = sum(sectors[family].values())
        sectors[family] = {k: r3(v / t) for k, v in sorted(sectors[family].items(), key=lambda x: -x[1])}
    center = cs & (np.abs(mid[:, 0]) < 0.003)
    # shared symmetry: identical UV triangles
    T = np.round(uv[F] * 1e4).astype(np.int64).tolist()
    keys = [tuple(sorted(map(tuple, t))) for t in T]
    from collections import Counter
    cc = Counter(keys)
    sharing = np.array([cc[k] > 1 for k in keys])
    # texel density
    sizes = {}
    out = {"islands": int(len(u)), "plus_large_island_part": r3(cnt.max() / cnt.sum()),
           "islands_part_faces": r3(np.sort(cnt)[::-1][:8] / cnt.sum()),
           "edges_seam_uv": int(cs.sum()), "length_seam_mm": r3(Lc[cs].sum() * 1000, 0),
           "seam_on_length_border": r3(Lc[cs].sum() / max(Lc[V.Ec == 1].sum(), 1e-9)),
           "cuts_normals_only": int((seam & ~uvd).sum()),
           "seams_per_sector": sectors, "seam_line_center_part": r3(Lc[center].sum() / max(Lc[cs].sum(), 1e-9)),
           "uv_shared_part": part(sharing, V.area)}
    out["texels"] = density_texels(V, sizes)
    out["texture"] = measure_texture(V, sizes)
    return out


def area_uv(T):
    a, b = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2


def family_region(r):
    if r in ("arm", "forearm", "hand", "finger"):
        return "arm"
    if r in ("thigh", "leg", "foot"):
        return "leg"
    return "trunk"


def load_image(M, t, cache):
    if t < 0 or t >= len(M.textures):
        return None
    f = os.path.normpath(os.path.join(os.path.dirname(M.path), M.textures[t].replace("\\", "/")))
    if f in cache:
        return cache[f]
    img = None
    if bpy is not None and os.path.exists(f):
        try:
            im = bpy.data.images.load(f)
            w, h = im.size
            px = np.empty(w * h * 4, np.float32)
            im.pixels.foreach_get(px)
            bpy.data.images.remove(im)
            img = {"name": M.textures[t], "w": w, "h": h, "px": px.reshape(h, w, 4)}
        except Exception as e:
            log("unreadable texture", f, e)
    cache[f] = img
    return img


def density_texels(V, cache):
    """px per meter (square root of the area ratio); garment, skin, underneath."""
    M = V.M

    def dens(fi):
        vals, weight = [], []
        for mi in np.unique(M.mf[fi]):
            img = load_image(M, M.mats[mi]["texture"], cache)
            if img is None:
                continue
            ff = fi[M.mf[fi] == mi]
            T = M.uv[M.F[ff]]
            auv = area_uv(T) * img["w"] * img["h"]
            a3 = M.area[ff]
            ok = a3 > 1e-12
            vals.append(np.sqrt(auv[ok] / a3[ok]))
            weight.append(a3[ok])
        if not vals or not sum(len(x) for x in vals):
            return None
        v, p = np.concatenate(vals), np.concatenate(weight)
        o = np.argsort(v)
        c = np.cumsum(p[o]) / p.sum()
        return {"p10": r3(v[o][np.searchsorted(c, 0.1)], 0), "p50": r3(v[o][np.searchsorted(c, 0.5)], 0),
                "p90": r3(v[o][np.searchsorted(c, 0.9)], 0)}
    g = dens(V.fg)
    pe = dens(np.where(V.skin & ~M.faces_role(["empty"]))[0])
    de = dens(V.fd)
    tex = sorted({(c["name"], c["w"], c["h"]) for c in cache.values() if c})
    return {"px_per_m": g, "skin_px_per_m": pe, "under_px_per_m": de,
            "report_skin": r3(g["p50"] / pe["p50"]) if g and pe else None,
            "report_under": r3(g["p50"] / de["p50"]) if g and de else None,
            "textures_views": [list(t) for t in tex]}


def blur(L, r):
    """Square moving average of radius r (cumulative sums), edges repeated."""
    if r < 1:
        return L
    P_ = np.pad(L, r + 1, mode="edge")
    S = P_.cumsum(0).cumsum(1)
    k = 2 * r + 1
    out = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]
    return out[:L.shape[0], :L.shape[1]] / (k * k)


def measure_texture(V, cache):
    """Samples drawn over the garment's UV surface: lightness, painted lines and shadows
    (offset from the local mean), fine pattern (offset from a 2 px blur), transparency."""
    M = V.M
    rng = np.random.default_rng(1)
    out = {"per_material": {}}
    for mi in np.unique(M.mf[V.fg]):
        m = M.mats[mi]
        img = load_image(M, m["texture"], cache)
        ff = V.fg[M.mf[V.fg] == mi]
        info = {"texture": M.textures[m["texture"]] if 0 <= m["texture"] < len(M.textures) else None,
                "sphere": M.textures[m["sphere"]] if 0 <= m.get("sphere", -1) < len(M.textures) else None,
                "sphere_mode": m.get("sphere_mode"),
                "toon": ("toon%02d.bmp (shared)" % (m["toon"] + 1) if m.get("toon_sharing") == 1 else
                         M.textures[m["toon"]] if 0 <= m.get("toon", -1) < len(M.textures) else None),
                "diffuse": r3(m["diffuse"]), "specular": r3(m.get("specular")), "ambient": r3(m.get("ambient"))}
        if img is not None:
            h, w = img["h"], img["w"]
            px = img["px"]
            Lm = MC._L(np.clip(px[..., :3], 0, 1))
            r = max(2, int(round(w / 128)))
            Lb = blur(Lm, r)
            L2 = blur(Lm, 2)
            T = M.uv[M.F[ff]]
            auv = area_uv(T)
            n = 60000
            j = rng.choice(len(ff), n, p=auv / auv.sum()) if auv.sum() > 0 else rng.integers(0, len(ff), n)
            r1, r2 = rng.random(n), rng.random(n)
            sq = np.sqrt(r1)
            bb = np.stack([1 - sq, sq * (1 - r2), sq * r2], 1)
            p = np.einsum("nk,nkd->nd", bb, T[j])
            x = (np.mod(p[:, 0], 1) * w).astype(int).clip(0, w - 1)
            y = ((1 - np.mod(p[:, 1], 1)) * h).astype(int).clip(0, h - 1)
            Ls, Lbs, L2s, a = Lm[y, x], Lb[y, x], L2[y, x], px[y, x, 3]
            info.update({"size": [w, h], "L": q(Ls, (5, 50, 95), 1),
                         "trait_sombre_part": r3((Ls - Lbs < -10).mean()),
                         "reflection_part": r3((Ls - Lbs > 10).mean()),
                         "gap_local_p5_p95": [r3(np.percentile(Ls - Lbs, 5), 1), r3(np.percentile(Ls - Lbs, 95), 1)],
                         "pattern_end_p90": r3(np.percentile(np.abs(Ls - L2s), 90), 2),
                         "transparent_part": r3((a < 0.5).mean()), "radius_blur_px": r})
        out["per_material"][M.names_mat[mi]] = info
    # additional UVs and per-vertex outline on the garment's vertices
    ua = M.uv_add[V.vr] if M.uv_add.shape[1] else None
    if ua is not None:
        out["uv_add"] = [{"min": r3(ua[:, k].min(0)), "max": r3(ua[:, k].max(0)),
                          "constant": bool(np.ptp(ua[:, k], 0).max() < 1e-6)} for k in range(ua.shape[1])]
    else:
        out["uv_add"] = []
    return out


# ====================================================================== D. weights and physics

def measure_weight(V):
    """-> manufacturing: copy the body's weights or not, number of bones per vertex, width of
    the transitions at the joints, the garment's own bones."""
    M = V.M
    vr = V.vr
    W, I = M.weight[vr], M.idx[vr]
    active = (W > 1e-3) & (I >= 0)
    nb = active.sum(1)
    typ = M.type_weight[vr]
    body = V.os_body
    own = np.where(active & ~body[np.maximum(I, 0)], W, 0).sum(1)
    # copied weights: nearest vertex underneath (within 20 mm)
    dr = np.unique(M.F[V.fd]) if len(V.fd) else np.array([], int)
    copied = None
    if len(dr):
        dist, j = MC.kd_plus_near(M.pos[dr], M.pos[vr])
        prox = dist < 0.02
        L1 = []
        for a, b in zip(np.where(prox)[0], dr[j[prox]]):
            wa = {int(i): float(w) for i, w in zip(I[a], W[a]) if i >= 0 and w > 1e-4}
            wb = {int(i): float(w) for i, w in zip(M.idx[b], M.weight[b]) if i >= 0 and w > 1e-4}
            L1.append(sum(abs(wa.get(k, 0) - wb.get(k, 0)) for k in set(wa) | set(wb)))
        L1 = np.array(L1)
        copied = {"vertices_near_part": r3(prox.mean()), "L1": q(L1, (10, 50, 90), 3),
                 "identical_part": r3((L1 < 0.05).mean()) if len(L1) else None,
                 "near_part": r3((L1 < 0.3).mean()) if len(L1) else None}
    # transitions at the joints: width (along the limb) of 0.1 < w_child < 0.9
    trans = {}
    nonv = np.setdiff1d(np.unique(M.F[~M.faces_role(["empty", "head"])]), vr)
    for name, j_os, p_os in JOINTS:
        if j_os not in M.names_os or p_os not in M.names_os:
            continue
        b, pb = M.names_os.index(j_os), M.names_os.index(p_os)
        under = under_tree(M, b)
        ax = M.heads[b] - M.heads[pb]
        ax /= np.linalg.norm(ax)

        def width(vv):
            P_ = M.pos[vv]
            pr = (P_ - M.heads[b]) @ ax
            radial = np.linalg.norm((P_ - M.heads[b]) - np.outer(pr, ax), axis=1)
            near = (np.abs(pr) < 0.12) & (radial < 0.12)
            wc = np.where(np.isin(M.idx[vv], under), M.weight[vv], 0).sum(1)
            t = near & (wc > 0.1) & (wc < 0.9)
            if t.sum() < 5:
                return None
            return {"vertices": int(t.sum()), "width_mm": r3((np.percentile(pr[t], 95) - np.percentile(pr[t], 5)) * 1000, 1),
                    "center_mm": r3(np.median(pr[t]) * 1000, 1)}
        trans[name] = {"garment": width(vr), "rest_of_model": width(nonv)}
    return {"influences": {str(k): int((nb == k).sum()) for k in range(1, 5)},
            "influences_avg": r3(nb.mean(), 2),
            "types": {PMX.TYPES_WEIGHT[int(k)]: int((typ == k).sum()) for k in np.unique(typ)},
            "weight_own_bones": {"part_vertices": r3((own > 1e-3).mean()), "average": r3(own.mean())},
            "copy_of_under": copied, "transitions": trans}


def under_tree(M, b):
    out, pile = [], [b]
    children = {}
    for i, p in enumerate(M.per):
        children.setdefault(int(p), []).append(i)
    while pile:
        x = pile.pop()
        out.append(x)
        pile += children.get(x, [])
    return out


def measure_physics(V):
    """-> manufacturing: bone chains of the free panels, rigid bodies (shape, mass, damping),
    joints, groups and collisions with the legs. Units: mm (rigid bodies read in MMD units × 80)."""
    M = V.M
    O, RG, J = M.p["bones"], M.rigid_bodies, M.joints
    vr = V.vr
    W, I = M.weight[vr], M.idx[vr]
    worn = np.unique(I[(W > 0.05) & (I >= 0)])
    own_ones = [int(b) for b in worn if not V.os_body[b]]
    if not own_ones:
        return {"own_bones": 0}
    sp = set(own_ones)
    # extend to non-body descendants (leaf bones without weights are part of the chain)
    children = {}
    for i, o in enumerate(O):
        children.setdefault(o["parent"], []).append(i)
    all_ = set(sp)
    pile = list(sp)
    while pile:
        x = pile.pop()
        for e in children.get(x, []):
            if not V.os_body[e] and e not in all_:
                all_.add(e)
                pile.append(e)
    # walk up to non-body ancestors
    for b in list(all_):
        p = O[b]["parent"]
        while p >= 0 and not V.os_body[p] and p not in all_:
            all_.add(p)
            p = O[p]["parent"]
    roots = [b for b in all_ if O[b]["parent"] < 0 or O[b]["parent"] not in all_]

    def plus_long(i):
        ch = [i]
        while True:
            e = [x for x in children.get(ch[-1], []) if x in all_]
            if not e:
                return ch
            ch.append(max(e, key=lambda x: len(plus_long(x))))
    rig_os = {}
    for k, r in enumerate(RG):
        rig_os.setdefault(r["bones"], []).append(k)
    rg_v = [k for b in all_ for k in rig_os.get(b, [])]
    dyn = [k for k in rg_v if RG[k]["mode"] in RIG_DYN]
    legs = [k for k, r in enumerate(RG) if r["bones"] >= 0 and r["mode"] == 0 and V.os_body[r["bones"]]
              and M.region_os[r["bones"]] in ("thigh", "leg", "pelvis", "buttocks")]
    trunk = [k for k, r in enumerate(RG) if r["bones"] >= 0 and r["mode"] == 0 and V.os_body[r["bones"]]
             and M.region_os[r["bones"]] in ("trunk_top", "belly", "chest", "arm", "forearm")]
    seg = {k: segment_rigid(RG[k]) for k in set(dyn) | set(legs) | set(trunk)}
    E_ = 80.0  # mm per MMD unit

    def contacts(target):
        hits, rec = [], []
        for k in dyn:
            a, b, ra = seg[k]
            for kk in target:
                a2, b2, r2 = seg[kk]
                dd = (dist_segments(a, b, a2, b2) - ra - r2) * E_
                if collide(RG[k], RG[kk]):
                    hits.append(kk)
                    rec.append(dd)
        return hits, rec
    hj, recj = contacts(legs)
    ht, rect = contacts(trunk)
    sd = set(dyn)
    jts = [j for j in J if j["a"] in sd or j["b"] in sd]
    # cross joints: between two dynamic bodies of different chains
    chain_of = {}
    chains = []
    for r0 in sorted(roots):
        ch = plus_long(r0)
        under = under_tree_set(r0, children, all_)
        for b in under:
            for k in rig_os.get(b, []):
                chain_of[k] = r0
        rr = [RG[rig_os[b][0]] if rig_os.get(b) else None for b in ch]
        pos = np.array([O[b]["pos"] for b in ch])
        L = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() * E_ if len(ch) > 1 else 0.0
        chains.append({"root": M.names_os[r0], "parent": M.names_os[O[r0]["parent"]] if O[r0]["parent"] >= 0 else None,
                        "chain_links": len(ch), "bones": len(under), "length_mm": r3(L, 1),
                        "modes": [r["mode"] if r else None for r in rr],
                        "masses": r3([r["mass"] if r else None for r in rr]),
                        "damping_disp": r3([r["damping_disp"] if r else None for r in rr]),
                        "damping_rot": r3([r["damping_rot"] if r else None for r in rr])})
    transverse = [j for j in jts if j["a"] in chain_of and j["b"] in chain_of and chain_of[j["a"]] != chain_of[j["b"]]]
    long_ = [j for j in jts if j not in transverse]

    def limits(js):
        if not js:
            return None
        lo = np.degrees(np.array([j["rot_min"] for j in js]))
        hi = np.degrees(np.array([j["rot_max"] for j in js]))
        amp = hi - lo
        disp = np.array([np.subtract(j["disp_max"], j["disp_min"]) for j in js]) * E_
        return {"n": len(js), "rot_min_deg_med": r3(np.median(lo, 0), 1), "rot_max_deg_med": r3(np.median(hi, 0), 1),
                "amplitude_deg_p90": r3(np.percentile(amp, 90, axis=0), 1),
                "blocked_part": r3((amp.max(1) < 0.01).mean()),
                "translation_free_part": r3((disp.max(1) > 1e-3).mean()),
                "spring_rot": sorted({tuple(r3(j["spring_rot"], 1)) for j in js})[:4],
                "spring_disp": sorted({tuple(r3(j["spring_disp"], 1)) for j in js})[:4]}
    groups = {}
    for k in dyn:
        groups[RG[k]["group"]] = groups.get(RG[k]["group"], 0) + 1
    rd = [RG[k] for k in dyn]
    return {
        "own_bones": len(all_), "bones_carrying_weight": len(own_ones), "chains": len(chains),
        "chain_links": q([c["chain_links"] for c in chains], (10, 50, 90), 1),
        "length_chain_mm": q([c["length_mm"] for c in chains], (10, 50, 90), 1),
        "rigid_bodies": len(rg_v), "rigid_bodies_dynamic": len(dyn),
        "modes": {str(m): sum(1 for k in rg_v if RG[k]["mode"] == m) for m in (0, 1, 2)},
        "shapes": {PMX.SHAPES_RIGID[f]: sum(1 for r in rd if r["shape"] == f) for f in (0, 1, 2)},
        "size_mm": q([max(r["size"]) * E_ for r in rd], (10, 50, 90), 1) if rd else None,
        "radius_mm": q([segment_rigid(r)[2] * E_ for r in rd], (10, 50, 90), 1) if rd else None,
        "mass": q([r["mass"] for r in rd], (10, 50, 90)) if rd else None,
        "damping_disp": q([r["damping_disp"] for r in rd], (10, 50, 90)) if rd else None,
        "damping_rot": q([r["damping_rot"] for r in rd], (10, 50, 90)) if rd else None,
        "restitution": sorted({r3(r["restitution"]) for r in rd}) if rd else None,
        "friction": sorted({r3(r["friction"]) for r in rd}) if rd else None,
        "groups": groups,
        "ignore_son_group_part": r3(np.mean([not ((r["mask"] >> r["group"]) & 1) for r in rd])) if rd else None,
        "ignores": sorted({tuple(ignores(r)) for r in rd})[:4] if rd else None,
        "rigid_bodies_legs": len(legs),
        "hits_legs_part": r3(np.mean([any(collide(RG[k], RG[kk]) for kk in legs) for k in dyn])) if dyn and legs else None,
        "hits_trunk_arm_part": r3(np.mean([any(collide(RG[k], RG[kk]) for kk in trunk) for k in dyn])) if dyn and trunk else None,
        "overlap_rest_legs_mm": q(recj, (0, 10, 50), 1) if recj else None,
        "overlap_rest_trunk_mm": q(rect, (0, 10, 50), 1) if rect else None,
        "joints_long": limits(long_), "joints_transverses": limits(transverse),
        "detail_chains": chains[:8]}


def under_tree_set(r0, children, all_):
    out, pile = [], [r0]
    while pile:
        x = pile.pop()
        out.append(x)
        pile += [e for e in children.get(x, []) if e in all_]
    return out


# ====================================================================== overlay layers

def measure_texture_layer(V):
    """"+" material: same geometry (shadow, pattern or outline overlay) or lining?"""
    M = V.M
    fc = np.where(V.texture_layer)[0]
    if not len(fc):
        return None
    Fc = M.Fs[fc]
    cg = set(map(tuple, np.sort(V.Fgs, 1).tolist()))
    same = np.array([tuple(f) in cg for f in np.sort(Fc, 1).tolist()])
    vv = np.unique(M.F[fc])
    d = []
    for p in M.pos[vv]:
        h = V.tree.find_nearest(Vector(p))
        d.append(h[3] if h[0] is not None else np.inf)
    d = np.array(d)
    cen = M.pos[M.F[fc]].mean(1)
    nc = M.nf[fc]
    # orientation relative to the garment: same direction or flipped (lining)
    direction = []
    for c_, n_ in zip(cen, nc):
        h = V.tree.find_nearest(Vector(c_))
        if h[0] is not None:
            direction.append(float(np.dot(n_, np.asarray(h[1]))))
    direction = np.array(direction)
    mats = sorted({int(i) for i in np.unique(M.mf[fc])})
    return {"faces": int(len(fc)), "same_faces_part": r3(same.mean()),
            "distance_mm": qmm(d[np.isfinite(d)], (50, 90, 100)),
            "same_direction_part": r3((direction > 0.5).mean()) if len(direction) else None,
            "flips_part": r3((direction < -0.5).mean()) if len(direction) else None,
            "materials": [{"name": M.names_mat[i], "texture": M.textures[M.mats[i]["texture"]] if 0 <= M.mats[i]["texture"] < len(M.textures) else None,
                           "diffuse": r3(M.mats[i]["diffuse"]), "flags": M.mats[i]["flags"],
                           "contour": bool(M.mats[i]["flags"] & 0x10)} for i in mats]}


# ====================================================================== checks

TIGHT = ("top_tight", "trousers", "shorts", "painted_on_body")


def check(r):
    """Guardrails and contracts of a garment, thresholds
    taken from the 10 official ones. Returns a list of {name, value, threshold, ok}; ok None = not applicable.
    -> manufacturing: a "no" on a guardrail makes the garment wrong; on a type contract
    (tight / loose), it departs from every official garment of that type."""
    m, a, u, p, ph = r["mesh"], r["adjustment"], r["uv"], r["weights"], r["physics"]
    C = []

    def c(name, val, threshold, ok):
        C.append({"name": name, "value": val, "threshold": threshold, "ok": None if val is None else bool(ok)})
    # guardrails (10/10 on the official ones, except named exceptions)
    c("non_manifold", m["edges_non_manifold"], "0", m["edges_non_manifold"] == 0)
    rec = r.get("texture_layer") or {}
    c("double_sided", bool(all(m["double_face_flag"]) or (rec.get("flips_part") or 0) > 0.9),
      "flag 0x01 or flipped overlay", all(m["double_face_flag"]) or (rec.get("flips_part") or 0) > 0.9)
    sizes = [x["size"] for x in _g(m, "contour", "materials") or [] if x["trace"]]
    c("outline_size", min(sizes) if sizes else 0.0, "0.5-0.6 on every material",
      len(sizes) == len(m["contour"]["materials"]) and all(0.45 <= t <= 0.65 for t in sizes))
    c("outline_vertex_p1", _g(m, "contour", "vertex", "p1"), "1.0", _g(m, "contour", "vertex", "p1") == 1.0)
    c("model_share_budget", m["part_faces_model"], "<= 0.19", m["part_faces_model"] <= 0.19)
    c("max_4_influences", max(int(k) for k, v in p["influences"].items() if v), "<= 4",
      max(int(k) for k, v in p["influences"].items() if v) <= 4)
    c("no_sdef", "SDEF" not in p["types"] or p["types"]["SDEF"] == 0, "no SDEF",
      "SDEF" not in p["types"] or p["types"]["SDEF"] == 0)
    c("texels_on_skin", _g(u, "texels", "report_skin"), "0.65-1.6",
      _g(u, "texels", "report_skin") is not None and 0.65 <= u["texels"]["report_skin"] <= 1.6)
    c("hidden_skin", _g(a, "cache", "area_hidden_skin_on_area_garment"), "<= 0.16",
      (_g(a, "cache", "area_hidden_skin_on_area_garment") or 0) <= 0.16)
    c("rest_crossing", _g(a, "rest_crossing", "fraction"), "<= 0.031",
      (_g(a, "rest_crossing", "fraction") or 0) <= 0.031)
    c("rest_crossing_p90_mm", _g(a, "rest_crossing", "depth_mm", "p90"), "<= 9",
      (_g(a, "rest_crossing", "depth_mm", "p90") or 0) <= 9)
    cs = _g(a, "poses", "track_body_alone_max_part")
    c("body_only_poses", cs, "<= 0.10 (target 0.01)", cs is not None and cs <= 0.10)
    if ph.get("own_bones"):
        c("phys_ignores_own_group", ph.get("ignore_son_group_part"), "1", ph.get("ignore_son_group_part") == 1)
        c("phys_hits_legs", ph.get("hits_legs_part"), "1", ph.get("hits_legs_part") == 1)
        jl = ph.get("joints_long") or {}
        c("phys_twist_locked", _g(jl, "rot_max_deg_med"), "Y = 0 (median of the chain joints)",
          bool(jl) and jl["rot_min_deg_med"][1] == 0 and jl["rot_max_deg_med"][1] == 0)
        c("phys_zero_springs", _g(jl, "spring_rot"), "0", bool(jl) and all(v == 0 for t in jl["spring_rot"] for v in t))
        c("phys_zero_bounce", ph.get("restitution"), "0", ph.get("restitution") == [0.0])
    # contracts per family
    cl = _g(a, "gap", "classes_part") or {}
    if r["type"] in TIGHT:
        c("tight_nothing_under", cl.get("nothing_under"), ">= 0.65", (cl.get("nothing_under") or 0) >= 0.65)
        c("tight_welded_edge", _g(a, "borders", "welded"), ">= 1 loop welded to the skin",
          (_g(a, "borders", "welded") or 0) >= 1)
        c("tight_folds_p90_mm", _g(m, "folds_geometric", "gap_smooth_abs_mm", "p90"), "<= 4",
          (_g(m, "folds_geometric", "gap_smooth_abs_mm", "p90") or 0) <= 4)
        c("tight_without_own_bones", ph.get("own_bones", 0), "0", ph.get("own_bones", 0) == 0)
    else:
        e50 = _g(a, "gap", "gap_mm", "p50")
        c("loose_gap_p50_mm", e50, "2-45", e50 is not None and 2 <= e50 <= 45)
        c("loose_free_edge", _g(a, "borders", "free"), ">= 1 free loop", (_g(a, "borders", "free") or 0) >= 1)
        c("loose_normals_split_on_seams", _g(m, "normals", "cuts_on_seam_uv_part"),
          ">= 0.99 if split", (_g(m, "normals", "cuts_on_seam_uv_part") or 1) >= 0.99)
    return C


# ====================================================================== main

class WithoutObj(Exception):
    """The measurement has nothing to measure: every piece of the part is set aside (rigid
    accessory, small object). This is not a failure of the part."""


def measure(spec, M, poses=True):
    t0 = time.time()
    V = Garment(M, spec)
    if not len(V.fg):
        raise WithoutObj("no piece kept (%s)" %", ".join(sorted({str(d.get("reason")) for d in V.discarded}))[:120])
    out = {"id": spec["id"], "character": spec["character"], "type": spec["type"], "why": spec["why"],
           "materials": spec["mats"], "texture_layers": spec.get("texture_layers", []),
           "isolation": {"parts_kept": [{k: d[k] for k in ("index", "faces", "region", "diag_mm", "closed", "os_dominants")}
                                            for d in V.kept],
                         "parts_discarded": [{k: d[k] for k in ("index", "faces", "region", "diag_mm", "closed", "os_dominants", "reason")}
                                             for d in V.discarded]},
           "layers_over": V.layers_over}
    log(spec["id"], "mesh")
    out["mesh"] = measure_mesh(V)
    log(spec["id"], "fit")
    out["adjustment"] = measure_adjustment(V, poses)
    log(spec["id"], "uv")
    out["uv"] = measure_uv(V)
    log(spec["id"], "weights")
    out["weights"] = measure_weight(V)
    out["physics"] = measure_physics(V)
    out["texture_layer"] = measure_texture_layer(V)
    out["checks"] = check(out)
    out["duration_s"] = r3(time.time() - t0, 1)
    return out


def _g(d, *ch):
    for c in ch:
        if d is None:
            return None
        d = d.get(c) if isinstance(d, dict) else None
    return d


def array(results, path):
    L = ["# ZZZ garments: overview table", "",
         "One row per garment. mm; shares between 0 and 1. See <id>.json for details.", "",
         "| id | type | faces | parts | boundary edges | facing | edge / skin | folds p90 mm | smoothed normals | vertex outline p50 | nothing underneath | stuck <1mm | offset p50 mm | clipping at rest | worst pose (tracked) | worst pose (body only) | UV islands | texels / skin | dark line | mean influences | body copy | own bones | dyn rigid bodies |",
         "|" + "---|" * 23]
    for r in results:
        m, a, u, p, ph = r["mesh"], r["adjustment"], r["uv"], r["weights"], r["physics"]
        tex = list(_g(u, "texture", "per_material").values())
        L.append("|" + " %s |" * 23 % (
            r["id"], r["type"], m["faces"], m["parts"], m["border_edges"], _g(m, "thickness", "part_faces_in_vis_a_vis"),
            _g(m, "density", "report_edge_skin"), _g(m, "folds_geometric", "gap_smooth_abs_mm", "p90"),
            _g(m, "normals", "plus_near_of_smooth_part"), _g(m, "contour", "vertex", "p50"),
            _g(a, "gap", "classes_part", "nothing_under"), _g(a, "gap", "classes_part", "glue_lt1mm"),
            _g(a, "gap", "gap_mm", "p50"), _g(a, "rest_crossing", "fraction"), _g(a, "poses", "track_max_part"),
            _g(a, "poses", "track_body_alone_max_part"), u["islands"], _g(u, "texels", "report_skin"), tex[0].get("trait_sombre_part") if tex else None,
            p["influences_avg"], _g(p, "copy_of_under", "identical_part"), ph.get("own_bones"),
            ph.get("rigid_bodies_dynamic")))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(tmp, path)


def folder_output(arg):
    if arg:
        return arg
    d = os.path.join(MD.R.path("paths.data"), "clothes")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    if os.path.isdir(d) and os.access(d, os.W_OK):
        return d
    return os.path.join(os.environ.get("TMPDIR", "/tmp"), "clothes")


def main(argv):
    output = folder_output(argv[argv.index("--output") + 1] if "--output" in argv else None)
    os.makedirs(output, exist_ok=True)
    alone = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    poses = "--no-poses" not in argv
    models = {}
    res = []
    specs = CLOTHES
    if "--pmx" in argv:
        # manufactured candidate: --pmx PATH --mats A,B [--id NAME] [--type T] [--layers X,Y]
        path = argv[argv.index("--pmx") + 1]
        val = lambda o, d=None: argv[argv.index(o) + 1] if o in argv else d  # noqa: E731
        name = val("--id", os.path.splitext(os.path.basename(path))[0])
        models[name] = MD.Model(name, path=path)
        specs = [{"id": name, "character": name, "type": val("--type", "loose_top"), "mats": val("--mats").split(","),
                  "texture_layers": val("--layers", "").split(",") if val("--layers") else [],
                  "why": "manufactured candidate (%s)" % path}]
        alone = [name]
    for spec in specs:
        if alone and spec["id"] not in alone:
            continue
        M = models.get(spec["character"]) or MD.model_of(spec["character"], spec.get("pmx"))
        models[spec["character"]] = M
        r = measure(spec, M, poses)
        write_json(os.path.join(output, spec["id"] + ".json"), r)
        log(spec["id"], "written (%.0f s)" % r["duration_s"])
        res.append(r)
        bad = [x for x in r["checks"] if x["ok"] is False]
        log(spec["id"], "checks: %d/%d passed%s" % (
            sum(1 for x in r["checks"] if x["ok"]), sum(1 for x in r["checks"] if x["ok"] is not None),
            "".join("\n    FAIL %s = %s (threshold %s)" % (x["name"], x["value"], x["threshold"]) for x in bad)))
    if not alone:
        write_json(os.path.join(output, "ensemble.json"), {"date": time.strftime("%Y-%m-%d"), "clothes": res})
        array(res, os.path.join(output, "tableau.md"))
    else:
        array(res, os.path.join(output, "partial_table.md"))
    print("outputs:", output)


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(a)
