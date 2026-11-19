"""Graft a head with its hair (face, hair, physics, morphs) onto a PMX body with the same skeleton.

    bin/blender-queue -b --factory-startup --python tools/clothes/head_graft.py -- \
        --body BODY.pmx --head HEAD.pmx --output FOLDER
        [--remove 颜,白目,…] [--take 颜,颜_A,…,髮] [--skin 肌] [--seam-mm 0] [--no-tint] [--keep-skull]
    (Blender to read and write the tint textures; without tint, python3 -I is enough.)

When to use it: the body and the head's model share the same upper skeleton (here, the neck and
head bones differ by a plain translation); retarget_head covers the other cases.

 - the head is placed by the translation that brings its 頭 bone onto the body's;
 - bones: a bone with the same name is shared, the others (hair) are added with their parents;
 - rigid bodies: those of the added bones, and those of the source attached to the upper body, which
   act as colliders for the hair; joints: those whose two rigid bodies are kept;
 - vertex morphs: carried over (other types are dropped, they refer to parts of the source
   model);
 - seam: ZZZ heads cover the body's skin (its edge runs 15-20 mm under the face);
   nothing to stitch, unless --seam-mm > 0: the nearby skin edge is then snapped onto the
   head's edge;
 - neck: the body skin that rises above the neck (scalp of a skin-completed body) is
   removed where the head's original model cut its own, as on a game model
   (`game_neck.py`), and the head brings its ears; --seam to weld it to the face edge instead (`neck_seam.py`),
   --keep-skull to leave it untouched;
 - tint: the body skin is matched in Lab to the head's, measured on both sides of the seam
   (lightness and hue shift, texture rewritten into <sortie>/tex).

Output: <sortie>/result.pmx, textures in <sortie>/tex (relative paths), graft.json.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
sys.path.insert(0, os.path.join(HERE, "..", "head"))
sys.path.insert(0, HERE)
import pmx as PMX  # noqa: E402
import pmx_write as pmx_write  # noqa: E402
import game_neck as neck_of_game  # noqa: E402
import neck_seam as join_neck  # noqa: E402

MM = 1.0 / 80.0
HEAD_DEFAULT = ("颜", "颜_A", "睫眉", "白目", "目", "目光", "目影", "眉", "睫", "颜2", "齿", "口线", "口舌", "髮")
REMOVE_DEFAULT = ("颜", "白目", "目", "目光", "二重", "睫", "眉", "口线", "口舌", "齿", "目影", "髮")
ISLAND_NECK = "肌_cou"  # ears brought by the head (taken from the top of its neck)
TOP_OF_BODY = ("頭", "首", "上半身3", "上半身2", "上半身", "左肩", "右肩", "左腕", "右腕", "左胸", "右胸")


def args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", dest="body", required=True)
    p.add_argument("--head", dest="head", required=True)
    p.add_argument("--output", dest="output", required=True)
    p.add_argument("--remove", dest="remove", default=",".join(REMOVE_DEFAULT), help="body materials that are removed")
    p.add_argument("--take", dest="take", default=",".join(HEAD_DEFAULT), help="materials taken from the head")
    p.add_argument("--skin", dest="skin", default="肌", help="body skin, whose edge is stitched to the head")
    p.add_argument("--head-skin", dest="skin_head", default="颜", help="head skin that touches the body")
    p.add_argument("--seam-mm", dest="seam_mm", type=float, default=0.0)
    p.add_argument("--max-neck-gap", dest="neck_gap_max", type=float, default=6.0,
                   help="tolerated offset (mm), once the head is placed, between the body's bones and those the grafted "
                        "vertices hang on (the neck for a whole head; 0 for hair on 頭 alone)")
    p.add_argument("--follow-bones", dest="follow_bones", action="store_true",
                   help="the taken vertices also follow the offset of the body's bones they hang on (hair resting on "
                        "the shoulders of a body with a longer neck); for parts that are not stitched to the body skin")
    p.add_argument("--no-tint", dest="without_tint", action="store_true")
    p.add_argument("--keep-skull", dest="keep_skull", action="store_true", help="leave the body skin under the head untouched")
    p.add_argument("--seam", dest="join", action="store_true",
                   help="weld the body skin to the face edge (neck seam) instead of cutting it at the neck as the game does")
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return p.parse_args(argv)


def weld(pos, tol=1e-4):
    """Welded index of each vertex (positions equal within tol)."""
    _, inv = np.unique(np.round(pos / tol).astype(np.int64), axis=0, return_inverse=True)
    return inv.ravel()


def edges_of_border(F, sub):
    """Edges (raw indices) that touch only one face, once the vertices are welded."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    S = np.sort(sub[E], axis=1)
    _, inv, cnt = np.unique(S, axis=0, return_inverse=True, return_counts=True)
    return E[cnt[inv.ravel()] == 1]


def point_segment(q, a, b):
    """Closest point of each q on the best segment (a, b): (point, index)."""
    ab = b - a
    t = np.clip(((q[:, None] - a) * ab).sum(2) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    pts = a + t[..., None] * ab
    d = ((q[:, None] - pts) ** 2).sum(2)
    k = d.argmin(1)
    return pts[np.arange(len(q)), k], k, np.sqrt(d[np.arange(len(q)), k])


def offsets_os(osB, osH, idxB, t):
    """Per head bone, offset (model units) between the body's bone and the head's once the head is moved
    by t. A head bone the body lacks (hair, ribbon) is added under the body's copy of its nearest shared
    ancestor: it takes that ancestor's offset."""
    ec = np.zeros((len(osH), 3))
    for i in range(len(osH)):
        j = i
        while j >= 0 and osH[j]["name"] not in idxB:
            j = osH[j]["parent"]
        if j >= 0:
            ec[i] = np.array(osB[idxB[osH[j]["name"]]]["pos"]) - np.array(osH[j]["pos"]) - t
    return ec


def gap_skeleton(ec, os_v, weight_v):
    """Rest offset vector of grafted vertices: the skinning-weighted offset of the bones they hang on
    (ec from offsets_os). It is how far the body's bones sit from where the vertex's skinning expects them."""
    os_v, w = np.asarray(os_v), np.asarray(weight_v, float)
    w = np.where(os_v >= 0, w, 0.0)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return (w[..., None] * ec[np.maximum(os_v, 0)]).sum(1)


def copy_texture(src, folder, made, names):
    if src in made:
        return made[src]
    base = os.path.basename(src)
    name, k = base, 0
    while name in names:
        k += 1
        name = "%d_%s" % (k, base)
    names.add(name)
    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(src, os.path.join(folder, name))
    made[src] = "tex/" + name
    return made[src]


def path_texture(pmx_path, rel):
    return os.path.normpath(os.path.join(os.path.dirname(pmx_path), rel.replace("\\", "/")))


def main(argv=None):
    a = args(argv)
    os.makedirs(a.output, exist_ok=True)
    B = PMX.read(a.body)
    H = PMX.read(a.head)
    tB, tH = PMX.to_arrays(B), PMX.to_arrays(H)
    osB = {o["name"]: i for i, o in enumerate(B["bones"])}
    osH = {o["name"]: i for i, o in enumerate(H["bones"])}
    t = np.array(B["bones"][osB["頭"]]["pos"]) - np.array(H["bones"][osH["頭"]]["pos"])
    print("head translation (mm):", np.round(t / MM, 2))
    gap_neck = (np.array(B["bones"][osB["首"]]["pos"]) - np.array(H["bones"][osH["首"]]["pos"]) - t) / MM
    print("neck offset once the head is placed (mm):", np.round(gap_neck, 2))

    remove = set(a.remove.split(","))
    take = [n for n in a.take.split(",")]
    namesB = [m["name"] for m in B["materials"]]
    namesH = [m["name"] for m in H["materials"]]
    missing = [n for n in take if n not in namesH]
    if missing:
        raise SystemExit("materials missing from the head: %s" % missing)

    # --- kept body vertices and faces
    keep_f = ~np.isin(tB["mat_face"], [i for i, n in enumerate(namesB) if n in remove])
    # nothing removed: the body keeps all its vertices, even those no face uses, and so their indices
    vB = np.arange(len(tB["pos"])) if keep_f.all() else np.unique(tB["faces"][keep_f])
    mapB = np.full(len(tB["pos"]), -1)
    mapB[vB] = np.arange(len(vB))
    # --- head vertices and faces
    iH = [namesH.index(n) for n in take]
    fH_m = np.isin(tH["mat_face"], iH)
    # the head's ears and scalp (what it carries alone or nearly so), taken when the body is cut at the
    # neck as in the game
    island_h = np.zeros(len(tH["faces"]), bool)
    if not a.keep_skull and not a.join:
        candidates_skin = [namesH.index(n) for n in (a.skin, "首", "体") if n in namesH]
        if candidates_skin:
            head_h = np.array(H["bones"][osH["頭"]]["pos"])
            vis_h = [namesH.index(n) for n in (a.skin_head, a.skin_head + "_A") if n in namesH]
            os_h = neck_of_game.os_of_head(H["bones"], osH["頭"])
            island_h = neck_of_game.island_of_neck(tH["pos"], tH["faces"], tH["mat_face"], candidates_skin, vis_h,
                                          (head_h[0], head_h[2]), tH["bones"], tH["weights"], os_h, tH["uv"])
            # and its scalp skin, which closes the head under the hair: in the ears' material, whose
            # texture it takes
            island_h |= neck_of_game.skull_of_game(tH["pos"], tH["nor"], tH["faces"], tH["mat_face"],
                                             [int(tH["mat_face"][island_h][0])] if island_h.any() else candidates_skin,
                                             vis_h, (head_h[0], head_h[2]), tH["bones"], tH["weights"], os_h)
    fH_m |= island_h
    vH = np.unique(tH["faces"][fH_m])
    # skeleton check on what is grafted: each taken vertex is moved by t (head placed by 頭) but then hangs on
    # the body's bones. Its offset is how far those bones sit from where its own skinning expects them.
    # A whole head (neck skin on 首) measures the neck as before; hair that hangs on 頭 alone measures 0,
    # whatever the body's neck length (our own shape has a longer neck than the game heads).
    offset_os = offsets_os(B["bones"], H["bones"], osB, t)
    offset_v = gap_skeleton(offset_os, tH["bones"][vH], tH["weights"][vH])
    gap_os = np.linalg.norm(offset_v, axis=1) / MM
    worst = int(np.argmax(gap_os)) if len(gap_os) else -1
    gap_graft = float(gap_os[worst]) if worst >= 0 else 0.0
    print("skeleton offset under the grafted vertices (mm): max %.2f" % gap_graft)
    if a.follow_bones:
        # each taken vertex (and added bone) also takes its bones' offset: hair resting on the shoulders follows
        # the body's shoulders, hair on 頭 alone does not move; nothing is left to tolerate
        print("vertices moved onto the body's bones (mm): max %.2f, %d vertices moved by more than 1 mm"
              % (gap_graft, int((gap_os > 1.0).sum())))
    else:
        offset_os[:] = 0.0
        offset_v[:] = 0.0
    if gap_graft > a.neck_gap_max and not a.follow_bones:
        heavy = tH["bones"][vH][worst][np.argmax(tH["weights"][vH][worst])]
        raise SystemExit("neck too different: a grafted vertex is %.1f mm from where its bones expect it (bone %s; "
                         "limit %.1f mm): this body and this head do not share the same skeleton, see retarget_head"
                         % (gap_graft, H["bones"][int(heavy)]["name"], a.neck_gap_max))
    mapH = np.full(len(tH["pos"]), -1)
    mapH[vH] = len(vB) + np.arange(len(vH))

    # --- bones
    useful = set(int(b) for b, w in zip(tH["bones"][vH].ravel(), tH["weights"][vH].ravel()) if b >= 0 and w > 0)
    for b in list(useful):
        x = H["bones"][b]["parent"]
        while x >= 0 and x not in useful:
            useful.add(int(x))
            x = H["bones"][x]["parent"]
    new = [i for i in sorted(useful) if H["bones"][i]["name"] not in osB]
    map_os = {}
    for i in sorted(useful):
        n = H["bones"][i]["name"]
        if n in osB:
            map_os[i] = osB[n]
    bones = [dict(o) for o in B["bones"]]
    for i in new:
        map_os[i] = len(bones)
        bones.append(dict(H["bones"][i]))
    for i in new:
        o = bones[map_os[i]]
        o["pos"] = tuple(np.array(o["pos"]) + t + offset_os[i])
        o["parent"] = map_os.get(o["parent"], osB["頭"])
        if o["tail_bone"] is not None:
            q = o["tail_bone"]
            if q in map_os:
                o["tail_bone"] = map_os[q]
            else:
                o["tail_bone"] = -1
        if o["addition"] is not None:
            aj = dict(o["addition"])
            if aj["source"] in map_os:
                aj["source"] = map_os[aj["source"]]
                o["addition"] = aj
            else:
                o["addition"] = None
                o["flags"] &= ~(PMX.OS_ADDITION_ROT | PMX.OS_ADDITION_DISP | PMX.OS_ADDITION_LOCAL)
        if o["ik"] is not None:
            raise SystemExit("IK bone in the head: not supported (%s)" % o["name"])
    print("bones: %d shared, %d added" % (len(useful) - len(new), len(new)))

    # --- vertex tables
    nadd = max(B["n_uv_add"], H["n_uv_add"])
    sB, sH = B["vertices"], H["vertices"]

    def uvadd(s, i, n0):
        u = list(s["uv_add"][i]) if n0 else []
        return u + [(0.0, 0.0, 0.0, 0.0)] * (nadd - n0)
    out = {k: [] for k in sB}
    for i in vB.tolist():
        for k in ("pos", "nor", "uv", "type", "weights", "sdef", "contour"):
            out[k].append(sB[k][i])
        out["bones"].append(sB["bones"][i])
        out["uv_add"].append(uvadd(sB, i, B["n_uv_add"]))
    for k_v, i in enumerate(vH.tolist()):
        p = sH["pos"][i]
        tt = (t + offset_v[k_v]).tolist()     # dec_v is zero unless --follow-bones
        out["pos"].append((p[0] + tt[0], p[1] + tt[1], p[2] + tt[2]))
        for k in ("nor", "uv", "type", "weights", "contour"):
            out[k].append(sH[k][i])
        c = sH["sdef"][i]
        out["sdef"].append(None if c is None else tuple(np.array(c) + np.tile(tt, 3)))
        out["bones"].append(tuple(map_os[int(b)] if b >= 0 else -1 for b in sH["bones"][i]))
        out["uv_add"].append(uvadd(sH, i, H["n_uv_add"]))
    P = np.array(out["pos"], float)
    N = np.array(out["nor"], float)

    # --- faces and materials
    back = os.path.join(a.output, "tex")
    made, names_tex = {}, set()
    textures, fabrics, faces = [], [], []
    ind_tex = {}

    def tex(pmx_ch, pmx, i):
        if i is None or i < 0:
            return -1
        path = path_texture(pmx_ch, pmx["textures"][i])
        if not os.path.exists(path):
            print("WARNING texture not found:", path)
            r = pmx["textures"][i]
        else:
            r = copy_texture(path, back, made, names_tex)
        if r not in ind_tex:
            ind_tex[r] = len(textures)
            textures.append(r)
        return ind_tex[r]

    def addition_mat(pmx_ch, pmx, im, Fm, map_):
        m = dict(pmx["materials"][im])
        m["texture"] = tex(pmx_ch, pmx, m["texture"])
        m["sphere"] = tex(pmx_ch, pmx, m["sphere"])
        if not m.get("toon_sharing"):
            m["toon"] = tex(pmx_ch, pmx, m["toon"])
        m["n_indices"] = int(3 * len(Fm))
        fabrics.append(m)
        faces.append(map_[Fm])

    for im, n in enumerate(namesB):
        if n in remove:
            continue
        addition_mat(a.body, B, im, tB["faces"][tB["mat_face"] == im], mapB)
    n_fB = sum(len(f) for f in faces)
    for im in iH:
        addition_mat(a.head, H, im, tH["faces"][tH["mat_face"] == im], mapH)
    if island_h.any():
        im_island = int(tH["mat_face"][island_h][0])
        addition_mat(a.head, H, im_island, tH["faces"][island_h], mapH)
        fabrics[-1]["name"] = ISLAND_NECK
    Ft = np.vstack(faces)

    # --- neck seam: body skin edge snapped onto the head skin edge
    i_skin = [m["name"] for m in fabrics].index(a.skin)
    names_m = [m["name"] for m in fabrics]
    i_neck = names_m.index(a.skin_head if a.skin_head in names_m else a.skin)   # single-skin model: nothing to stitch
    disp = np.cumsum([0] + [m["n_indices"] // 3 for m in fabrics])
    F_skin, F_neck = Ft[disp[i_skin]:disp[i_skin + 1]], Ft[disp[i_neck]:disp[i_neck + 1]]
    sub = weld(P)
    border_head = edges_of_border(F_neck, sub)
    border_body = edges_of_border(F_skin, sub)
    vc = np.unique(border_body)
    pa, pb = P[border_head[:, 0]], P[border_head[:, 1]]
    y_min = B["bones"][osB["首"]]["pos"][1] + 20 * MM        # (a body with closed skin has no edge: empty vc)
    vc = vc[P[vc, 1] > y_min]
    if not len(border_head):                               # head with closed skin too: no edge to follow
        vc = vc[:0]
        pa = pb = np.zeros((1, 3))
    pts, k, d = point_segment(P[vc], pa, pb)
    taken = d < a.seam_mm * MM if a.seam_mm > 0 else np.zeros(len(d), bool)
    print("seam: %d skin edge vertices above the neck, %d snapped; offset mm median %.1f, max %.1f, unsnapped max %.1f"
          % (len(vc), int(taken.sum()), float(np.median(d[taken]) / MM) if taken.any() else 0.0,
             float(d[taken].max() / MM) if taken.any() else 0.0,
             float(d[~taken].max() / MM) if (~taken).any() else 0.0))
    for v_, p_, k_ in zip(vc[taken].tolist(), pts[taken], k[taken].tolist()):
        n_ = N[border_head[k_, 0]] + N[border_head[k_, 1]]
        n_ = n_ / max(np.linalg.norm(n_), 1e-9)
        for x in np.where(sub == sub[v_])[0].tolist():
            if x < len(vB):
                out["pos"][x] = tuple(p_.tolist())
                out["nor"][x] = tuple(n_.tolist())

    # --- rigid bodies and joints
    kept = {}
    rigs = []
    for ir, r in enumerate(H["rigid_bodies"]):
        if r["bones"] < 0 or (r["bones"] not in map_os and H["bones"][r["bones"]]["name"] not in osB):
            continue
        name_os = H["bones"][r["bones"]]["name"]
        added = r["bones"] in map_os and name_os not in osB
        top = name_os in TOP_OF_BODY and r["mode"] == 0
        if not (added or top):
            continue
        c = dict(r)
        c["bones"] = map_os[r["bones"]] if r["bones"] in map_os else osB[name_os]
        c["pos"] = tuple(np.array(r["pos"]) + t)
        kept[ir] = len(rigs)
        rigs.append(c)
    jts = []
    for j in H["joints"]:
        if j["a"] in kept and j["b"] in kept:
            c = dict(j)
            c["a"], c["b"] = kept[j["a"]], kept[j["b"]]
            c["pos"] = tuple(np.array(j["pos"]) + t)
            jts.append(c)
    print("physics: %d rigid bodies, %d joints kept" % (len(rigs), len(jts)))

    # --- vertex morphs
    morphs = []
    for m in H["morphs"]:
        if m["type"] != 1:
            continue
        offset = [(int(mapH[i]), d) for i, d in m["offsets"] if mapH[i] >= 0]
        if offset:
            c = dict(m)
            c["offsets"] = offset
            morphs.append(c)
    print("morphs: %d kept" % len(morphs))

    # --- tint: body skin matched to the head's, on both sides of the seam
    tint = None
    if not a.without_tint:
        tint = adjust_tint(a, out, fabrics, textures, Ft, disp, i_skin, i_neck)

    # --- face-neck join: done after seam and tint, which read the original skin edge
    skull = None
    if not a.keep_skull:
        names_m = [m["name"] for m in fabrics]
        i_vis = [names_m.index(n) for n in (a.skin_head, a.skin_head + "_A") if n in names_m]
        mat_face = np.repeat(np.arange(len(fabrics)), np.diff(disp))
        n_front = len(Ft)
        if a.join:
            Ft, skull = join_neck.apply(out, Ft, mat_face, fabrics, morphs, i_skin, i_vis,
                                              folder=None if a.without_tint else a.output, textures=textures)
        else:
            # the donor head's neck, where its own body stopped
            # (without the ears or the scalp taken, which would widen and raise the reference neck), measured over its whole surface.
            # Without the clothes carried by the skin material: a neck skin vertex only follows bones that
            # the common body also has. Nicole's collar (bone Bn_LINGZI, in 体, up to half of the weight)
            # gave the body's neck a 55 mm ruff and a second jaw line; likewise the
            # collars, necklaces, ties, badges and pendants of the others.
            f_neck = np.isin(tH["mat_face"], [namesH.index(n) for n in (a.skin, "首", "体") if n in namesH]) & ~island_h
            v_neck = np.unique(tH["faces"][f_neck])
            o_v, p_v = tH["bones"][v_neck], tH["weights"][v_neck]
            of_body = np.isin(o_v, [i for i, o in enumerate(H["bones"]) if o["name"] in osB]) & (o_v >= 0)
            guard = np.zeros(len(tH["pos"]), bool)
            guard[v_neck[(p_v * of_body).sum(1) >= 0.99 * p_v.sum(1)]] = True
            # Nor the top of the shoulders: the neck wraps around the axis, the top of the shoulder faces up.
            F_neck = tH["faces"][f_neck]
            F_neck = F_neck[guard[F_neck].all(1) & (tH["nor"][F_neck, 1].mean(1) < 0.5)]
            neck_h = neck_of_game.surface(tH["pos"] + t, F_neck)
            head_os = np.array(B["bones"][osB["頭"]]["pos"])
            i_island = names_m.index(ISLAND_NECK) if ISLAND_NECK in names_m else None
            Ft, skull = neck_of_game.apply(out, Ft, mat_face, fabrics, i_skin, i_vis, neck_h, (head_os[0], head_os[2]), i_island,
                                             neck_of_game.os_of_head(B["bones"], osB["頭"]))
        n_fB -= n_front - len(Ft)
        print("face-neck join:", skull)

    R = dict(B)
    R["n_uv_add"] = nadd
    R["vertices"] = out
    R["faces"] = Ft.tolist()
    R["textures"] = textures
    R["materials"] = fabrics
    R["bones"] = bones
    R["morphs"] = morphs
    R["rigid_bodies"] = rigs
    R["joints"] = jts
    R["frames"] = [dict(c, elements=[(ty, x) for ty, x in c["elements"] if ty == 0 and x < len(B["bones"])])
                   for c in B["frames"]]
    path = os.path.join(a.output, "result.pmx")
    tmp = path + ".tmp"
    pmx_write.write(R, tmp)
    os.replace(tmp, path)
    info = {"body": a.body, "head": a.head, "translation_mm": [round(float(x / MM), 2) for x in t],
            "neck_gap_mm": [round(float(x), 2) for x in gap_neck], "graft_gap_mm": round(gap_graft, 2), "body_faces": int(n_fB), "head_faces": int(len(Ft) - n_fB),
            "vertices": len(out["pos"]), "added_bones": len(new), "rigid_bodies": len(rigs), "joints": len(jts),
            "morphs": len(morphs), "tint": tint, "skull": skull}
    with open(os.path.join(a.output, "graft.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print("written:", path, "%d vertices, %d faces" % (len(out["pos"]), len(Ft)))


# ---------------------------------------------------------------------- tint

def srgb_to_lab(c):
    c = np.asarray(c, float)
    l = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    x = (l @ M.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(x > 0.008856, np.cbrt(x), 7.787 * x + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    x = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    l = np.clip(x @ M.T, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


def read_image(path):
    """sRGB RGBA, rows top to bottom (bpy: Blender's Python lacks Pillow)."""
    import bpy
    im = bpy.data.images.load(path, check_existing=False)
    im.colorspace_settings.name = "Non-Color"
    w, h = im.size
    px = np.empty(4 * w * h, np.float32)
    im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    return px.reshape(h, w, 4)[::-1].copy()


def write_image(path, px):
    import bpy
    h, w = px.shape[:2]
    im = bpy.data.images.new("tmp", w, h, alpha=True)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(np.clip(px[::-1], 0, 1).astype(np.float32).ravel())
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)


def adjust_tint(a, out, fabrics, textures, Ft, disp, i_skin, i_neck):
    """Lab median of the skin within 25 mm of the seam, on each side."""
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("tint: must run in Blender (texture reading); nothing changed")
        return None
    P = np.array(out["pos"], float)
    UV = np.array(out["uv"], float)
    Fp = Ft[disp[i_skin]:disp[i_skin + 1]]
    Fc = Ft[disp[i_neck]:disp[i_neck + 1]]
    cp, cc = P[Fp].mean(1), P[Fc].mean(1)
    def neighbors(X, Y, r):
        tree = None
        d = np.full(len(X), np.inf)
        for k in range(0, len(Y), 200):
            dd = np.sqrt(((X[:, None] - Y[k:k + 200][None]) ** 2).sum(2)).min(1)
            d = np.minimum(d, dd)
        del tree
        return d < r
    r = 25 * MM
    mp = neighbors(cp, cc, r)
    mc = neighbors(cc, cp, r)

    def lab_of(F_, m, itex):
        img = read_image(os.path.join(a.output, textures[itex]))
        h, w = img.shape[:2]
        u = np.mod(UV[F_[m]].mean(1), 1.0)
        px = img[(u[:, 1] * h).astype(int).clip(0, h - 1), (u[:, 0] * w).astype(int).clip(0, w - 1)]
        return srgb_to_lab(px[:, :3])
    Lp = np.median(lab_of(Fp, mp, fabrics[i_skin]["texture"]), 0)
    Lc = np.median(lab_of(Fc, mc, fabrics[i_neck]["texture"]), 0)
    d = Lc - Lp
    print("seam tint: body Lab %s, head Lab %s, ΔE %.1f" % (np.round(Lp, 1), np.round(Lc, 1), float(np.linalg.norm(d))))
    if np.linalg.norm(d) < 1.5:
        return {"body": Lp.round(1).tolist(), "head": Lc.round(1).tolist(), "applied": False}
    ch = os.path.join(a.output, textures[fabrics[i_skin]["texture"]])
    im = read_image(ch)
    lab = srgb_to_lab(im[..., :3])
    lab = lab + d
    rgb = lab_to_srgb(lab)
    res = np.dstack([rgb, im[..., 3]])
    tmp = ch + ".tmp.png"
    write_image(tmp, res)
    os.replace(tmp, ch)
    return {"body": Lp.round(1).tolist(), "head": Lc.round(1).tolist(), "applied": True}


if __name__ == "__main__":
    main()
