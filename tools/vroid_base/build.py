"""Builds Chimera's base (bare body, base head, hairstyle) from a VRoid Studio character.

    python3 -I tools/vroid_base/build.py <file.vrm | name of an imported model> [--output <folder>] [--no-sheets]

 1. converts the .vrm (VRM 0.x or 1.0) into a PMX in the common language (tools/vroid/importer.py), in
    <output>/model/<Name>/, keeping all of the body skin (the ordinary import removes the faces that the
    texture erases: VRoid erases the skin under the outfit); a bare name: <the game's sources.vrm>/<name>.vrm if it
    exists, otherwise the already imported PMX (<mmd_extracted>/<name>/<name>.pmx);
 2. repaints the body skin (skin.py): VRoid paints tights, socks, a tight top and underwear on it, or
    makes it transparent under the outfit; whatever is not skin is filled from the neighbouring skin;
 3. gives a shared normal to the split vertices of the skin seams, then writes <output>/body.pmx (the skin alone, 肌: the base body, setting workshop.body), head.pmx (the same skin,
    vertices in the same order, plus the face, its eyes and its expressions) and hairstyle.pmx (the model's
    hair), with their textures. The model's outfit is not included: it enters the catalogue like that of any
    imported character;
 4. measures the head (head.Head: distance between the eyes and their midpoint) and stores in <output>/base.json what the "vroid" game
    reads of its base head (base_head: name, first_pass, eye_gap, eyes_center) and the display names;
 5. checks (checks.json): humanoid bones with MMD names, arms-down pose, skin closed below the head, opaque skin
    texture, eyes found; a failed blocking check stops everything (exit code 1);
 6. writes the provenance and the terms of use read from the file (VRM meta): provenance.json, PROVENANCE.md;
 7. renders grey contact sheets (without Blender): corps.png, head.png.
By default, output in {data}/base-vroid. Without Blender; a few seconds.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path[:0] = [HERE, TOOLS] + [os.path.join(TOOLS, d) for d in ("vroid", "atlas", "head", "rebuilt_head", "rebuilt_body")]
import settings as R          # noqa: E402
import importer as IM         # noqa: E402
import pmx as PMX             # noqa: E402
import pmx_write as pmx_write             # noqa: E402
import geom as G              # noqa: E402
import images                 # noqa: E402
import png                    # noqa: E402
import skin as PE             # noqa: E402

SKIN, FACE_SKIN = "肌", "颜"
FACE = ("颜", "口舌", "齿", "眉", "睫", "二重", "白目", "目", "目光", "目光2")
HAIR = re.compile(r"^髪\d*$")
OS_REQUIRED = ("全ての親", "センター", "下半身", "上半身", "首", "頭", "左肩", "左腕", "左ひじ", "左手首", "右肩", "右腕",
             "右ひじ", "右手首", "左足", "左ひざ", "左足首", "右足", "右ひざ", "右足首")
ARM = (40.0, 56.0)          # degrees below the horizontal of the bone 腕 -> ひじ: Chimera's A-pose (47-50°)
HOLE_MAX = 6                 # vertices: a smaller opening in the skin is not a hole (seam of the model)
WELD = 0.01               # MMD units: two vertices closer than this are the same point (seams, neck cut)
SKIN_KEPT_MIN = 0.10       # share of the body texture still skin: below it, almost everything is repainted
CM = 8.0
META = ("title", "version", "author", "contactInformation", "reference", "allowedUserName", "violentUssageName",
        "sexualUssageName", "commercialUssageName", "otherPermissionUrl", "licenseName", "otherLicenseUrl")
META_1 = ("name", "version", "authors", "copyrightInformation", "contactInformation", "references", "thirdPartyLicenses",
          "licenseUrl", "avatarPermission", "allowExcessivelyViolentUsage", "allowExcessivelySexualUsage", "commercialUsage",
          "allowPoliticalOrReligiousUsage", "allowAntisocialOrHateUsage", "creditNotation", "allowRedistribution",
          "modification", "otherLicenseUrl")


# ------------------------------------------------------------------ source
def source(arg, game=None):
    """(vrm or None, already imported pmx or None, name) from the argument: a .vrm, or the name of an imported model."""
    if arg.lower().endswith(".vrm"):
        if not os.path.isfile(arg):
            raise SystemExit("VRoid base: %s does not exist" % arg)
        return os.path.abspath(arg), None, os.path.splitext(os.path.basename(arg))[0]
    if game is None:
        import games as games
        game = games.load("vroid")
    vrm = os.path.join(game.path("vrm"), arg + ".vrm")
    if os.path.isfile(vrm):
        return vrm, None, arg
    pmx = os.path.join(game.path("mmd_extracted"), arg, arg + ".pmx")
    if os.path.isfile(pmx):
        return None, pmx, arg
    raise SystemExit("VRoid base: neither %s nor %s. Give the path of a .vrm file exported from VRoid Studio or taken from VRoid Hub (VRM 0.x or 1.0)." % (vrm, pmx))


def vrm_meta(vrm):
    """Terms of use declared in the file, under their own keys: VRM 0.x (extensions.VRM.meta) or 1.0
    (extensions.VRMC_vrm.meta, other names and values). "title" is the character's name in both."""
    j, _ = IM.read_glb(vrm)
    ext = j.get("extensions", {})
    if "VRMC_vrm" in ext:
        m = ext["VRMC_vrm"].get("meta") or {}
        r = {"vrm": "1.0"}
        r.update({k: m.get(k) for k in META_1})
        r["title"] = m.get("name")
        return r
    m = (ext.get("VRM") or {}).get("meta", {})
    r = {"vrm": "0.x"}
    r.update({k: m.get(k) for k in META})
    return r


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ PMX
def under_pmx(P, T, mats, first=()):
    """PMX reduced to the materials `mats` (names), all bones kept. The vertices `premiers` come first, in
    that order, then the other vertices used, in the original order. Returns (pmx, old index -> new)."""
    names = [m["name"] for m in P["materials"]]
    guard = [i for i, n in enumerate(names) if n in mats]
    sel = np.isin(T["mat_face"], guard)
    useful = np.unique(T["faces"][sel])
    first = np.asarray(first, np.int64)
    order = np.concatenate([first, np.setdiff1d(useful, first)])
    nv = -np.ones(len(T["pos"]), np.int64)
    nv[order] = np.arange(len(order))
    s = P["vertices"]
    vert = {k: [s[k][i] for i in order] for k in ("pos", "nor", "uv", "uv_add", "type", "bones", "weights", "sdef", "contour")}
    faces, materials, textures, tex_nv = [], [], [], {}
    for i in guard:
        F = T["faces"][T["mat_face"] == i]
        if not len(F):
            continue
        m = dict(P["materials"][i])
        t = m["texture"]
        if 0 <= t < len(P["textures"]):
            if t not in tex_nv:
                tex_nv[t] = len(textures)
                textures.append(P["textures"][t])
            m["texture"] = tex_nv[t]
        m["n_indices"] = 3 * len(F)
        materials.append(m)
        faces += [tuple(int(x) for x in f) for f in nv[F]]
    morphs = []
    for mo in P["morphs"]:
        if mo["type"] != 1:
            continue
        d = [(int(nv[a]), v) for a, v in mo["offsets"] if nv[a] >= 0]
        if d:
            morphs.append(dict(mo, offsets=d))
    Q = dict(P, vertices=vert, faces=faces, materials=materials, textures=textures, morphs=morphs, rigid_bodies=[], joints=[],
             frames=[{"name": "Root", "name_en": "Root", "special": 1, "elements": [(0, 0)]},
                     {"name": "表情", "name_en": "Exp", "special": 1, "elements": [(1, k) for k in range(len(morphs))]}])
    return Q, nv


def write(Q, folder_src, output, name, replaces=None):
    """Writes `name` (pmx) into `sortie` with its textures, copied from `dossier_src`; `replaces`: {old: new}."""
    replaces = replaces or {}
    Q = dict(Q, textures=[replaces.get(t, t) for t in Q["textures"]])
    for t in Q["textures"]:
        dest = os.path.join(output, t)
        if t not in replaces.values() and not os.path.isfile(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(os.path.join(folder_src, t), dest)
    pmx_write.write(Q, os.path.join(output, name + ".tmp.pmx"))
    os.replace(os.path.join(output, name + ".tmp.pmx"), os.path.join(output, name))


def borders_of_seams(T, i_mat, weld=WELD, stage=6):
    """Points of the skin seams: (uv_a, uv_b), two places in the texture for the same point of the model.
    The vertices split at the same spot, and `pas` points along each seam edge."""
    F = T["faces"][T["mat_face"] == i_mat]
    idx = np.unique(F)
    pos, uv = np.asarray(T["pos"]), np.asarray(T["uv"], float)
    _, inv = np.unique(np.round(pos[idx] / weld).astype(np.int64), axis=0, return_inverse=True)
    point = -np.ones(len(pos), np.int64)
    point[idx] = inv.ravel()
    # edge by its two welded points -> its real edges (vertices)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    the_key = np.sort(point[E], 1)
    order = np.lexsort((the_key[:, 1], the_key[:, 0]))
    the_key, E = the_key[order], E[order]
    ua, ub = [], []
    start = np.concatenate([[True], (the_key[1:] != the_key[:-1]).any(1)])
    groups = np.split(np.arange(len(E)), np.where(start)[0][1:])
    t = (np.arange(stage) + 0.5) / stage
    for g in groups:
        if len(g) < 2:
            continue
        a = E[g[0]]
        if point[a[0]] != the_key[g[0], 0]:
            a = a[::-1]
        for j in g[1:]:
            b = E[j]
            if point[b[0]] != the_key[j, 0]:
                b = b[::-1]
            if np.abs(uv[a] - uv[b]).max() < 1e-4:
                continue                              # the same edge seen from both triangles: not a seam
            ua.append(uv[a[0]] + t[:, None] * (uv[a[1]] - uv[a[0]]))
            ub.append(uv[b[0]] + t[:, None] * (uv[b[1]] - uv[b[0]]))
    if not ua:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(ua), np.concatenate(ub)


def weld_normals(P, T, i_mat, weld=WELD):
    """The skin vertices split at the same point (UV seams: ankles, wrists…) take the average of
    their normals. VRoid leaves them separate: under a sock or a glove the break could not be seen; on
    bare skin, toon shading draws a line there. Returns the number of vertices changed."""
    idx = np.unique(T["faces"][T["mat_face"] == i_mat])
    pos = np.asarray(T["pos"])[idx]
    nor = np.asarray(P["vertices"]["nor"], float)[idx]
    _, inv, cpt = np.unique(np.round(pos / weld).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    total = np.zeros((len(cpt), 3))
    np.add.at(total, inv, nor)
    avg = total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-9)
    change = (cpt[inv] > 1) & (np.abs(avg[inv] - nor).max(1) > 1e-4)
    for i in np.where(change)[0]:
        P["vertices"]["nor"][idx[i]] = tuple(float(x) for x in avg[inv[i]])
    return int(change.sum())


def flip_diagonals(P, T, i_mat):
    """The skin's shared edges swapped where they cross a fold instead of following it (geom.flip_diagonals):
    under the breasts, in the armpits and at the elbows the VRoid mesh has diagonals across the crease, which grazing
    light and toon shadows draw as small steps. Vertices and normals do not move; P and T are changed in place, before
    under_pmx, so that body.pmx and head.pmx keep the same body faces. Returns the number of swaps."""
    sel = np.flatnonzero(T["mat_face"] == i_mat)
    F, n = G.flip_diagonals(np.asarray(T["pos"], float), T["faces"][sel])
    T["faces"][sel] = F
    for k, f in zip(sel, F):
        P["faces"][k] = [int(x) for x in f]
    return n


# ------------------------------------------------------------------ checks
def openings(pos, F, weld=WELD):
    """Free edges of the surface (vertices welded within `weld`): [(number of vertices, centre)] from largest to smallest."""
    _, inv = np.unique(np.round(np.asarray(pos) / weld).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    W = inv[np.asarray(F)]
    W = W[(W[:, 0] != W[:, 1]) & (W[:, 1] != W[:, 2]) & (W[:, 0] != W[:, 2])]
    E = np.sort(np.concatenate([W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    border = u[c == 1]
    per = {}

    def root(x):
        while per.setdefault(x, x) != x:
            per[x] = per[per[x]]
            x = per[x]
        return x
    for a, b in border:
        per[root(a)] = root(b)
    groups = {}
    for a, b in border:
        groups.setdefault(root(a), set()).update((int(a), int(b)))
    rep = np.zeros(inv.max() + 1, np.int64)
    rep[inv] = np.arange(len(inv))
    res = [(len(g), np.asarray(pos)[rep[list(g)]].mean(0)) for g in groups.values()]
    return sorted(res, key=lambda x: -x[0])


def angle_arm(bones):
    p = {o["name"]: np.asarray(o["pos"], float) for o in bones}
    res = []
    for c in "左右":
        d = p[c + "ひじ"] - p[c + "腕"]
        res.append(float(np.degrees(np.arctan2(-d[1], abs(d[0])))))
    return res


def check(P_corps, T_corps, fraction_skin, head, alpha_min):
    """{name: {passed, value, blocking}} of the base's checks."""
    names_os = {o["name"] for o in P_corps["bones"]}
    missing = [n for n in OS_REQUIRED if n not in names_os]
    c = {"humanoid_bones": {"passed": not missing, "blocking": True, "value": missing or "all"}}
    if not missing:
        a = angle_arm(P_corps["bones"])
        c["arm_pose_deg"] = {"passed": all(ARM[0] <= x <= ARM[1] for x in a), "blocking": True, "value": [round(x, 1) for x in a]}
        y_neck = next(o["pos"][1] for o in P_corps["bones"] if o["name"] == "首")
        holes = [(n, ctr) for n, ctr in openings(T_corps["pos"], T_corps["faces"]) if ctr[1] < y_neck and n > HOLE_MAX]
        c["closed_skin"] = {"passed": not holes, "blocking": True,
                            "value": [{"vertices": n, "center_cm": [round(float(x) * CM, 1) for x in ctr]} for n, ctr in holes] or "no hole below the neck"}
    c["opaque_skin"] = {"passed": alpha_min >= 0.999, "blocking": True, "value": round(float(alpha_min), 3)}
    c["original_skin"] = {"passed": fraction_skin >= SKIN_KEPT_MIN, "blocking": False, "value": round(fraction_skin, 3),
                           "note": "share of the body texture kept as it is; the rest is repainted from the neighbouring skin"}
    c["height_cm"] = {"passed": True, "blocking": False, "value": round(float(np.ptp(T_corps["pos"][:, 1])) * CM, 1)}
    c["eyes"] = {"passed": head is not None and not isinstance(head, str), "blocking": True,
                 "value": head if isinstance(head, str) else ("distance %.4f" % head.iod if head else "not measured")}
    for v in c.values():
        v["passed"] = bool(v["passed"])
    return c


# ------------------------------------------------------------------ provenance
def origin(meta, src, name, sha):
    pj = {"model": name, "file": os.path.basename(src), "sha256": sha, "vrm_meta": meta,
          "tool": "tools/vroid_base/build.py"}
    warnings = []
    if meta and meta.get("vrm") == "1.0":   # an absent key takes the VRM 1.0 default, the strictest one
        if meta.get("avatarPermission") != "everyone":
            warnings.append("the file restricts the use of the avatar (avatarPermission = %s)" % meta.get("avatarPermission"))
        if meta.get("allowRedistribution") is not True:
            warnings.append("the file prohibits redistribution (allowRedistribution = false)")
        if meta.get("commercialUsage") in (None, "personalNonProfit"):
            warnings.append("the file prohibits commercial use (commercialUsage = personalNonProfit)")
        if meta.get("modification") in (None, "prohibited"):
            warnings.append("the file prohibits modification (modification = prohibited), and the base is one")
    elif meta:
        if meta.get("allowedUserName") not in (None, "Everyone"):
            warnings.append("the file restricts the use of the avatar (allowedUserName = %s)" % meta.get("allowedUserName"))
        if meta.get("licenseName") in ("Redistribution_Prohibited",):
            warnings.append("the file prohibits redistribution (licenseName = Redistribution_Prohibited)")
        if meta.get("commercialUssageName") == "Disallow":
            warnings.append("the file prohibits commercial use")
    pj["warnings"] = warnings
    lines = ["# Provenance of the VRoid base", "",
              "Built by `tools/vroid_base/build.py` from `%s` (sha256 %s), character \"%s\"."
              % (os.path.basename(src), sha[:16], name), ""]
    if meta:
        keys = META_1 if meta.get("vrm") == "1.0" else META
        lines += ["Terms of use declared in the file (VRM %s meta):" % meta.get("vrm", "0.x"), ""]
        lines += ["- `%s` : %s" % (k, meta[k]) for k in keys if meta.get(k) not in (None, "", [])]
    else:
        lines += ["No .vrm file: terms of use not read. See the comment of the imported PMX."]
    lines += ["", "These terms are those of the model's author; they apply to the base and to every model "
               "assembled on it (body, face and hairstyle come from it)."]
    if warnings:
        lines += ["", "**Warning**: " + "; ".join(warnings) + "."]
    return pj, "\n".join(lines) + "\n"


# ------------------------------------------------------------------ fabrication
def build(arg, output, sheets=True, log=print):
    vrm, pmx_ingested, name = source(arg)
    os.makedirs(output, exist_ok=True)
    prev = _read(os.path.join(output, "provenance.json"))
    if vrm:
        folder = os.path.join(output, "model", name)
        shutil.rmtree(folder, ignore_errors=True)
        log("— converting %s (body skin kept whole)" % vrm)
        IM.convert(vrm, folder, log=log, keep={SKIN})
        path = os.path.join(folder, name + ".pmx")
    else:
        log("— %s: no .vrm, the imported PMX is taken as it is (the skin that its texture erases may be missing)" % pmx_ingested)
        path = pmx_ingested
    src = vrm or pmx_ingested
    sha = fingerprint(src)
    if prev and prev.get("sha256") != sha:
        log("Note: %s held the base of another model (%s). Chimera's cache is only valid for one body: "
                "give it a new one (setting workshop.cache)." % (output, prev.get("model")))
    folder_src = os.path.dirname(path)
    P = PMX.read(path)
    T = PMX.to_arrays(P)
    names = [m["name"] for m in P["materials"]]
    for required in (SKIN, FACE_SKIN):
        if required not in names:
            raise SystemExit("VRoid base: %s has no material %s (body or face skin): is it a VRoid Studio export?" % (path, required))
    i_skin = names.index(SKIN)
    # --- repainted skin
    log("— body skin")
    tex = lambda name_mat: os.path.join(folder_src, P["textures"][P["materials"][names.index(name_mat)]["texture"]])   # noqa: E731
    own, known = PE.clean(images.read(tex(SKIN)), images.read(tex(FACE_SKIN)))
    part = float(known.mean())
    uv_a, uv_b = borders_of_seams(T, i_skin)
    own = PE.match_seams(own, known, uv_a, uv_b)
    log("texture seams: %d points matched" % len(uv_a))
    png.write(os.path.join(output, "skin.png"), (own * 255).round().clip(0, 255).astype(np.uint8))
    log("skin kept as it is: %.0f %% of the texture, the rest repainted" % (100 * part))
    old = P["textures"][P["materials"][i_skin]["texture"]]
    n_seams = weld_normals(P, T, i_skin)
    log("skin seams: %d split vertices get a shared normal" % n_seams)
    n_diag = flip_diagonals(P, T, i_skin)
    log("skin diagonals: %d swapped to follow the folds" % n_diag)
    # --- body, head, hairstyle
    body, nv = under_pmx(P, T, {SKIN})
    body["name"], body["name_en"] = "corps " + name, "body"
    write(body, folder_src, output, "body.pmx", {old: "skin.png"})
    first = np.where(nv >= 0)[0][np.argsort(nv[nv >= 0])]     # the body vertices, in the order of corps.pmx
    head, _ = under_pmx(P, T, {SKIN} | set(FACE), first)
    head["name"], head["name_en"] = "head " + name, "head"
    write(head, folder_src, output, "head.pmx", {old: "skin.png"})
    mats_hair = {n for n in names if HAIR.match(n)}
    if not mats_hair:
        log("Note: no hair (material 髪): no base hairstyle")
    else:
        hairstyle, _ = under_pmx(P, T, mats_hair)
        hairstyle["name"], hairstyle["name_en"] = "coiffure " + name, "hair"
        write(hairstyle, folder_src, output, "hairstyle.pmx")
    # --- head measurements
    Pc = PMX.read(os.path.join(output, "body.pmx"))
    Tc = PMX.to_arrays(Pc)
    try:
        import head as TE
        t = TE.Head(os.path.join(output, "head.pmx"))
        center = (np.asarray(t.center_eye["g"]) + np.asarray(t.center_eye["d"])) / 2
        measure = t
    except Exception as e:                     # noqa: BLE001  (an unreadable head is a failed check, not a crash)
        measure, center = "%s: %s" % (type(e).__name__, e), None
    alpha = images.read(os.path.join(output, "skin.png"))[..., 3].min()
    ctl = check(Pc, Tc, part, measure, alpha)
    meta = vrm_meta(vrm) if vrm else None
    title = (meta or {}).get("title") or name
    base = {"model": name, "title": title,
            "base_head": {"name": name, "first_pass": name}}
    if center is not None:
        base["base_head"].update({"eye_gap": round(float(measure.iod), 4), "eyes_center": [round(float(x), 4) for x in center]})
    _write_json(os.path.join(output, "base.json"), base)
    _write_json(os.path.join(output, "checks.json"), ctl)
    pj, md = origin(meta, src, name, sha)
    _write_json(os.path.join(output, "provenance.json"), pj)
    with open(os.path.join(output, "PROVENANCE.md"), "w", encoding="utf-8") as f:
        f.write(md)
    for a in pj["warnings"]:
        log("WARNING: " + a)
    if sheets:
        look(output, Tc, PMX.read(os.path.join(output, "head.pmx")))
    rates = [k for k, v in ctl.items() if not v["passed"]]
    for k in rates:
        log("check %s %s: %s" % ("FAILED" if ctl[k]["blocking"] else "to look at", k, ctl[k]["value"]))
    return ctl, base


def look(output, Tc, Pt):
    """Grey contact sheets: corps.png (front, three-quarter, profile, back), head.png (close-up)."""
    import view as VU
    views = [VU.VIEWS[k] for k in ("face", "three_quarter", "profile", "back")]
    VU.sheet(Tc["pos"] * CM, Tc["faces"], views, os.path.join(output, "body.png"), T=700)
    Tt = PMX.to_arrays(Pt)
    h = next(np.asarray(o["pos"]) * CM for o in Pt["bones"] if o["name"] == "頭")
    VU.sheet(Tt["pos"] * CM, Tt["faces"], views, os.path.join(output, "head.png"), T=700,
               box=[-14, h[1] - 12, -14, 14, h[1] + 20, 14], thread=True)


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path, d):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(path + ".tmp", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help=".vrm file (VRM 0.x or 1.0, VRoid Studio or VRoid Hub), or name of an already imported VRoid model")
    p.add_argument("--output", dest="output", default=os.path.join(R.path("paths.data"), "vroid-base"))
    p.add_argument("--no-sheets", dest="without_sheets", action="store_true")
    a = p.parse_args(argv)
    ctl, base = build(a.model, a.output, sheets=not a.without_sheets)
    if any(not v["passed"] and v["blocking"] for v in ctl.values()):
        print("VRoid base: a blocking check failed (see %s); this base must not be used." % os.path.join(a.output, "checks.json"))
        return 1
    body = os.path.join(a.output, "body.pmx")
    print("base body: %s (with head.pmx and hairstyle.pmx: face and hairstyle of \"%s\")" % (body, base["title"]))
    if os.path.realpath(body) != os.path.realpath(R.path("workshop.body")):
        print("Chimera reads another body (%s). For this one: \"atelier\": {\"corps\": \"%s\", \"cache\": <a cache of its own>} "
              "in settings.json" % (R.path("workshop.body"), body))
    else:
        print("Next: python3 -I tools/workshop/catalog.py prepare, then bin/chimera")
    print("Its outfit in the catalogue: python3 -I tools/vroid/importer.py <file.vrm>, then the libraries (COMPATIBILITY.md, VRoid).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
