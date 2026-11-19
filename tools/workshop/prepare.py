"""Preparing the cache (Blender process): body, heads, hairstyles, parts.

    bin/blender-queue -b --factory-startup --python tools/workshop/prepare.py -- --travail LIST.json

LIST.json: [{"kind": "body"|"head"|"hairstyle"|"build"|"part"|"covers", ...}, ...]. Each item
writes its meta.json record (with "kept" and "reason" if it is rejected); an error on one
item does not stop the run. Not run by hand: `catalog.py prepare` splits the work.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import glb as GLB  # noqa: E402

import bpy  # noqa: E402

import pmx as PMX  # noqa: E402  (via common's sys.path)
import pmx_write as pmx_write  # noqa: E402

# checks of measure.py that reject a part; the others do not apply on the bare body or
# come from the source (see the library notes) and stay recorded in the record
BLOCKING = ("under_skin", "rest_crossing", "rest_crossing_p90_mm", "body_only_poses", "foot_in_shoe")
FOOT_SUNK_MAX = 0.015  # metres: a shoe lowered further onto its sole to let the foot through it does not
                          # hold the foot, which stands above it (a bare foot over a crushed-looking shoe)
UNDER_SKIN_MAX = 0.01   # share of vertices more than 1 mm inside the body (wear.under_skin, no depth limit)
SIZE_TEXTURE = 1024


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ textures
def _scene_webp():
    s = bpy.context.scene
    s.render.image_settings.file_format = "WEBP"
    s.render.image_settings.color_mode = "RGBA"
    s.render.image_settings.quality = 85
    s.view_settings.view_transform = "Standard"
    s.view_settings.look = "None"
    return s


def texture(path):
    """Converts an image to a shared WebP in the cache; returns its URL (or None if missing)."""
    if not path or not os.path.isfile(path):
        return None
    n = C.fingerprint_file(path) + ".webp"
    dest = os.path.join(C.CACHE, "tex", n)
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        im = bpy.data.images.load(path, check_existing=False)
        try:
            w, h = im.size
            if max(w, h) > SIZE_TEXTURE:
                k = SIZE_TEXTURE / max(w, h)
                im.scale(max(1, int(w * k)), max(1, int(h * k)))
            tmp = dest + ".tmp.webp"
            im.save_render(tmp, scene=_scene_webp())
            os.replace(tmp, dest)
        finally:
            bpy.data.images.remove(im)
    return "/data/tex/" + n


def path_tex(pmx_path, rel):
    return os.path.normpath(os.path.join(os.path.dirname(pmx_path), rel.replace("\\", "/")))


def mat_glb(P, m, pmx_path, role):
    t = m["texture"]
    url = texture(path_tex(pmx_path, P["textures"][t])) if 0 <= t < len(P["textures"]) else None
    return {"name": m["name"], "image": url, "diffuse": list(m["diffuse"]), "role": role,
            "contour": list(m["outline_color"]) + [float(m["outline_size"])],
            "with_outline": bool(m["flags"] & 0x10), "double": True}


def extract_glb(pmx_path, names_mats, output, role="fabric", T=None, P=None):
    """Chosen materials of a PMX -> GLB. Returns (vertex count, face count, materials)."""
    P = P or PMX.read(pmx_path)
    T = T or PMX.to_arrays(P)
    names = [m["name"] for m in P["materials"]]
    guard = [i for i, n in enumerate(names) if n in names_mats and P["materials"][i]["n_indices"] > 0]
    sel = np.isin(T["mat_face"], guard)
    F = T["faces"][sel]
    mf = T["mat_face"][sel]
    vs, inv = np.unique(F, return_inverse=True)
    Fl = inv.reshape(-1, 3)
    pos, nor, Fl = GLB.to_gltf(T["pos"][vs], T["nor"][vs], Fl)
    prims, mats = [], []
    for i in guard:
        prims.append({"faces": Fl[mf == i], "material": len(mats)})
        mats.append(mat_glb(P, P["materials"][i], pmx_path, role))
    GLB.write(output, pos, nor, T["uv"][vs], prims, mats)
    return len(pos), len(Fl), [m["name"] for m in mats]


def write_foot(P, names_mats, pose, meta, d):
    """A shoe worn on a posed foot (porter, foot.py): its pose in meta.json, and per vertex of piece.glb
    the weights that pose it again (pied.bin: float32 ankle weight, toe weight, side 0 left / 1 right / -1)."""
    import foot as PD
    path = os.path.join(d, "foot.bin")
    if not pose:
        meta.pop("foot", None)
        if os.path.exists(path):
            os.remove(path)
        return
    meta["foot"] = pose
    T = PMX.to_arrays(P)
    names = [m["name"] for m in P["materials"]]
    guard = [i for i, n in enumerate(names) if n in names_mats and P["materials"][i]["n_indices"] > 0]
    vs = np.unique(T["faces"][np.isin(T["mat_face"], guard)])
    wa, wt, side = PD.weight_pmx(P, PD.chains(P["bones"]))
    C.write_bytes(path, np.stack([wa[vs], wt[vs], side[vs]], 1).astype("<f4").tobytes())


# ------------------------------------------------------------------ corps
def make_body(_):
    P = PMX.read(C.BODY)
    d = os.path.join(C.CACHE, "body")
    n, f, _ = extract_glb(C.BODY, {"肌"}, os.path.join(d, "body.glb"), role="skin", P=P)
    C.write_json(os.path.join(d, "meta.json"), {"vertices": n, "faces": f, "fingerprint": C.fingerprint_file(C.BODY)})
    log("body", n, "vertices", f, "faces")


# ------------------------------------------------------------------ heads
def join_skin(R, d):
    """Skin of the common body as the base head leaves it: faces removed, vertices adjusted.

    The page puts the head on corps.glb: it needs the body faces gone under the head
    (hidden_skin.bin, faces of 肌 in corps.glb) and the vertices whole_head adjusted at the neck and at the
    edge of the skull (corps.bin, glTF positions then normals of corps.glb).
    """
    import head_graft as GT
    Co = PMX.read(C.BODY)
    Tc, Tr = PMX.to_arrays(Co), PMX.to_arrays(R)
    names = [m["name"] for m in Co["materials"]]
    guard = ~np.isin(Tc["mat_face"], [i for i, n in enumerate(names) if n in GT.REMOVE_DEFAULT])
    vB = np.unique(Tc["faces"][guard])          # body vertices in the graft, in order (mapB)
    Fc = Tc["faces"][Tc["mat_face"] == names.index("肌")]
    Fr = Tr["faces"][Tr["mat_face"] == [m["name"] for m in R["materials"]].index("肌")]
    out_of = int((Fr >= len(vB)).any(axis=1).sum())
    rest = {tuple(sorted(int(x) for x in f_)) for f_ in vB[Fr[(Fr < len(vB)).all(axis=1)]]}
    cache = np.array([k for k, f_ in enumerate(Fc) if tuple(sorted(int(x) for x in f_)) not in rest], np.uint32)
    pos, nor = np.asarray(Tc["pos"], float).copy(), np.asarray(Tc["nor"], float).copy()
    pos[vB], nor[vB] = np.asarray(Tr["pos"], float)[:len(vB)], np.asarray(Tr["nor"], float)[:len(vB)]
    vs = np.unique(Fc)                            # vertex order of corps.glb
    p, n, _ = GLB.to_gltf(pos[vs], nor[vs], np.zeros((0, 3), np.int64))
    C.write_bytes(os.path.join(d, "hidden_skin.bin"), cache.tobytes())
    C.write_bytes(os.path.join(d, "body.bin"), np.concatenate([p.ravel(), n.ravel()]).astype(np.float32).tobytes())
    C.write_bytes(os.path.join(d, "body_pos.bin"), np.asarray(Tr["pos"], "<f4")[:len(vB)].tobytes())
    moves = int((np.abs(np.asarray(Tr["pos"], float)[:len(vB)] - np.asarray(Tc["pos"], float)[vB]).max(axis=1) > 1e-6).sum())
    return {"hidden_faces": int(len(cache)), "moved_vertices": moves, "faces_off_body": out_of}


def make_base(it):
    """Base head sewn once onto the common body (clothes/whole_head.py), and the body skin
    as it leaves it. A result.pmx already there (the approved one) is not sewn again."""
    if it.get("free"):      # free head: the base body already wears its head, nothing to sew; copied with its textures
        import shutil
        os.makedirs(C.BASE, exist_ok=True)
        for rel in [os.path.basename(it["head"])] + PMX.read(it["head"])["textures"]:
            shutil.copyfile(path_tex(it["head"], rel), path_tex(C.BASE_PMX, rel))
        os.replace(path_tex(C.BASE_PMX, os.path.basename(it["head"])), C.BASE_PMX)
    elif not os.path.isfile(C.BASE_PMX):
        import whole_head as head_whole
        head_whole.main(["--body", C.BODY, "--head", it["head"], "--output", C.BASE,
                           "--base-tint", C.GAME.value("base_head.tint", "loop")])
    R = PMX.read(C.BASE_PMX)
    join = join_skin(R, C.BASE)
    C.write_json(os.path.join(C.BASE, "meta.json"), {"head": it["head"], "fingerprint": C.fingerprint_file(C.BASE_PMX),
                                                      "join": join, "free": bool(it.get("free"))})
    log("base", "free head" if it.get("free") else C.BASE_HEAD, join)


def _key_pos(P):
    return [bytes(x) for x in np.round(np.asarray(P, float), 5).astype("<f4")]


def _islands(T, i_mat):
    """UV islands of material i_mat: list of face arrays."""
    F = T["faces"][T["mat_face"] == i_mat]
    per = np.arange(len(T["uv"]))

    def r(a):
        while per[a] != a:
            per[a] = per[per[a]]
            a = per[a]
        return a
    for a, b, c in F:
        per[r(a)] = r(b)
        per[r(b)] = r(c)
    grp = {}
    for f in F:
        grp.setdefault(r(f[0]), []).append(f)
    return [np.array(g) for g in grp.values()]


def _frame(T, g):
    uv = T["uv"][g.reshape(-1)]
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    return (x0, y0, x1, y1)


def _iris_and_texture_layers(T, i_mat):
    """Frames (x0, y0, x1, y1 in UV, v downwards) of the irises of material i_mat: on each side (sign of x), the island
    largest in space (the others are layers laid in front: pupil, highlights); and the faces of the
    islands outside these frames, which the chosen iris also replaces. An island whose UVs fall inside an iris frame
    reads the same pixels (a layer of the same drawing): it stays."""
    islands = _islands(T, i_mat)
    frames = [_frame(T, g) for g in islands]
    large = {}
    for k, g in enumerate(islands):
        p = T["pos"][np.unique(g)]
        side = 1 if p[:, 0].mean() >= 0 else -1
        e = p.max(0) - p.min(0)
        area = e[0] * e[1]
        if side not in large or area > large[side][0]:
            large[side] = (area, k)
    iris = list(dict.fromkeys(frames[k] for _, k in large.values()))
    inside = lambda a, b: a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]
    texture_layers = [g for g, c in zip(islands, frames) if not any(inside(c, b) for b in iris)]
    return iris, (np.concatenate(texture_layers) if texture_layers else np.zeros((0, 3), int))


def _mask_faces(T, faces, W, H, margin=4):
    """Pixels (row 0 at the top) covered by the UV triangles of faces, widened by marge pixels."""
    m = np.zeros((H, W), bool)
    for f in faces:
        p = T["uv"][f] * (W, H)
        x0, y0 = np.floor(p.min(0)).astype(int) - 1
        x1, y1 = np.ceil(p.max(0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-9:
            continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / d
        l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / d
        e = 1e-6
        m[y0:y1, x0:x1] |= (l1 >= -e) & (l2 >= -e) & (1 - l1 - l2 >= -e)
    for _ in range(margin):
        n = m.copy()
        n[1:] |= m[:-1]
        n[:-1] |= m[1:]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n
    return m


def zone_mouth(T, mats, W, H):
    """Rectangle in pixels (x0, y0, x1, y1, rows from the top) of the UVs of materials mats (teeth, inside of the
    mouth), with common's margin: the area cut out to be repainted and pasted back."""
    uv = T["uv"][np.unique(T["faces"][np.isin(T["mat_face"], list(mats))])]
    m = C.MARGIN_MOUTH
    return (int(uv[:, 0].min() * W) - m, int(uv[:, 1].min() * H) - m,
            int(uv[:, 0].max() * W) + m, min(H, int(uv[:, 1].max() * H) + m))


def texture_iris(path_tex_face, frames, path_iris, output, T=None, wiped=(), mouth=None, mats_mouth=()):
    """Copy of the face texture where each iris frame receives the image path_iris (blended by its alpha);
    the pixels under the faces listed in wiped become transparent (the viewer discards them). mouth: repainted image of the
    mouth area (zone_mouth of materials mats_mouth), pasted as is (blended by its alpha)."""
    background = bpy.data.images.load(path_tex_face, check_existing=False)
    W, H = background.size
    px = np.empty(W * H * 4, np.float32)
    background.pixels.foreach_get(px)
    px = px.reshape(H, W, 4)   # row 0 at the bottom
    if len(wiped):
        px[_mask_faces(T, wiped, W, H)[::-1], 3] = 0.0
    for x0, y0, x1, y1 in frames:
        c0, c1 = int(round(x0 * W)), int(round(x1 * W))
        l0, l1 = int(round(y0 * H)), int(round(y1 * H))   # rows from the top
        w, h = max(1, c1 - c0), max(1, l1 - l0)
        im = bpy.data.images.load(path_iris, check_existing=False)
        im.scale(w, h)
        ip = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(ip)
        bpy.data.images.remove(im)
        ip = ip.reshape(h, w, 4)
        zone = px[H - l1:H - l0, c0:c1]
        a = ip[..., 3:4]
        zone[..., :3] = ip[..., :3] * a + zone[..., :3] * (1 - a)
        zone[..., 3:4] = np.maximum(zone[..., 3:4], a)
    if mouth:
        x0, l0, x1, l1 = zone_mouth(T, mats_mouth, W, H)
        w, h = x1 - x0, l1 - l0
        im = bpy.data.images.load(mouth, check_existing=False)
        if tuple(im.size) != (w, h):
            print(f"mouth: image {tuple(im.size)} instead of {(w, h)}, rescaled")
            im.scale(w, h)
        ip = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(ip)
        bpy.data.images.remove(im)
        ip = ip.reshape(h, w, 4)
        zone = px[H - l1:H - l0, x0:x1]
        a = ip[..., 3:4]
        zone[..., :3] = ip[..., :3] * a + zone[..., :3] * (1 - a)
    background.pixels.foreach_set(px.reshape(-1))
    background.filepath_raw = output
    background.file_format = "PNG"
    background.save()
    bpy.data.images.remove(background)


def glb_head(res, R, output, iris=None, mouth=None):
    """head.glb: everything the head puts on corps.glb, that is every material except the body
    skin, and of that skin only the seam strips (faces with a vertex that is not a body vertex)."""
    T = PMX.to_arrays(R)
    names = [m["name"] for m in R["materials"]]
    i_skin = names.index("肌")
    of_body = set(_key_pos(np.fromfile(os.path.join(C.BASE, "body_pos.bin"), "<f4").reshape(-1, 3)))
    vp = np.unique(T["faces"][T["mat_face"] == i_skin])
    out_of = np.zeros(len(T["pos"]), bool)
    out_of[vp] = [k not in of_body for k in _key_pos(T["pos"][vp])]
    band = (T["mat_face"] == i_skin) & out_of[T["faces"]].any(1)
    sel = (T["mat_face"] != i_skin) | band
    F, mf = T["faces"][sel], T["mat_face"][sel]
    vs, inv = np.unique(F, return_inverse=True)
    pos, nor, Fl = GLB.to_gltf(T["pos"][vs], T["nor"][vs], inv.reshape(-1, 3))
    # chosen iris: a copy of the 目 texture, iris pasted, original pupil and highlights erased (layers of 目
    # outside the irises, and material 目光 if it reads the same texture: the copy takes them over)
    # repainted mouth: the same copy receives the area of the teeth and inside of the mouth, and those materials read it
    url_eyes, with_eyes = None, set()
    mats_mouth = [j for j, n in enumerate(names) if n in ("齿", "口舌")]
    t_vis = R["materials"][names.index("目")]["texture"] if "目" in names else -1
    if (iris or mouth) and 0 <= t_vis < len(R["textures"]):
        i_eye = names.index("目")
        frames, wiped = _iris_and_texture_layers(T, i_eye) if iris else ([], np.zeros((0, 3), int))
        if frames:
            with_eyes = {i_eye}
            for j, n in enumerate(names):
                if n == "目光" and R["materials"][j]["texture"] == t_vis:
                    with_eyes.add(j)
                    wiped = np.concatenate([wiped, T["faces"][T["mat_face"] == j]])
        mats_mouth = [j for j in mats_mouth if R["materials"][j]["texture"] == t_vis] if mouth else []
        if mats_mouth:
            with_eyes |= {j for j, n in enumerate(names) if n in ("齿", "口舌", "口线") and R["materials"][j]["texture"] == t_vis}
        if with_eyes:
            dest = os.path.join(os.path.dirname(output), "eyes.png")
            texture_iris(path_tex(res, R["textures"][t_vis]), frames, iris, dest, T=T, wiped=wiped,
                         mouth=mouth if mats_mouth else None, mats_mouth=mats_mouth)
            url_eyes = texture(dest)
    prims, mats = [], []
    for i in np.unique(mf):
        m = mat_glb(R, R["materials"][i], res, "skin" if i == i_skin else "face")
        if i in with_eyes:
            m["image"] = url_eyes
        if i == i_skin:
            m["name"] = "肌_bande"
        prims.append({"faces": Fl[mf == i], "material": len(mats)})
        mats.append(m)
    GLB.write(output, pos, nor, T["uv"][vs], prims, mats)
    return len(pos), len(Fl), [m["name"] for m in mats], int(band.sum())


def make_head(it):
    """A face of the catalogue: that of it["pmx"], sewn into the oval of the base head by the face transfer
    tool, as is (retarget_head visage). The skull, ears, neck and neck seam to the
    body are the base's, the same for all."""
    d = os.path.join(C.CACHE, "heads", it["id"])
    os.makedirs(d, exist_ok=True)
    measures = None
    if it["library"] == C.BASE_HEAD or it.get("free"):
        res = C.BASE_PMX
    else:
        mod = os.path.join(d, "model")   # written before, by catalog.transfer (outside the Blender queue)
        res = os.path.join(mod, "result.pmx")
        measures = C.read_json(os.path.join(mod, "measures.json"))
        if measures is None or not os.path.isfile(res):
            raise RuntimeError("face transfer: " + (C.read_json(os.path.join(mod, "error.json")) or "not done"))
    R = PMX.read(res)
    iris, mouth = C.iris(it["id"]), C.mouth(it["id"])
    n, f, mats, n_band = glb_head(res, R, os.path.join(d, "head.glb"), iris=iris, mouth=mouth)
    for name in ("body.bin", "hidden_skin.bin"):
        C.link(os.path.join(C.BASE, name), os.path.join(d, name))
    skin = [m for m in R["materials"] if m["name"] == "肌"]
    url = None
    if skin and 0 <= skin[0]["texture"] < len(R["textures"]):
        url = texture(path_tex(res, R["textures"][skin[0]["texture"]]))
    oc = {o["name"]: o["pos"] for o in R["bones"]}
    ot = {o["name"]: o["pos"] for o in PMX.read((C.read_json(os.path.join(C.BASE, "meta.json")))["head"])["bones"]}
    tr = [float(a - b) for a, b in zip(oc["頭"], ot["頭"])]
    failures = sorted(k for k, v in (measures or {}).items() if isinstance(v, dict) and v.get("passed") is False)
    C.write_json(os.path.join(d, "meta.json"), {
        "id": it["id"], "name": it.get("name", it["id"]), "pmx": it["pmx"], "source": it["library"], "library": it.get("base_library", C.BASE_HEAD),
        "model": C.relative(res), "materials": mats, "vertices": n, "faces": f, "band_faces": n_band, "skin": url,
        "translation": tr, "kept": True, "fingerprint": C.fingerprint_file(it["pmx"]), "v_skin": C.V_SKIN,
        "iris": C.fingerprint_file(iris) if iris else None, "mouth": C.fingerprint_file(mouth) if mouth else None, "measure_failures": failures})
    log("head", it["id"], f, "faces;", "transfer: " + (", ".join(failures) or "all pass") if measures is not None else "base head")


# ------------------------------------------------------------------ made parts
def make_built(it):
    """Cuts a made garment (materials of a complete model) into the library format."""
    import importlib.util
    import model as MD
    spec = importlib.util.spec_from_file_location("library_clothes", os.path.join(C.TOOLS, "clothes", "library.py"))
    BV = importlib.util.module_from_spec(spec)
    sys.modules["library_clothes"] = BV
    spec.loader.exec_module(BV)
    body_os = {o["name"] for o in PMX.read(C.BODY)["bones"]}
    M = MD.Model(it["character"], path=it["pmx"])
    chosen = [i for i, n in enumerate(M.names_mat) if n in it["mats"]]
    faces = np.where(np.isin(M.mf, chosen))[0]
    if not len(faces):
        raise ValueError("materials missing: %s" % it["mats"])
    T = SimpleNamespace(M=M, character=it["character"], os_body=[n in body_os for n in M.names_os])
    d = os.path.join(C.CACHE, "built", it["character"], it["id"])
    os.makedirs(d, exist_ok=True)
    copies = {}
    part_name = BV.part_pmx(T, faces, it["id"], "tex", copies)
    if isinstance(part_name, tuple):
        part_name = part_name[0]
    for base, src in copies.items():
        C.link(src, os.path.join(d, "tex", base))
    pmx_write.write(part_name, os.path.join(d, it["id"] + ".pmx"))
    C.write_json(os.path.join(d, "record.json"), {
        "character": it["character"], "id": it["id"], "class": "garment", "type": it["type"], "sort": None,
        "pmx": it["id"] + ".pmx", "thumbnail": None, "faces": int(len(faces)), "materials": list(it["mats"]),
        "own_bones": [o["name"] for o in part_name["bones"] if o["name"] not in body_os],
        "source": {"pmx": it["pmx"]}, "workshop": it.get("workshop") or {},
        "fingerprint": C.fingerprint_file(it["pmx"])})
    log("made", it["character"], it["id"], len(faces), "faces")


# ------------------------------------------------------------------ parts (fitting + measurement)
def slot(card):
    a = card.get("workshop") or {}
    if a.get("tab"):
        return a["tab"], a["slot"], a["rank"], a.get("name") or C.FR_TYPE.get(card.get("type"), card["id"])
    if card.get("class") == "accessory":
        rank, excl = C.ACCESSORIES[card["sort"]]
        return "accessories", (card["sort"] if excl else None), rank, C.FR_TYPE.get(card["sort"], card["sort"])
    tab, emp, rank = C.CLOTHES[card["type"]]
    return tab, emp, rank, C.FR_TYPE.get(card["type"], card["type"])


TYPES_MEASURE = {"underwear": "shorts"}
EXTENT_MIN = 2.0         # mm of diagonal: below it, the fitted part is a point (an earring measures 9)


def make_part(it):
    """A shoe is first worn on feet posed into it (porter, pied="search"); if that fit is not
    kept, it is redone on feet at rest, the shoe still placed in one block (a heel then sinks into
    the ground, the shoe stays whole); if that is not kept either, the last resort is the flat foot
    with the shoe carried by its bones and left as it is around the foot (porter, pied="flat"). A
    shoe is never pushed out of the foot vertex by vertex: that tears it to shards."""
    meta = prepare_part(it, "search")
    if not meta["kept"] and meta.get("foot"):
        import foot as PD
        refusals = meta["reason"]
        rest_pose = {name: {"ankle": 0.0, "toes": 0.0} for name, _, _ in PD.SIDES}
        if any(abs(v.get("ankle", 0.0)) > 0 for v in meta["foot"].values() if isinstance(v, dict)):
            meta = prepare_part(it, rest_pose)
        if not meta["kept"]:
            meta = prepare_part(it, "flat")
        meta["foot_refused"] = refusals
        C.write_json(os.path.join(C.CACHE, "parts", it["id"], "meta.json"), meta)
    return meta


def prepare_part(it, foot):
    import model as MD
    import measure as measure
    import wear as PO
    src = it["folder"]
    with open(os.path.join(src, "record.json"), encoding="utf-8") as fh:
        card = json.load(fh)
    if card.get("type") == "socks" or card.get("sort") == "anklet":
        # a "sock" or an "anklet" with a sole is a shoe (porter.charger_piece)
        real = PO.load_part(src)[0]
        card["class"], card["type"] = real["class"], real.get("type")
    d = os.path.join(C.CACHE, "parts", it["id"])
    os.makedirs(d, exist_ok=True)
    tab, emp, rank, type_name = slot(card)
    trimmed = card.get("trimmed") or {}
    pmx_src = os.path.join(src, trimmed.get("pmx") or card["pmx"])
    if not os.path.exists(pmx_src):
        trimmed, pmx_src = {}, os.path.join(src, card["pmx"])
    meta = {"id": it["id"], "character": card["character"], "part_name": card["id"], "class": card["class"],
            "type": card.get("type") or card.get("sort"), "type_name": type_name, "tab": tab, "slot": emp,
            "rank": rank, "occupies": C.OCCUPIES.get(card.get("type"), []), "folder": C.relative(src),
            "fingerprint": C.fingerprint_file(pmx_src), "v_fitting": C.V_FITTING, "body": C.fingerprint_file(C.BODY),
            "kept": False, "reason": None, "checks": [], "faces": 0,
            "trims": trimmed.get("parts") or []}
    tmp = tempfile.mkdtemp(prefix="piece-", dir=os.environ.get("TMPDIR") or None)
    t0 = time.time()
    try:
        try:
            res = PO.wear(src, C.BODY, tmp, name="part", margin_mm=3.0, skin="remove", foot=foot)
        except Exception as e:
            meta["reason"] = "fitting failed: %s" % str(e)[:160]
            return meta
        if res.get("extent_mm", EXTENT_MIN) < EXTENT_MIN:
            # an object the game stores folded into a point (Hugo's hat: 3 microns) is not a part
            meta["reason"] = "part with no extent (%.1f mm)" % res["extent_mm"]
            return meta
        path = os.path.join(tmp, "piece.pmx")
        P = PMX.read(path)
        n, f, mats = extract_glb(path, set(res["mats"]), os.path.join(d, "part.glb"), P=P)
        meta["faces"], meta["vertices"], meta["materials"] = f, n, mats
        write_foot(P, set(res["mats"]), res.get("foot"), meta, d)
        # hidden skin: faces of the body's 肌 missing from the output (porter keeps the indices)
        Co = PMX.read(C.BODY)
        Tc = PMX.to_arrays(Co)
        i_skin = [m["name"] for m in Co["materials"]].index("肌")
        Fc = Tc["faces"][Tc["mat_face"] == i_skin]
        To = PMX.to_arrays(P)
        Fo = To["faces"][To["mat_face"] == [m["name"] for m in P["materials"]].index("肌")]
        rest = {tuple(int(x) for x in sorted(f_)) for f_ in Fo}
        cache = np.array([k for k, f_ in enumerate(Fc) if tuple(int(x) for x in sorted(f_)) not in rest], np.uint32)
        C.write_bytes(os.path.join(d, "hidden_skin.bin"), cache.tobytes())
        meta["hidden_skin"] = int(len(cache))
        vi = card.get("thumbnail")
        if vi and os.path.isfile(os.path.join(src, vi)):
            C.link(os.path.join(src, vi), os.path.join(d, "thumbnail.png"))
            meta["thumbnail"] = "/data/parts/%s/thumbnail.png" % it["id"]
        spec = {"id": it["id"], "character": it["id"], "type": TYPES_MEASURE.get(card.get("type"), res["measure_type"]),
                "mats": res["mats"], "texture_layers": [], "why": "workshop"}
        try:
            r = measure.measure(spec, MD.Model(it["id"], path=path), True)
        except measure.WithoutObj as e:
            # garment measurement does not apply here (rigid accessory): the part is judged
            # on what fitting says about it, and the missing checks are named
            meta["measure_not_applicable"] = str(e)
            r = {"checks": [{"name": n, "value": None, "threshold": None, "ok": None} for n in BLOCKING if n != "under_skin"]}
        except Exception as e:
            meta["reason"] = "measurement failed: %s" % str(e)[:160]
            return meta
        # fitting itself says what it left inside the body: rest_crossing (measure.py) only
        # sees up to 10 mm, a part sunk several centimetres deep escapes it
        sp = res.get("under_skin") or {}
        # a part placed in one block (wing, bag: rigid_pose; a shoe's foot part: foot_as_block) sits
        # in the body by nature, and cannot be pulled out without deforming: the check does not apply to it
        without_obj = sp.get("fraction") is None or res.get("rigid_pose") or res.get("foot_as_block")
        r["checks"].append({"name": "under_skin", "value": sp.get("fraction"), "threshold": "<= 0.01",
                               "ok": None if without_obj else sp["fraction"] <= UNDER_SKIN_MAX})
        # a shoe placed in one block holds the foot only if it did not have to be lowered by more than
        # FOOT_SUNK_MAX for the foot to stop going through its sole (wear.put_on_shoes)
        ch = res.get("on_foot") or {}
        if ch:
            sunk = max(float(v.get("sole", 0.0)) for v in ch.values())
            r["checks"].append({"name": "foot_in_shoe", "value": round(100 * sunk, 1),
                                   "threshold": "<= %g cm" % (100 * FOOT_SUNK_MAX), "ok": sunk <= FOOT_SUNK_MAX})
        # the depth (9th decile) of the clipping points says nothing about a handful of points:
        # under 1 % of the part, the share is what judges (rest_crossing, under_skin)
        part_cross = next((x["value"] for x in r["checks"] if x["name"] == "rest_crossing"), None)
        for x in r["checks"]:
            if x["name"] == "rest_crossing_p90_mm" and part_cross is not None and part_cross < UNDER_SKIN_MAX:
                x["ok"] = None
        meta["fitting"] = {"under_skin": sp, "stays_inside": res.get("stays_inside"), "on_foot": ch or None, "sheath": res.get("sheath"), "transport": res.get("transport"), "vertex_rigid_bodies": res.get("vertex_rigid_bodies"), "attempts": res.get("attempts"), "per_limb": res.get("per_limb")}
        meta["checks"] = [{"name": x["name"], "value": x["value"], "threshold": x["threshold"], "ok": x["ok"]} for x in r["checks"]]
        # a blocking check without a value is not a passed check: it is named
        meta["not_measured"] = [x["name"] for x in r["checks"] if x["ok"] is None and x["name"] in BLOCKING
                               and not (x["name"] == "rest_crossing_p90_mm" and part_cross is not None and part_cross < UNDER_SKIN_MAX)]
        bad = [x for x in r["checks"] if x["ok"] is False and x["name"] in BLOCKING]
        if bad:
            meta["reason"] = "; ".join("%s = %s (threshold %s)" % (x["name"], _court(x["value"]), x["threshold"]) for x in bad)
        else:
            meta["kept"] = True
        meta["other_failures"] = [x["name"] for x in r["checks"] if x["ok"] is False and x["name"] not in BLOCKING]
        if meta["kept"]:
            C.write_json(os.path.join(d, "meta.json"), meta)  # coverage reads the record
            complete_covers(meta, d)
        return meta
    finally:
        meta["duration_s"] = round(time.time() - t0, 1)
        C.write_json(os.path.join(d, "meta.json"), meta)
        shutil.rmtree(tmp, ignore_errors=True)
        log("fraction", it["id"], "kept" if meta["kept"] else "rejected: %s" % meta["reason"], "%.1f s" % meta["duration_s"])


# ------------------------------------------------------------------ what each face can do
_alphas = {}


def alpha_of(url):
    """Alpha channel (0..1, first row at the top) of a cache texture."""
    if url not in _alphas:
        path = os.path.join(C.CACHE, url[len("/data/"):])
        im = bpy.data.images.load(path, check_existing=False)
        w, h = im.size
        px = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(px)
        bpy.data.images.remove(im)
        _alphas[url] = px.reshape(h, w, 4)[::-1, :, 3]
    return _alphas[url]


def opaque_of(doc, uv, F, prim):
    """Per face: the material is opaque, and so is the texture everywhere on the face."""
    op = np.ones(len(F), bool)
    for i, pr in enumerate(doc["meshes"][0]["primitives"]):
        mat = doc["materials"][pr["material"]]["pbrMetallicRoughness"]
        sel = np.where(prim == i)[0]
        if len(mat.get("baseColorFactor", [1, 1, 1, 1])) > 3 and mat["baseColorFactor"][3] < 0.95:
            op[sel] = False
            continue
        bt = mat.get("baseColorTexture")
        if bt is None:
            continue
        a = alpha_of(doc["images"][doc["textures"][bt["index"]]["source"]]["uri"])
        import coverage as K
        op[sel] = K.opaque_per_face(a, uv, F[sel])
    return op


def complete_covers(meta, d):
    """covers.bin of the part; a shoe that does not cover the top of the foot is rejected."""
    import coverage as K
    _, part = K.compute(meta["id"], opaque_of)
    if meta.get("class") == "garment" and os.path.isfile(os.path.join(d, "hidden_skin.bin")):
        # fitting counted all the held geometry; only opaque faces hide the skin
        import skin as PE
        k = PE.of_part(meta["id"])
        C.write_bytes(os.path.join(d, "hidden_skin.bin"), k.tobytes())
        meta["hidden_skin"] = int(len(k))
    if part is not None:
        meta["instep"] = round(part, 3)
        if part < K.FOOT_MIN:
            meta["kept"] = False
            meta["reason"] = "sole without an upper: %d %% of the top of the foot covered (minimum %d %%)" % (round(100 * part), round(100 * K.FOOT_MIN))
    return meta


def make_covers(it):
    """Completion for a part prepared before covers.bin existed."""
    d = os.path.join(C.CACHE, "parts", it["id"])
    m = C.read_json(os.path.join(d, "meta.json"))
    complete_covers(m, d)
    C.write_json(os.path.join(d, "meta.json"), m)
    log("covers", it["id"], "kept" if m["kept"] else "rejected: %s" % m["reason"])


def _court(v):
    return round(v, 3) if isinstance(v, float) else v


# ------------------------------------------------------------------ hairstyles
def hairstyle_already_placed(it, d):
    """New heads that are the same target as a head already measured (same base head, same translation:
    hairstyles.Preparation.place_on gives them the same pose): the pair is copied, nothing is loaded.
    True when every new head was copied. Each extraction lot adds heads, and posing every hairstyle again
    for an answer it already had cost about 25 s per hairstyle and per lot."""
    m = C.read_json(os.path.join(d, "meta.json")) or {}
    before = m.get("pairs") or {}
    keys = {}
    for t in it.get("keep") or []:
        mt = C.read_json(os.path.join(C.CACHE, "heads", t, "meta.json")) or {}
        if t in before and mt.get("library") and os.path.isfile(os.path.join(d, t + ".bin")):
            keys.setdefault((mt["library"], tuple(float(x) for x in mt.get("translation") or ())), t)
    src = {t["id"]: keys.get((t["library"], tuple(float(x) for x in t["translation"]))) for t in it["heads"]}
    if not src or not all(src.values()) or not os.path.isfile(os.path.join(d, "hairstyle.glb")):
        return False
    pairs = {t: before[t] for t in it["keep"] if t in before}
    for n, s in src.items():
        C.link(os.path.join(d, s + ".bin"), os.path.join(d, n + ".bin"))   # same bytes: a link, not a copy
        pairs[n] = dict(before[s])
    own = it.get("own")
    kept, reason = C.hairstyle_kept(pairs, own)
    C.write_json(os.path.join(d, "meta.json"), dict(m, name=it.get("name", it["id"]), pairs=pairs,
                                                     kept=bool(kept), reason=reason, own=own,
                                                     fingerprint=it.get("fingerprint")))
    n_ok = sum(1 for r in pairs.values() if r["passed"])
    log("hairstyle", it["id"], "%d/%d heads" % (n_ok, len(pairs)), "kept" if kept else "rejected: %s" % reason)
    return True


def make_hairstyle(it):
    import hairstyles as CF
    d = os.path.join(C.CACHE, "hairstyles", it["id"])
    if it.get("keep") and hairstyle_already_placed(it, d):
        return
    hair = CF.Preparation(it["library"], it["id"])
    os.makedirs(d, exist_ok=True)
    A = hair.A
    mats = [mat_glb(A.pmx, A.pmx["materials"][i], A.path, "hair") for i in hair.mats]
    for k, i in enumerate(hair.mats):
        if i in hair.tex_reflections:
            mats[k]["image"] = texture(hair.tex_reflections[i])
    GLB.write(os.path.join(d, "hairstyle.glb"), *hair.glb_base(), mats)
    pairs = {}
    if it.get("keep"):   # pairs measured on an earlier pass (catalogue.plan_suite): only the new heads are posed
        before = (C.read_json(os.path.join(d, "meta.json")) or {}).get("pairs") or {}
        pairs = {t: before[t] for t in it["keep"] if t in before}
    # Heads that are the same target (same base head, same translation) get the same pose: it is computed once
    # and the other heads link to its file. With hundreds of models, posing and storing every pair separately
    # grew as heads x hairstyles in time and on disk (one identical file per pair).
    already = {}
    for head in it["heads"]:
        the_key = (head.get("library"), tuple(float(x) for x in head.get("translation") or ()))
        output = os.path.join(d, head["id"] + ".bin")
        if the_key in already:
            C.link(os.path.join(d, already[the_key] + ".bin"), output)
            pairs[head["id"]] = dict(pairs[already[the_key]])
            continue
        res = hair.place_on(head)
        C.write_bytes(output, res.pop("positions").astype("<f4").tobytes())
        pairs[head["id"]] = res
        already[the_key] = head["id"]
    n_ok = sum(1 for r in pairs.values() if r["passed"])
    own = it.get("own")
    kept, reason = C.hairstyle_kept(pairs, own)
    C.write_json(os.path.join(d, "meta.json"), {
        "id": it["id"], "name": it.get("name", it["id"]), "faces": int(sum(len(f) for f in hair.per_mat)),
        "vertices": int(len(hair.vs)), "pairs": pairs, "kept": bool(kept), "reason": reason,
        "own": own, "fingerprint": it.get("fingerprint")})
    log("hairstyle", it["id"], "%d/%d heads" % (n_ok, len(pairs)), "kept" if kept else "rejected: %s" % reason)


def make_hairstyle_free(it):
    """The base's hairstyle (tools/free_base, tools/vroid_base): made for its head, placed as is."""
    d = os.path.join(C.CACHE, "hairstyles", it["id"])
    os.makedirs(d, exist_ok=True)
    P = PMX.read(it["pmx"])   # the file holds only the hairstyle: all its materials (VRoid: 髪, 髪2…)
    n, f, _ = extract_glb(it["pmx"], {m["name"] for m in P["materials"]}, os.path.join(d, "hairstyle.glb"), role="hair", P=P)
    C.write_json(os.path.join(d, "meta.json"), {
        "id": it["id"], "name": it.get("name", it["id"]), "faces": f, "vertices": n,
        "pairs": {t: {"passed": True, "failures": []} for t in it["heads"]}, "kept": True, "reason": None,
        "own": None, "fingerprint": it.get("fingerprint"), "free": True, "pmx": it["pmx"]})
    log("free hairstyle", f, "faces")


MAKE = {"hairstyle_free": make_hairstyle_free, "body": make_body, "base": make_base, "head": make_head, "build": make_built, "part": make_part, "hairstyle": make_hairstyle, "covers": make_covers}


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    with open(a[a.index("--travail") + 1], encoding="utf-8") as fh:
        items = json.load(fh)
    for it in items:
        try:
            MAKE[it["kind"]](it)
        except (Exception, SystemExit) as e:   # head_graft rejects through SystemExit
            log("FAILED", it["kind"], it.get("id"), repr(e)[:300])
            traceback.print_exc()
            if it["kind"] in ("head", "hairstyle"):
                under = {"head": "heads", "hairstyle": "hairstyles"}[it["kind"]]
                C.write_json(os.path.join(C.CACHE, under, it["id"], "meta.json"), {
                    "id": it["id"], "name": it.get("name", it["id"]), "kept": False,
                    "reason": "preparation failed: %s" % (str(e)[:160] if isinstance(e, SystemExit) else "%s %s" % (type(e).__name__, str(e)[:140]))})


if __name__ == "__main__":
    main()
