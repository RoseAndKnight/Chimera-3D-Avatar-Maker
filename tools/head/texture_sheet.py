"""Textured contact sheet (Eevee, texture colour without shading): rows = models, columns = front, 3/4,
profile, blink, smile, あ.
    blender -b --factory-startup --python texture_sheet.py -- [--hairstyle] <sortie.png> <pmx> <dossier_tex> [...]
--hairstyle: front, 3/4, profile, back, framed on the whole head including long hair.
Relative texture paths are looked up next to the PMX, then in dossier_tex.
SHEET_FRAME=head (with --hairstyle): frames the face and hair, without shoulders or raised arms: the head stays large in the tile.
SHEET_FRAME=eyes: tiles framed on the eyes (box of the eye whites 白目, otherwise of the 目 plate, margin included), at the same
pixel scale as the full tile (SHEET_SCALE included); for close-ups of shading and highlights.
"""
import math, os, sys
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atlas"))
import pmx as PMX

COLUMNS = (("face", 0, {}), ("three_quarters", 35, {}), ("profile", 90, {}),
            ("blink", 0, {"まばたき": 1.0}), ("smile", 0, {"笑い": 1.0, "口角上げ": 1.0}),
            ("a", 0, {"あ": 1.0}))
HIDDEN = ("髮+",)  # duplicated hair outline
COLUMNS_HAIRSTYLE = (("face", 0, {}), ("three_quarters", 35, {}), ("profile", 90, {}), ("back", 180, {}))
TX, TY = 400, 460
_E = int(os.environ.get("SHEET_SCALE", "1"))  # enlarge the tiles for an image to show
# SHEET_COLUMNS=face,three_quarters: render only these columns (compare candidates more cheaply)
TX, TY = TX * _E, TY * _E
FRAME_EYES = os.environ.get("SHEET_FRAME") == "eyes"
MARGIN_EYES = (0.35, 1.2)  # margin around the box of both eye whites, as a fraction of its width / height


def offsets(P, weight):
    D = np.zeros((len(P["vertices"]["pos"]), 3))
    names = {m["name"]: m for m in P["morphs"]}
    for name, w in weight.items():
        m = names.get(name)
        if m is None:
            continue
        pile = [(m, w)]
        if m["type"] == 0:
            pile = [(P["morphs"][i], w * x) for i, x in m["offsets"]]
        for mm, ww in pile:
            if mm["type"] == 1:
                for v, d in mm["offsets"]:
                    D[v] += ww * np.asarray(d)
    return D


def conv(p):
    return np.stack([p[:, 0], p[:, 2], p[:, 1]], 1)


def find(rel, folders):
    rel = rel.replace("\\", "/")
    for d in folders:
        f = os.path.normpath(os.path.join(d, rel))
        if os.path.exists(f):
            return f
    return None


def texture_layer_eye(name):
    """Overlay laid on the eye (目光, 目光2, 目影, 目色…), not the iris 目: blended in PMX order."""
    return name.startswith("目") and name != "目"


def material(name, image, diffuse=(1.0, 1.0, 1.0, 1.0), blend=False):
    """Colour = texture × diffuse (RGB), opacity = texture alpha × diffuse alpha, as in MMD
    (otherwise an eye highlight meant to be at 35 % covers the white of the eye). melange: sorted alpha
    blending (BLENDED) instead of dithering (DITHERED), for the eye overlays: dithering left a light grain
    on the eyelid shadow, even at 256 samples."""
    m = bpy.data.materials.new(name)

    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mix.outputs[0], out.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    if image is not None:
        tx = nt.nodes.new("ShaderNodeTexImage")
        tx.image = image
        mc = nt.nodes.new("ShaderNodeMix")
        mc.data_type, mc.blend_type = "RGBA", "MULTIPLY"
        mc.inputs["Factor"].default_value = 1.0
        nt.links.new(tx.outputs["Color"], mc.inputs["A"])
        mc.inputs["B"].default_value = (*diffuse[:3], 1.0)
        nt.links.new(mc.outputs["Result"], em.inputs["Color"])
        ma = nt.nodes.new("ShaderNodeMath")
        ma.operation = "MULTIPLY"
        ma.inputs[1].default_value = diffuse[3]
        nt.links.new(tx.outputs["Alpha"], ma.inputs[0])
        nt.links.new(ma.outputs[0], mix.inputs[0])
    else:
        em.inputs["Color"].default_value = (0.8 * diffuse[0], 0.8 * diffuse[1], 0.8 * diffuse[2], 1)
        mix.inputs[0].default_value = diffuse[3]
    try:
        m.surface_render_method = "BLENDED" if blend else "DITHERED"
    except Exception:
        m.blend_method = "BLEND" if blend else "HASHED"
    m.use_backface_culling = False
    return m


def load(path, dtex, name, hairstyle=False):
    P = PMX.read(path)
    pos = np.asarray(P["vertices"]["pos"], dtype=float)
    uv = np.asarray(P["vertices"]["uv"], dtype=float)
    F = np.asarray(P["faces"], dtype=np.int64)
    mf = np.repeat(np.arange(len(P["materials"])), [m["n_indices"] // 3 for m in P["materials"]])
    names = [m["name"] for m in P["materials"]]
    guard = np.array([not any(names[k] == c or names[k].startswith(c + "_") for c in HIDDEN) for k in mf])
    skin = np.isin(mf, [i for i, n in enumerate(names) if n in ("颜", "颜_A")])
    vp = np.unique(F[skin])
    lo, hi = pos[vp].min(0), pos[vp].max(0)
    bottom = lo[1] - (1.6 if hairstyle else 0.35) * (hi[1] - lo[1])
    cf = pos[F].mean(1)
    guard &= cf[:, 1] > bottom
    if hairstyle:  # box around the head: no ponytail, no outstretched arms
        h, cx, cz = hi[1] - lo[1], (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
        guard &= (np.abs(cf[:, 0] - cx) < 2.2 * h) & (np.abs(cf[:, 2] - cz) < 2.2 * h)
    F, mf = F[guard], mf[guard]
    if hairstyle:  # frame what is kept, hair included, within the width of the head
        vg = np.unique(F)
        if os.environ.get("SHEET_FRAME") == "head":     # face and hair only: no shoulders, no raised arms
            vg = np.unique(F[np.isin(mf, [i for i, n in enumerate(names) if n in ("颜", "颜_A") or n.startswith("髮")])])
        lo, hi = pos[vg].min(0), pos[vg].max(0)
    folders = [os.path.dirname(path), dtex]
    cache = {}
    mats = []
    for k, m in enumerate(P["materials"]):
        t = m["texture"]
        img = None
        if 0 <= t < len(P["textures"]):
            f = find(P["textures"][t], folders)
            if f:
                img = cache.get(f) or bpy.data.images.load(f)
                img.colorspace_settings.name = "Non-Color"  # sRGB values as they are: blending as in MMD
                cache[f] = img
            else:
                print("missing texture:", P["textures"][t])
        mats.append(material("%s_%d" % (name, k), img, m["diffuse"], texture_layer_eye(m["name"])))
    # eye overlays: one object each, in PMX order (Eevee sorts blended objects by the
    # position of their origin along the view; see placer())
    texture_layers = [k for k in range(len(names)) if texture_layer_eye(names[k]) and (mf == k).any()]
    obs = []
    for rank, sel in enumerate([~np.isin(mf, texture_layers)] + [mf == k for k in texture_layers]):
        me = bpy.data.meshes.new("%s_%d" % (name, rank))
        me.from_pydata([tuple(p) for p in conv(pos)], [], [tuple(int(x) for x in f) for f in F[sel]])
        me.update()
        lay = me.uv_layers.new(name="UV")
        lu = uv[F[sel].ravel()].copy()
        lu[:, 1] = 1 - lu[:, 1]
        lay.data.foreach_set("uv", lu.astype(np.float32).ravel())
        for m in mats:
            me.materials.append(m)
        me.polygons.foreach_set("material_index", mf[sel].astype(np.int32))
        me.update()
        o = bpy.data.objects.new("%s_%d" % (name, rank), me)
        bpy.context.scene.collection.objects.link(o)
        obs.append(o)
    ob = obs[0]
    center = conv(((lo + hi) / 2)[None])[0]
    size = float(max(hi[1] - lo[1], hi[0] - lo[0]))
    M = {"P": P, "pos": pos, "ob": ob, "obs": obs, "center": Vector(center), "size": size}
    eyes = np.isin(mf, [i for i, n in enumerate(names) if n == "白目"])
    if not eyes.any():  # Outerplane eyes: the white is in the 目 plate
        eyes = np.isin(mf, [i for i, n in enumerate(names) if n == "目"])
    if FRAME_EYES and eyes.any():
        vy = np.unique(F[eyes])
        ly, hy = pos[vy].min(0), pos[vy].max(0)
        M["eyes"] = (Vector(conv(((ly + hy) / 2)[None])[0]),
                     (hy[0] - ly[0]) * (1 + 2 * MARGIN_EYES[0]), (hy[1] - ly[1]) * (1 + 2 * MARGIN_EYES[1]))
    return M


def put(M, p, to_camera):
    """Vertices p (Blender frame) set on each object of the model. The eye overlays have their origin
    at the centre of the head, moved one step towards the camera per rank: Eevee draws blended objects
    from farthest to nearest, hence in the order of the PMX materials, as in MMD."""
    stage = 1e-3 * M["size"]
    for rank, o in enumerate(M["obs"]):
        loc = Vector(M["center"]) + to_camera * (stage * rank) if rank else Vector((0.0, 0.0, 0.0))
        o.location = loc
        o.data.vertices.foreach_set("co", (p - np.asarray(loc)).astype(np.float32).ravel())
        o.data.update()


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    hairstyle = a[0] == "--hairstyle"
    a = a[1:] if hairstyle else a
    columns = COLUMNS_HAIRSTYLE if hairstyle else COLUMNS
    keep = os.environ.get("SHEET_COLUMNS")  # e.g. "face,three_quarters": quick comparison sheet
    if keep:
        columns = tuple(c for c in columns if c[0] in keep.split(","))
    output, pairs = a[0], list(zip(a[1::2], a[2::2]))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    # dithered transparency (hair, eyelashes); the eye overlays are blended (material()).
    # SHEET_SAMPLES=32 to compare candidates (software render without GPU: ~3 min per tile at 256)
    sc.eevee.taa_render_samples = int(os.environ.get("SHEET_SAMPLES", "256"))
    # as in MMD: sRGB colours blended as they are (Non-Color textures, Raw output). In Standard,
    # blending happens in linear space: a black overlay at alpha 0.17 darkens by only 8 % (17 % in MMD)
    sc.view_settings.view_transform = "Raw"
    sc.render.resolution_x, sc.render.resolution_y = TX, TY
    sc.render.image_settings.file_format = "PNG"
    sc.world = bpy.data.worlds.new("w")
    
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.93, 0.93, 0.93, 1)  # 0.85 linear
    cam = bpy.data.cameras.new("c")
    cam.type = "ORTHO"
    cam.clip_end = 50
    co = bpy.data.objects.new("c", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    tiles = os.path.splitext(output)[0] + "_tiles"
    os.makedirs(tiles, exist_ok=True)
    models = [load(p, d, "m%d" % i, hairstyle) for i, (p, d) in enumerate(pairs)]
    images = []
    for i, M in enumerate(models):
        for o in models:
            for ob in o["obs"]:
                ob.hide_render = o is not M
        rank = []
        for nc, az, weight in columns:
            p = conv(M["pos"] + offsets(M["P"], weight))
            cam.ortho_scale = (1.1 if hairstyle else 1.9) * M["size"]
            cen = M["center"]
            sc.render.resolution_x, sc.render.resolution_y = TX, TY
            if "eyes" in M:  # same scale: px per unit = max(TX, TY) / ortho_scale of the full tile
                cen, lx, ly = M["eyes"]
                px = max(TX, TY) / cam.ortho_scale
                sc.render.resolution_x, sc.render.resolution_y = int(round(lx * px)), int(round(ly * px))
                cam.ortho_scale = max(lx, ly)
            r = math.radians(az)
            put(M, p, Vector((math.sin(r), -math.cos(r), 0.0)))
            co.location = cen + Vector((math.sin(r), -math.cos(r), 0.0)) * 10
            co.rotation_euler = (cen - co.location).to_track_quat("-Z", "Y").to_euler()
            f = os.path.join(tiles, "%d_%s.png" % (i, nc))
            sc.render.filepath = f
            bpy.ops.render.render(write_still=True)
            rank.append(f)
        images.append(rank)
    nl, nc = len(images), len(columns)
    pix = []
    for rank in images:
        pix.append([])
        for f in rank:
            im = bpy.data.images.load(f)
            w, h = im.size
            pix[-1].append(np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4))
            bpy.data.images.remove(im)
    ty = max(p.shape[0] for r in pix for p in r)  # tiles framed on the eyes: sizes specific to each head
    tx = max(p.shape[1] for r in pix for p in r)
    all_ = np.ones((nl * ty, nc * tx, 4), dtype=np.float32)
    for i, rank in enumerate(pix):
        for j, px in enumerate(rank):
            y0 = (nl - 1 - i) * ty
            all_[y0:y0 + px.shape[0], j * tx:j * tx + px.shape[1]] = px
    for i in range(1, nl):
        all_[i * ty - 1:i * ty + 1] = (0.4, 0.4, 0.4, 1)
    for j in range(1, nc):
        all_[:, j * tx - 1:j * tx + 1] = (0.4, 0.4, 0.4, 1)
    out = bpy.data.images.new("sheet", nc * tx, nl * ty, alpha=True)
    out.pixels[:] = all_.ravel()
    out.filepath_raw = output
    out.file_format = "PNG"
    out.save()
    print("contact sheet:", output)


main()
