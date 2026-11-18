"""Textured contact sheet (Eevee, texture colour without shading): rows = models, columns = front, 3/4,
profile, blink, smile, あ.
    blender -b --factory-startup --python planche_textures.py -- [--coiffure] <sortie.png> <pmx> <dossier_tex> [...]
--coiffure: front, 3/4, profile, back, framed on the whole head including long hair.
Relative texture paths are looked up next to the PMX, then in dossier_tex.
PLANCHE_CADRE=tete (with --coiffure): frames the face and hair, without shoulders or raised arms: the head stays large in the tile.
PLANCHE_CADRE=yeux: tiles framed on the eyes (box of the eye whites 白目, otherwise of the 目 plate, margin included), at the same
pixel scale as the full tile (PLANCHE_ECHELLE included); for close-ups of shading and highlights.
"""
import math, os, sys
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atlas"))
import pmx as PMX

COLONNES = (("face", 0, {}), ("trois_quarts", 35, {}), ("profil", 90, {}),
            ("clignement", 0, {"まばたき": 1.0}), ("sourire", 0, {"笑い": 1.0, "口角上げ": 1.0}),
            ("a", 0, {"あ": 1.0}))
CACHES = ("髮+",)  # duplicated hair outline
COLONNES_COIFFURE = (("face", 0, {}), ("trois_quarts", 35, {}), ("profil", 90, {}), ("dos", 180, {}))
TX, TY = 400, 460
_E = int(os.environ.get("PLANCHE_ECHELLE", "1"))  # enlarge the tiles for an image to show
# PLANCHE_COLONNES=face,trois_quarts: render only these columns (compare candidates more cheaply)
TX, TY = TX * _E, TY * _E
CADRE_YEUX = os.environ.get("PLANCHE_CADRE") == "yeux"
MARGE_YEUX = (0.35, 1.2)  # margin around the box of both eye whites, as a fraction of its width / height


def decalages(P, poids):
    D = np.zeros((len(P["sommets"]["pos"]), 3))
    noms = {m["nom"]: m for m in P["morphes"]}
    for nom, w in poids.items():
        m = noms.get(nom)
        if m is None:
            continue
        pile = [(m, w)]
        if m["type"] == 0:
            pile = [(P["morphes"][i], w * x) for i, x in m["decalages"]]
        for mm, ww in pile:
            if mm["type"] == 1:
                for v, d in mm["decalages"]:
                    D[v] += ww * np.asarray(d)
    return D


def conv(p):
    return np.stack([p[:, 0], p[:, 2], p[:, 1]], 1)


def trouver(rel, dossiers):
    rel = rel.replace("\\", "/")
    for d in dossiers:
        f = os.path.normpath(os.path.join(d, rel))
        if os.path.exists(f):
            return f
    return None


def calque_oeil(nom):
    """Overlay laid on the eye (目光, 目光2, 目影, 目色…), not the iris 目: blended in PMX order."""
    return nom.startswith("目") and nom != "目"


def materiau(nom, image, diffus=(1.0, 1.0, 1.0, 1.0), melange=False):
    """Colour = texture × diffuse (RGB), opacity = texture alpha × diffuse alpha, as in MMD
    (otherwise an eye highlight meant to be at 35 % covers the white of the eye). melange: sorted alpha
    blending (BLENDED) instead of dithering (DITHERED), for the eye overlays: dithering left a light grain
    on the eyelid shadow, even at 256 samples."""
    m = bpy.data.materials.new(nom)

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
        mc.inputs["B"].default_value = (*diffus[:3], 1.0)
        nt.links.new(mc.outputs["Result"], em.inputs["Color"])
        ma = nt.nodes.new("ShaderNodeMath")
        ma.operation = "MULTIPLY"
        ma.inputs[1].default_value = diffus[3]
        nt.links.new(tx.outputs["Alpha"], ma.inputs[0])
        nt.links.new(ma.outputs[0], mix.inputs[0])
    else:
        em.inputs["Color"].default_value = (0.8 * diffus[0], 0.8 * diffus[1], 0.8 * diffus[2], 1)
        mix.inputs[0].default_value = diffus[3]
    try:
        m.surface_render_method = "BLENDED" if melange else "DITHERED"
    except Exception:
        m.blend_method = "BLEND" if melange else "HASHED"
    m.use_backface_culling = False
    return m


def charger(chemin, dtex, nom, coiffure=False):
    P = PMX.lire(chemin)
    pos = np.asarray(P["sommets"]["pos"], dtype=float)
    uv = np.asarray(P["sommets"]["uv"], dtype=float)
    F = np.asarray(P["faces"], dtype=np.int64)
    mf = np.repeat(np.arange(len(P["materiaux"])), [m["n_indices"] // 3 for m in P["materiaux"]])
    noms = [m["nom"] for m in P["materiaux"]]
    garde = np.array([not any(noms[k] == c or noms[k].startswith(c + "_") for c in CACHES) for k in mf])
    peau = np.isin(mf, [i for i, n in enumerate(noms) if n in ("颜", "颜_A")])
    vp = np.unique(F[peau])
    lo, hi = pos[vp].min(0), pos[vp].max(0)
    bas = lo[1] - (1.6 if coiffure else 0.35) * (hi[1] - lo[1])
    cf = pos[F].mean(1)
    garde &= cf[:, 1] > bas
    if coiffure:  # box around the head: no ponytail, no outstretched arms
        h, cx, cz = hi[1] - lo[1], (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
        garde &= (np.abs(cf[:, 0] - cx) < 2.2 * h) & (np.abs(cf[:, 2] - cz) < 2.2 * h)
    F, mf = F[garde], mf[garde]
    if coiffure:  # frame what is kept, hair included, within the width of the head
        vg = np.unique(F)
        if os.environ.get("PLANCHE_CADRE") == "tete":     # face and hair only: no shoulders, no raised arms
            vg = np.unique(F[np.isin(mf, [i for i, n in enumerate(noms) if n in ("颜", "颜_A") or n.startswith("髮")])])
        lo, hi = pos[vg].min(0), pos[vg].max(0)
    dossiers = [os.path.dirname(chemin), dtex]
    cache = {}
    mats = []
    for k, m in enumerate(P["materiaux"]):
        t = m["texture"]
        img = None
        if 0 <= t < len(P["textures"]):
            f = trouver(P["textures"][t], dossiers)
            if f:
                img = cache.get(f) or bpy.data.images.load(f)
                img.colorspace_settings.name = "Non-Color"  # sRGB values as they are: blending as in MMD
                cache[f] = img
            else:
                print("missing texture:", P["textures"][t])
        mats.append(materiau("%s_%d" % (nom, k), img, m["diffus"], calque_oeil(m["nom"])))
    # eye overlays: one object each, in PMX order (Eevee sorts blended objects by the
    # position of their origin along the view; see placer())
    calques = [k for k in range(len(noms)) if calque_oeil(noms[k]) and (mf == k).any()]
    obs = []
    for rang, sel in enumerate([~np.isin(mf, calques)] + [mf == k for k in calques]):
        me = bpy.data.meshes.new("%s_%d" % (nom, rang))
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
        o = bpy.data.objects.new("%s_%d" % (nom, rang), me)
        bpy.context.scene.collection.objects.link(o)
        obs.append(o)
    ob = obs[0]
    centre = conv(((lo + hi) / 2)[None])[0]
    taille = float(max(hi[1] - lo[1], hi[0] - lo[0]))
    M = {"P": P, "pos": pos, "ob": ob, "obs": obs, "centre": Vector(centre), "taille": taille}
    yeux = np.isin(mf, [i for i, n in enumerate(noms) if n == "白目"])
    if not yeux.any():  # Outerplane eyes: the white is in the 目 plate
        yeux = np.isin(mf, [i for i, n in enumerate(noms) if n == "目"])
    if CADRE_YEUX and yeux.any():
        vy = np.unique(F[yeux])
        ly, hy = pos[vy].min(0), pos[vy].max(0)
        M["yeux"] = (Vector(conv(((ly + hy) / 2)[None])[0]),
                     (hy[0] - ly[0]) * (1 + 2 * MARGE_YEUX[0]), (hy[1] - ly[1]) * (1 + 2 * MARGE_YEUX[1]))
    return M


def placer(M, p, vers_camera):
    """Vertices p (Blender frame) set on each object of the model. The eye overlays have their origin
    at the centre of the head, moved one step towards the camera per rank: Eevee draws blended objects
    from farthest to nearest, hence in the order of the PMX materials, as in MMD."""
    pas = 1e-3 * M["taille"]
    for rang, o in enumerate(M["obs"]):
        loc = Vector(M["centre"]) + vers_camera * (pas * rang) if rang else Vector((0.0, 0.0, 0.0))
        o.location = loc
        o.data.vertices.foreach_set("co", (p - np.asarray(loc)).astype(np.float32).ravel())
        o.data.update()


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    coiffure = a[0] == "--coiffure"
    a = a[1:] if coiffure else a
    colonnes = COLONNES_COIFFURE if coiffure else COLONNES
    garder = os.environ.get("PLANCHE_COLONNES")  # e.g. "face,trois_quarts": quick comparison sheet
    if garder:
        colonnes = tuple(c for c in colonnes if c[0] in garder.split(","))
    sortie, paires = a[0], list(zip(a[1::2], a[2::2]))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    # dithered transparency (hair, eyelashes); the eye overlays are blended (materiau()).
    # PLANCHE_ECHANTILLONS=32 to compare candidates (software render without GPU: ~3 min per tile at 256)
    sc.eevee.taa_render_samples = int(os.environ.get("PLANCHE_ECHANTILLONS", "256"))
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
    tuiles = os.path.splitext(sortie)[0] + "_tuiles"
    os.makedirs(tuiles, exist_ok=True)
    modeles = [charger(p, d, "m%d" % i, coiffure) for i, (p, d) in enumerate(paires)]
    images = []
    for i, M in enumerate(modeles):
        for o in modeles:
            for ob in o["obs"]:
                ob.hide_render = o is not M
        rang = []
        for nc, az, poids in colonnes:
            p = conv(M["pos"] + decalages(M["P"], poids))
            cam.ortho_scale = (1.1 if coiffure else 1.9) * M["taille"]
            cen = M["centre"]
            sc.render.resolution_x, sc.render.resolution_y = TX, TY
            if "yeux" in M:  # same scale: px per unit = max(TX, TY) / ortho_scale of the full tile
                cen, lx, ly = M["yeux"]
                px = max(TX, TY) / cam.ortho_scale
                sc.render.resolution_x, sc.render.resolution_y = int(round(lx * px)), int(round(ly * px))
                cam.ortho_scale = max(lx, ly)
            r = math.radians(az)
            placer(M, p, Vector((math.sin(r), -math.cos(r), 0.0)))
            co.location = cen + Vector((math.sin(r), -math.cos(r), 0.0)) * 10
            co.rotation_euler = (cen - co.location).to_track_quat("-Z", "Y").to_euler()
            f = os.path.join(tuiles, "%d_%s.png" % (i, nc))
            sc.render.filepath = f
            bpy.ops.render.render(write_still=True)
            rang.append(f)
        images.append(rang)
    nl, nc = len(images), len(colonnes)
    pix = []
    for rang in images:
        pix.append([])
        for f in rang:
            im = bpy.data.images.load(f)
            w, h = im.size
            pix[-1].append(np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4))
            bpy.data.images.remove(im)
    ty = max(p.shape[0] for r in pix for p in r)  # tiles framed on the eyes: sizes specific to each head
    tx = max(p.shape[1] for r in pix for p in r)
    tout = np.ones((nl * ty, nc * tx, 4), dtype=np.float32)
    for i, rang in enumerate(pix):
        for j, px in enumerate(rang):
            y0 = (nl - 1 - i) * ty
            tout[y0:y0 + px.shape[0], j * tx:j * tx + px.shape[1]] = px
    for i in range(1, nl):
        tout[i * ty - 1:i * ty + 1] = (0.4, 0.4, 0.4, 1)
    for j in range(1, nc):
        tout[:, j * tx - 1:j * tx + 1] = (0.4, 0.4, 0.4, 1)
    out = bpy.data.images.new("planche", nc * tx, nl * ty, alpha=True)
    out.pixels[:] = tout.ravel()
    out.filepath_raw = sortie
    out.file_format = "PNG"
    out.save()
    print("contact sheet:", sortie)


main()
