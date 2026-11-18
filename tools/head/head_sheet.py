"""Grey contact sheet (Workbench, orthographic): source / target / result as rows; front, 3/4,
profile, then blink (まばたき), smile (笑い + 口角上げ) and "あ" from the front as columns.
Hair and shadow overlays hidden: the skull, the face and the seam are visible.

    blender -b --factory-startup --python head_sheet.py -- <mode> <source> <cible> <resultat> <sortie>
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
import pmx as PMX  # noqa: E402

PEAUX = ("颜", "颜_A", "颜_A2")
CACHES = ("髮", "发", "目影", "二重影", "瞳光")
COLONNES = (("face", 0, {}), ("trois_quarts", 35, {}), ("profil", 90, {}),
            ("clignement", 0, {"まばたき": 1.0}), ("sourire", 0, {"笑い": 1.0, "口角上げ": 1.0}),
            ("a", 0, {"あ": 1.0}))
TX, TY = 360, 420


def decalages(P, poids):
    """Sum of the requested vertex morphs (a group morph is expanded one level)."""
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


def charger(chemin, nom):
    P = PMX.lire(chemin)
    pos = np.asarray(P["sommets"]["pos"], dtype=float)
    F = np.asarray(P["faces"], dtype=np.int64)
    mf = np.repeat(np.arange(len(P["materiaux"])), [m["n_indices"] // 3 for m in P["materiaux"]])
    noms = [m["nom"] for m in P["materiaux"]]
    garde = np.array([not any(c in noms[k] for c in CACHES) for k in mf])
    peau = np.isin(mf, [i for i, n in enumerate(noms) if n in PEAUX])
    vp = np.unique(F[peau])
    lo, hi = pos[vp].min(0), pos[vp].max(0)
    # the body below the neck is not used: faces below the bottom of the face skin are removed
    bas = lo[1] - 0.35 * (hi[1] - lo[1])
    garde &= pos[F].mean(1)[:, 1] > bas
    F = F[garde]
    me = bpy.data.meshes.new(nom)
    me.from_pydata([tuple(p) for p in conv(pos)], [], [tuple(int(x) for x in f) for f in F])
    me.update()
    ob = bpy.data.objects.new(nom, me)
    bpy.context.scene.collection.objects.link(ob)
    centre = conv(((lo + hi) / 2)[None])[0]
    taille = float(max(hi[1] - lo[1], hi[0] - lo[0]))
    return {"P": P, "pos": pos, "ob": ob, "centre": Vector(centre), "taille": taille}


def conv(p):
    """PMX (Y up, face towards -Z) -> Blender (Z up, face towards -Y)."""
    return np.stack([p[:, 0], p[:, 2], p[:, 1]], 1)


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    mode, source, cible, resultat, sortie = a[:5]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    d = sc.display.shading
    d.light = "STUDIO"
    d.color_type = "SINGLE"
    d.single_color = (0.7, 0.7, 0.7)
    d.show_backface_culling = False
    d.show_cavity = False
    sc.render.resolution_x, sc.render.resolution_y = TX, TY
    sc.render.image_settings.file_format = "PNG"
    sc.world = bpy.data.worlds.new("w")
    sc.world.color = (0.93, 0.93, 0.93)
    cam = bpy.data.cameras.new("c")
    cam.type = "ORTHO"
    cam.clip_end = 50
    co = bpy.data.objects.new("c", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    tuiles = os.path.join(sortie, "planche")
    os.makedirs(tuiles, exist_ok=True)
    lignes = (("source", source), ("cible", cible), ("resultat", resultat))
    modeles = [charger(c, n) for n, c in lignes]
    images = []
    for (nl, _), M in zip(lignes, modeles):
        for o in modeles:
            o["ob"].hide_render = o is not M
        rang = []
        for nc, az, poids in COLONNES:
            p = conv(M["pos"] + decalages(M["P"], poids))
            M["ob"].data.vertices.foreach_set("co", p.astype(np.float32).ravel())
            M["ob"].data.update()
            cam.ortho_scale = 1.9 * M["taille"]
            r = math.radians(az)
            cen = M["centre"]
            co.location = cen + Vector((math.sin(r), -math.cos(r), 0.0)) * 10
            co.rotation_euler = (cen - co.location).to_track_quat("-Z", "Y").to_euler()
            f = os.path.join(tuiles, "%s_%s.png" % (nl, nc))
            sc.render.filepath = f
            bpy.ops.render.render(write_still=True)
            rang.append(f)
        images.append(rang)
    # assembly (the top row of the image is the last row of pixels)
    nl, nc = len(images), len(COLONNES)
    tout = np.ones((nl * TY, nc * TX, 4), dtype=np.float32)
    for i, rang in enumerate(images):
        for j, f in enumerate(rang):
            im = bpy.data.images.load(f)
            px = np.array(im.pixels[:], dtype=np.float32).reshape(TY, TX, 4)
            y0 = (nl - 1 - i) * TY
            tout[y0:y0 + TY, j * TX:(j + 1) * TX] = px
            bpy.data.images.remove(im)
    for i in range(1, nl):
        tout[i * TY - 1:i * TY + 1] = (0.5, 0.5, 0.5, 1)
    for j in range(1, nc):
        tout[:, j * TX - 1:j * TX + 1] = (0.5, 0.5, 0.5, 1)
    out = bpy.data.images.new("planche", nc * TX, nl * TY, alpha=True)
    out.pixels[:] = tout.ravel()
    chemin = os.path.join(sortie, "planche.png")
    tmp = os.path.join(sortie, "planche.tmp.png")
    out.filepath_raw = tmp
    out.file_format = "PNG"
    out.save()
    os.replace(tmp, chemin)
    print("contact sheet:", chemin, "(rows: source, target, result; columns: %s)" % ", ".join(c[0] for c in COLONNES))


main()
