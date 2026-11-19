"""Grey contact sheet (Workbench, orthographic): source / target / result as rows; front, 3/4,
profile, then blink (まばたき), smile (笑い + 口角上げ) and "あ" from the front as columns.
Hair and shadow overlays hidden: the skull, the face and the seam are visible.

    blender -b --factory-startup --python head_sheet.py -- <mode> <source> <target> <result> <sortie>
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
import pmx as PMX  # noqa: E402

SKINS = ("颜", "颜_A", "颜_A2")
HIDDEN = ("髮", "发", "目影", "二重影", "瞳光")
COLUMNS = (("face", 0, {}), ("three_quarters", 35, {}), ("profile", 90, {}),
            ("blink", 0, {"まばたき": 1.0}), ("smile", 0, {"笑い": 1.0, "口角上げ": 1.0}),
            ("a", 0, {"あ": 1.0}))
TX, TY = 360, 420


def offsets(P, weight):
    """Sum of the requested vertex morphs (a group morph is expanded one level)."""
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


def load(path, name):
    P = PMX.read(path)
    pos = np.asarray(P["vertices"]["pos"], dtype=float)
    F = np.asarray(P["faces"], dtype=np.int64)
    mf = np.repeat(np.arange(len(P["materials"])), [m["n_indices"] // 3 for m in P["materials"]])
    names = [m["name"] for m in P["materials"]]
    guard = np.array([not any(c in names[k] for c in HIDDEN) for k in mf])
    skin = np.isin(mf, [i for i, n in enumerate(names) if n in SKINS])
    vp = np.unique(F[skin])
    lo, hi = pos[vp].min(0), pos[vp].max(0)
    # the body below the neck is not used: faces below the bottom of the face skin are removed
    bottom = lo[1] - 0.35 * (hi[1] - lo[1])
    guard &= pos[F].mean(1)[:, 1] > bottom
    F = F[guard]
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in conv(pos)], [], [tuple(int(x) for x in f) for f in F])
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    center = conv(((lo + hi) / 2)[None])[0]
    size = float(max(hi[1] - lo[1], hi[0] - lo[0]))
    return {"P": P, "pos": pos, "ob": ob, "center": Vector(center), "size": size}


def conv(p):
    """PMX (Y up, face towards -Z) -> Blender (Z up, face towards -Y)."""
    return np.stack([p[:, 0], p[:, 2], p[:, 1]], 1)


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    mode, source, target, result, output = a[:5]
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
    tiles = os.path.join(output, "sheet")
    os.makedirs(tiles, exist_ok=True)
    lines = (("source", source), ("target", target), ("result", result))
    models = [load(c, n) for n, c in lines]
    images = []
    for (nl, _), M in zip(lines, models):
        for o in models:
            o["ob"].hide_render = o is not M
        rank = []
        for nc, az, weight in COLUMNS:
            p = conv(M["pos"] + offsets(M["P"], weight))
            M["ob"].data.vertices.foreach_set("co", p.astype(np.float32).ravel())
            M["ob"].data.update()
            cam.ortho_scale = 1.9 * M["size"]
            r = math.radians(az)
            cen = M["center"]
            co.location = cen + Vector((math.sin(r), -math.cos(r), 0.0)) * 10
            co.rotation_euler = (cen - co.location).to_track_quat("-Z", "Y").to_euler()
            f = os.path.join(tiles, "%s_%s.png" % (nl, nc))
            sc.render.filepath = f
            bpy.ops.render.render(write_still=True)
            rank.append(f)
        images.append(rank)
    # assembly (the top row of the image is the last row of pixels)
    nl, nc = len(images), len(COLUMNS)
    all_ = np.ones((nl * TY, nc * TX, 4), dtype=np.float32)
    for i, rank in enumerate(images):
        for j, f in enumerate(rank):
            im = bpy.data.images.load(f)
            px = np.array(im.pixels[:], dtype=np.float32).reshape(TY, TX, 4)
            y0 = (nl - 1 - i) * TY
            all_[y0:y0 + TY, j * TX:(j + 1) * TX] = px
            bpy.data.images.remove(im)
    for i in range(1, nl):
        all_[i * TY - 1:i * TY + 1] = (0.5, 0.5, 0.5, 1)
    for j in range(1, nc):
        all_[:, j * TX - 1:j * TX + 1] = (0.5, 0.5, 0.5, 1)
    out = bpy.data.images.new("sheet", nc * TX, nl * TY, alpha=True)
    out.pixels[:] = all_.ravel()
    path = os.path.join(output, "sheet.png")
    tmp = os.path.join(output, "sheet.tmp.png")
    out.filepath_raw = tmp
    out.file_format = "PNG"
    out.save()
    os.replace(tmp, path)
    print("contact sheet:", path, "(rows: source, target, result; columns: %s)" % ", ".join(c[0] for c in COLUMNS))


main()
