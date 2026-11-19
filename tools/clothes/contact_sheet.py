"""Garment contact sheets (Eevee, unshaded texture colour, orthographic camera).

    bin/blender-queue -b --factory-startup --python tools/clothes/contact_sheet.py -- \
        --clothes <output.png> [--only id1,id2 | --pmx CANDIDATE.pmx --mats A,B [--id NAME]]
        Three tiles per garment of measure.CLOTHES (two garments per row): whole model
        (front), isolated garment from the front and from the back; parts of the same materials that
        were set aside (accessories, outside the regions) in magenta, the rest of the model hidden.
    bin/blender-queue -b --factory-startup --python tools/clothes/contact_sheet.py -- \
        --preview <output.png> Char1,Char2=Folder/x.pmx,...
        Whole models from the front and the back (for choosing garments).

Textures are read next to the PMX (read-only). Nothing is written except the requested PNG.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as MD  # noqa: E402

TX, TY = 300, 520
MAGENTA = (1.0, 0.0, 1.0, 1.0)


def empty():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras):
        for x in list(coll):
            coll.remove(x)


def material(name, image, diffuse=(1, 1, 1, 1), color=None):
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
    if color is not None:
        em.inputs["Color"].default_value = color
        mix.inputs[0].default_value = 1.0
    elif image is not None:
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
        m.surface_render_method = "DITHERED"
    except Exception:
        m.blend_method = "HASHED"
    m.use_backface_culling = False
    return m


def obj(M, faces, name, cache, magenta=None):
    """Blender object of the given faces (indices into M.F), with the PMX textures; magenta: mask of
    faces (within `faces`) rendered in magenta."""
    F = M.F[faces]
    mf = M.mf[faces].copy()
    mats = []
    folder = os.path.dirname(M.path)
    for k, m in enumerate(M.mats):
        t = m["texture"]
        img = None
        if 0 <= t < len(M.textures):
            f = os.path.normpath(os.path.join(folder, M.textures[t].replace("\\", "/")))
            if os.path.exists(f):
                img = cache.get(f) or bpy.data.images.load(f)
                img.colorspace_settings.name = "sRGB"
                cache[f] = img
        mats.append(material("%s_%d" % (name, k), img, m["diffuse"]))
    mats.append(material(name + "_magenta", None, color=MAGENTA))
    if magenta is not None:
        mf[magenta] = len(M.mats)
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in M.pos], [], [tuple(int(x) for x in f) for f in F])
    me.update()
    lay = me.uv_layers.new(name="UV")
    lu = M.uv[F.ravel()].copy()
    lu[:, 1] = 1 - lu[:, 1]
    lay.data.foreach_set("uv", lu.astype(np.float32).ravel())
    for m in mats:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", mf.astype(np.int32))
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def scene_rendered():
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
        e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = TX, TY
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("background") if not bpy.data.worlds else bpy.data.worlds[0]
    sc.world = w
    w.color = (0.93, 0.93, 0.93)
    try:
        w.use_nodes = True
        bg = w.node_tree.nodes.get("Background")
        bg.inputs[0].default_value = (0.93, 0.93, 0.93, 1)
    except Exception:
        pass
    try:
        sc.eevee.taa_render_samples = 16
    except Exception:
        pass
    return sc


def render(lo, hi, angle, path):
    """Orthographic camera framed on the box [lo, hi], rotated by `angle` degrees around Z
    (0 = front: the camera is at -Y looking towards +Y)."""
    sc = bpy.context.scene
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    c = (lo + hi) / 2
    h, l = hi[2] - lo[2], max(hi[0] - lo[0], hi[1] - lo[1])
    cam_d.ortho_scale = 1.08 * max(h, l * TY / TX)
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    a = math.radians(angle)
    d = 5.0
    cam.location = Vector((c[0] + d * math.sin(a), c[1] - d * math.cos(a), c[2]))
    cam.rotation_euler = (math.pi / 2, 0, a)
    cam_d.clip_end = 20
    sc.camera = cam
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


def assemble(tiles, path, columns, labels=None):
    """Pastes the tile PNGs (list of rows) into one contact sheet, using Blender (no PIL)."""
    lines = len(tiles)
    W, H = TX * columns, TY * lines
    out = np.ones((H, W, 4), np.float32)
    for i, line in enumerate(tiles):
        for j, f in enumerate(line):
            if f is None or not os.path.exists(f):
                continue
            im = bpy.data.images.load(f)
            px = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)
            y0 = H - (i + 1) * TY
            out[y0:y0 + TY, j * TX:(j + 1) * TX] = px
            bpy.data.images.remove(im)
    # separator line between rows
    for i in range(1, lines):
        out[H - i * TY - 1:H - i * TY + 1] = (0.3, 0.3, 0.3, 1)
    img = bpy.data.images.new("sheet", W, H, alpha=True)
    img.pixels.foreach_set(out.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()


def main(argv):
    tmp = os.environ.get("TMPDIR", "/tmp")
    if argv[0] == "--preview":
        output, characters = argv[1], argv[2].split(",")
        tiles = []
        for p in characters:
            empty()
            scene_rendered()
            # "Name" (library) or "Name=path.pmx" (relative to the official sources)
            p, _, pmx = p.partition("=")
            M = MD.model_of(p, pmx or None)
            guard = ~M.faces_role(["empty"])
            obj(M, np.where(guard)[0], p, {})
            lo, hi = M.pos.min(0), M.pos.max(0)
            line = []
            for ang, name in ((0, "face"), (180, "back")):
                f = os.path.join(tmp, "ap_%s_%s.png" % (p, name))
                render(lo, hi, ang, f)
                line.append(f)
            tiles.append(line)
        # preview: two models per row
        lines = [tiles[i] + (tiles[i + 1] if i + 1 < len(tiles) else [None, None])
                  for i in range(0, len(tiles), 2)]
        assemble(lines, output, 4)
        print("contact sheet written:", output)
        return
    if argv[0] == "--clothes":
        import measure as measure
        output = argv[1]
        alone = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
        tiles = []
        models = {}
        specs = measure.CLOTHES
        if "--pmx" in argv:
            path = argv[argv.index("--pmx") + 1]
            name = argv[argv.index("--id") + 1] if "--id" in argv else os.path.splitext(os.path.basename(path))[0]
            models[name] = MD.Model(name, path=path)
            specs = [{"id": name, "character": name, "mats": argv[argv.index("--mats") + 1].split(",")}]
            alone = None
        for spec in specs:
            if alone and spec["id"] not in alone:
                continue
            empty()
            scene_rendered()
            M = models.get(spec["character"]) or MD.model_of(spec["character"], spec.get("pmx"))
            models[spec["character"]] = M
            sel, gard, gap = MD.isolate(M, spec)
            cache = {}
            line = []
            # 1. whole model, from the front
            all_ = obj(M, np.where(~M.faces_role(["empty"]))[0], "all", cache)
            lo, hi = M.pos.min(0), M.pos.max(0)
            f = os.path.join(tmp, "pv_%s_tout.png" % spec["id"])
            render(lo, hi, 0, f)
            line.append(f)
            bpy.data.objects.remove(all_, do_unlink=True)
            # 2-3. isolated garment + set-aside parts in magenta
            discarded = np.zeros(len(M.F), bool)
            mats = [i for i, n in enumerate(M.names_mat) if n in spec["mats"]]
            in_ = np.isin(M.mf, mats)
            if spec.get("uv_boxes"):
                in_ &= MD.zone_uv(M, spec)
            for fk in M.parts(in_):
                if not sel[fk[0]]:
                    discarded[fk] = True
            faces = np.where(sel | discarded)[0]
            o = obj(M, faces, "garment", cache, magenta=discarded[faces])
            vs = np.unique(M.F[sel])
            lo, hi = M.pos[vs].min(0), M.pos[vs].max(0)
            for ang, name in ((0, "face"), (180, "back")):
                f = os.path.join(tmp, "pv_%s_%s.png" % (spec["id"], name))
                render(lo, hi, ang, f)
                line.append(f)
            tiles.append(line)
            print("rendered", spec["id"], "kept", len(gard), "set aside", len(gap))
        # two garments per row
        lines = [tiles[i] + (tiles[i + 1] if i + 1 < len(tiles) else [None] * 3)
                  for i in range(0, len(tiles), 2)]
        assemble(lines, output, 6)
        print("contact sheet written:", output)


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(a)
