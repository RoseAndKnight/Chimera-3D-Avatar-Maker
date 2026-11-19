"""Toon renders of a built garment: two-tone shading, outline via an inverted shell, test-bench poses.

    bin/blender-queue -b --factory-startup --python tools/clothes/render.py -- \
        --pmx result.pmx --output FOLDER [--views face,three_quarters,back,arms_raised,collar,sleeve] (+ shoulder)
        [--samples 16] [--size 720] [--sheet sheet.png] [--mat T恤] [--frame top|body]
    Lower body: --frame body (overall views of the whole body, feet included) and the views
    squat, step, waist, waist_back, buttocks, ankle, e.g. --mat 裤子 --frame body --views
    face,three_quarters,back,squat,arms_raised,waist_back,waist,ankle,face_close

Why not contact_sheet.py: emission alone shows the texture but not the shape (the bridge over the
cleavage, the thickness of hems, painted folds against the actual shading).
Here: Diffuse -> Shader to RGB -> constant ramp (two tones, like MMD toon), x texture x
diffuse; outline = flipped Solidify shell, with the PMX outline colour and size (flag 0x10),
thickness following each vertex's outline weight.
Views: orthographic camera; "face" looks at the character from the front (-Y in Blender = front).
Poses: test bench from tools/atlas/bench.py (P1 = arms raised), skinning by primitives.deform.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as MD  # noqa: E402
from model import P  # noqa: E402
import bench as BENCH  # noqa: E402

# name: (pose, azimuth deg, framing); framing = (centre z or bone, half-height m, x offset m)
VIEWS = {
    "face": (None, 0, ("top", None)),
    "three_quarters": (None, 35, ("top", None)),
    "back": (None, 180, ("top", None)),
    "arms_raised": ("P1", 20, ("top_raised", None)),
    "collar": (None, 15, ("首", 0.13)),
    "sleeve": (None, 70, ("左腕", 0.16)),
    "shoulder": ("P1", 20, ("右肩", 0.07)),     # close-up, arms raised: drawstring, strands on the shoulder
    "face_close": (None, 0, ("頭", 0.10)),       # face close-up (tint of the grafted head)
    "face_34": (None, 35, ("頭", 0.10)),   # face seam: cheek, jaw, neck
    "face_profile": (None, 90, ("頭", 0.10)),
    "profile": (None, 90, ("頭", 0.11)),
    "under_chin": (None, 22, ("首", 0.11), -28),  # low angle under the chin and jaw
    "head_back": (None, 180, ("頭", 0.12)),
    "nape": (None, 125, ("首", 0.14)),       # back seam: top of the head's neck over the body's neck
    # lower body (trousers); with --frame body, the "top" views take in the whole body
    "squat": ("P7@D", 35, ("top", None)),   # bench squat: hips 110 deg, knees 140 deg
    "step": ("P12", 60, ("top", None)),      # walking step
    "seat": ("ASSIS", 40, ("top", None)),    # seated on a chair: hips and knees at 90 deg
    "leg_bent": ("P5@60", 40, ("top", None)),  # left thigh raised 60 deg
    "waist": (None, 30, ("下半身", 0.16)),      # close-up of the top / trousers seam
    "waist_back": (None, 180, ("下半身", 0.16)),
    "buttocks": (None, 180, ("下半身", 0.13)),     # back of the waist and buttocks: back pockets
    "ankle": (None, 30, ("左足首", 0.11)),    # close-up of the lower leg
}
SHEET = ["face", "three_quarters", "back", "arms_raised", "collar", "sleeve"]


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--pmx", required=True)
    p.add_argument("--output", dest="output", required=True)
    p.add_argument("--views", dest="views", default=",".join(SHEET))
    p.add_argument("--samples", dest="samples", type=int, default=16)
    p.add_argument("--size", dest="size", type=int, default=720)
    p.add_argument("--sheet", dest="sheet", default=None)
    p.add_argument("--mat", default="T恤")
    p.add_argument("--frame", dest="frame", choices=("top", "body"), default="top",
                   help="overall views: top = garment and head (default), body = whole body")
    return p.parse_args(a)


def empty():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for b in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.worlds):
        for x in list(b):
            b.remove(x)


def mat_toon(name, image, diffuse, shade, skin=False):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    dif = nt.nodes.new("ShaderNodeBsdfDiffuse")
    s2r = nt.nodes.new("ShaderNodeShaderToRGB")
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    rp = nt.nodes.new("ShaderNodeValToRGB")
    rp.color_ramp.interpolation = "EASE" if skin else "CONSTANT"
    rp.color_ramp.elements[0].position = 0.20 if skin else 0.0
    rp.color_ramp.elements[0].color = (*shade, 1)
    rp.color_ramp.elements[1].position = 0.75 if skin else 0.45
    rp.color_ramp.elements[1].color = (1, 1, 1, 1)
    nt.links.new(dif.outputs[0], s2r.inputs[0])
    nt.links.new(s2r.outputs["Color"], bw.inputs[0])
    nt.links.new(bw.outputs[0], rp.inputs[0])
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    nt.links.new(rp.outputs["Color"], mul.inputs["B"])
    em = nt.nodes.new("ShaderNodeEmission")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mul.outputs["Result"], em.inputs["Color"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    if image is not None:
        tx = nt.nodes.new("ShaderNodeTexImage")
        tx.image = image
        md = nt.nodes.new("ShaderNodeMix")
        md.data_type, md.blend_type = "RGBA", "MULTIPLY"
        md.inputs["Factor"].default_value = 1.0
        nt.links.new(tx.outputs["Color"], md.inputs["A"])
        md.inputs["B"].default_value = (*diffuse[:3], 1)
        nt.links.new(md.outputs["Result"], mul.inputs["A"])
        ma = nt.nodes.new("ShaderNodeMath")
        ma.operation = "MULTIPLY"
        ma.inputs[1].default_value = diffuse[3]
        nt.links.new(tx.outputs["Alpha"], ma.inputs[0])
        nt.links.new(ma.outputs[0], mix.inputs[0])
    else:
        mul.inputs["A"].default_value = (*[0.8 * x for x in diffuse[:3]], 1)
        mix.inputs[0].default_value = diffuse[3]
    try:
        m.surface_render_method = "DITHERED"
    except Exception:
        pass
    m.use_backface_culling = False
    return m


def mat_outline(name, color, visible):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if visible:
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*color[:3], 1)
        nt.links.new(em.outputs[0], out.inputs[0])
    else:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        nt.links.new(tr.outputs[0], out.inputs[0])
    m.use_backface_culling = True
    try:
        m.surface_render_method = "DITHERED"
    except Exception:
        pass
    return m


def srgb_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def obj(M, pos, name_garment):
    folder = os.path.dirname(M.path)
    n = len(M.mats)
    mats, outlines = [], []
    cache = {}
    for k, m in enumerate(M.mats):
        t = m["texture"]
        img = None
        if 0 <= t < len(M.textures):
            f = os.path.normpath(os.path.join(folder, M.textures[t].replace("\\", "/")))
            if os.path.exists(f):
                img = cache.get(f) or bpy.data.images.load(f)
                cache[f] = img
        skin = M.role_material(k) == "skin" or m["name"] in ("颜", "颜_A", "肌_cou", "首")
        shade = (0.86, 0.70, 0.68) if skin else (0.74, 0.74, 0.84)
        mats.append(mat_toon("m%d" % k, img, m["diffuse"], shade, skin=skin))
        cc = srgb_lin(m["outline_color"][:3])
        outlines.append(mat_outline("c%d" % k, cc, bool(m["flags"] & 0x10)))
    ns = np.asarray(M.nor, float)
    ns = ns / np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-12)
    pos_c = pos + (0.0011 * np.asarray(M.outline, float))[:, None] * ns
    nv = len(pos)
    all_pos = np.vstack([pos, pos_c])
    with_c = np.array([bool(m["flags"] & 0x10) for m in M.mats], bool)
    sel_c = with_c[M.mf] & (np.asarray(M.outline, float)[M.F] > 0.05).any(1)
    Fc = M.F[sel_c][:, [0, 2, 1]] + nv
    tout_F = np.vstack([M.F, Fc])
    me = bpy.data.meshes.new("model")
    me.from_pydata([tuple(p) for p in all_pos], [], [tuple(int(x) for x in f) for f in tout_F])
    me.update()
    lay = me.uv_layers.new(name="UV")
    lu = M.uv[M.F.ravel()].copy()
    lu[:, 1] = 1 - lu[:, 1]
    read_c = M.uv[M.F[sel_c][:, [0, 2, 1]].ravel()].copy()
    read_c[:, 1] = 1 - read_c[:, 1]
    lay.data.foreach_set("uv", np.vstack([lu, read_c]).astype(np.float32).ravel())
    for m in mats + outlines:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", np.concatenate([M.mf, M.mf[sel_c] + n]).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(tout_F), bool))
    me.update()
    try:
        me.normals_split_custom_set_from_vertices([tuple(float(x) for x in v) for v in np.vstack([ns, -ns])])
    except Exception:
        pass
    o = bpy.data.objects.new("model", me)
    bpy.context.scene.collection.objects.link(o)
    return o


def scene(size, samp):
    sc = bpy.context.scene
    names = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in names else "BLENDER_EEVEE"
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("background")
    sc.world = w
    w.color = (0.9, 0.9, 0.92)
    if w.node_tree is not None:
        bg = w.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs[0].default_value = (0.8, 0.8, 0.83, 1)
    try:
        sc.eevee.taa_render_samples = samp
        sc.eevee.use_raytracing = False
        sc.eevee.use_shadows = False
    except Exception:
        pass
    sl = bpy.data.lights.new("soleil", "SUN")
    sl.energy = 3.0
    try:
        sl.use_shadow = False
    except Exception:
        pass
    so = bpy.data.objects.new("soleil", sl)
    so.rotation_euler = (math.radians(50), 0, math.radians(-35))
    sc.collection.objects.link(so)
    return sc


def render(center, half, azimuth, path, elev=0):
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("cam")
    cd.type = "ORTHO"
    cd.ortho_scale = 2 * half
    cd.clip_end = 20
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    a, e = math.radians(azimuth), math.radians(elev)
    cam.location = Vector((center[0] + 5 * math.sin(a) * math.cos(e),
                           center[1] - 5 * math.cos(a) * math.cos(e),
                           center[2] + 5 * math.sin(e)))
    cam.rotation_euler = (math.pi / 2 - e, 0, a)
    sc.camera = cam
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


def assemble(paths, path, columns, size):
    lines = (len(paths) + columns - 1) // columns
    W, H = size * columns, size * lines
    out = np.ones((H, W, 4), np.float32)
    for k, f in enumerate(paths):
        i, j = divmod(k, columns)
        im = bpy.data.images.load(f)
        px = np.empty(im.size[0] * im.size[1] * 4, np.float32)
        im.pixels.foreach_get(px)
        px = px.reshape(im.size[1], im.size[0], 4)
        y0 = H - (i + 1) * size
        out[y0:y0 + size, j * size:(j + 1) * size] = px[:size, :size]
        bpy.data.images.remove(im)
    img = bpy.data.images.new("sheet", W, H, alpha=True)
    img.pixels.foreach_set(out.ravel())
    tmp = path[:-4] + ".tmp.png"
    img.filepath_raw = tmp
    img.file_format = "PNG"
    img.save()
    os.replace(tmp, path)


def main():
    a = args()
    os.makedirs(a.output, exist_ok=True)
    M = MD.Model("render", path=os.path.abspath(a.pmx))
    garment = np.isin(M.mf, [i for i, m in enumerate(M.mats) if m["name"] == a.mat])
    made = []
    pose_prev = "__aucune__"
    pos = M.pos
    for name in a.views.split(","):
        v_info = VIEWS[name]
        pose, az, (frame, half) = v_info[:3]
        elev = v_info[3] if len(v_info) > 3 else 0
        if pose != pose_prev:
            pos = M.pos
            if pose:
                rot = BENCH.pose_bench(pose, M.sq)
                pos = P.deform(M.pos, M.idx, M.weight, M.sq, rotations=rot)[0]
            empty()
            scene(a.size, a.samples)
            obj(M, pos, a.mat)
            pose_prev = pose
        if frame in ("top", "top_raised"):
            pv = pos[np.unique(M.F[garment])] if a.frame == "top" else pos
            lo, hi = pv.min(0), pv.max(0)
            ztop = pos[:, 2].max()
            zlo = lo[2] - (0.12 if a.frame == "top" else 0.03)
            c = np.array([0.0, 0.5 * (lo[1] + hi[1]), 0.5 * (zlo + ztop)])
            half = 0.53 * max(ztop - zlo, (hi[0] - lo[0]) + 0.2)
        else:
            b = M.names_os.index(frame)
            c = M.heads[b].copy()
            if pose:   # posed bone: centre of the vertices it weights most
                c = pos[M.dom == b].mean(0)
            if frame == "首":
                c[2] += 0.01 if name == "under_chin" else -0.04
            if frame == "頭":
                c[2] += 0.07
            if frame == "下半身":
                c[2] -= 0.12 if name == "buttocks" else 0.03
            if frame == "左足首":
                c[2] += 0.03
            if frame == "左腕":
                i = M.names_os.index("左ひじ")
                c = 0.5 * (c + M.heads[i])
        f = os.path.join(a.output, name + ".png")
        render(c, half, az, f, elev=elev)
        made.append(f)
        print("view", name, "->", f)
    if a.sheet:
        assemble(made, os.path.join(a.output, a.sheet), 3, a.size)
        print("contact sheet", os.path.join(a.output, a.sheet))


if __name__ == "__main__":
    main()
