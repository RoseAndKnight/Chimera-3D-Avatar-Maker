"""Reads a .vrm back in Blender and poses it: does the file read back, do the bones move without breaking?

    blender -b --factory-startup --python tools/workshop/check_vrm.py -- file.vrm [--output folder]

With Blender's VRM add-on if it loads (import_scene.vrm), otherwise through the native glTF import, VRM
being a GLB. Test pose: arms lowered, one leg raised, head turned, torso leaning, plus the "blink"
expression. Returns a JSON report and, with --output, an image of the pose. Exit code 1 if broken.
"""
import json
import math
import os
import shutil
import sys

import numpy as np
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vrm as VR  # noqa: E402

STRETCHED_MAX = 4.0    # an edge does not stretch beyond this factor in the pose (broken beyond)


def importer(path, tmp):
    """Returns the name of the method used."""
    try:
        import addon_utils
        for m in addon_utils.modules():
            if "vrm" in m.__name__.lower():
                addon_utils.enable(m.__name__)
        if hasattr(bpy.ops.import_scene, "vrm"):
            r = bpy.ops.import_scene.vrm(filepath=path)
            if "FINISHED" in r:
                return "VRM add-on"
    except Exception as e:
        print("VRM add-on unavailable:", e)
    copied = os.path.join(tmp, "trial.glb")
    shutil.copy(path, copied)
    bpy.ops.import_scene.gltf(filepath=copied)
    return "import glTF"


def turn(arm, name, axis, degrees):
    pb = arm.pose.bones[name]
    t = Matrix.Translation(pb.head)
    pb.matrix = t @ Matrix.Rotation(math.radians(degrees), 4, axis) @ t.inverted() @ pb.matrix
    bpy.context.view_layer.update()


def positions(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = ob.evaluated_get(dg).to_mesh()
    v = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", v)
    a = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", a)
    ob.evaluated_get(dg).to_mesh_clear()
    return v.reshape(-1, 3), a.reshape(-1, 2)


def without_reflections(meshes):
    """Removes the "+" highlight layers from the image: lying exactly on top of the hair, they
    make it vanish under ray tracing (a real-time render does not have this defect)."""
    import bmesh
    for o in meshes:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if "+" in o.data.materials[f.material_index].name], context="FACES")
        bm.to_mesh(o.data)
        bm.free()


def place_camera(target, view):
    cam = bpy.data.objects.new("c", bpy.data.cameras.new("c"))
    bpy.context.collection.objects.link(cam)
    cam.data.lens = 70
    cam.location = Vector(view)
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    path = os.path.abspath(a[0])
    output = a[a.index("--output") + 1] if "--output" in a else None
    bpy.ops.wm.read_factory_settings(use_empty=True)
    tmp = os.path.join(os.environ.get("TMPDIR") or "/tmp", "verif-vrm-%d" % os.getpid())
    os.makedirs(tmp, exist_ok=True)
    defaults = VR.validate(path)
    doc, _ = VR.read(path)
    ext = doc["extensions"]["VRMC_vrm"]
    method = importer(path, tmp)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]
    os_of = {k: doc["nodes"][v["node"]]["name"] for k, v in ext["humanoid"]["humanBones"].items()}
    missing = [v for v in os_of.values() if v not in arm.pose.bones]
    if missing:
        defaults.append("humanoid bones missing after import: %s" % ", ".join(missing[:5]))
    before = {o.name: positions(o) for o in meshes}
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    x, y, z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    # Blender frame after glTF import: character facing -Y, its left towards +X, Z up
    for name, axis, deg in (("leftUpperArm", y, 70), ("rightUpperArm", y, -70), ("leftLowerArm", z, 50), ("leftUpperLeg", x, 60), ("leftLowerLeg", x, -70),
                          ("rightUpperLeg", x, -20), ("head", z, 25), ("spine", x, 15)):
        turn(arm, os_of[name], axis, deg)
    bpy.ops.object.mode_set(mode="OBJECT")
    after = {}
    assessment_stretched = []
    moves = 0.0
    for o in meshes:
        p0, ar = before[o.name]
        p1, _ = positions(o)
        after[o.name] = p1
        moves = max(moves, float(np.linalg.norm(p1 - p0, axis=1).max()))
        if np.isnan(p1).any():
            defaults.append("%s: NaN positions in the pose" % o.name)
        l0 = np.linalg.norm(p0[ar[:, 0]] - p0[ar[:, 1]], axis=1)
        l1 = np.linalg.norm(p1[ar[:, 0]] - p1[ar[:, 1]], axis=1)
        r = l1 / np.maximum(l0, 1e-5)
        ok = l0 > 5e-4   # below this, edges say nothing
        q = float(np.percentile(r[ok], 99.9))
        assessment_stretched.append({"mesh": o.name, "edges": int(ok.sum()), "p99_9": round(q, 2), "max": round(float(r[ok].max()), 2)})
        if q > STRETCHED_MAX:
            defaults.append("%s: edges stretched x%.1f in the pose" % (o.name, q))
    if moves < 0.1:
        defaults.append("the pose moves nothing (%.3f m)" % moves)
    # expressions: the "blink" key must move vertices
    expr = ext["expressions"]["preset"]
    n_keys = 0
    moves_blink = 0.0
    for o in meshes:
        keys = o.data.shape_keys.key_blocks if o.data.shape_keys else []
        n_keys = max(n_keys, len(keys) - 1 if len(keys) else 0)
        if len(keys) and "blink" in expr:
            i = expr["blink"]["morphTargetBinds"][0]["index"]
            p0, _ = positions(o)
            keys[i + 1].value = 1.0
            bpy.context.view_layer.update()
            p1, _ = positions(o)
            moves_blink = float(np.linalg.norm(p1 - p0, axis=1).max())
            keys[i + 1].value = 0.0
    if expr and moves_blink < 1e-3:
        defaults.append("the blink expression moves nothing")
    report = {"method": method, "bones": len(arm.data.bones), "humanoids": len(os_of), "expressions": sorted(expr), "targets": n_keys,
               "pose_moved_m": round(moves, 3), "blink_moved_m": round(moves_blink, 4), "etirement": assessment_stretched,
               "materials_mtoon": sum(1 for m in doc["materials"] if "VRMC_materials_mtoon" in m.get("extensions", {})), "defaults": defaults}
    print("vrm check:", json.dumps(report, ensure_ascii=False))
    if output:
        os.makedirs(output, exist_ok=True)
        sc = bpy.context.scene
        without_reflections(meshes)
        sc.view_settings.view_transform = "Standard"
        sc.render.engine = "CYCLES"
        sc.cycles.samples = 8
        sc.render.resolution_x, sc.render.resolution_y = 640, 900
        sc.world = bpy.data.worlds.new("m")
        sc.world.use_nodes = True
        sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.85, 0.85, 0.85, 1)
        name = os.path.splitext(os.path.basename(path))[0]
        place_camera((0, 0, 0.88), (0.8, -3.6, 1.1))
        sc.render.filepath = os.path.join(output, "pose.png")
        bpy.ops.render.render(write_still=True)
        with open(os.path.join(output, "verif.json"), "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print("rendered", name)
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if defaults else 0)


if __name__ == "__main__":
    main()
