"""Toon renders of a built garment: two-tone shading, outline via an inverted shell, test-bench poses.

    bin/blender-file -b --factory-startup --python outils/vetements/rendu.py -- \
        --pmx resultat.pmx --sortie DOSSIER [--vues face,trois_quarts,dos,bras_leves,col,manche] (+ epaule)
        [--echantillons 16] [--taille 720] [--planche planche.png] [--mat T恤] [--cadre haut|corps]
    Lower body: --cadre corps (overall views of the whole body, feet included) and the views
    accroupi, marche, taille, taille_dos, fesses, cheville, e.g. --mat 裤子 --cadre corps --vues
    face,trois_quarts,dos,accroupi,bras_leves,taille_dos,taille,cheville,visage

Why not planche.py: emission alone shows the texture but not the shape (the bridge over the
cleavage, the thickness of hems, painted folds against the actual shading).
Here: Diffuse -> Shader to RGB -> constant ramp (two tones, like MMD toon), x texture x
diffuse; outline = flipped Solidify shell, with the PMX outline colour and size (flag 0x10),
thickness following each vertex's outline weight.
Views: orthographic camera; "face" looks at the character from the front (-Y in Blender = front).
Poses: test bench from outils/atlas/banc.py (P1 = arms raised), skinning by primitives.deformer.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import modele as MD  # noqa: E402
from modele import P  # noqa: E402
import banc as BANC  # noqa: E402

# name: (pose, azimuth deg, framing); framing = (centre z or bone, half-height m, x offset m)
VUES = {
    "face": (None, 0, ("haut", None)),
    "trois_quarts": (None, 35, ("haut", None)),
    "dos": (None, 180, ("haut", None)),
    "bras_leves": ("P1", 20, ("haut_leve", None)),
    "col": (None, 15, ("首", 0.13)),
    "manche": (None, 70, ("左腕", 0.16)),
    "epaule": ("P1", 20, ("右肩", 0.07)),     # close-up, arms raised: drawstring, strands on the shoulder
    "visage": (None, 0, ("頭", 0.10)),       # face close-up (tint of the grafted head)
    "visage_34": (None, 35, ("頭", 0.10)),   # face seam: cheek, jaw, neck
    "visage_profil": (None, 90, ("頭", 0.10)),
    "profil": (None, 90, ("頭", 0.11)),
    "dessous_menton": (None, 22, ("首", 0.11), -28),  # low angle under the chin and jaw
    "tete_dos": (None, 180, ("頭", 0.12)),
    "nuque": (None, 125, ("首", 0.14)),       # back seam: top of the head's neck over the body's neck
    # lower body (trousers); with --cadre corps, the "haut" views take in the whole body
    "accroupi": ("P7@D", 35, ("haut", None)),   # bench squat: hips 110 deg, knees 140 deg
    "marche": ("P12", 60, ("haut", None)),      # walking step
    "assise": ("ASSIS", 40, ("haut", None)),    # seated on a chair: hips and knees at 90 deg
    "jambe_pliee": ("P5@60", 40, ("haut", None)),  # left thigh raised 60 deg
    "taille": (None, 30, ("下半身", 0.16)),      # close-up of the top / trousers seam
    "taille_dos": (None, 180, ("下半身", 0.16)),
    "fesses": (None, 180, ("下半身", 0.13)),     # back of the waist and buttocks: back pockets
    "cheville": (None, 30, ("左足首", 0.11)),    # close-up of the lower leg
}
PLANCHE = ["face", "trois_quarts", "dos", "bras_leves", "col", "manche"]


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--pmx", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--vues", default=",".join(PLANCHE))
    p.add_argument("--echantillons", type=int, default=16)
    p.add_argument("--taille", type=int, default=720)
    p.add_argument("--planche", default=None)
    p.add_argument("--mat", default="T恤")
    p.add_argument("--cadre", choices=("haut", "corps"), default="haut",
                   help="overall views: haut = garment and head (default), corps = whole body")
    return p.parse_args(a)


def vider():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for b in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.worlds):
        for x in list(b):
            b.remove(x)


def mat_toon(nom, image, diffus, ombre, peau=False):
    m = bpy.data.materials.new(nom)
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    dif = nt.nodes.new("ShaderNodeBsdfDiffuse")
    s2r = nt.nodes.new("ShaderNodeShaderToRGB")
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    rp = nt.nodes.new("ShaderNodeValToRGB")
    rp.color_ramp.interpolation = "EASE" if peau else "CONSTANT"
    rp.color_ramp.elements[0].position = 0.20 if peau else 0.0
    rp.color_ramp.elements[0].color = (*ombre, 1)
    rp.color_ramp.elements[1].position = 0.75 if peau else 0.45
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
        md.inputs["B"].default_value = (*diffus[:3], 1)
        nt.links.new(md.outputs["Result"], mul.inputs["A"])
        ma = nt.nodes.new("ShaderNodeMath")
        ma.operation = "MULTIPLY"
        ma.inputs[1].default_value = diffus[3]
        nt.links.new(tx.outputs["Alpha"], ma.inputs[0])
        nt.links.new(ma.outputs[0], mix.inputs[0])
    else:
        mul.inputs["A"].default_value = (*[0.8 * x for x in diffus[:3]], 1)
        mix.inputs[0].default_value = diffus[3]
    try:
        m.surface_render_method = "DITHERED"
    except Exception:
        pass
    m.use_backface_culling = False
    return m


def mat_contour(nom, couleur, visible):
    m = bpy.data.materials.new(nom)
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if visible:
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*couleur[:3], 1)
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


def objet(M, pos, nom_vet):
    dossier = os.path.dirname(M.chemin)
    n = len(M.mats)
    mats, contours = [], []
    cache = {}
    for k, m in enumerate(M.mats):
        t = m["texture"]
        img = None
        if 0 <= t < len(M.textures):
            f = os.path.normpath(os.path.join(dossier, M.textures[t].replace("\\", "/")))
            if os.path.exists(f):
                img = cache.get(f) or bpy.data.images.load(f)
                cache[f] = img
        peau = M.role_materiau(k) == "peau" or m["nom"] in ("颜", "颜_A", "肌_cou", "首")
        ombre = (0.86, 0.70, 0.68) if peau else (0.74, 0.74, 0.84)
        mats.append(mat_toon("m%d" % k, img, m["diffus"], ombre, peau=peau))
        cc = srgb_lin(m["contour_couleur"][:3])
        contours.append(mat_contour("c%d" % k, cc, bool(m["drapeaux"] & 0x10)))
    ns = np.asarray(M.nor, float)
    ns = ns / np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-12)
    pos_c = pos + (0.0011 * np.asarray(M.contour, float))[:, None] * ns
    nv = len(pos)
    tout_pos = np.vstack([pos, pos_c])
    avec_c = np.array([bool(m["drapeaux"] & 0x10) for m in M.mats], bool)
    sel_c = avec_c[M.mf] & (np.asarray(M.contour, float)[M.F] > 0.05).any(1)
    Fc = M.F[sel_c][:, [0, 2, 1]] + nv
    tout_F = np.vstack([M.F, Fc])
    me = bpy.data.meshes.new("modele")
    me.from_pydata([tuple(p) for p in tout_pos], [], [tuple(int(x) for x in f) for f in tout_F])
    me.update()
    lay = me.uv_layers.new(name="UV")
    lu = M.uv[M.F.ravel()].copy()
    lu[:, 1] = 1 - lu[:, 1]
    lu_c = M.uv[M.F[sel_c][:, [0, 2, 1]].ravel()].copy()
    lu_c[:, 1] = 1 - lu_c[:, 1]
    lay.data.foreach_set("uv", np.vstack([lu, lu_c]).astype(np.float32).ravel())
    for m in mats + contours:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", np.concatenate([M.mf, M.mf[sel_c] + n]).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(tout_F), bool))
    me.update()
    try:
        me.normals_split_custom_set_from_vertices([tuple(float(x) for x in v) for v in np.vstack([ns, -ns])])
    except Exception:
        pass
    o = bpy.data.objects.new("modele", me)
    bpy.context.scene.collection.objects.link(o)
    return o


def scene(taille, ech):
    sc = bpy.context.scene
    noms = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in noms else "BLENDER_EEVEE"
    sc.render.resolution_x = sc.render.resolution_y = taille
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("fond")
    sc.world = w
    w.color = (0.9, 0.9, 0.92)
    if w.node_tree is not None:
        bg = w.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs[0].default_value = (0.8, 0.8, 0.83, 1)
    try:
        sc.eevee.taa_render_samples = ech
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


def rendre(centre, demi, azimut, chemin, elev=0):
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("cam")
    cd.type = "ORTHO"
    cd.ortho_scale = 2 * demi
    cd.clip_end = 20
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    a, e = math.radians(azimut), math.radians(elev)
    cam.location = Vector((centre[0] + 5 * math.sin(a) * math.cos(e),
                           centre[1] - 5 * math.cos(a) * math.cos(e),
                           centre[2] + 5 * math.sin(e)))
    cam.rotation_euler = (math.pi / 2 - e, 0, a)
    sc.camera = cam
    sc.render.filepath = chemin
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


def assembler(chemins, chemin, colonnes, taille):
    lignes = (len(chemins) + colonnes - 1) // colonnes
    W, H = taille * colonnes, taille * lignes
    out = np.ones((H, W, 4), np.float32)
    for k, f in enumerate(chemins):
        i, j = divmod(k, colonnes)
        im = bpy.data.images.load(f)
        px = np.empty(im.size[0] * im.size[1] * 4, np.float32)
        im.pixels.foreach_get(px)
        px = px.reshape(im.size[1], im.size[0], 4)
        y0 = H - (i + 1) * taille
        out[y0:y0 + taille, j * taille:(j + 1) * taille] = px[:taille, :taille]
        bpy.data.images.remove(im)
    img = bpy.data.images.new("planche", W, H, alpha=True)
    img.pixels.foreach_set(out.ravel())
    tmp = chemin[:-4] + ".tmp.png"
    img.filepath_raw = tmp
    img.file_format = "PNG"
    img.save()
    os.replace(tmp, chemin)


def main():
    a = args()
    os.makedirs(a.sortie, exist_ok=True)
    M = MD.Modele("rendu", chemin=os.path.abspath(a.pmx))
    vet = np.isin(M.mf, [i for i, m in enumerate(M.mats) if m["nom"] == a.mat])
    faites = []
    pose_prec = "__aucune__"
    pos = M.pos
    for nom in a.vues.split(","):
        v_info = VUES[nom]
        pose, az, (cadre, demi) = v_info[:3]
        elev = v_info[3] if len(v_info) > 3 else 0
        if pose != pose_prec:
            pos = M.pos
            if pose:
                rot = BANC.pose_banc(pose, M.sq)
                pos = P.deformer(M.pos, M.idx, M.poids, M.sq, rotations=rot)[0]
            vider()
            scene(a.taille, a.echantillons)
            objet(M, pos, a.mat)
            pose_prec = pose
        if cadre in ("haut", "haut_leve"):
            pv = pos[np.unique(M.F[vet])] if a.cadre == "haut" else pos
            lo, hi = pv.min(0), pv.max(0)
            ztop = pos[:, 2].max()
            zlo = lo[2] - (0.12 if a.cadre == "haut" else 0.03)
            c = np.array([0.0, 0.5 * (lo[1] + hi[1]), 0.5 * (zlo + ztop)])
            demi = 0.53 * max(ztop - zlo, (hi[0] - lo[0]) + 0.2)
        else:
            b = M.noms_os.index(cadre)
            c = M.tetes[b].copy()
            if pose:   # posed bone: centre of the vertices it weights most
                c = pos[M.dom == b].mean(0)
            if cadre == "首":
                c[2] += 0.01 if nom == "dessous_menton" else -0.04
            if cadre == "頭":
                c[2] += 0.07
            if cadre == "下半身":
                c[2] -= 0.12 if nom == "fesses" else 0.03
            if cadre == "左足首":
                c[2] += 0.03
            if cadre == "左腕":
                i = M.noms_os.index("左ひじ")
                c = 0.5 * (c + M.tetes[i])
        f = os.path.join(a.sortie, nom + ".png")
        rendre(c, demi, az, f, elev=elev)
        faites.append(f)
        print("view", nom, "->", f)
    if a.planche:
        assembler(faites, os.path.join(a.sortie, a.planche), 3, a.taille)
        print("contact sheet", os.path.join(a.sortie, a.planche))


if __name__ == "__main__":
    main()
