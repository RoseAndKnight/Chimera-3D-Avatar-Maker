"""Garment contact sheets (Eevee, unshaded texture colour, orthographic camera).

    bin/blender-file -b --factory-startup --python outils/vetements/planche.py -- \
        --vetements <sortie.png> [--seuls id1,id2 | --pmx CANDIDAT.pmx --mats A,B [--id NOM]]
        Three tiles per garment of mesure.VETEMENTS (two garments per row): whole model
        (front), isolated garment from the front and from the back; parts of the same materials that
        were set aside (accessories, outside the regions) in magenta, the rest of the model hidden.
    bin/blender-file -b --factory-startup --python outils/vetements/planche.py -- \
        --apercu <sortie.png> Perso1,Perso2=Dossier/x.pmx,...
        Whole models from the front and the back (for choosing garments).

Textures are read next to the PMX (read-only). Nothing is written except the requested PNG.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import modele as MD  # noqa: E402

TX, TY = 300, 520
MAGENTA = (1.0, 0.0, 1.0, 1.0)


def vider():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras):
        for x in list(coll):
            coll.remove(x)


def materiau(nom, image, diffus=(1, 1, 1, 1), couleur=None):
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
    if couleur is not None:
        em.inputs["Color"].default_value = couleur
        mix.inputs[0].default_value = 1.0
    elif image is not None:
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
        m.surface_render_method = "DITHERED"
    except Exception:
        m.blend_method = "HASHED"
    m.use_backface_culling = False
    return m


def objet(M, faces, nom, cache, magenta=None):
    """Blender object of the given faces (indices into M.F), with the PMX textures; magenta: mask of
    faces (within `faces`) rendered in magenta."""
    F = M.F[faces]
    mf = M.mf[faces].copy()
    mats = []
    dossier = os.path.dirname(M.chemin)
    for k, m in enumerate(M.mats):
        t = m["texture"]
        img = None
        if 0 <= t < len(M.textures):
            f = os.path.normpath(os.path.join(dossier, M.textures[t].replace("\\", "/")))
            if os.path.exists(f):
                img = cache.get(f) or bpy.data.images.load(f)
                img.colorspace_settings.name = "sRGB"
                cache[f] = img
        mats.append(materiau("%s_%d" % (nom, k), img, m["diffus"]))
    mats.append(materiau(nom + "_magenta", None, couleur=MAGENTA))
    if magenta is not None:
        mf[magenta] = len(M.mats)
    me = bpy.data.meshes.new(nom)
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
    o = bpy.data.objects.new(nom, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def scene_rendu():
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
        e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = TX, TY
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("fond") if not bpy.data.worlds else bpy.data.worlds[0]
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


def rendre(lo, hi, angle, chemin):
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
    sc.render.filepath = chemin
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


def assembler(tuiles, chemin, colonnes, etiquettes=None):
    """Pastes the tile PNGs (list of rows) into one contact sheet, using Blender (no PIL)."""
    lignes = len(tuiles)
    W, H = TX * colonnes, TY * lignes
    out = np.ones((H, W, 4), np.float32)
    for i, ligne in enumerate(tuiles):
        for j, f in enumerate(ligne):
            if f is None or not os.path.exists(f):
                continue
            im = bpy.data.images.load(f)
            px = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)
            y0 = H - (i + 1) * TY
            out[y0:y0 + TY, j * TX:(j + 1) * TX] = px
            bpy.data.images.remove(im)
    # separator line between rows
    for i in range(1, lignes):
        out[H - i * TY - 1:H - i * TY + 1] = (0.3, 0.3, 0.3, 1)
    img = bpy.data.images.new("planche", W, H, alpha=True)
    img.pixels.foreach_set(out.ravel())
    img.filepath_raw = chemin
    img.file_format = "PNG"
    img.save()


def main(argv):
    tmp = os.environ.get("TMPDIR", "/tmp")
    if argv[0] == "--apercu":
        sortie, persos = argv[1], argv[2].split(",")
        tuiles = []
        for p in persos:
            vider()
            scene_rendu()
            # "Name" (library) or "Name=path.pmx" (relative to the official sources)
            p, _, pmx = p.partition("=")
            M = MD.modele_de(p, pmx or None)
            garde = ~M.faces_role(["vide"])
            objet(M, np.where(garde)[0], p, {})
            lo, hi = M.pos.min(0), M.pos.max(0)
            ligne = []
            for ang, nom in ((0, "face"), (180, "dos")):
                f = os.path.join(tmp, "ap_%s_%s.png" % (p, nom))
                rendre(lo, hi, ang, f)
                ligne.append(f)
            tuiles.append(ligne)
        # preview: two models per row
        lignes = [tuiles[i] + (tuiles[i + 1] if i + 1 < len(tuiles) else [None, None])
                  for i in range(0, len(tuiles), 2)]
        assembler(lignes, sortie, 4)
        print("contact sheet written:", sortie)
        return
    if argv[0] == "--vetements":
        import mesure
        sortie = argv[1]
        seuls = argv[argv.index("--seuls") + 1].split(",") if "--seuls" in argv else None
        tuiles = []
        modeles = {}
        specs = mesure.VETEMENTS
        if "--pmx" in argv:
            chemin = argv[argv.index("--pmx") + 1]
            nom = argv[argv.index("--id") + 1] if "--id" in argv else os.path.splitext(os.path.basename(chemin))[0]
            modeles[nom] = MD.Modele(nom, chemin=chemin)
            specs = [{"id": nom, "perso": nom, "mats": argv[argv.index("--mats") + 1].split(",")}]
            seuls = None
        for spec in specs:
            if seuls and spec["id"] not in seuls:
                continue
            vider()
            scene_rendu()
            M = modeles.get(spec["perso"]) or MD.modele_de(spec["perso"], spec.get("pmx"))
            modeles[spec["perso"]] = M
            sel, gard, ecart = MD.isoler(M, spec)
            cache = {}
            ligne = []
            # 1. whole model, from the front
            tout = objet(M, np.where(~M.faces_role(["vide"]))[0], "tout", cache)
            lo, hi = M.pos.min(0), M.pos.max(0)
            f = os.path.join(tmp, "pv_%s_tout.png" % spec["id"])
            rendre(lo, hi, 0, f)
            ligne.append(f)
            bpy.data.objects.remove(tout, do_unlink=True)
            # 2-3. isolated garment + set-aside parts in magenta
            ecartes = np.zeros(len(M.F), bool)
            mats = [i for i, n in enumerate(M.noms_mat) if n in spec["mats"]]
            dans = np.isin(M.mf, mats)
            if spec.get("uv_boites"):
                dans &= MD.zone_uv(M, spec)
            for fk in M.pieces(dans):
                if not sel[fk[0]]:
                    ecartes[fk] = True
            faces = np.where(sel | ecartes)[0]
            o = objet(M, faces, "vet", cache, magenta=ecartes[faces])
            vs = np.unique(M.F[sel])
            lo, hi = M.pos[vs].min(0), M.pos[vs].max(0)
            for ang, nom in ((0, "face"), (180, "dos")):
                f = os.path.join(tmp, "pv_%s_%s.png" % (spec["id"], nom))
                rendre(lo, hi, ang, f)
                ligne.append(f)
            tuiles.append(ligne)
            print("rendered", spec["id"], "kept", len(gard), "set aside", len(ecart))
        # two garments per row
        lignes = [tuiles[i] + (tuiles[i + 1] if i + 1 < len(tuiles) else [None] * 3)
                  for i in range(0, len(tuiles), 2)]
        assembler(lignes, sortie, 6)
        print("contact sheet written:", sortie)


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(a)
