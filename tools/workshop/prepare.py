"""Preparing the cache (Blender process): body, heads, hairstyles, parts.

    bin/blender-queue -b --factory-startup --python tools/workshop/prepare.py -- --travail LISTE.json

LISTE.json: [{"genre": "corps"|"tete"|"coiffure"|"fabriquer"|"piece"|"couvre", ...}, ...]. Each item
writes its meta.json record (with "retenue" and "raison" if it is rejected); an error on one
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

import pmx as PMX  # noqa: E402  (via commun's sys.path)
import pmx_write as pmx_ecrire  # noqa: E402

# checks of measure.py that reject a part; the others do not apply on the bare body or
# come from the source (see BIBLIOTHEQUE.md) and stay recorded in the record
BLOQUANTS = ("sous_la_peau", "traversee_repos", "traversee_repos_p90_mm", "poses_corps_seul", "pied_dans_chaussure")
PIED_ENFONCE_MAX = 0.015  # metres: a shoe lowered further onto its sole to let the foot through it does not
                          # hold the foot, which stands above it (a bare foot over a crushed-looking shoe)
SOUS_LA_PEAU_MAX = 0.01   # share of vertices more than 1 mm inside the body (porter.sous_la_peau, no depth limit)
TAILLE_TEXTURE = 1024


def journal(*a):
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


def texture(chemin):
    """Converts an image to a shared WebP in the cache; returns its URL (or None if missing)."""
    if not chemin or not os.path.isfile(chemin):
        return None
    n = C.empreinte_fichier(chemin) + ".webp"
    dest = os.path.join(C.CACHE, "tex", n)
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        im = bpy.data.images.load(chemin, check_existing=False)
        try:
            w, h = im.size
            if max(w, h) > TAILLE_TEXTURE:
                k = TAILLE_TEXTURE / max(w, h)
                im.scale(max(1, int(w * k)), max(1, int(h * k)))
            tmp = dest + ".tmp.webp"
            im.save_render(tmp, scene=_scene_webp())
            os.replace(tmp, dest)
        finally:
            bpy.data.images.remove(im)
    return "/data/tex/" + n


def chemin_tex(pmx_chemin, rel):
    return os.path.normpath(os.path.join(os.path.dirname(pmx_chemin), rel.replace("\\", "/")))


def mat_glb(P, m, pmx_chemin, role):
    t = m["texture"]
    url = texture(chemin_tex(pmx_chemin, P["textures"][t])) if 0 <= t < len(P["textures"]) else None
    return {"nom": m["nom"], "image": url, "diffus": list(m["diffus"]), "role": role,
            "contour": list(m["contour_couleur"]) + [float(m["contour_taille"])],
            "avec_contour": bool(m["drapeaux"] & 0x10), "double": True}


def extraire_glb(pmx_chemin, noms_mats, sortie, role="tissu", T=None, P=None):
    """Chosen materials of a PMX -> GLB. Returns (vertex count, face count, materials)."""
    P = P or PMX.lire(pmx_chemin)
    T = T or PMX.vers_tableaux(P)
    noms = [m["nom"] for m in P["materiaux"]]
    garde = [i for i, n in enumerate(noms) if n in noms_mats and P["materiaux"][i]["n_indices"] > 0]
    sel = np.isin(T["mat_face"], garde)
    F = T["faces"][sel]
    mf = T["mat_face"][sel]
    vs, inv = np.unique(F, return_inverse=True)
    Fl = inv.reshape(-1, 3)
    pos, nor, Fl = GLB.vers_gltf(T["pos"][vs], T["nor"][vs], Fl)
    prims, mats = [], []
    for i in garde:
        prims.append({"faces": Fl[mf == i], "materiau": len(mats)})
        mats.append(mat_glb(P, P["materiaux"][i], pmx_chemin, role))
    GLB.ecrire(sortie, pos, nor, T["uv"][vs], prims, mats)
    return len(pos), len(Fl), [m["nom"] for m in mats]


def ecrire_pied(P, noms_mats, pose, meta, d):
    """A shoe worn on a posed foot (porter, foot.py): its pose in meta.json, and per vertex of piece.glb
    the weights that pose it again (pied.bin: float32 ankle weight, toe weight, side 0 left / 1 right / -1)."""
    import foot as PD
    chemin = os.path.join(d, "pied.bin")
    if not pose:
        meta.pop("pied", None)
        if os.path.exists(chemin):
            os.remove(chemin)
        return
    meta["pied"] = pose
    T = PMX.vers_tableaux(P)
    noms = [m["nom"] for m in P["materiaux"]]
    garde = [i for i, n in enumerate(noms) if n in noms_mats and P["materiaux"][i]["n_indices"] > 0]
    vs = np.unique(T["faces"][np.isin(T["mat_face"], garde)])
    wa, wt, cote = PD.poids_pmx(P, PD.chaines(P["os"]))
    C.ecrire_octets(chemin, np.stack([wa[vs], wt[vs], cote[vs]], 1).astype("<f4").tobytes())


# ------------------------------------------------------------------ corps
def faire_corps(_):
    P = PMX.lire(C.CORPS)
    d = os.path.join(C.CACHE, "corps")
    n, f, _ = extraire_glb(C.CORPS, {"肌"}, os.path.join(d, "corps.glb"), role="peau", P=P)
    C.ecrire_json(os.path.join(d, "meta.json"), {"sommets": n, "faces": f, "empreinte": C.empreinte_fichier(C.CORPS)})
    journal("body", n, "vertices", f, "faces")


# ------------------------------------------------------------------ heads
def raccord_peau(R, d):
    """Skin of the common body as the base head leaves it: faces removed, vertices adjusted.

    The page puts the head on corps.glb: it needs the body faces gone under the head
    (peau_cachee.bin, faces of 肌 in corps.glb) and the vertices tete_entiere adjusted at the neck and at the
    edge of the skull (corps.bin, glTF positions then normals of corps.glb).
    """
    import head_graft as GT
    Co = PMX.lire(C.CORPS)
    Tc, Tr = PMX.vers_tableaux(Co), PMX.vers_tableaux(R)
    noms = [m["nom"] for m in Co["materiaux"]]
    garde = ~np.isin(Tc["mat_face"], [i for i, n in enumerate(noms) if n in GT.RETIRER_DEFAUT])
    vB = np.unique(Tc["faces"][garde])          # body vertices in the graft, in order (carteB)
    Fc = Tc["faces"][Tc["mat_face"] == noms.index("肌")]
    Fr = Tr["faces"][Tr["mat_face"] == [m["nom"] for m in R["materiaux"]].index("肌")]
    hors = int((Fr >= len(vB)).any(axis=1).sum())
    reste = {tuple(sorted(int(x) for x in f_)) for f_ in vB[Fr[(Fr < len(vB)).all(axis=1)]]}
    cache = np.array([k for k, f_ in enumerate(Fc) if tuple(sorted(int(x) for x in f_)) not in reste], np.uint32)
    pos, nor = np.asarray(Tc["pos"], float).copy(), np.asarray(Tc["nor"], float).copy()
    pos[vB], nor[vB] = np.asarray(Tr["pos"], float)[:len(vB)], np.asarray(Tr["nor"], float)[:len(vB)]
    vs = np.unique(Fc)                            # vertex order of corps.glb
    p, n, _ = GLB.vers_gltf(pos[vs], nor[vs], np.zeros((0, 3), np.int64))
    C.ecrire_octets(os.path.join(d, "peau_cachee.bin"), cache.tobytes())
    C.ecrire_octets(os.path.join(d, "corps.bin"), np.concatenate([p.ravel(), n.ravel()]).astype(np.float32).tobytes())
    C.ecrire_octets(os.path.join(d, "corps_pos.bin"), np.asarray(Tr["pos"], "<f4")[:len(vB)].tobytes())
    bouge = int((np.abs(np.asarray(Tr["pos"], float)[:len(vB)] - np.asarray(Tc["pos"], float)[vB]).max(axis=1) > 1e-6).sum())
    return {"faces_cachees": int(len(cache)), "sommets_bouges": bouge, "faces_hors_corps": hors}


def faire_base(it):
    """Base head sewn once onto the common body (clothes/whole_head.py), and the body skin
    as it leaves it. A resultat.pmx already there (the approved one) is not sewn again."""
    if it.get("libre"):      # free head: the base body already wears its head, nothing to sew; copied with its textures
        import shutil
        os.makedirs(C.BASE, exist_ok=True)
        for rel in [os.path.basename(it["tete"])] + PMX.lire(it["tete"])["textures"]:
            shutil.copyfile(chemin_tex(it["tete"], rel), chemin_tex(C.BASE_PMX, rel))
        os.replace(chemin_tex(C.BASE_PMX, os.path.basename(it["tete"])), C.BASE_PMX)
    elif not os.path.isfile(C.BASE_PMX):
        import whole_head as tete_entiere
        tete_entiere.main(["--corps", C.CORPS, "--tete", it["tete"], "--sortie", C.BASE,
                           "--teinte-base", C.JEU.valeur("tete_base.teinte", "anneau")])
    R = PMX.lire(C.BASE_PMX)
    raccord = raccord_peau(R, C.BASE)
    C.ecrire_json(os.path.join(C.BASE, "meta.json"), {"tete": it["tete"], "empreinte": C.empreinte_fichier(C.BASE_PMX),
                                                      "raccord": raccord, "libre": bool(it.get("libre"))})
    journal("base", "free head" if it.get("libre") else C.TETE_BASE, raccord)


def _cle_pos(P):
    return [bytes(x) for x in np.round(np.asarray(P, float), 5).astype("<f4")]


def _ilots(T, i_mat):
    """UV islands of material i_mat: list of face arrays."""
    F = T["faces"][T["mat_face"] == i_mat]
    par = np.arange(len(T["uv"]))

    def r(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a, b, c in F:
        par[r(a)] = r(b)
        par[r(b)] = r(c)
    grp = {}
    for f in F:
        grp.setdefault(r(f[0]), []).append(f)
    return [np.array(g) for g in grp.values()]


def _cadre(T, g):
    uv = T["uv"][g.reshape(-1)]
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    return (x0, y0, x1, y1)


def _iris_et_calques(T, i_mat):
    """Frames (x0, y0, x1, y1 in UV, v downwards) of the irises of material i_mat: on each side (sign of x), the island
    largest in space (the others are layers laid in front: pupil, highlights); and the faces of the
    islands outside these frames, which the chosen iris also replaces. An island whose UVs fall inside an iris frame
    reads the same pixels (a layer of the same drawing): it stays."""
    ilots = _ilots(T, i_mat)
    cadres = [_cadre(T, g) for g in ilots]
    grands = {}
    for k, g in enumerate(ilots):
        p = T["pos"][np.unique(g)]
        cote = 1 if p[:, 0].mean() >= 0 else -1
        e = p.max(0) - p.min(0)
        aire = e[0] * e[1]
        if cote not in grands or aire > grands[cote][0]:
            grands[cote] = (aire, k)
    iris = list(dict.fromkeys(cadres[k] for _, k in grands.values()))
    dedans = lambda a, b: a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]
    calques = [g for g, c in zip(ilots, cadres) if not any(dedans(c, b) for b in iris)]
    return iris, (np.concatenate(calques) if calques else np.zeros((0, 3), int))


def _masque_faces(T, faces, W, H, marge=4):
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
    for _ in range(marge):
        n = m.copy()
        n[1:] |= m[:-1]
        n[:-1] |= m[1:]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n
    return m


def zone_bouche(T, mats, W, H):
    """Rectangle in pixels (x0, y0, x1, y1, rows from the top) of the UVs of materials mats (teeth, inside of the
    mouth), with commun's margin: the area cut out to be repainted and pasted back."""
    uv = T["uv"][np.unique(T["faces"][np.isin(T["mat_face"], list(mats))])]
    m = C.MARGE_BOUCHE
    return (int(uv[:, 0].min() * W) - m, int(uv[:, 1].min() * H) - m,
            int(uv[:, 0].max() * W) + m, min(H, int(uv[:, 1].max() * H) + m))


def texture_iris(chemin_tex_visage, cadres, chemin_iris, sortie, T=None, effacer=(), bouche=None, mats_bouche=()):
    """Copy of the face texture where each iris frame receives the image chemin_iris (blended by its alpha);
    the pixels under the faces effacer become transparent (the viewer discards them). bouche: repainted image of the
    mouth area (zone_bouche of materials mats_bouche), pasted as is (blended by its alpha)."""
    fond = bpy.data.images.load(chemin_tex_visage, check_existing=False)
    W, H = fond.size
    px = np.empty(W * H * 4, np.float32)
    fond.pixels.foreach_get(px)
    px = px.reshape(H, W, 4)   # row 0 at the bottom
    if len(effacer):
        px[_masque_faces(T, effacer, W, H)[::-1], 3] = 0.0
    for x0, y0, x1, y1 in cadres:
        c0, c1 = int(round(x0 * W)), int(round(x1 * W))
        l0, l1 = int(round(y0 * H)), int(round(y1 * H))   # rows from the top
        w, h = max(1, c1 - c0), max(1, l1 - l0)
        im = bpy.data.images.load(chemin_iris, check_existing=False)
        im.scale(w, h)
        ip = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(ip)
        bpy.data.images.remove(im)
        ip = ip.reshape(h, w, 4)
        zone = px[H - l1:H - l0, c0:c1]
        a = ip[..., 3:4]
        zone[..., :3] = ip[..., :3] * a + zone[..., :3] * (1 - a)
        zone[..., 3:4] = np.maximum(zone[..., 3:4], a)
    if bouche:
        x0, l0, x1, l1 = zone_bouche(T, mats_bouche, W, H)
        w, h = x1 - x0, l1 - l0
        im = bpy.data.images.load(bouche, check_existing=False)
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
    fond.pixels.foreach_set(px.reshape(-1))
    fond.filepath_raw = sortie
    fond.file_format = "PNG"
    fond.save()
    bpy.data.images.remove(fond)


def glb_tete(res, R, sortie, iris=None, bouche=None):
    """tete.glb: everything the head puts on corps.glb, that is every material except the body
    skin, and of that skin only the seam strips (faces with a vertex that is not a body vertex)."""
    T = PMX.vers_tableaux(R)
    noms = [m["nom"] for m in R["materiaux"]]
    i_peau = noms.index("肌")
    du_corps = set(_cle_pos(np.fromfile(os.path.join(C.BASE, "corps_pos.bin"), "<f4").reshape(-1, 3)))
    vp = np.unique(T["faces"][T["mat_face"] == i_peau])
    hors = np.zeros(len(T["pos"]), bool)
    hors[vp] = [k not in du_corps for k in _cle_pos(T["pos"][vp])]
    bande = (T["mat_face"] == i_peau) & hors[T["faces"]].any(1)
    sel = (T["mat_face"] != i_peau) | bande
    F, mf = T["faces"][sel], T["mat_face"][sel]
    vs, inv = np.unique(F, return_inverse=True)
    pos, nor, Fl = GLB.vers_gltf(T["pos"][vs], T["nor"][vs], inv.reshape(-1, 3))
    # chosen iris: a copy of the 目 texture, iris pasted, original pupil and highlights erased (layers of 目
    # outside the irises, and material 目光 if it reads the same texture: the copy takes them over)
    # repainted mouth: the same copy receives the area of the teeth and inside of the mouth, and those materials read it
    url_yeux, avec_yeux = None, set()
    mats_bouche = [j for j, n in enumerate(noms) if n in ("齿", "口舌")]
    t_vis = R["materiaux"][noms.index("目")]["texture"] if "目" in noms else -1
    if (iris or bouche) and 0 <= t_vis < len(R["textures"]):
        i_oeil = noms.index("目")
        cadres, effacer = _iris_et_calques(T, i_oeil) if iris else ([], np.zeros((0, 3), int))
        if cadres:
            avec_yeux = {i_oeil}
            for j, n in enumerate(noms):
                if n == "目光" and R["materiaux"][j]["texture"] == t_vis:
                    avec_yeux.add(j)
                    effacer = np.concatenate([effacer, T["faces"][T["mat_face"] == j]])
        mats_bouche = [j for j in mats_bouche if R["materiaux"][j]["texture"] == t_vis] if bouche else []
        if mats_bouche:
            avec_yeux |= {j for j, n in enumerate(noms) if n in ("齿", "口舌", "口线") and R["materiaux"][j]["texture"] == t_vis}
        if avec_yeux:
            dest = os.path.join(os.path.dirname(sortie), "yeux.png")
            texture_iris(chemin_tex(res, R["textures"][t_vis]), cadres, iris, dest, T=T, effacer=effacer,
                         bouche=bouche if mats_bouche else None, mats_bouche=mats_bouche)
            url_yeux = texture(dest)
    prims, mats = [], []
    for i in np.unique(mf):
        m = mat_glb(R, R["materiaux"][i], res, "peau" if i == i_peau else "visage")
        if i in avec_yeux:
            m["image"] = url_yeux
        if i == i_peau:
            m["nom"] = "肌_bande"
        prims.append({"faces": Fl[mf == i], "materiau": len(mats)})
        mats.append(m)
    GLB.ecrire(sortie, pos, nor, T["uv"][vs], prims, mats)
    return len(pos), len(Fl), [m["nom"] for m in mats], int(bande.sum())


def faire_tete(it):
    """A face of the catalogue: that of it["pmx"], sewn into the oval of the base head by the face transfer
    tool, as is (retarget_tete visage). The skull, ears, neck and neck seam to the
    body are the base's, the same for all."""
    d = os.path.join(C.CACHE, "tetes", it["id"])
    os.makedirs(d, exist_ok=True)
    mesures = None
    if it["biblio"] == C.TETE_BASE or it.get("libre"):
        res = C.BASE_PMX
    else:
        mod = os.path.join(d, "modele")   # written before, by catalogue.transferer (outside the Blender queue)
        res = os.path.join(mod, "resultat.pmx")
        mesures = C.lire_json(os.path.join(mod, "mesures.json"))
        if mesures is None or not os.path.isfile(res):
            raise RuntimeError("face transfer: " + (C.lire_json(os.path.join(mod, "erreur.json")) or "not done"))
    R = PMX.lire(res)
    iris, bouche = C.iris(it["id"]), C.bouche(it["id"])
    n, f, mats, n_bande = glb_tete(res, R, os.path.join(d, "tete.glb"), iris=iris, bouche=bouche)
    for nom in ("corps.bin", "peau_cachee.bin"):
        C.lier(os.path.join(C.BASE, nom), os.path.join(d, nom))
    peau = [m for m in R["materiaux"] if m["nom"] == "肌"]
    url = None
    if peau and 0 <= peau[0]["texture"] < len(R["textures"]):
        url = texture(chemin_tex(res, R["textures"][peau[0]["texture"]]))
    oc = {o["nom"]: o["pos"] for o in R["os"]}
    ot = {o["nom"]: o["pos"] for o in PMX.lire((C.lire_json(os.path.join(C.BASE, "meta.json")))["tete"])["os"]}
    tr = [float(a - b) for a, b in zip(oc["頭"], ot["頭"])]
    echecs = sorted(k for k, v in (mesures or {}).items() if isinstance(v, dict) and v.get("passe") is False)
    C.ecrire_json(os.path.join(d, "meta.json"), {
        "id": it["id"], "nom": it.get("nom", it["id"]), "pmx": it["pmx"], "source": it["biblio"], "biblio": it.get("base_biblio", C.TETE_BASE),
        "modele": C.relatif(res), "materiaux": mats, "sommets": n, "faces": f, "faces_bande": n_bande, "peau": url,
        "translation": tr, "retenue": True, "empreinte": C.empreinte_fichier(it["pmx"]), "v_peau": C.V_PEAU,
        "iris": C.empreinte_fichier(iris) if iris else None, "bouche": C.empreinte_fichier(bouche) if bouche else None, "mesures_echecs": echecs})
    journal("head", it["id"], f, "faces;", "transfer: " + (", ".join(echecs) or "all pass") if mesures is not None else "base head")


# ------------------------------------------------------------------ made parts
def faire_fabriquee(it):
    """Cuts a made garment (materials of a complete model) into the library format."""
    import importlib.util
    import model as MD
    spec = importlib.util.spec_from_file_location("bibliotheque_vetements", os.path.join(C.OUTILS, "clothes", "library.py"))
    BV = importlib.util.module_from_spec(spec)
    sys.modules["bibliotheque_vetements"] = BV
    spec.loader.exec_module(BV)
    corps_os = {o["nom"] for o in PMX.lire(C.CORPS)["os"]}
    M = MD.Modele(it["perso"], chemin=it["pmx"])
    choisis = [i for i, n in enumerate(M.noms_mat) if n in it["mats"]]
    faces = np.where(np.isin(M.mf, choisis))[0]
    if not len(faces):
        raise ValueError("materials missing: %s" % it["mats"])
    T = SimpleNamespace(M=M, perso=it["perso"], os_corps=[n in corps_os for n in M.noms_os])
    d = os.path.join(C.CACHE, "fabriquees", it["perso"], it["id"])
    os.makedirs(d, exist_ok=True)
    copies = {}
    piece = BV.piece_pmx(T, faces, it["id"], "tex", copies)
    if isinstance(piece, tuple):
        piece = piece[0]
    for base, src in copies.items():
        C.lier(src, os.path.join(d, "tex", base))
    pmx_ecrire.ecrire(piece, os.path.join(d, it["id"] + ".pmx"))
    C.ecrire_json(os.path.join(d, "fiche.json"), {
        "perso": it["perso"], "id": it["id"], "classe": "vetement", "type": it["type"], "sorte": None,
        "pmx": it["id"] + ".pmx", "vignette": None, "faces": int(len(faces)), "materiaux": list(it["mats"]),
        "os_propres": [o["nom"] for o in piece["os"] if o["nom"] not in corps_os],
        "source": {"pmx": it["pmx"]}, "atelier": it.get("atelier") or {},
        "empreinte": C.empreinte_fichier(it["pmx"])})
    journal("made", it["perso"], it["id"], len(faces), "faces")


# ------------------------------------------------------------------ parts (fitting + measurement)
def emplacement(fiche):
    a = fiche.get("atelier") or {}
    if a.get("onglet"):
        return a["onglet"], a["emplacement"], a["rang"], a.get("nom") or C.FR_TYPE.get(fiche.get("type"), fiche["id"])
    if fiche.get("classe") == "accessoire":
        rang, excl = C.ACCESSOIRES[fiche["sorte"]]
        return "accessoires", (fiche["sorte"] if excl else None), rang, C.FR_TYPE.get(fiche["sorte"], fiche["sorte"])
    onglet, emp, rang = C.VETEMENTS[fiche["type"]]
    return onglet, emp, rang, C.FR_TYPE.get(fiche["type"], fiche["type"])


TYPES_MESURE = {"culotte": "short"}
ETENDUE_MIN = 2.0         # mm of diagonal: below it, the fitted part is a point (an earring measures 9)


def faire_piece(it):
    """A shoe is first worn on feet posed into it (porter, pied="chercher"); if that fit is not
    kept, it is redone on feet at rest, the shoe still placed in one block (a heel then sinks into
    the ground, the shoe stays whole); if that is not kept either, the last resort is the flat foot
    with the shoe carried by its bones and left as it is around the foot (porter, pied="plat"). A
    shoe is never pushed out of the foot vertex by vertex: that tears it to shards."""
    meta = preparer_piece(it, "chercher")
    if not meta["retenue"] and meta.get("pied"):
        import foot as PD
        refus = meta["raison"]
        repos = {nom: {"cheville": 0.0, "orteils": 0.0} for nom, _, _ in PD.COTES}
        if any(abs(v.get("cheville", 0.0)) > 0 for v in meta["pied"].values() if isinstance(v, dict)):
            meta = preparer_piece(it, repos)
        if not meta["retenue"]:
            meta = preparer_piece(it, "plat")
        meta["pied_refuse"] = refus
        C.ecrire_json(os.path.join(C.CACHE, "pieces", it["id"], "meta.json"), meta)
    return meta


def preparer_piece(it, pied):
    import model as MD
    import measure as mesure
    import wear as PO
    src = it["dossier"]
    with open(os.path.join(src, "fiche.json"), encoding="utf-8") as fh:
        fiche = json.load(fh)
    if fiche.get("type") == "chaussettes" or fiche.get("sorte") == "chevillere":
        # a "sock" or an "anklet" with a sole is a shoe (porter.charger_piece)
        vraie = PO.charger_piece(src)[0]
        fiche["classe"], fiche["type"] = vraie["classe"], vraie.get("type")
    d = os.path.join(C.CACHE, "pieces", it["id"])
    os.makedirs(d, exist_ok=True)
    onglet, emp, rang, type_fr = emplacement(fiche)
    garnie = fiche.get("garnie") or {}
    pmx_src = os.path.join(src, garnie.get("pmx") or fiche["pmx"])
    if not os.path.exists(pmx_src):
        garnie, pmx_src = {}, os.path.join(src, fiche["pmx"])
    meta = {"id": it["id"], "perso": fiche["perso"], "piece": fiche["id"], "classe": fiche["classe"],
            "type": fiche.get("type") or fiche.get("sorte"), "type_fr": type_fr, "onglet": onglet, "emplacement": emp,
            "rang": rang, "occupe": C.OCCUPE.get(fiche.get("type"), []), "dossier": C.relatif(src),
            "empreinte": C.empreinte_fichier(pmx_src), "v_portage": C.V_PORTAGE, "corps": C.empreinte_fichier(C.CORPS),
            "retenue": False, "raison": None, "controles": [], "faces": 0,
            "garnitures": garnie.get("pieces") or []}
    tmp = tempfile.mkdtemp(prefix="piece-", dir=os.environ.get("TMPDIR") or None)
    t0 = time.time()
    try:
        try:
            res = PO.porter(src, C.CORPS, tmp, nom="piece", marge_mm=3.0, peau="retirer", pied=pied)
        except Exception as e:
            meta["raison"] = "fitting failed: %s" % str(e)[:160]
            return meta
        if res.get("etendue_mm", ETENDUE_MIN) < ETENDUE_MIN:
            # an object the game stores folded into a point (Hugo's hat: 3 microns) is not a part
            meta["raison"] = "part with no extent (%.1f mm)" % res["etendue_mm"]
            return meta
        chemin = os.path.join(tmp, "piece.pmx")
        P = PMX.lire(chemin)
        n, f, mats = extraire_glb(chemin, set(res["mats"]), os.path.join(d, "piece.glb"), P=P)
        meta["faces"], meta["sommets"], meta["materiaux"] = f, n, mats
        ecrire_pied(P, set(res["mats"]), res.get("pied"), meta, d)
        # hidden skin: faces of the body's 肌 missing from the output (porter keeps the indices)
        Co = PMX.lire(C.CORPS)
        Tc = PMX.vers_tableaux(Co)
        i_peau = [m["nom"] for m in Co["materiaux"]].index("肌")
        Fc = Tc["faces"][Tc["mat_face"] == i_peau]
        To = PMX.vers_tableaux(P)
        Fo = To["faces"][To["mat_face"] == [m["nom"] for m in P["materiaux"]].index("肌")]
        reste = {tuple(int(x) for x in sorted(f_)) for f_ in Fo}
        cache = np.array([k for k, f_ in enumerate(Fc) if tuple(int(x) for x in sorted(f_)) not in reste], np.uint32)
        C.ecrire_octets(os.path.join(d, "peau_cachee.bin"), cache.tobytes())
        meta["peau_cachee"] = int(len(cache))
        vi = fiche.get("vignette")
        if vi and os.path.isfile(os.path.join(src, vi)):
            C.lier(os.path.join(src, vi), os.path.join(d, "vignette.png"))
            meta["vignette"] = "/data/pieces/%s/vignette.png" % it["id"]
        spec = {"id": it["id"], "perso": it["id"], "type": TYPES_MESURE.get(fiche.get("type"), res["type_mesure"]),
                "mats": res["mats"], "calques": [], "pourquoi": "atelier"}
        try:
            r = mesure.mesurer(spec, MD.Modele(it["id"], chemin=chemin), True)
        except mesure.SansObjet as e:
            # garment measurement does not apply here (rigid accessory): the part is judged
            # on what fitting says about it, and the missing checks are named
            meta["mesure_sans_objet"] = str(e)
            r = {"controles": [{"nom": n, "valeur": None, "seuil": None, "ok": None} for n in BLOQUANTS if n != "sous_la_peau"]}
        except Exception as e:
            meta["raison"] = "measurement failed: %s" % str(e)[:160]
            return meta
        # fitting itself says what it left inside the body: traversee_repos (measure.py) only
        # sees up to 10 mm, a part sunk several centimetres deep escapes it
        sp = res.get("sous_la_peau") or {}
        # a part placed in one block (wing, bag: pose_rigide; a shoe's foot part: pied_en_bloc) sits
        # in the body by nature, and cannot be pulled out without deforming: the check does not apply to it
        sans_objet = sp.get("part") is None or res.get("pose_rigide") or res.get("pied_en_bloc")
        r["controles"].append({"nom": "sous_la_peau", "valeur": sp.get("part"), "seuil": "<= 0.01",
                               "ok": None if sans_objet else sp["part"] <= SOUS_LA_PEAU_MAX})
        # a shoe placed in one block holds the foot only if it did not have to be lowered by more than
        # PIED_ENFONCE_MAX for the foot to stop going through its sole (porter.chausser)
        ch = res.get("chausse") or {}
        if ch:
            enfonce = max(float(v.get("semelle", 0.0)) for v in ch.values())
            r["controles"].append({"nom": "pied_dans_chaussure", "valeur": round(100 * enfonce, 1),
                                   "seuil": "<= %g cm" % (100 * PIED_ENFONCE_MAX), "ok": enfonce <= PIED_ENFONCE_MAX})
        # the depth (9th decile) of the clipping points says nothing about a handful of points:
        # under 1 % of the part, the share is what judges (traversee_repos, sous_la_peau)
        part_trav = next((x["valeur"] for x in r["controles"] if x["nom"] == "traversee_repos"), None)
        for x in r["controles"]:
            if x["nom"] == "traversee_repos_p90_mm" and part_trav is not None and part_trav < SOUS_LA_PEAU_MAX:
                x["ok"] = None
        meta["portage"] = {"sous_la_peau": sp, "reste_dedans": res.get("reste_dedans"), "chausse": ch or None, "gaine": res.get("gaine"), "transport": res.get("transport"), "rigides_sommets": res.get("rigides_sommets"), "essais": res.get("essais"), "par_membre": res.get("par_membre")}
        meta["controles"] = [{"nom": x["nom"], "valeur": x["valeur"], "seuil": x["seuil"], "ok": x["ok"]} for x in r["controles"]]
        # a blocking check without a value is not a passed check: it is named
        meta["non_mesures"] = [x["nom"] for x in r["controles"] if x["ok"] is None and x["nom"] in BLOQUANTS
                               and not (x["nom"] == "traversee_repos_p90_mm" and part_trav is not None and part_trav < SOUS_LA_PEAU_MAX)]
        mauvais = [x for x in r["controles"] if x["ok"] is False and x["nom"] in BLOQUANTS]
        if mauvais:
            meta["raison"] = "; ".join("%s = %s (threshold %s)" % (x["nom"], _court(x["valeur"]), x["seuil"]) for x in mauvais)
        else:
            meta["retenue"] = True
        meta["autres_echecs"] = [x["nom"] for x in r["controles"] if x["ok"] is False and x["nom"] not in BLOQUANTS]
        if meta["retenue"]:
            C.ecrire_json(os.path.join(d, "meta.json"), meta)  # couvre reads the record
            completer_couvre(meta, d)
        return meta
    finally:
        meta["duree_s"] = round(time.time() - t0, 1)
        C.ecrire_json(os.path.join(d, "meta.json"), meta)
        shutil.rmtree(tmp, ignore_errors=True)
        journal("part", it["id"], "kept" if meta["retenue"] else "rejected: %s" % meta["raison"], "%.1f s" % meta["duree_s"])


# ------------------------------------------------------------------ what each face can do
_alphas = {}


def alpha_de(url):
    """Alpha channel (0..1, first row at the top) of a cache texture."""
    if url not in _alphas:
        chemin = os.path.join(C.CACHE, url[len("/data/"):])
        im = bpy.data.images.load(chemin, check_existing=False)
        w, h = im.size
        px = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(px)
        bpy.data.images.remove(im)
        _alphas[url] = px.reshape(h, w, 4)[::-1, :, 3]
    return _alphas[url]


def opaque_de(doc, uv, F, prim):
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
        a = alpha_de(doc["images"][doc["textures"][bt["index"]]["source"]]["uri"])
        import coverage as K
        op[sel] = K.opaque_par_face(a, uv, F[sel])
    return op


def completer_couvre(meta, d):
    """couvre.bin of the part; a shoe that does not cover the top of the foot is rejected."""
    import coverage as K
    _, part = K.calculer(meta["id"], opaque_de)
    if meta.get("classe") == "vetement" and os.path.isfile(os.path.join(d, "peau_cachee.bin")):
        # fitting counted all the held geometry; only opaque faces hide the skin
        import skin as PE
        k = PE.de_piece(meta["id"])
        C.ecrire_octets(os.path.join(d, "peau_cachee.bin"), k.tobytes())
        meta["peau_cachee"] = int(len(k))
    if part is not None:
        meta["dessus_du_pied"] = round(part, 3)
        if part < K.PIED_MIN:
            meta["retenue"] = False
            meta["raison"] = "sole without an upper: %d %% of the top of the foot covered (minimum %d %%)" % (round(100 * part), round(100 * K.PIED_MIN))
    return meta


def faire_couvre(it):
    """Completion for a part prepared before couvre.bin existed."""
    d = os.path.join(C.CACHE, "pieces", it["id"])
    m = C.lire_json(os.path.join(d, "meta.json"))
    completer_couvre(m, d)
    C.ecrire_json(os.path.join(d, "meta.json"), m)
    journal("couvre", it["id"], "kept" if m["retenue"] else "rejected: %s" % m["raison"])


def _court(v):
    return round(v, 3) if isinstance(v, float) else v


# ------------------------------------------------------------------ hairstyles
def coiffure_deja_posee(it, d):
    """New heads that are the same target as a head already measured (same base head, same translation:
    coiffures.Preparation.poser_sur gives them the same pose): the pair is copied, nothing is loaded.
    True when every new head was copied. Each extraction lot adds heads, and posing every hairstyle again
    for an answer it already had cost about 25 s per hairstyle and per lot."""
    m = C.lire_json(os.path.join(d, "meta.json")) or {}
    avant = m.get("paires") or {}
    cles = {}
    for t in it.get("garder") or []:
        mt = C.lire_json(os.path.join(C.CACHE, "tetes", t, "meta.json")) or {}
        if t in avant and mt.get("biblio") and os.path.isfile(os.path.join(d, t + ".bin")):
            cles.setdefault((mt["biblio"], tuple(float(x) for x in mt.get("translation") or ())), t)
    src = {t["id"]: cles.get((t["biblio"], tuple(float(x) for x in t["translation"]))) for t in it["tetes"]}
    if not src or not all(src.values()) or not os.path.isfile(os.path.join(d, "coiffure.glb")):
        return False
    paires = {t: avant[t] for t in it["garder"] if t in avant}
    for n, s in src.items():
        C.lier(os.path.join(d, s + ".bin"), os.path.join(d, n + ".bin"))   # same bytes: a link, not a copy
        paires[n] = dict(avant[s])
    propre = it.get("propre")
    retenue, raison = C.coiffure_retenue(paires, propre)
    C.ecrire_json(os.path.join(d, "meta.json"), dict(m, nom=it.get("nom", it["id"]), paires=paires,
                                                     retenue=bool(retenue), raison=raison, propre=propre,
                                                     empreinte=it.get("empreinte")))
    n_ok = sum(1 for r in paires.values() if r["passe"])
    journal("hairstyle", it["id"], "%d/%d heads" % (n_ok, len(paires)), "kept" if retenue else "rejected: %s" % raison)
    return True


def faire_coiffure(it):
    import hairstyles as CF
    d = os.path.join(C.CACHE, "coiffures", it["id"])
    if it.get("garder") and coiffure_deja_posee(it, d):
        return
    hair = CF.Preparation(it["biblio"], it["id"])
    os.makedirs(d, exist_ok=True)
    A = hair.A
    mats = [mat_glb(A.pmx, A.pmx["materiaux"][i], A.chemin, "cheveux") for i in hair.mats]
    for k, i in enumerate(hair.mats):
        if i in hair.tex_reflets:
            mats[k]["image"] = texture(hair.tex_reflets[i])
    GLB.ecrire(os.path.join(d, "coiffure.glb"), *hair.glb_base(), mats)
    paires = {}
    if it.get("garder"):   # pairs measured on an earlier pass (catalogue.plan_suite): only the new heads are posed
        avant = (C.lire_json(os.path.join(d, "meta.json")) or {}).get("paires") or {}
        paires = {t: avant[t] for t in it["garder"] if t in avant}
    # Heads that are the same target (same base head, same translation) get the same pose: it is computed once
    # and the other heads link to its file. With hundreds of models, posing and storing every pair separately
    # grew as heads x hairstyles in time and on disk (one identical file per pair).
    deja = {}
    for tete in it["tetes"]:
        cle = (tete.get("biblio"), tuple(float(x) for x in tete.get("translation") or ()))
        sortie = os.path.join(d, tete["id"] + ".bin")
        if cle in deja:
            C.lier(os.path.join(d, deja[cle] + ".bin"), sortie)
            paires[tete["id"]] = dict(paires[deja[cle]])
            continue
        res = hair.poser_sur(tete)
        C.ecrire_octets(sortie, res.pop("positions").astype("<f4").tobytes())
        paires[tete["id"]] = res
        deja[cle] = tete["id"]
    n_ok = sum(1 for r in paires.values() if r["passe"])
    propre = it.get("propre")
    retenue, raison = C.coiffure_retenue(paires, propre)
    C.ecrire_json(os.path.join(d, "meta.json"), {
        "id": it["id"], "nom": it.get("nom", it["id"]), "faces": int(sum(len(f) for f in hair.par_mat)),
        "sommets": int(len(hair.vs)), "paires": paires, "retenue": bool(retenue), "raison": raison,
        "propre": propre, "empreinte": it.get("empreinte")})
    journal("hairstyle", it["id"], "%d/%d heads" % (n_ok, len(paires)), "kept" if retenue else "rejected: %s" % raison)


def faire_coiffure_libre(it):
    """The base's hairstyle (tools/free_base, tools/vroid_base): made for its head, placed as is."""
    d = os.path.join(C.CACHE, "coiffures", it["id"])
    os.makedirs(d, exist_ok=True)
    P = PMX.lire(it["pmx"])   # the file holds only the hairstyle: all its materials (VRoid: 髪, 髪2…)
    n, f, _ = extraire_glb(it["pmx"], {m["nom"] for m in P["materiaux"]}, os.path.join(d, "coiffure.glb"), role="cheveux", P=P)
    C.ecrire_json(os.path.join(d, "meta.json"), {
        "id": it["id"], "nom": it.get("nom", it["id"]), "faces": f, "sommets": n,
        "paires": {t: {"passe": True, "echecs": []} for t in it["tetes"]}, "retenue": True, "raison": None,
        "propre": None, "empreinte": it.get("empreinte"), "libre": True, "pmx": it["pmx"]})
    journal("free hairstyle", f, "faces")


FAIRE = {"coiffure_libre": faire_coiffure_libre, "corps": faire_corps, "base": faire_base, "tete": faire_tete, "fabriquer": faire_fabriquee, "piece": faire_piece, "coiffure": faire_coiffure, "couvre": faire_couvre}


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    with open(a[a.index("--travail") + 1], encoding="utf-8") as fh:
        liste = json.load(fh)
    for it in liste:
        try:
            FAIRE[it["genre"]](it)
        except (Exception, SystemExit) as e:   # greffe_tete rejects through SystemExit
            journal("FAILED", it["genre"], it.get("id"), repr(e)[:300])
            traceback.print_exc()
            if it["genre"] in ("tete", "coiffure"):
                sous = {"tete": "tetes", "coiffure": "coiffures"}[it["genre"]]
                C.ecrire_json(os.path.join(C.CACHE, sous, it["id"], "meta.json"), {
                    "id": it["id"], "nom": it.get("nom", it["id"]), "retenue": False,
                    "raison": "preparation failed: %s" % (str(e)[:160] if isinstance(e, SystemExit) else "%s %s" % (type(e).__name__, str(e)[:140]))})


if __name__ == "__main__":
    main()
