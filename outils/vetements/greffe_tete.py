"""Graft a head with its hair (face, hair, physics, morphs) onto a PMX body with the same skeleton.

    bin/blender-file -b --factory-startup --python outils/vetements/greffe_tete.py -- \
        --corps BODY.pmx --tete HEAD.pmx --sortie FOLDER
        [--retirer 颜,白目,…] [--prendre 颜,颜_A,…,髮] [--peau 肌] [--couture-mm 0] [--sans-teinte] [--garder-crane]
    (Blender to read and write the tint textures; without tint, python3 -I is enough.)

When to use it: the body and the head's model share the same upper skeleton (here, the neck and
head bones differ by a plain translation); retarget_tete covers the other cases.

 - the head is placed by the translation that brings its 頭 bone onto the body's;
 - bones: a bone with the same name is shared, the others (hair) are added with their parents;
 - rigid bodies: those of the added bones, and those of the source attached to the upper body, which
   act as colliders for the hair; joints: those whose two rigid bodies are kept;
 - vertex morphs: carried over (other types are dropped, they refer to parts of the source
   model);
 - seam: ZZZ heads cover the body's skin (its edge runs 15-20 mm under the face);
   nothing to stitch, unless --couture-mm > 0: the nearby skin edge is then snapped onto the
   head's edge;
 - neck: the body skin that rises above the neck (scalp of a body from `peau_complete.py`) is
   removed where the head's original model cut its own, as on a game model
   (`cou_du_jeu.py`), and the head brings its ears; --raccord to weld it to the face edge instead (`raccord_cou.py`),
   --garder-crane to leave it untouched;
 - tint: the body skin is matched in Lab to the head's, measured on both sides of the seam
   (lightness and hue shift, texture rewritten into <sortie>/tex).

Output: <sortie>/resultat.pmx, textures in <sortie>/tex (relative paths), greffe.json.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
sys.path.insert(0, os.path.join(ICI, "..", "tete"))
sys.path.insert(0, ICI)
import pmx as PMX  # noqa: E402
import pmx_ecrire  # noqa: E402
import cou_du_jeu  # noqa: E402
import raccord_cou  # noqa: E402

MM = 1.0 / 80.0
TETE_DEFAUT = ("颜", "颜_A", "睫眉", "白目", "目", "目光", "目影", "眉", "睫", "颜2", "齿", "口线", "口舌", "髮")
RETIRER_DEFAUT = ("颜", "白目", "目", "目光", "二重", "睫", "眉", "口线", "口舌", "齿", "目影", "髮")
ILE_COU = "肌_cou"  # ears brought by the head (taken from the top of its neck)
HAUT_DU_CORPS = ("頭", "首", "上半身3", "上半身2", "上半身", "左肩", "右肩", "左腕", "右腕", "左胸", "右胸")


def args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corps", required=True)
    p.add_argument("--tete", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--retirer", default=",".join(RETIRER_DEFAUT), help="body materials that are removed")
    p.add_argument("--prendre", default=",".join(TETE_DEFAUT), help="materials taken from the head")
    p.add_argument("--peau", default="肌", help="body skin, whose edge is stitched to the head")
    p.add_argument("--peau-tete", default="颜", help="head skin that touches the body")
    p.add_argument("--couture-mm", type=float, default=0.0)
    p.add_argument("--ecart-cou-max", type=float, default=6.0,
                   help="tolerated offset (mm), once the head is placed, between the body's bones and those the grafted "
                        "vertices hang on (the neck for a whole head; 0 for hair on 頭 alone)")
    p.add_argument("--suivre-os", action="store_true",
                   help="the taken vertices also follow the offset of the body's bones they hang on (hair resting on "
                        "the shoulders of a body with a longer neck); for parts that are not stitched to the body skin")
    p.add_argument("--sans-teinte", action="store_true")
    p.add_argument("--garder-crane", action="store_true", help="leave the body skin under the head untouched")
    p.add_argument("--raccord", action="store_true",
                   help="weld the body skin to the face edge (raccord_cou) instead of cutting it at the neck as the game does")
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return p.parse_args(argv)


def soudure(pos, tol=1e-4):
    """Welded index of each vertex (positions equal within tol)."""
    _, inv = np.unique(np.round(pos / tol).astype(np.int64), axis=0, return_inverse=True)
    return inv.ravel()


def aretes_de_bord(F, sou):
    """Edges (raw indices) that touch only one face, once the vertices are welded."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    S = np.sort(sou[E], axis=1)
    _, inv, cnt = np.unique(S, axis=0, return_inverse=True, return_counts=True)
    return E[cnt[inv.ravel()] == 1]


def point_segment(q, a, b):
    """Closest point of each q on the best segment (a, b): (point, index)."""
    ab = b - a
    t = np.clip(((q[:, None] - a) * ab).sum(2) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    pts = a + t[..., None] * ab
    d = ((q[:, None] - pts) ** 2).sum(2)
    k = d.argmin(1)
    return pts[np.arange(len(q)), k], k, np.sqrt(d[np.arange(len(q)), k])


def decalages_os(osB, osH, idxB, t):
    """Per head bone, offset (model units) between the body's bone and the head's once the head is moved
    by t. A head bone the body lacks (hair, ribbon) is added under the body's copy of its nearest shared
    ancestor: it takes that ancestor's offset."""
    ec = np.zeros((len(osH), 3))
    for i in range(len(osH)):
        j = i
        while j >= 0 and osH[j]["nom"] not in idxB:
            j = osH[j]["parent"]
        if j >= 0:
            ec[i] = np.array(osB[idxB[osH[j]["nom"]]]["pos"]) - np.array(osH[j]["pos"]) - t
    return ec


def ecart_squelette(ec, os_v, poids_v):
    """Rest offset vector of grafted vertices: the skinning-weighted offset of the bones they hang on
    (ec from decalages_os). It is how far the body's bones sit from where the vertex's skinning expects them."""
    os_v, w = np.asarray(os_v), np.asarray(poids_v, float)
    w = np.where(os_v >= 0, w, 0.0)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return (w[..., None] * ec[np.maximum(os_v, 0)]).sum(1)


def copier_texture(src, dossier, faites, noms):
    if src in faites:
        return faites[src]
    base = os.path.basename(src)
    nom, k = base, 0
    while nom in noms:
        k += 1
        nom = "%d_%s" % (k, base)
    noms.add(nom)
    os.makedirs(dossier, exist_ok=True)
    shutil.copyfile(src, os.path.join(dossier, nom))
    faites[src] = "tex/" + nom
    return faites[src]


def chemin_texture(pmx_chemin, rel):
    return os.path.normpath(os.path.join(os.path.dirname(pmx_chemin), rel.replace("\\", "/")))


def main(argv=None):
    a = args(argv)
    os.makedirs(a.sortie, exist_ok=True)
    B = PMX.lire(a.corps)
    H = PMX.lire(a.tete)
    tB, tH = PMX.vers_tableaux(B), PMX.vers_tableaux(H)
    osB = {o["nom"]: i for i, o in enumerate(B["os"])}
    osH = {o["nom"]: i for i, o in enumerate(H["os"])}
    t = np.array(B["os"][osB["頭"]]["pos"]) - np.array(H["os"][osH["頭"]]["pos"])
    print("head translation (mm):", np.round(t / MM, 2))
    ecart_cou = (np.array(B["os"][osB["首"]]["pos"]) - np.array(H["os"][osH["首"]]["pos"]) - t) / MM
    print("neck offset once the head is placed (mm):", np.round(ecart_cou, 2))

    retirer = set(a.retirer.split(","))
    prendre = [n for n in a.prendre.split(",")]
    nomsB = [m["nom"] for m in B["materiaux"]]
    nomsH = [m["nom"] for m in H["materiaux"]]
    manque = [n for n in prendre if n not in nomsH]
    if manque:
        raise SystemExit("materials missing from the head: %s" % manque)

    # --- kept body vertices and faces
    garde_f = ~np.isin(tB["mat_face"], [i for i, n in enumerate(nomsB) if n in retirer])
    # nothing removed: the body keeps all its vertices, even those no face uses, and so their indices
    vB = np.arange(len(tB["pos"])) if garde_f.all() else np.unique(tB["faces"][garde_f])
    carteB = np.full(len(tB["pos"]), -1)
    carteB[vB] = np.arange(len(vB))
    # --- head vertices and faces
    iH = [nomsH.index(n) for n in prendre]
    fH_m = np.isin(tH["mat_face"], iH)
    # the head's ears and scalp (what it carries alone or nearly so), taken when the body is cut at the
    # neck as in the game
    ile_h = np.zeros(len(tH["faces"]), bool)
    if not a.garder_crane and not a.raccord:
        cands_peau = [nomsH.index(n) for n in (a.peau, "首", "体") if n in nomsH]
        if cands_peau:
            tete_h = np.array(H["os"][osH["頭"]]["pos"])
            vis_h = [nomsH.index(n) for n in (a.peau_tete, a.peau_tete + "_A") if n in nomsH]
            os_h = cou_du_jeu.os_de_tete(H["os"], osH["頭"])
            ile_h = cou_du_jeu.ile_du_cou(tH["pos"], tH["faces"], tH["mat_face"], cands_peau, vis_h,
                                          (tete_h[0], tete_h[2]), tH["os"], tH["poids"], os_h, tH["uv"])
            # and its scalp skin, which closes the head under the hair: in the ears' material, whose
            # texture it takes
            ile_h |= cou_du_jeu.crane_du_jeu(tH["pos"], tH["nor"], tH["faces"], tH["mat_face"],
                                             [int(tH["mat_face"][ile_h][0])] if ile_h.any() else cands_peau,
                                             vis_h, (tete_h[0], tete_h[2]), tH["os"], tH["poids"], os_h)
    fH_m |= ile_h
    vH = np.unique(tH["faces"][fH_m])
    # skeleton check on what is grafted: each taken vertex is moved by t (head placed by 頭) but then hangs on
    # the body's bones. Its offset is how far those bones sit from where its own skinning expects them.
    # A whole head (neck skin on 首) measures the neck as before; hair that hangs on 頭 alone measures 0,
    # whatever the body's neck length (our own shape has a longer neck than the game heads).
    dec_os = decalages_os(B["os"], H["os"], osB, t)
    dec_v = ecart_squelette(dec_os, tH["os"][vH], tH["poids"][vH])
    ecart_os = np.linalg.norm(dec_v, axis=1) / MM
    pire = int(np.argmax(ecart_os)) if len(ecart_os) else -1
    ecart_greffe = float(ecart_os[pire]) if pire >= 0 else 0.0
    print("skeleton offset under the grafted vertices (mm): max %.2f" % ecart_greffe)
    if a.suivre_os:
        # each taken vertex (and added bone) also takes its bones' offset: hair resting on the shoulders follows
        # the body's shoulders, hair on 頭 alone does not move; nothing is left to tolerate
        print("vertices moved onto the body's bones (mm): max %.2f, %d vertices moved by more than 1 mm"
              % (ecart_greffe, int((ecart_os > 1.0).sum())))
    else:
        dec_os[:] = 0.0
        dec_v[:] = 0.0
    if ecart_greffe > a.ecart_cou_max and not a.suivre_os:
        lourd = tH["os"][vH][pire][np.argmax(tH["poids"][vH][pire])]
        raise SystemExit("neck too different: a grafted vertex is %.1f mm from where its bones expect it (bone %s; "
                         "limit %.1f mm): this body and this head do not share the same skeleton, see retarget_tete"
                         % (ecart_greffe, H["os"][int(lourd)]["nom"], a.ecart_cou_max))
    carteH = np.full(len(tH["pos"]), -1)
    carteH[vH] = len(vB) + np.arange(len(vH))

    # --- bones
    utiles = set(int(b) for b, w in zip(tH["os"][vH].ravel(), tH["poids"][vH].ravel()) if b >= 0 and w > 0)
    for b in list(utiles):
        x = H["os"][b]["parent"]
        while x >= 0 and x not in utiles:
            utiles.add(int(x))
            x = H["os"][x]["parent"]
    nouveaux = [i for i in sorted(utiles) if H["os"][i]["nom"] not in osB]
    carte_os = {}
    for i in sorted(utiles):
        n = H["os"][i]["nom"]
        if n in osB:
            carte_os[i] = osB[n]
    oss = [dict(o) for o in B["os"]]
    for i in nouveaux:
        carte_os[i] = len(oss)
        oss.append(dict(H["os"][i]))
    for i in nouveaux:
        o = oss[carte_os[i]]
        o["pos"] = tuple(np.array(o["pos"]) + t + dec_os[i])
        o["parent"] = carte_os.get(o["parent"], osB["頭"])
        if o["queue_os"] is not None:
            q = o["queue_os"]
            if q in carte_os:
                o["queue_os"] = carte_os[q]
            else:
                o["queue_os"] = -1
        if o["ajout"] is not None:
            aj = dict(o["ajout"])
            if aj["source"] in carte_os:
                aj["source"] = carte_os[aj["source"]]
                o["ajout"] = aj
            else:
                o["ajout"] = None
                o["drapeaux"] &= ~(PMX.OS_AJOUT_ROT | PMX.OS_AJOUT_DEP | PMX.OS_AJOUT_LOCAL)
        if o["ik"] is not None:
            raise SystemExit("IK bone in the head: not supported (%s)" % o["nom"])
    print("bones: %d shared, %d added" % (len(utiles) - len(nouveaux), len(nouveaux)))

    # --- vertex tables
    nadd = max(B["n_uv_add"], H["n_uv_add"])
    sB, sH = B["sommets"], H["sommets"]

    def uvadd(s, i, n0):
        u = list(s["uv_add"][i]) if n0 else []
        return u + [(0.0, 0.0, 0.0, 0.0)] * (nadd - n0)
    out = {k: [] for k in sB}
    for i in vB.tolist():
        for k in ("pos", "nor", "uv", "type", "poids", "sdef", "contour"):
            out[k].append(sB[k][i])
        out["os"].append(sB["os"][i])
        out["uv_add"].append(uvadd(sB, i, B["n_uv_add"]))
    for k_v, i in enumerate(vH.tolist()):
        p = sH["pos"][i]
        tt = (t + dec_v[k_v]).tolist()     # dec_v is zero unless --suivre-os
        out["pos"].append((p[0] + tt[0], p[1] + tt[1], p[2] + tt[2]))
        for k in ("nor", "uv", "type", "poids", "contour"):
            out[k].append(sH[k][i])
        c = sH["sdef"][i]
        out["sdef"].append(None if c is None else tuple(np.array(c) + np.tile(tt, 3)))
        out["os"].append(tuple(carte_os[int(b)] if b >= 0 else -1 for b in sH["os"][i]))
        out["uv_add"].append(uvadd(sH, i, H["n_uv_add"]))
    P = np.array(out["pos"], float)
    N = np.array(out["nor"], float)

    # --- faces and materials
    dos = os.path.join(a.sortie, "tex")
    faites, noms_tex = {}, set()
    textures, matieres, faces = [], [], []
    ind_tex = {}

    def tex(pmx_ch, pmx, i):
        if i is None or i < 0:
            return -1
        chemin = chemin_texture(pmx_ch, pmx["textures"][i])
        if not os.path.exists(chemin):
            print("WARNING texture not found:", chemin)
            r = pmx["textures"][i]
        else:
            r = copier_texture(chemin, dos, faites, noms_tex)
        if r not in ind_tex:
            ind_tex[r] = len(textures)
            textures.append(r)
        return ind_tex[r]

    def ajout_mat(pmx_ch, pmx, im, Fm, carte):
        m = dict(pmx["materiaux"][im])
        m["texture"] = tex(pmx_ch, pmx, m["texture"])
        m["sphere"] = tex(pmx_ch, pmx, m["sphere"])
        if not m.get("toon_partage"):
            m["toon"] = tex(pmx_ch, pmx, m["toon"])
        m["n_indices"] = int(3 * len(Fm))
        matieres.append(m)
        faces.append(carte[Fm])

    for im, n in enumerate(nomsB):
        if n in retirer:
            continue
        ajout_mat(a.corps, B, im, tB["faces"][tB["mat_face"] == im], carteB)
    n_fB = sum(len(f) for f in faces)
    for im in iH:
        ajout_mat(a.tete, H, im, tH["faces"][tH["mat_face"] == im], carteH)
    if ile_h.any():
        im_ile = int(tH["mat_face"][ile_h][0])
        ajout_mat(a.tete, H, im_ile, tH["faces"][ile_h], carteH)
        matieres[-1]["nom"] = ILE_COU
    Ft = np.vstack(faces)

    # --- neck seam: body skin edge snapped onto the head skin edge
    i_peau = [m["nom"] for m in matieres].index(a.peau)
    noms_m = [m["nom"] for m in matieres]
    i_cou = noms_m.index(a.peau_tete if a.peau_tete in noms_m else a.peau)   # single-skin model: nothing to stitch
    dep = np.cumsum([0] + [m["n_indices"] // 3 for m in matieres])
    F_peau, F_cou = Ft[dep[i_peau]:dep[i_peau + 1]], Ft[dep[i_cou]:dep[i_cou + 1]]
    sou = soudure(P)
    bord_tete = aretes_de_bord(F_cou, sou)
    bord_corps = aretes_de_bord(F_peau, sou)
    vc = np.unique(bord_corps)
    pa, pb = P[bord_tete[:, 0]], P[bord_tete[:, 1]]
    y_min = B["os"][osB["首"]]["pos"][1] + 20 * MM        # (a body with closed skin has no edge: empty vc)
    vc = vc[P[vc, 1] > y_min]
    if not len(bord_tete):                               # head with closed skin too: no edge to follow
        vc = vc[:0]
        pa = pb = np.zeros((1, 3))
    pts, k, d = point_segment(P[vc], pa, pb)
    pris = d < a.couture_mm * MM if a.couture_mm > 0 else np.zeros(len(d), bool)
    print("seam: %d skin edge vertices above the neck, %d snapped; offset mm median %.1f, max %.1f, unsnapped max %.1f"
          % (len(vc), int(pris.sum()), float(np.median(d[pris]) / MM) if pris.any() else 0.0,
             float(d[pris].max() / MM) if pris.any() else 0.0,
             float(d[~pris].max() / MM) if (~pris).any() else 0.0))
    for v_, p_, k_ in zip(vc[pris].tolist(), pts[pris], k[pris].tolist()):
        n_ = N[bord_tete[k_, 0]] + N[bord_tete[k_, 1]]
        n_ = n_ / max(np.linalg.norm(n_), 1e-9)
        for x in np.where(sou == sou[v_])[0].tolist():
            if x < len(vB):
                out["pos"][x] = tuple(p_.tolist())
                out["nor"][x] = tuple(n_.tolist())

    # --- rigid bodies and joints
    gardes = {}
    rigs = []
    for ir, r in enumerate(H["rigides"]):
        if r["os"] < 0 or (r["os"] not in carte_os and H["os"][r["os"]]["nom"] not in osB):
            continue
        nom_os = H["os"][r["os"]]["nom"]
        ajoute = r["os"] in carte_os and nom_os not in osB
        haut = nom_os in HAUT_DU_CORPS and r["mode"] == 0
        if not (ajoute or haut):
            continue
        c = dict(r)
        c["os"] = carte_os[r["os"]] if r["os"] in carte_os else osB[nom_os]
        c["pos"] = tuple(np.array(r["pos"]) + t)
        gardes[ir] = len(rigs)
        rigs.append(c)
    jts = []
    for j in H["joints"]:
        if j["a"] in gardes and j["b"] in gardes:
            c = dict(j)
            c["a"], c["b"] = gardes[j["a"]], gardes[j["b"]]
            c["pos"] = tuple(np.array(j["pos"]) + t)
            jts.append(c)
    print("physics: %d rigid bodies, %d joints kept" % (len(rigs), len(jts)))

    # --- vertex morphs
    morphes = []
    for m in H["morphes"]:
        if m["type"] != 1:
            continue
        dec = [(int(carteH[i]), d) for i, d in m["decalages"] if carteH[i] >= 0]
        if dec:
            c = dict(m)
            c["decalages"] = dec
            morphes.append(c)
    print("morphs: %d kept" % len(morphes))

    # --- tint: body skin matched to the head's, on both sides of the seam
    teinte = None
    if not a.sans_teinte:
        teinte = regler_teinte(a, out, matieres, textures, Ft, dep, i_peau, i_cou)

    # --- face-neck join: done after seam and tint, which read the original skin edge
    crane = None
    if not a.garder_crane:
        noms_m = [m["nom"] for m in matieres]
        i_vis = [noms_m.index(n) for n in (a.peau_tete, a.peau_tete + "_A") if n in noms_m]
        mat_face = np.repeat(np.arange(len(matieres)), np.diff(dep))
        n_avant = len(Ft)
        if a.raccord:
            Ft, crane = raccord_cou.appliquer(out, Ft, mat_face, matieres, morphes, i_peau, i_vis,
                                              dossier=None if a.sans_teinte else a.sortie, textures=textures)
        else:
            # the donor head's neck, where its own body stopped
            # (without the ears or the scalp taken, which would widen and raise the reference neck), measured over its whole surface.
            # Without the clothes carried by the skin material: a neck skin vertex only follows bones that
            # the common body also has. Nicole's collar (bone Bn_LINGZI, in 体, up to half of the weight)
            # gave the body's neck a 55 mm ruff and a second jaw line; likewise the
            # collars, necklaces, ties, badges and pendants of the others.
            f_cou = np.isin(tH["mat_face"], [nomsH.index(n) for n in (a.peau, "首", "体") if n in nomsH]) & ~ile_h
            v_cou = np.unique(tH["faces"][f_cou])
            o_v, p_v = tH["os"][v_cou], tH["poids"][v_cou]
            du_corps = np.isin(o_v, [i for i, o in enumerate(H["os"]) if o["nom"] in osB]) & (o_v >= 0)
            garde = np.zeros(len(tH["pos"]), bool)
            garde[v_cou[(p_v * du_corps).sum(1) >= 0.99 * p_v.sum(1)]] = True
            # Nor the top of the shoulders: the neck wraps around the axis, the top of the shoulder faces up.
            F_cou = tH["faces"][f_cou]
            F_cou = F_cou[garde[F_cou].all(1) & (tH["nor"][F_cou, 1].mean(1) < 0.5)]
            cou_h = cou_du_jeu.surface(tH["pos"] + t, F_cou)
            tete_os = np.array(B["os"][osB["頭"]]["pos"])
            i_ile = noms_m.index(ILE_COU) if ILE_COU in noms_m else None
            Ft, crane = cou_du_jeu.appliquer(out, Ft, mat_face, matieres, i_peau, i_vis, cou_h, (tete_os[0], tete_os[2]), i_ile,
                                             cou_du_jeu.os_de_tete(B["os"], osB["頭"]))
        n_fB -= n_avant - len(Ft)
        print("face-neck join:", crane)

    R = dict(B)
    R["n_uv_add"] = nadd
    R["sommets"] = out
    R["faces"] = Ft.tolist()
    R["textures"] = textures
    R["materiaux"] = matieres
    R["os"] = oss
    R["morphes"] = morphes
    R["rigides"] = rigs
    R["joints"] = jts
    R["cadres"] = [dict(c, elements=[(ty, x) for ty, x in c["elements"] if ty == 0 and x < len(B["os"])])
                   for c in B["cadres"]]
    chemin = os.path.join(a.sortie, "resultat.pmx")
    tmp = chemin + ".tmp"
    pmx_ecrire.ecrire(R, tmp)
    os.replace(tmp, chemin)
    info = {"corps": a.corps, "tete": a.tete, "translation_mm": [round(float(x / MM), 2) for x in t],
            "ecart_cou_mm": [round(float(x), 2) for x in ecart_cou], "ecart_greffe_mm": round(ecart_greffe, 2), "faces_corps": int(n_fB), "faces_tete": int(len(Ft) - n_fB),
            "sommets": len(out["pos"]), "os_ajoutes": len(nouveaux), "rigides": len(rigs), "joints": len(jts),
            "morphes": len(morphes), "teinte": teinte, "crane": crane}
    with open(os.path.join(a.sortie, "greffe.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print("written:", chemin, "%d vertices, %d faces" % (len(out["pos"]), len(Ft)))


# ---------------------------------------------------------------------- tint

def srgb_vers_lab(c):
    c = np.asarray(c, float)
    l = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    x = (l @ M.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(x > 0.008856, np.cbrt(x), 7.787 * x + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_vers_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    x = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    l = np.clip(x @ M.T, 0, 1)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


def lire_image(chemin):
    """sRGB RGBA, rows top to bottom (bpy: Blender's Python lacks Pillow)."""
    import bpy
    im = bpy.data.images.load(chemin, check_existing=False)
    im.colorspace_settings.name = "Non-Color"
    w, h = im.size
    px = np.empty(4 * w * h, np.float32)
    im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    return px.reshape(h, w, 4)[::-1].copy()


def ecrire_image(chemin, px):
    import bpy
    h, w = px.shape[:2]
    im = bpy.data.images.new("tmp", w, h, alpha=True)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(np.clip(px[::-1], 0, 1).astype(np.float32).ravel())
    im.filepath_raw = chemin
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)


def regler_teinte(a, out, matieres, textures, Ft, dep, i_peau, i_cou):
    """Lab median of the skin within 25 mm of the seam, on each side."""
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("tint: must run in Blender (texture reading); nothing changed")
        return None
    P = np.array(out["pos"], float)
    UV = np.array(out["uv"], float)
    Fp = Ft[dep[i_peau]:dep[i_peau + 1]]
    Fc = Ft[dep[i_cou]:dep[i_cou + 1]]
    cp, cc = P[Fp].mean(1), P[Fc].mean(1)
    def voisins(X, Y, r):
        tree = None
        d = np.full(len(X), np.inf)
        for k in range(0, len(Y), 200):
            dd = np.sqrt(((X[:, None] - Y[k:k + 200][None]) ** 2).sum(2)).min(1)
            d = np.minimum(d, dd)
        del tree
        return d < r
    r = 25 * MM
    mp = voisins(cp, cc, r)
    mc = voisins(cc, cp, r)

    def lab_de(F_, m, itex):
        img = lire_image(os.path.join(a.sortie, textures[itex]))
        h, w = img.shape[:2]
        u = np.mod(UV[F_[m]].mean(1), 1.0)
        px = img[(u[:, 1] * h).astype(int).clip(0, h - 1), (u[:, 0] * w).astype(int).clip(0, w - 1)]
        return srgb_vers_lab(px[:, :3])
    Lp = np.median(lab_de(Fp, mp, matieres[i_peau]["texture"]), 0)
    Lc = np.median(lab_de(Fc, mc, matieres[i_cou]["texture"]), 0)
    d = Lc - Lp
    print("seam tint: body Lab %s, head Lab %s, ΔE %.1f" % (np.round(Lp, 1), np.round(Lc, 1), float(np.linalg.norm(d))))
    if np.linalg.norm(d) < 1.5:
        return {"corps": Lp.round(1).tolist(), "tete": Lc.round(1).tolist(), "applique": False}
    ch = os.path.join(a.sortie, textures[matieres[i_peau]["texture"]])
    im = lire_image(ch)
    lab = srgb_vers_lab(im[..., :3])
    lab = lab + d
    rgb = lab_vers_srgb(lab)
    res = np.dstack([rgb, im[..., 3]])
    tmp = ch + ".tmp.png"
    ecrire_image(tmp, res)
    os.replace(tmp, ch)
    return {"corps": Lp.round(1).tolist(), "tete": Lc.round(1).tolist(), "applique": True}


if __name__ == "__main__":
    main()
