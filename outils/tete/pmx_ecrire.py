"""PMX 2.0 / 2.1 writer, exact mirror of `outils/atlas/pmx.lire`.

`ecrire(pmx, chemin)` takes the dict returned by `pmx.lire` (possibly modified:
vertices, faces, materials, morphs...) and writes it. Index sizes are recomputed
from the content; the encoding is kept. A round trip lire -> ecrire -> lire returns
the same values (up to 32-bit floats).
"""
import os
import struct
import tempfile

import pmx as PMX


def _taille(n, sommet=False):
    if sommet:
        return 1 if n < 256 else 2 if n < 65536 else 4
    return 1 if n < 128 else 2 if n < 32768 else 4


def _fmt(taille, sommet=False):
    return PMX._fmt_index(taille, sommet)


def ecrire(pmx, chemin):
    enc = "utf-16-le" if pmx.get("encodage", "UTF-16") == "UTF-16" else "utf-8"
    s = pmx["sommets"]
    n_uv_add = pmx["n_uv_add"]
    t_som = _taille(len(s["pos"]), True)
    t_tex = _taille(len(pmx["textures"]))
    t_mat = _taille(len(pmx["materiaux"]))
    t_os = _taille(len(pmx["os"]))
    t_mor = _taille(len(pmx["morphes"]))
    t_rig = _taille(len(pmx["rigides"]))
    fs, ft, fm, fo, fmo, fr = (_fmt(t_som, True), _fmt(t_tex), _fmt(t_mat), _fmt(t_os),
                               _fmt(t_mor), _fmt(t_rig))
    out = bytearray()

    def w(fmt, *v):
        out.extend(struct.pack("<" + fmt, *v))

    def texte(x):
        b = (x or "").encode(enc)
        w("i", len(b))
        out.extend(b)

    out.extend(b"PMX ")
    w("f", float(pmx.get("version", 2.0)))
    w("B", 8)
    w("8B", 0 if enc == "utf-16-le" else 1, n_uv_add, t_som, t_tex, t_mat, t_os, t_mor, t_rig)
    texte(pmx.get("nom", ""))
    texte(pmx.get("nom_en", ""))
    texte(pmx.get("commentaire", ""))
    texte(pmx.get("commentaire_en", ""))

    # --- vertices
    n = len(s["pos"])
    w("i", n)
    for i in range(n):
        w("8f", *s["pos"][i], *s["nor"][i], *s["uv"][i])
        for k in range(n_uv_add):
            w("4f", *s["uv_add"][i][k])
        t = int(s["type"][i])
        w("B", t)
        o, p = s["os"][i], s["poids"][i]
        if t == 0:
            w(fo, int(o[0]))
        elif t == 1:
            w(fo * 2, int(o[0]), int(o[1]))
            w("f", float(p[0]))
        elif t in (2, 4):
            w(fo * 4, *[int(x) for x in o])
            w("4f", *[float(x) for x in p])
        elif t == 3:
            w(fo * 2, int(o[0]), int(o[1]))
            w("f", float(p[0]))
            w("9f", *s["sdef"][i])
        w("f", float(s["contour"][i]))

    # --- faces
    fl = [int(x) for f in pmx["faces"] for x in f]
    w("i", len(fl))
    out.extend(struct.pack("<%d%s" % (len(fl), fs), *fl))

    # --- textures
    w("i", len(pmx["textures"]))
    for x in pmx["textures"]:
        texte(x)

    # --- materials
    w("i", len(pmx["materiaux"]))
    for m in pmx["materiaux"]:
        texte(m["nom"])
        texte(m["nom_en"])
        w("4f", *m["diffus"])
        w("3f", *m["speculaire"])
        w("f", m["brillance"])
        w("3f", *m["ambiant"])
        w("B", m["drapeaux"])
        w("4f", *m["contour_couleur"])
        w("f", m["contour_taille"])
        w(ft, m["texture"])
        w(ft, m["sphere"])
        w("B", m["sphere_mode"])
        w("B", m["toon_partage"])
        if m["toon_partage"] == 1:
            w("B", m["toon"])
        else:
            w(ft, m["toon"])
        texte(m["memo"])
        w("i", m["n_indices"])

    # --- bones
    w("i", len(pmx["os"]))
    for o in pmx["os"]:
        texte(o["nom"])
        texte(o["nom_en"])
        w("3f", *o["pos"])
        w(fo, o["parent"])
        w("i", o["couche"])
        f = o["drapeaux"]
        w("H", f)
        if f & PMX.OS_QUEUE_INDEX:
            w(fo, o["queue_os"])
        else:
            w("3f", *o["queue_dec"])
        if f & (PMX.OS_AJOUT_ROT | PMX.OS_AJOUT_DEP):
            w(fo, o["ajout"]["source"])
            w("f", o["ajout"]["facteur"])
        if f & PMX.OS_AXE_FIXE:
            w("3f", *o["axe_fixe"])
        if f & PMX.OS_AXES_LOCAUX:
            w("3f", *o["axes_locaux"][0])
            w("3f", *o["axes_locaux"][1])
        if f & PMX.OS_PARENT_EXTERNE:
            w("i", o["parent_externe"])
        if f & PMX.OS_IK:
            ik = o["ik"]
            w(fo, ik["cible"])
            w("i", ik["boucles"])
            w("f", ik["angle_limite"])
            w("i", len(ik["liens"]))
            for l in ik["liens"]:
                w(fo, l["os"])
                if l["min"] is not None:
                    w("B", 1)
                    w("3f", *l["min"])
                    w("3f", *l["max"])
                else:
                    w("B", 0)

    # --- morphs
    w("i", len(pmx["morphes"]))
    for m in pmx["morphes"]:
        texte(m["nom"])
        texte(m["nom_en"])
        w("B", m["panneau"])
        t = m["type"]
        w("B", t)
        w("i", len(m["decalages"]))
        for d in m["decalages"]:
            if t in (0, 9):
                w(fmo, d[0])
                w("f", d[1])
            elif t == 1:
                w(fs, d[0])
                w("3f", *d[1])
            elif t == 2:
                w(fo, d[0])
                w("3f", *d[1])
                w("4f", *d[2])
            elif 3 <= t <= 7:
                w(fs, d[0])
                w("4f", *d[1])
            elif t == 8:
                w(fm, d["materiau"])
                w("B", d["operation"])
                w("28f", *d["valeurs"])
            elif t == 10:
                w(fr, d[0])
                w("B", d[1])
                w("3f", *d[2])
                w("3f", *d[3])

    # --- display frames
    w("i", len(pmx["cadres"]))
    for c in pmx["cadres"]:
        texte(c["nom"])
        texte(c["nom_en"])
        w("B", c["special"])
        w("i", len(c["elements"]))
        for ty, ix in c["elements"]:
            w("B", ty)
            w(fo if ty == 0 else fmo, ix)

    # --- rigid bodies
    w("i", len(pmx["rigides"]))
    for c in pmx["rigides"]:
        texte(c["nom"])
        texte(c["nom_en"])
        w(fo, c["os"])
        w("B", c["groupe"])
        w("H", c["masque"])
        w("B", c["forme"])
        w("3f", *c["taille"])
        w("3f", *c["pos"])
        w("3f", *c["rot"])
        for k in ("masse", "amort_dep", "amort_rot", "rebond", "friction"):
            w("f", c[k])
        w("B", c["mode"])

    # --- joints
    w("i", len(pmx["joints"]))
    for j in pmx["joints"]:
        texte(j["nom"])
        texte(j["nom_en"])
        w("B", j["type"])
        w(fr, j["a"])
        w(fr, j["b"])
        for k in ("pos", "rot", "dep_min", "dep_max", "rot_min", "rot_max", "ressort_dep",
                  "ressort_rot"):
            w("3f", *j[k])

    d = os.path.dirname(os.path.abspath(chemin))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(out)
    os.replace(tmp, chemin)
