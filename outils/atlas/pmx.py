"""PMX 2.0 / 2.1 reader in pure Python (struct), without mmd_tools.

`lire(chemin)` returns a dict of Python lists (no dependency).
`vers_tableaux(pmx)` converts it into numpy arrays (Blender's numpy).

Material names: `lire` is the only place where the core translates a game's names into the common language
(outils/jeux/README.md): a PMX stored in a source folder of a game whose description has a
"materiaux" table gets its names translated (the original name stays in "nom_jeu"). Elsewhere, and for ZZZ, nothing changes.

Frame: positions stay in MMD units (left-handed, Y up, facing -Z).
The conversion to Blender is in `repere.py`.
"""
import os
import struct
import sys

# Bone flags (PMX 2.0 specification)
OS_QUEUE_INDEX = 0x0001
OS_ROTATION = 0x0002
OS_DEPLACEMENT = 0x0004
OS_VISIBLE = 0x0008
OS_ACTIF = 0x0010
OS_IK = 0x0020
OS_AJOUT_LOCAL = 0x0080
OS_AJOUT_ROT = 0x0100
OS_AJOUT_DEP = 0x0200
OS_AXE_FIXE = 0x0400
OS_AXES_LOCAUX = 0x0800
OS_APRES_PHYSIQUE = 0x1000
OS_PARENT_EXTERNE = 0x2000

# Material flags
MAT_DOUBLE_FACE = 0x01
MAT_OMBRE_SOL = 0x02
MAT_PROJETTE = 0x04
MAT_RECOIT = 0x08
MAT_CONTOUR = 0x10
MAT_COULEUR_SOMMET = 0x20  # 2.1
MAT_POINTS = 0x40  # 2.1
MAT_LIGNES = 0x80  # 2.1

TYPES_POIDS = {0: "BDEF1", 1: "BDEF2", 2: "BDEF4", 3: "SDEF", 4: "QDEF"}
TYPES_MORPHE = {0: "groupe", 1: "sommet", 2: "os", 3: "uv", 4: "uv1", 5: "uv2",
                6: "uv3", 7: "uv4", 8: "materiau", 9: "bascule", 10: "impulsion"}
PANNEAUX = {0: "systeme", 1: "sourcils", 2: "yeux", 3: "bouche", 4: "autre"}
MODES_PHYSIQUE = {0: "suit_os", 1: "physique", 2: "physique_position_os"}
FORMES_RIGIDE = {0: "sphere", 1: "boite", 2: "capsule"}


class Lecteur:
    def __init__(self, donnees):
        self.d = donnees
        self.p = 0

    def lit(self, fmt):
        t = struct.calcsize("<" + fmt)
        v = struct.unpack_from("<" + fmt, self.d, self.p)
        self.p += t
        return v

    def un(self, fmt):
        return self.lit(fmt)[0]

    def octets(self, n):
        v = self.d[self.p:self.p + n]
        self.p += n
        return v


def _fmt_index(taille, sommet=False):
    if sommet:
        return {1: "B", 2: "H", 4: "i"}[taille]
    return {1: "b", 2: "h", 4: "i"}[taille]


def _jeu_du_fichier(chemin):
    """The game (outils/jeux) whose "materiaux" table translates this file, otherwise None."""
    outils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if outils not in sys.path:
        sys.path.append(outils)
    import jeux
    return jeux.jeu_du_fichier(chemin)


def lire(chemin):
    with open(chemin, "rb") as f:
        donnees = f.read()
    r = Lecteur(donnees)
    if r.octets(4) != b"PMX ":
        raise ValueError("not a PMX file: %s" % chemin)
    version = round(r.un("f"), 2)
    n_glob = r.un("B")
    glob = list(r.lit("%dB" % n_glob))
    encodage, n_uv_add, t_som, t_tex, t_mat, t_os, t_morph, t_rig = glob[:8]
    enc = "utf-16-le" if encodage == 0 else "utf-8"

    def texte():
        n = r.un("i")
        return r.octets(n).decode(enc, errors="replace")

    fs = _fmt_index(t_som, True)
    ft = _fmt_index(t_tex)
    fm = _fmt_index(t_mat)
    fo = _fmt_index(t_os)
    fmo = _fmt_index(t_morph)
    fr = _fmt_index(t_rig)

    pmx = {
        "version": version,
        "encodage": "UTF-16" if encodage == 0 else "UTF-8",
        "n_uv_add": n_uv_add,
        "tailles_index": {"sommet": t_som, "texture": t_tex, "materiau": t_mat,
                          "os": t_os, "morphe": t_morph, "rigide": t_rig},
    }
    pmx["nom"] = texte()
    pmx["nom_en"] = texte()
    pmx["commentaire"] = texte()
    pmx["commentaire_en"] = texte()

    # --- Vertices
    n = r.un("i")
    pos, nor, uv, uva, typ, idx, poids, sdef, contour = [], [], [], [], [], [], [], [], []
    for _ in range(n):
        v = r.lit("8f")
        pos.append(v[0:3])
        nor.append(v[3:6])
        uv.append(v[6:8])
        uva.append([r.lit("4f") for _ in range(n_uv_add)])
        t = r.un("B")
        typ.append(t)
        if t == 0:
            b = r.un(fo)
            idx.append((b, -1, -1, -1))
            poids.append((1.0, 0.0, 0.0, 0.0))
            sdef.append(None)
        elif t == 1:
            b1, b2 = r.lit(fo * 2)
            w = r.un("f")
            idx.append((b1, b2, -1, -1))
            poids.append((w, 1.0 - w, 0.0, 0.0))
            sdef.append(None)
        elif t in (2, 4):
            idx.append(r.lit(fo * 4))
            poids.append(r.lit("4f"))
            sdef.append(None)
        elif t == 3:
            b1, b2 = r.lit(fo * 2)
            w = r.un("f")
            c = r.lit("9f")
            idx.append((b1, b2, -1, -1))
            poids.append((w, 1.0 - w, 0.0, 0.0))
            sdef.append(c)
        else:
            raise ValueError("unknown weight type %d" % t)
        contour.append(r.un("f"))
    pmx["sommets"] = {"pos": pos, "nor": nor, "uv": uv, "uv_add": uva, "type": typ,
                      "os": idx, "poids": poids, "sdef": sdef, "contour": contour}

    # --- Faces
    n = r.un("i")
    ind = r.lit(fs * n)
    pmx["faces"] = [ind[i:i + 3] for i in range(0, n, 3)]

    # --- Textures
    n = r.un("i")
    pmx["textures"] = [texte() for _ in range(n)]

    # --- Materials
    n = r.un("i")
    mats = []
    for _ in range(n):
        m = {"nom": texte(), "nom_en": texte()}
        m["diffus"] = r.lit("4f")
        m["speculaire"] = r.lit("3f")
        m["brillance"] = r.un("f")
        m["ambiant"] = r.lit("3f")
        m["drapeaux"] = r.un("B")
        m["contour_couleur"] = r.lit("4f")
        m["contour_taille"] = r.un("f")
        m["texture"] = r.un(ft)
        m["sphere"] = r.un(ft)
        m["sphere_mode"] = r.un("B")
        m["toon_partage"] = r.un("B")
        m["toon"] = r.un("B") if m["toon_partage"] == 1 else r.un(ft)
        m["memo"] = texte()
        m["n_indices"] = r.un("i")
        mats.append(m)
    J = _jeu_du_fichier(chemin)
    if J is not None:
        for m in mats:
            t = J.traduire(m["nom"])
            if t != m["nom"]:
                m["nom_jeu"], m["nom"] = m["nom"], t
    pmx["materiaux"] = mats

    # --- Bones
    n = r.un("i")
    oss = []
    for _ in range(n):
        o = {"nom": texte(), "nom_en": texte()}
        o["pos"] = r.lit("3f")
        o["parent"] = r.un(fo)
        o["couche"] = r.un("i")
        fl = r.un("H")
        o["drapeaux"] = fl
        if fl & OS_QUEUE_INDEX:
            o["queue_os"] = r.un(fo)
            o["queue_dec"] = None
        else:
            o["queue_os"] = None
            o["queue_dec"] = r.lit("3f")
        o["ajout"] = None
        if fl & (OS_AJOUT_ROT | OS_AJOUT_DEP):
            src = r.un(fo)
            w = r.un("f")
            o["ajout"] = {"source": src, "facteur": w,
                          "rotation": bool(fl & OS_AJOUT_ROT),
                          "deplacement": bool(fl & OS_AJOUT_DEP),
                          "local": bool(fl & OS_AJOUT_LOCAL)}
        o["axe_fixe"] = r.lit("3f") if fl & OS_AXE_FIXE else None
        o["axes_locaux"] = (r.lit("3f"), r.lit("3f")) if fl & OS_AXES_LOCAUX else None
        o["parent_externe"] = r.un("i") if fl & OS_PARENT_EXTERNE else None
        o["ik"] = None
        if fl & OS_IK:
            cible = r.un(fo)
            boucles = r.un("i")
            limite = r.un("f")
            nl = r.un("i")
            liens = []
            for _ in range(nl):
                b = r.un(fo)
                lim = r.un("B")
                if lim:
                    liens.append({"os": b, "min": r.lit("3f"), "max": r.lit("3f")})
                else:
                    liens.append({"os": b, "min": None, "max": None})
            o["ik"] = {"cible": cible, "boucles": boucles, "angle_limite": limite, "liens": liens}
        o["apres_physique"] = bool(fl & OS_APRES_PHYSIQUE)
        oss.append(o)
    pmx["os"] = oss

    # --- Morphs
    n = r.un("i")
    morphs = []
    for _ in range(n):
        m = {"nom": texte(), "nom_en": texte()}
        m["panneau"] = r.un("B")
        t = r.un("B")
        m["type"] = t
        k = r.un("i")
        dec = []
        for _ in range(k):
            if t in (0, 9):
                dec.append((r.un(fmo), r.un("f")))
            elif t == 1:
                dec.append((r.un(fs), r.lit("3f")))
            elif t == 2:
                dec.append((r.un(fo), r.lit("3f"), r.lit("4f")))
            elif 3 <= t <= 7:
                dec.append((r.un(fs), r.lit("4f")))
            elif t == 8:
                mi = r.un(fm)
                op = r.un("B")
                dec.append({"materiau": mi, "operation": op, "valeurs": r.lit("28f")})
            elif t == 10:
                dec.append((r.un(fr), r.un("B"), r.lit("3f"), r.lit("3f")))
            else:
                raise ValueError("unknown morph type %d" % t)
        m["decalages"] = dec
        morphs.append(m)
    pmx["morphes"] = morphs

    # --- Display frames
    n = r.un("i")
    cadres = []
    for _ in range(n):
        c = {"nom": texte(), "nom_en": texte(), "special": r.un("B")}
        k = r.un("i")
        el = []
        for _ in range(k):
            ty = r.un("B")
            el.append((ty, r.un(fo) if ty == 0 else r.un(fmo)))
        c["elements"] = el
        cadres.append(c)
    pmx["cadres"] = cadres

    # --- Rigid bodies
    n = r.un("i")
    rig = []
    for _ in range(n):
        c = {"nom": texte(), "nom_en": texte()}
        c["os"] = r.un(fo)
        c["groupe"] = r.un("B")
        c["masque"] = r.un("H")
        c["forme"] = r.un("B")
        c["taille"] = r.lit("3f")
        c["pos"] = r.lit("3f")
        c["rot"] = r.lit("3f")
        c["masse"] = r.un("f")
        c["amort_dep"] = r.un("f")
        c["amort_rot"] = r.un("f")
        c["rebond"] = r.un("f")
        c["friction"] = r.un("f")
        c["mode"] = r.un("B")
        rig.append(c)
    pmx["rigides"] = rig

    # --- Joints
    n = r.un("i")
    jts = []
    for _ in range(n):
        j = {"nom": texte(), "nom_en": texte()}
        j["type"] = r.un("B")
        j["a"] = r.un(fr)
        j["b"] = r.un(fr)
        j["pos"] = r.lit("3f")
        j["rot"] = r.lit("3f")
        j["dep_min"] = r.lit("3f")
        j["dep_max"] = r.lit("3f")
        j["rot_min"] = r.lit("3f")
        j["rot_max"] = r.lit("3f")
        j["ressort_dep"] = r.lit("3f")
        j["ressort_rot"] = r.lit("3f")
        jts.append(j)
    pmx["joints"] = jts
    pmx["octets_restants"] = len(donnees) - r.p
    return pmx


def vers_tableaux(pmx):
    """Numpy arrays of the vertices and faces (MMD units and frame)."""
    import numpy as np
    s = pmx["sommets"]
    t = {
        "pos": np.asarray(s["pos"], dtype=np.float64),
        "nor": np.asarray(s["nor"], dtype=np.float64),
        "uv": np.asarray(s["uv"], dtype=np.float64),
        "type_poids": np.asarray(s["type"], dtype=np.int8),
        "os": np.asarray(s["os"], dtype=np.int32),
        "poids": np.asarray(s["poids"], dtype=np.float64),
        "contour": np.asarray(s["contour"], dtype=np.float64),
        "faces": np.asarray(pmx["faces"], dtype=np.int64).reshape(-1, 3),
    }
    if pmx["n_uv_add"]:
        t["uv_add"] = np.asarray(s["uv_add"], dtype=np.float64)  # (n, k, 4)
    else:
        t["uv_add"] = np.zeros((len(s["pos"]), 0, 4))
    sd = np.zeros((len(s["pos"]), 9))
    for i, c in enumerate(s["sdef"]):
        if c is not None:
            sd[i] = c
    t["sdef"] = sd
    # material of each face, in file order
    mat = np.zeros(len(t["faces"]), dtype=np.int32)
    deb = 0
    for i, m in enumerate(pmx["materiaux"]):
        k = m["n_indices"] // 3
        mat[deb:deb + k] = i
        deb += k
    t["mat_face"] = mat
    return t


def resume_json(pmx):
    """Everything except the heavy geometry, serializable to JSON."""
    out = {k: v for k, v in pmx.items() if k not in ("sommets", "faces")}
    out["n_sommets"] = len(pmx["sommets"]["pos"])
    out["n_faces"] = len(pmx["faces"])
    # vertex morph offsets are heavy: they are kept in the npz
    morphs = []
    for m in pmx["morphes"]:
        mm = dict(m)
        if m["type"] in (1, 3, 4, 5, 6, 7):
            mm["decalages"] = None
            mm["n_decalages"] = len(m["decalages"])
        else:
            mm["n_decalages"] = len(m["decalages"])
        morphs.append(mm)
    out["morphes"] = morphs
    return out
