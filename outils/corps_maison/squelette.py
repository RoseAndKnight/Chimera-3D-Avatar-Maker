"""The house skeleton: bones of our own, placed by measurement on our mesh, with their twist.

    python3 -I squelette.py [--sortie squelette.json]      # places the bones, compares them with those of ZZZ

What sets it apart from VRoid and ZZZ:
 - the joints are placed by measurements on the rings of the grid (mesures.py) and by
   anatomy rules written here, not taken from another model;
 - the twist is in the skeleton: the forearm follows the roll of the hand, the upper arm and the thigh
   hold back the roll of their bone (VRoid has no twist bones; those of ZZZ are turned by hand);
 - one bone per toe, under a common bone "orteils" (toes) that keeps the usual role;
 - readable names (French): bassin, colonne, bras_G, avant_bras_G, index1_G...

Frame: cm, y up, front = -z, character's left = +x. As in MMD, the rest frame of each
bone is aligned with the world: a local rotation is a rotation about the world axes, under the parent.
"""
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
for d in (ICI, os.path.join(ICI, "..", "atlas")):
    if d not in sys.path:
        sys.path.insert(0, d)
import cinematique as CIN     # noqa: E402
import mesures as M           # noqa: E402

DOIGTS = ("index", "majeur", "annulaire", "petit_doigt")
# anatomy rules (fractions of length), each one stated where it is used
PART_COUDE = 0.54          # shoulder -> elbow, over shoulder -> wrist (humerus 0.186 H; forearm 0.146 H... rounded for the posed arm)
PART_PHALANGES = (0.45, 0.28, 0.27)   # the three phalanges, fingertip included
PART_DOIGT_LIBRE = 0.75    # the web is in the middle of the first phalanx: the free finger is 3/4 of it
PART_POUCE = 0.52          # first phalanx of the thumb over its free part
PART_COLONNE_ARRIERE = 0.35   # the spine is in the back third of the trunk, not in the middle
PART_CAGE = 0.6            # the trunk is a cage, not a line: crossbars up to 60 % of the half-width
TORSION = (1 / 3, 2 / 3)   # twist steps between the two ends of a bone

# roles (VRM 1.0, as in the squelettes tool) and equivalent MMD name, to translate poses and animations
ROLES = {"racine": ("x.root", "全ての親"), "bassin": ("hips", "下半身"), "colonne": ("spine", "上半身"),
         "poitrine": ("chest", "上半身1"), "haut_poitrine": ("upperChest", "上半身2"), "cou": ("neck", "首"),
         "tete": ("head", "頭")}
for _c, _C, _m in (("G", "left", "左"), ("D", "right", "右")):
    for _n, _r, _j in (("epaule", "Shoulder", "肩"), ("bras", "UpperArm", "腕"), ("avant_bras", "LowerArm", "ひじ"),
                       ("main", "Hand", "手首"), ("cuisse", "UpperLeg", "足"), ("jambe", "LowerLeg", "ひざ"),
                       ("pied", "Foot", "足首"), ("orteils", "Toes", "つま先")):
        ROLES["%s_%s" % (_n, _c)] = (_C + _r, _m + _j)
    for _n, _r, _j in (("pouce", "Thumb", "親指"), ("index", "Index", "人指"), ("majeur", "Middle", "中指"),
                       ("annulaire", "Ring", "薬指"), ("petit_doigt", "Little", "小指")):
        for _k, _p in enumerate(("Metacarpal", "Proximal", "Distal") if _n == "pouce" else ("Proximal", "Intermediate", "Distal")):
            ROLES["%s%d_%s" % (_n, _k + 1, _c)] = (_C + _r + _p, _m + _j + "０１２"[_k] if _n == "pouce" else _m + _j + "１２３"[_k])


def _le_long(C, s):
    """Point at abscissa s (cm) along a line of centres; before the start: the first slope is extended."""
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    if s <= 0:
        d = (C[1] - C[0]) / np.linalg.norm(C[1] - C[0])
        return C[0] + s * d
    s = min(s, L[-1])
    k = min(int(np.searchsorted(L, s, "right")) - 1, len(C) - 2)
    return C[k] + (C[k + 1] - C[k]) * (s - L[k]) / (L[k + 1] - L[k])


def placer(g):
    """Positions of the joints of the left side and the middle, with the measurement that gives them."""
    P = g["P"]
    J, dit = {}, {}

    def pose(nom, p, comment):
        J[nom], dit[nom] = np.asarray(p, float), comment

    # ---- leg: ankle, knee, hip
    r, C, L = M.ligne(g, "jambe")
    sol = float(P[:, 1].min())
    fin = int(np.argmin(L[40:56])) + 40                      # the narrowest ring of the lower leg
    che = next(v for v in range(fin, len(L)) if L[v] > 1.15 * L[fin])
    pose("pied_G", C[che], "first ring below the narrowest of the lower leg (rank %d) that widens by 15 per cent: the malleoli" % fin)
    y_fourche = float(P[M.anneau(g, "bande", 4)][:, 1].min())
    larg = float(np.abs(P[M.anneau(g, "tronc", 0)][:, 0]).max())
    axe = (C[6] - C[20]) / np.linalg.norm(C[6] - C[20])
    y_h = y_fourche + 0.10 * (y_fourche - sol)
    h = C[6] + axe * (y_h - C[6][1]) / axe[1]
    pose("cuisse_G", [0.45 * larg, y_h, h[2]],
         "height: crotch + 10 % of the leg; spread: 45 % of the half-width of the hips; depth: axis of the thigh")
    creux = int(np.argmin(L[20:34])) + 20                    # the hollow under the knee
    y_g = C[creux][1] + 0.04 * (y_h - C[che][1])
    k = int(np.argmin(np.abs(C[:, 1] - y_g)))
    pose("jambe_G", [C[k][0], y_g, C[k][2]], "4 per cent of the leg above the hollow under the knee (narrowest ring, rank %d)" % creux)
    pose("orteils_G", C[-2], "centre of the second-to-last ring of the foot: the line where the toes attach")
    talon = P[M.anneau(g, "jambe", che + 2)]
    J["_talon_G"] = talon[talon[:, 2].argmax()] * [1, 1, 0.8] + [0, 0, 0]
    for k in range(5):
        _, Cd, _ = M.ligne(g, "doigt_orteil%d" % k)
        lng = np.linalg.norm(np.diff(Cd, axis=0), axis=1).sum()
        pose("orteil%d_G" % (k + 1), _le_long(Cd, -min(1.0, 0.35 * lng)), "a little behind the base of the toe")
        J["_bout_orteil%d_G" % (k + 1)] = Cd[-1]

    # ---- trunk
    rt = M.rangs(g, "tronc")
    per = np.array([M.perimetre(g, "tronc", v) for v in rt])
    v_t = int(np.argmin(per))

    def dos(carte, v):
        """Point of the spine at this ring: in the back third, on the middle plane."""
        p = P[M.anneau(g, carte, v)]
        m = p[np.abs(p[:, 0]) < 3.0]
        za, zb = m[:, 2].min(), m[:, 2].max()
        return np.array([0.0, p[:, 1].mean(), zb - PART_COLONNE_ARRIERE * (zb - za)])
    taille = dos("tronc", v_t)
    y_aisselle = float(P[M.anneau(g, "bras", 0)][:, 1].min())
    pose("bassin", [0, y_h + 0.3 * (taille[1] - y_h), taille[2]], "30 % of the way from the hips to the waist")
    pose("colonne", [0, y_h + 0.4 * (taille[1] - y_h), taille[2]], "40 % of the way from the hips to the waist: the bottom of the lumbar spine")
    pose("poitrine", taille, "the waist: the narrowest ring of the trunk (rank %d)" % v_t)
    v_s = int(np.argmin(np.abs(np.array([P[M.anneau(g, "tronc", v)][:, 1].mean() for v in rt]) - (taille[1] + y_aisselle) / 2)))
    pose("haut_poitrine", dos("tronc", v_s), "halfway from the waist to the armpit: under the chest")
    rc = M.rangs(g, "cou")
    pose("cou", dos("cou", rc[0]), "the base of the neck")
    pose("tete", dos("cou", rc[int(round(0.6 * (len(rc) - 1)))]), "at 60 % of the neck")
    J["_haut_tete"] = J["tete"] + [0, 5.0, 0]

    # ---- arm: wrist, shoulder, elbow
    r, C, L = M.ligne(g, "bras")
    poi = int(np.argmin(L[20:40])) + 20
    pose("main_G", C[poi], "the narrowest ring of the arm (rank %d)" % poi)
    axe = (C[4] - C[12]) / np.linalg.norm(C[4] - C[12])
    ray = L[4] / (2 * np.pi)
    pas = np.arange(0.0, 20.0, 0.25)
    libre = np.array([np.linalg.norm(P - (C[4] + x * axe), axis=1).min() for x in pas])
    fin = next((k for k in range(len(pas)) if libre[k] < 0.5 * ray), len(pas))      # the axis exits through the top of the shoulder
    s = pas[int(np.argmax(libre[:fin]))]
    pose("bras_G", C[4] + s * axe, "on the axis of the arm extended into the shoulder: the point where the largest ball fits under the skin (%.1f cm)" % libre[:fin].max())
    d = J["main_G"] - J["bras_G"]
    y_c = J["bras_G"] + PART_COUDE * d
    k = int(np.argmin(np.linalg.norm(C[:poi] - y_c, axis=1)))
    pose("avant_bras_G", C[k], "%d per cent from the shoulder to the wrist, brought back to the centre of the ring" % round(100 * PART_COUDE))
    pose("epaule_G", [0.18 * J["bras_G"][0], J["bras_G"][1] + 0.25 * (J["cou"][1] - J["bras_G"][1]), J["cou"][2]],
         "near the sternum: 18 % of the shoulder spread, a quarter of the way to the base of the neck")

    # ---- hand
    for k, nom in enumerate(DOIGTS):
        _, Cd, _ = M.ligne(g, "doigt_doigt%d" % k)
        lng = np.linalg.norm(np.diff(Cd, axis=0), axis=1).sum()
        tout = lng / PART_DOIGT_LIBRE
        s0 = lng - tout
        for i, part in enumerate((0.0, PART_PHALANGES[0], PART_PHALANGES[0] + PART_PHALANGES[1])):
            pose("%s%d_G" % (nom, i + 1), _le_long(Cd, s0 + part * tout),
                 "phalanges at 45, 28 and 27 % of the finger, 3/4 of which is free")
        J["_bout_%s_G" % nom] = Cd[-1]
    _, Cd, _ = M.ligne(g, "doigt_pouce")
    lng = np.linalg.norm(np.diff(Cd, axis=0), axis=1).sum()
    pose("pouce1_G", J["main_G"] + 0.35 * (Cd[0] - J["main_G"]), "35 % from the wrist to the free base of the thumb")
    pose("pouce2_G", Cd[0], "the free base of the thumb")
    pose("pouce3_G", _le_long(Cd, PART_POUCE * lng), "52 % of the free part of the thumb")
    J["_bout_pouce_G"] = Cd[-1]
    J["_sol"] = np.array([0.0, sol, 0.0])
    return J, dit


def cage(g, y0, y1, x_max):
    """Crossbars through the trunk between two heights (left-right and front-back, every other ring): the
    skin of the trunk is far from the spine; without them, the chest and the flanks take the heat of the arm."""
    P = g["P"]
    segs = []
    for v in M.rangs(g, "tronc")[::2]:
        p = P[M.anneau(g, "tronc", v)]
        y = float(p[:, 1].mean())
        if not (y0 <= y < y1):
            continue
        m = p[np.abs(p[:, 0]) < 3.0]
        za, zb = m[:, 2].min(), m[:, 2].max()
        zc = 0.5 * (za + zb)
        dx = PART_CAGE * min(float(np.abs(p[:, 0]).max()), x_max)
        dz = PART_CAGE * 0.5 * (zb - za)
        segs.append(([-dx, y, zc], [dx, y, zc]))
        segs.append(([0.0, y, zc - dz], [0.0, y, zc + dz]))
    return segs


def construire(g, place=None):
    """The whole skeleton: list of bones {nom, parent, pos, segments, role, mmd, torsion}.
    place: (joints, measurements) already placed (base_libre/corps.py); otherwise placer(g)."""
    J, dit = place or placer(g)
    os_ = []

    def os(nom, parent, pos, segs, torsion=None):
        role, mmd = ROLES.get(nom, (None, None))
        os_.append({"nom": nom, "parent": parent, "pos": [float(x) for x in pos],
                    "segments": [[[float(x) for x in a], [float(x) for x in b]] for a, b in segs],
                    "role": role, "mmd": mmd, "torsion": torsion, "mesure": dit.get(nom.replace("_D", "_G"))})

    mir = np.array([-1.0, 1.0, 1.0])
    os("racine", None, [0, J["_sol"][1], 0], [])
    hanches = J["cuisse_G"] * [0, 1, 1]
    large = 1e9
    os("bassin", "racine", J["bassin"], [(J["bassin"], hanches), (J["cuisse_G"], J["cuisse_G"] * mir)]
       + cage(g, J["cuisse_G"][1], J["colonne"][1], large))
    for a, b in (("colonne", "poitrine"), ("poitrine", "haut_poitrine"), ("haut_poitrine", "cou"), ("cou", "tete")):
        haut = min(J[b][1], J["epaule_G"][1] - 2.0)          # above, the shoulder holds the skin
        barres = cage(g, J[a][1], haut, J["bras_G"][0]) if a != "cou" else []
        os(a, "bassin" if a == "colonne" else os_[-1]["nom"], J[a], [(J[a], J[b])] + barres)
    os("tete", "cou", J["tete"], [(J["tete"], J["_haut_tete"])])
    for c, m in (("G", np.ones(3)), ("D", mir)):
        def j(n):
            return J[n + "_G"] * m

        def tors(nom, parent, a, b, sorte, source, paliers):
            """Twist bones along a -> b. kind "retient" (holds back): child of the bone, undoes a share of its
            roll (the top stays in place); "suit" (follows): child of the bone, takes a share of the roll of the source."""
            axe = (b - a) / np.linalg.norm(b - a)
            for i, t in enumerate(paliers):
                os("%s_torsion%d_%s" % (nom, i + 1, c), parent, a + t * (b - a), [],
                   {"sorte": sorte, "source": source, "axe": [float(x) for x in axe], "part": float(t)})
        os("epaule_" + c, "haut_poitrine", j("epaule"), [(j("epaule"), j("bras"))])
        os("bras_" + c, "epaule_" + c, j("bras"), [(j("bras"), j("avant_bras"))])
        tors("bras", "bras_" + c, j("bras"), j("avant_bras"), "retient", "bras_" + c, (0.0,) + TORSION)
        os("avant_bras_" + c, "bras_" + c, j("avant_bras"), [(j("avant_bras"), j("main"))])
        tors("avant_bras", "avant_bras_" + c, j("avant_bras"), j("main"), "suit", "main_" + c, TORSION + (1.0,))
        os("main_" + c, "avant_bras_" + c, j("main"), [(j("main"), j(d + "1")) for d in DOIGTS])
        for d in ("pouce",) + DOIGTS:
            pts = [j(d + "1"), j(d + "2"), j(d + "3"), J["_bout_%s_G" % d] * m]
            for i in range(3):
                os("%s%d_%s" % (d, i + 1, c), "main_" + c if i == 0 else "%s%d_%s" % (d, i, c), pts[i], [(pts[i], pts[i + 1])])
        os("cuisse_" + c, "bassin", j("cuisse"), [(j("cuisse"), j("jambe"))])
        tors("cuisse", "cuisse_" + c, j("cuisse"), j("jambe"), "retient", "cuisse_" + c, (0.0,) + TORSION)
        os("jambe_" + c, "cuisse_" + c, j("jambe"), [(j("jambe"), j("pied"))])
        os("pied_" + c, "jambe_" + c, j("pied"), [(j("pied"), j("orteils")), (j("pied"), J["_talon_G"] * m)])
        os("orteils_" + c, "pied_" + c, j("orteils"), [])
        for k in range(1, 6):
            os("orteil%d_%s" % (k, c), "orteils_" + c, j("orteil%d" % k), [(j("orteil%d" % k), J["_bout_orteil%d_G" % k] * m)])
    return os_


# ------------------------------------------------------------------------------------------ kinematics
def _roulis(q, axe):
    """Part of a rotation that turns about an axis (swing / twist decomposition)."""
    p = np.dot(q[1:], axe) * np.asarray(axe)
    r = np.r_[q[0], p]
    n = np.linalg.norm(r)
    return r / n if n > 1e-9 else CIN.IDENTITE.copy()


class Squelette:
    def __init__(self, os_):
        self.os = os_
        self.noms = [o["nom"] for o in os_]
        self.i = {n: k for k, n in enumerate(self.noms)}
        self.tetes = np.array([o["pos"] for o in os_])
        self.parents = np.array([-1 if o["parent"] is None else self.i[o["parent"]] for o in os_])
        self.de_mmd = {o["mmd"]: o["nom"] for o in os_ if o["mmd"]}
        n = len(os_)
        self._rien = (np.full(n, -1), np.zeros(n), np.zeros(n, bool), np.zeros(n, bool))

    def miroir(self):
        """Index of the opposite bone."""
        return np.array([self.i.get(n[:-2] + ("_D" if n.endswith("_G") else "_G"), k) if n[-2:] in ("_G", "_D") else k
                         for k, n in enumerate(self.noms)])

    def poser(self, rotations):
        """World matrices (R, t) for local rotations given by bone name; the twist follows on its own."""
        rot = {self.i[n]: np.asarray(q, float) for n, q in rotations.items()}
        for k, o in enumerate(self.os):
            t = o["torsion"]
            if not t:
                continue
            r = _roulis(rot.get(self.i[t["source"]], CIN.IDENTITE), t["axe"])
            rot[k] = CIN.q_puissance(r, t["part"] - 1.0 if t["sorte"] == "retient" else t["part"])
        return CIN.cinematique(self.tetes, self.parents, *self._rien, rotations=rot)


def ecrire(os_, chemin):
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({"unite": "cm", "repere": "y up, front -z, character's left +x",
                   "torsion": "retient: the bone undoes (1 - part) of the roll of its source about the axis; "
                              "suit: the bone takes part of the roll of its source about the axis",
                   "os": os_}, f, ensure_ascii=False, indent=1)


def lire(chemin):
    with open(chemin, encoding="utf-8") as f:
        return Squelette(json.load(f)["os"])


def main():
    import pmx as PMX
    g = M.charger()
    os_ = construire(g)
    a = sys.argv[1:]
    if "--sortie" in a:
        ecrire(os_, a[a.index("--sortie") + 1])
    Z = PMX.lire(os.path.join(os.path.dirname(M.FORME), "corps.pmx"))
    z = {o["nom"]: np.array(o["pos"]) * 8 for o in Z["os"]}
    print("%d bones. Gap to the ZZZ bone with the same role (cm):" % len(os_))
    for o in os_:
        if o["mmd"] in z and not o["nom"].endswith("_D"):
            p = np.array(o["pos"])
            print("  %-16s %-7s (%6.1f %6.1f %6.1f)  ZZZ (%6.1f %6.1f %6.1f)  gap %4.1f   %s" % (
                o["nom"], o["mmd"], *p, *z[o["mmd"]], np.linalg.norm(p - z[o["mmd"]]), o["mesure"] or ""))


if __name__ == "__main__":
    main()
