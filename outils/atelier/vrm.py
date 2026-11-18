"""VRM 1.0 export of an assembled model (PMX read by pmx.py). Numpy only, except for the texture
conversion (`images`, supplied by Blender).

    ecrire_vrm(P, "sortie.vrm", images=dict(file -> (bytes, mimetype)), nom="Outfit", auteur="Chimera")

The cache's glTF frame is kept: metres, character facing +Z, its left towards +X. The model is put
into T-pose right here (MMD arms are in A-pose): VRM wants bones without rotation, arms horizontal.
Only useful bones are kept: those carrying weight, the humanoid ones and their ancestors.
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

METRE = 0.08
ALPHA_MASQUE = 0.4
PART_DECOUPE = 0.001  # share of pixels below the threshold from which the texture has holes

# VRM humanoid bone -> MMD bones, from the most precise to the fallback
HUMANOIDES = {
    "hips": ["下半身"], "spine": ["上半身"], "chest": ["上半身1"], "upperChest": ["上半身2", "上半身3"],
    "neck": ["首"], "head": ["頭"], "leftEye": ["左目"], "rightEye": ["右目"],
    "leftShoulder": ["左肩"], "leftUpperArm": ["左腕"], "leftLowerArm": ["左ひじ"], "leftHand": ["左手首"],
    "rightShoulder": ["右肩"], "rightUpperArm": ["右腕"], "rightLowerArm": ["右ひじ"], "rightHand": ["右手首"],
    "leftUpperLeg": ["左足D", "左足"], "leftLowerLeg": ["左ひざD", "左ひざ"], "leftFoot": ["左足首D", "左足首"], "leftToes": ["左足先EX"],
    "rightUpperLeg": ["右足D", "右足"], "rightLowerLeg": ["右ひざD", "右ひざ"], "rightFoot": ["右足首D", "右足首"], "rightToes": ["右足先EX"],
}
for cote, vrm in (("左", "left"), ("右", "right")):
    for doigt, mmd, tronc in (("Thumb", "親指", ("Metacarpal", "Proximal", "Distal")), ("Index", "人指", None), ("Middle", "中指", None),
                              ("Ring", "薬指", None), ("Little", "小指", None)):
        if tronc:  # the MMD thumb goes from ０ to ２
            for k, nom in zip("０１２", tronc):
                HUMANOIDES["%s%s%s" % (vrm, doigt, nom)] = ["%s%s%s" % (cote, mmd, k)]
        else:
            for k, nom in zip("１２３", ("Proximal", "Intermediate", "Distal")):
                HUMANOIDES["%s%s%s" % (vrm, doigt, nom)] = ["%s%s%s" % (cote, mmd, k)]
OBLIGATOIRES = ["hips", "spine", "head", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand",
                "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]

# Weights placed on the control chain go to the deforming chain
DOUBLES = {"左足": "左足D", "左ひざ": "左ひざD", "左足首": "左足首D", "右足": "右足D", "右ひざ": "右ひざD", "右足首": "右足首D"}

# VRM expression -> MMD morphs (the first one present)
EXPRESSIONS = {
    "happy": ["笑い", "にこり"], "angry": ["怒り", "怒り目"], "sad": ["悲しむ", "困る"], "surprised": ["びっくり"], "relaxed": ["じと目"],
    "blink": ["まばたき"], "blinkLeft": ["ウィンク", "ウィンク２"], "blinkRight": ["ウィンク右", "ウィンク２右"],
    "aa": ["あ"], "ih": ["い"], "ou": ["う"], "ee": ["え"], "oh": ["お"],
}


def _vers_gltf(v):
    v = np.asarray(v, float) * METRE
    return v * np.array([1.0, 1.0, -1.0])


def _rotation(a, b):
    """Smallest rotation bringing direction a onto b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.9999:
        return np.diag([-1.0, 1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


class Squelette:
    """Kept bones (PMX indices), parents corrected for VRM, and T-pose."""

    def __init__(self, P):
        self.P = P
        self.noms = [o["nom"] for o in P["os"]]
        self.par_nom = {n: i for i, n in enumerate(self.noms)}
        self.pos = _vers_gltf([o["pos"] for o in P["os"]])
        self.parent = [o["parent"] for o in P["os"]]
        I = np.array(P["sommets"]["os"], np.int64)
        W = np.array(P["sommets"]["poids"], float)
        self.I, self.W = self._sans_doubles(I, W)
        self.humain = {}
        # house skeleton (corps_maison/brancher.py): each bone carries its VRM role as its English name
        roles = {o.get("nom_en"): i for i, o in enumerate(P["os"]) if o.get("nom_en") in HUMANOIDES}
        self.maison = all(b in roles for b in OBLIGATOIRES)
        for vrm, candidats in HUMANOIDES.items():
            if self.maison and vrm in roles:
                self.humain[vrm] = roles[vrm]
                continue
            for n in candidats:
                if n in self.par_nom:
                    self.humain[vrm] = self.par_nom[n]
                    break
        manque = [b for b in OBLIGATOIRES if b not in self.humain]
        if manque:
            raise ValueError("missing humanoid bones: %s" % ", ".join(manque))
        # in MMD, upper and lower body are siblings under 腰: VRM wants the spine under the hips
        self.parent[self.humain["spine"]] = self.humain["hips"]
        self._reprendre_chaine()
        self.garde = self._gardes()

    def _sans_doubles(self, I, W):
        carte = np.arange(len(self.noms))
        for a, b in DOUBLES.items():
            if a in self.par_nom and b in self.par_nom:
                carte[self.par_nom[a]] = self.par_nom[b]
        J = I.copy()
        J[J >= 0] = carte[J[J >= 0]]
        return J, W

    def _reprendre_chaine(self):
        """Each humanoid bone descends from the humanoid bone before it: the parent is moved up if
        a control branch sits in between without going through it."""
        def ancetres(i):
            out, vus = [], set()
            while i >= 0 and i not in vus:
                vus.add(i)
                out.append(i)
                i = self.parent[i]
            return out
        chaine = {"chest": "spine", "upperChest": "chest", "neck": "upperChest", "head": "neck",
                  "leftUpperArm": "leftShoulder", "rightUpperArm": "rightShoulder", "leftShoulder": "upperChest",
                  "rightShoulder": "upperChest", "leftLowerArm": "leftUpperArm", "rightLowerArm": "rightUpperArm",
                  "leftHand": "leftLowerArm", "rightHand": "rightLowerArm", "leftLowerLeg": "leftUpperLeg", "rightLowerLeg": "rightUpperLeg",
                  "leftFoot": "leftLowerLeg", "rightFoot": "rightLowerLeg", "leftToes": "leftFoot", "rightToes": "rightFoot",
                  "leftUpperLeg": "hips", "rightUpperLeg": "hips"}
        for b, p in chaine.items():
            if b in self.humain and p in self.humain and self.humain[p] not in ancetres(self.humain[b]):
                self.parent[self.humain[b]] = self.humain[p]

    def _gardes(self):
        utiles = set(int(x) for x in np.unique(self.I[(self.I >= 0) & (self.W > 0)]))
        utiles |= set(self.humain.values())
        for i in list(utiles):
            while i >= 0:
                utiles.add(i)
                i = self.parent[i]
        return sorted(utiles)

    def tpose(self, jambes=False):
        """Returns (bone positions in T-pose, rotation per bone, deformation per vertex)."""
        n = len(self.noms)
        cible = {}
        for cote, x in (("left", 1.0), ("right", -1.0)):
            for bone, suite in (("UpperArm", "LowerArm"), ("LowerArm", "Hand"), ("Hand", "MiddleProximal")):
                a, b = cote + bone, cote + suite
                if a in self.humain and b in self.humain:
                    cible[self.humain[a]] = (self.humain[b], np.array([x, 0.0, 0.0]))
            a, b = cote + "UpperLeg", cote + "LowerLeg"
            if jambes and self.maison:   # the thigh straight down: its twist is then expressed along a frame axis
                cible[self.humain[a]] = (self.humain[b], np.array([0.0, -1.0, 0.0]))
        ordre, vus = [], set()

        def visiter(i):
            if i < 0 or i in vus:
                return
            vus.add(i)
            visiter(self.parent[i])
            ordre.append(i)
        for i in range(n):
            visiter(i)
        Rw = np.tile(np.eye(3), (n, 1, 1))
        Pw = self.pos.copy()
        for i in ordre:
            p = self.parent[i]
            Rp = Rw[p] if p >= 0 else np.eye(3)
            Pw[i] = (Pw[p] + Rp @ (self.pos[i] - self.pos[p])) if p >= 0 else self.pos[i]
            R = Rp
            if i in cible:
                c, t = cible[i]
                R = _rotation(Rp @ (self._vers(i, c)), t) @ Rp
            Rw[i] = R
        return Pw, Rw

    def _vers(self, a, b):
        return self.pos[b] - self.pos[a]


class Bin:
    def __init__(self):
        self.o = bytearray()
        self.vues, self.acc = [], []

    def vue(self, octets, cible=None):
        while len(self.o) % 4:
            self.o.append(0)
        v = {"buffer": 0, "byteOffset": len(self.o), "byteLength": len(octets)}
        if cible:
            v["target"] = cible
        self.o += octets
        self.vues.append(v)
        return len(self.vues) - 1

    def acces(self, a, composant, type_, cible=None, bornes=False, norm=False):
        a = np.asarray(a)
        v = self.vue(a.tobytes(), cible)
        d = {"bufferView": v, "componentType": composant, "count": int(len(a)), "type": type_}
        if norm:
            d["normalized"] = True
        if bornes:
            d["min"] = [float(x) for x in a.min(0)]
            d["max"] = [float(x) for x in a.max(0)]
        self.acc.append(d)
        return len(self.acc) - 1


def _mtoon(m, tex, double, nom, decoupe=True):
    """`decoupe`: the texture has real holes. Without them, a MASK material gains nothing and
    breaks in viewers that plug the alpha channel in as is: it stays opaque."""
    d = list(m["diffus"])
    base = {"baseColorFactor": [float(d[0]), float(d[1]), float(d[2]), float(d[3])], "metallicFactor": 0.0, "roughnessFactor": 0.9}
    if tex is not None:
        base["baseColorTexture"] = {"index": tex}
    a = {"name": nom, "pbrMetallicRoughness": base, "doubleSided": double,
         "extensions": {"KHR_materials_unlit": {}, "VRMC_materials_mtoon": {
             "specVersion": "1.0", "transparentWithZWrite": False, "renderQueueOffsetNumber": 0,
             "shadeColorFactor": [float(d[0]) * 0.8, float(d[1]) * 0.72, float(d[2]) * 0.72],
             "shadingShiftFactor": -0.1, "shadingToonyFactor": 0.9, "giEqualizationFactor": 0.9,
             "matcapFactor": [0.0, 0.0, 0.0], "parametricRimColorFactor": [0.0, 0.0, 0.0], "rimLightingMixFactor": 1.0,
             "outlineLightingMixFactor": 1.0, "uvAnimationScrollXSpeedFactor": 0.0, "uvAnimationScrollYSpeedFactor": 0.0,
             "uvAnimationRotationSpeedFactor": 0.0}}}
    mt = a["extensions"]["VRMC_materials_mtoon"]
    if tex is not None:
        mt["shadeMultiplyTexture"] = {"index": tex}
    if m["drapeaux"] & 0x10 and m["contour_taille"] > 0:
        mt["outlineWidthMode"] = "worldCoordinates"
        mt["outlineWidthFactor"] = float(m["contour_taille"]) * 0.0012
        mt["outlineColorFactor"] = [float(x) for x in m["contour_couleur"][:3]]
    if "+" in nom:
        a["alphaMode"] = "BLEND"
    elif tex is not None and decoupe:
        a["alphaMode"] = "MASK"
        a["alphaCutoff"] = ALPHA_MASQUE
    return a


def ecrire_vrm(P, sortie, images, nom="Character", auteur="Chimera", torsion=None, viseurs=False):
    """Writes `sortie`. `images`: PMX texture (path as written in P) -> (bytes, mimetype).
    `torsion`: twist bones of the house skeleton, name -> {sorte, source, part} (squelette.json). A bone that "follows"
    (suit) becomes a VRM 1.0 roll constraint, changing nothing for a viewer that ignores constraints. A bone that
    "holds back" (retient) would need a negative weight, which VRM lacks: its rule is in `extras`. With `viseurs`, it
    goes under an aimer (aim constraint: its source's swing without its roll) and takes back its share of the
    roll: exact in a VRM viewer (three-vrm), but the bone no longer follows its arm in a viewer that ignores
    constraints (Blender, Godot). Hence the default: without."""
    S = Squelette(P)
    Pw, Rw = S.tpose(jambes=viseurs)
    # ---- vertices in T-pose
    X = _vers_gltf(P["sommets"]["pos"])
    N = np.array(P["sommets"]["nor"], float) * np.array([1.0, 1.0, -1.0])
    UV = np.array(P["sommets"]["uv"], float)
    I, W = S.I, S.W
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Y, M = np.zeros_like(X), np.zeros_like(N)
    for k in range(4):
        b = I[:, k]
        ok = (b >= 0) & (W[:, k] > 0)
        rel = (X[ok] - S.pos[b[ok]])[:, :, None]
        Y[ok] += W[ok, k][:, None] * (Pw[b[ok]] + (Rw[b[ok]] @ rel)[:, :, 0])
        M[ok] += W[ok, k][:, None] * (Rw[b[ok]] @ N[ok][:, :, None])[:, :, 0]
    M /= np.maximum(np.linalg.norm(M, axis=1), 1e-9)[:, None]
    # ---- joints
    joints = S.garde
    idx_joint = {b: k for k, b in enumerate(joints)}
    J = np.zeros((len(I), 4), np.uint16)
    for k in range(4):
        ok = I[:, k] >= 0
        J[ok, k] = [idx_joint[int(x)] for x in I[ok, k]]
    Wn = np.where(I >= 0, W, 0.0).astype(np.float32)
    Wn /= np.maximum(Wn.sum(1, keepdims=True), 1e-6)
    # ---- faces (glTF order, winding reversed by the switch to +Z), per material
    F = np.array(P["faces"], np.int64)
    mats, k = [], 0
    for m in P["materiaux"]:
        n = m["n_indices"] // 3
        mats.append(F[k:k + n])
        k += n
    # ---- selected morphs and the vertices they touch
    morphes = {m["nom"]: m for m in P["morphes"] if m["type"] == 1}
    choisis = {}
    for expr, cands in EXPRESSIONS.items():
        for c in cands:
            if c in morphes:
                choisis[expr] = morphes[c]
                break
    avec_morphe = set()
    for m in choisis.values():
        avec_morphe.update(i for i, _ in m["decalages"])
    tete = [i for i, f in enumerate(mats) if len(f) and any(int(x) in avec_morphe for x in np.unique(f))]
    autres = [i for i, f in enumerate(mats) if len(f) and i not in tete]
    # ---- writing
    B = Bin()
    meshes, images_doc, textures, materiaux, tex_de = [], [], [], [], {}

    def tex(i):
        if i < 0 or i >= len(P["textures"]):
            return None
        cle = P["textures"][i]
        if cle not in images:
            return None
        if cle not in tex_de:
            octets, mime = images[cle][:2]
            images_doc.append({"bufferView": B.vue(octets), "mimeType": mime, "name": os.path.basename(cle.replace("\\", "/"))})
            textures.append({"source": len(images_doc) - 1, "sampler": 0})
            tex_de[cle] = len(textures) - 1
        return tex_de[cle]

    def maillage(liste, nom_m, cibles):
        vs = np.unique(np.concatenate([mats[i].ravel() for i in liste]))
        carte = -np.ones(len(X), np.int64)
        carte[vs] = np.arange(len(vs))
        at = {"POSITION": B.acces(Y[vs].astype("<f4"), 5126, "VEC3", 34962, bornes=True),
              "NORMAL": B.acces(M[vs].astype("<f4"), 5126, "VEC3", 34962),
              "TEXCOORD_0": B.acces(UV[vs].astype("<f4"), 5126, "VEC2", 34962),
              "JOINTS_0": B.acces(J[vs].astype("<u2"), 5123, "VEC4", 34962),
              "WEIGHTS_0": B.acces(Wn[vs].astype("<f4"), 5126, "VEC4", 34962)}
        cib = []
        for m in cibles:
            d = np.zeros((len(vs), 3), np.float32)
            for i, v in m["decalages"]:
                if carte[i] >= 0:
                    d[carte[i]] = (float(v[0]) * METRE, float(v[1]) * METRE, -float(v[2]) * METRE)
            cib.append({"POSITION": B.acces(d.astype("<f4"), 5126, "VEC3", 34962, bornes=True)})
        prims = []
        for i in liste:
            m = P["materiaux"][i]
            g = carte[mats[i]]
            if not len(materiaux) > i or materiaux[i] is None:
                while len(materiaux) <= i:
                    materiaux.append(None)
                entree = images.get(P["textures"][m["texture"]]) if 0 <= m["texture"] < len(P["textures"]) else None
                decoupe = entree is None or len(entree) < 3 or bool(entree[2])
                materiaux[i] = _mtoon(m, tex(m["texture"]), bool(m["drapeaux"] & 1), m["nom"], decoupe)
            pr = {"attributes": at, "indices": B.acces(g[:, ::-1].astype("<u4").ravel(), 5125, "SCALAR", 34963),
                  "material": i, "mode": 4}
            if cib:
                pr["targets"] = cib
            prims.append(pr)
        d = {"name": nom_m, "primitives": prims}
        if cib:
            d["extras"] = {"targetNames": [m["nom"] for m in cibles]}
        meshes.append(d)
        return len(meshes) - 1

    cibles = list(dict.fromkeys(id(m) for m in choisis.values()))
    par_id = {id(m): m for m in choisis.values()}
    cibles_m = [par_id[c] for c in cibles]
    if tete:
        i_tete = maillage(tete, "tete", cibles_m)
    i_autres = maillage(autres, "corps", []) if autres else None
    while None in materiaux:   # materials without faces: an empty material keeps the numbering
        materiaux[materiaux.index(None)] = {"name": "vide"}

    # ---- bones -> nodes
    n_joints = len(joints)
    noeuds = []
    for b in joints:
        p = S.parent[b]
        while p >= 0 and p not in idx_joint:
            p = S.parent[p]
        t = Pw[b] - (Pw[p] if p >= 0 else np.zeros(3))
        noeuds.append({"name": S.noms[b], "translation": [float(x) for x in t]})
    for k, b in enumerate(joints):
        p = S.parent[b]
        while p >= 0 and p not in idx_joint:
            p = S.parent[p]
        if p >= 0:
            noeuds[idx_joint[p]].setdefault("children", []).append(k)
    # ---- twist of the house skeleton
    torsion = torsion or {}
    contraintes = 0
    pos_n = [Pw[b] for b in joints]
    noeud_de = {S.noms[b]: k for k, b in enumerate(joints)}
    pere = {c: k for k, n in enumerate(noeuds) for c in n.get("children", [])}
    vis = {}

    def axe_de(src):
        """The bone extending src and the frame axis it follows in T-pose: (node, "X", "PositiveX"), or None."""
        suite = [c for c in noeuds[src].get("children", []) if noeuds[c]["name"] not in torsion and c < len(joints)]
        if not suite:
            return None
        c = max(suite, key=lambda c: np.linalg.norm(pos_n[c] - pos_n[src]))
        d = pos_n[c] - pos_n[src]
        d = d / max(np.linalg.norm(d), 1e-9)
        a = int(np.abs(d).argmax())
        if abs(d[a]) < 0.999:
            return None
        return c, "XYZ"[a], ("Positive" if d[a] > 0 else "Negative") + "XYZ"[a]

    def contrainte(sorte, **kw):
        return {"VRMC_node_constraint": {"specVersion": "1.0", "constraint": {sorte: dict(kw, weight=float(kw.get("weight", 1.0)))}}}
    for n, t in torsion.items():
        if n not in noeud_de or t["source"] not in noeud_de:
            continue
        k, s = noeud_de[n], noeud_de[t["source"]]
        noeuds[k]["extras"] = {"torsion": {"sorte": t["sorte"], "source": t["source"], "part": t["part"]}}
        if t["sorte"] == "suit":
            a = axe_de(pere[k])
            if a and t["part"] > 0:
                noeuds[k]["extensions"] = contrainte("roll", source=s, rollAxis=a[1], weight=t["part"])
                contraintes += 1
        elif viseurs and s in pere and axe_de(s):
            # "retient" (holds back): the bone keeps a share of its source's roll. It leaves its source for an aimer
            # placed at the same spot, aiming at the next bone (the swing only, no roll), and takes back a share of the roll.
            a = axe_de(s)
            if s not in vis:
                noeuds.append({"name": t["source"] + "_viseur", "translation": list(noeuds[s]["translation"]),
                               "extensions": contrainte("aim", source=a[0], aimAxis=a[2])})
                vis[s] = len(noeuds) - 1
                pos_n.append(pos_n[s])
                noeuds[pere[s]]["children"].append(vis[s])
                contraintes += 1
            noeuds[s]["children"].remove(k)
            noeuds[vis[s]].setdefault("children", []).append(k)
            if t["part"] > 0:
                noeuds[k]["extensions"] = contrainte("roll", source=s, rollAxis=a[1], weight=t["part"])
                contraintes += 1
    n_joints = len(noeuds)
    racines = [k for k in range(n_joints) if not any(k in n.get("children", []) for n in noeuds)]
    ibm = np.tile(np.eye(4), (n_joints, 1, 1))
    ibm[:, :3, 3] = -np.array(pos_n)
    ibm_acc = B.acces(np.transpose(ibm, (0, 2, 1)).astype("<f4"), 5126, "MAT4")
    skin = {"joints": list(range(n_joints)), "inverseBindMatrices": ibm_acc, "skeleton": racines[0]}
    i_noeud_tete = None
    if tete:
        noeuds.append({"name": "tete", "mesh": i_tete, "skin": 0})
        i_noeud_tete = len(noeuds) - 1
    if i_autres is not None:
        noeuds.append({"name": "corps", "mesh": i_autres, "skin": 0})
    scene = racines + [k for k in (i_noeud_tete, len(noeuds) - 1 if i_autres is not None else None) if k is not None]
    # ---- VRM extension
    humain = {v: {"node": idx_joint[b]} for v, b in S.humain.items() if b in idx_joint}
    preset = {}
    if tete:
        for expr, m in choisis.items():
            preset[expr] = {"morphTargetBinds": [{"node": i_noeud_tete, "index": cibles.index(id(m)), "weight": 1.0}], "isBinary": expr.startswith("blink"),
                            "overrideBlink": "none", "overrideLookAt": "none", "overrideMouth": "none"}
    ext = {"specVersion": "1.0", "meta": {
        "name": nom, "version": "1.0", "authors": [auteur], "licenseUrl": "https://vrm.dev/licenses/1.0/", "avatarPermission": "onlyAuthor",
        "allowExcessivelyViolentUsage": False, "allowExcessivelySexualUsage": False, "commercialUsage": "personalNonProfit",
        "allowPoliticalOrReligiousUsage": False, "allowAntisocialOrHateUsage": False, "creditNotation": "required",
        "allowRedistribution": False, "modification": "prohibited"},
        "humanoid": {"humanBones": humain}, "expressions": {"preset": preset}}
    doc = {"asset": {"version": "2.0", "generator": "Chimera"}, "scene": 0, "scenes": [{"nodes": scene}], "nodes": noeuds,
           "meshes": meshes, "skins": [skin], "materials": materiaux, "accessors": B.acc, "bufferViews": B.vues,
           "buffers": [{"byteLength": 0}], "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}],
           "extensionsUsed": ["KHR_materials_unlit", "VRMC_materials_mtoon", "VRMC_vrm"], "extensions": {"VRMC_vrm": ext}}
    if images_doc:
        doc["images"], doc["textures"] = images_doc, textures
    if contraintes:
        doc["extensionsUsed"].append("VRMC_node_constraint")
    while len(B.o) % 4:
        B.o.append(0)
    doc["buffers"][0]["byteLength"] = len(B.o)
    j = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    j += b" " * ((4 - len(j) % 4) % 4)
    total = 12 + 8 + len(j) + 8 + len(B.o)
    tmp = sortie + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(j), 0x4E4F534A) + j)
        f.write(struct.pack("<II", len(B.o), 0x004E4942) + bytes(B.o))
    os.replace(tmp, sortie)
    return {"sommets": int(len(X)), "os": n_joints, "humanoides": len(humain), "contraintes": contraintes, "expressions": len(preset), "materiaux": len(materiaux), "octets": total}


def images_blender(P, dossier, plafond=2048):
    """PMX textures as PNG (in Blender): path written in P -> (bytes, mimetype, has holes). Maximum side length."""
    import tempfile
    import bpy
    out = {}
    tmp = tempfile.mkdtemp(prefix="vrm-", dir=os.environ.get("TMPDIR") or None)
    for k, t in enumerate(P["textures"]):
        ch = os.path.join(dossier, t.replace("\\", "/"))
        if not os.path.isfile(ch):
            continue
        im = bpy.data.images.load(ch, check_existing=False)
        w, h = im.size
        a = np.empty(w * h * im.channels, np.float32)
        im.pixels.foreach_get(a)
        decoupe = im.channels == 4 and float((a[3::4] < ALPHA_MASQUE).mean()) > PART_DECOUPE
        if max(w, h) > plafond:
            r = plafond / max(w, h)
            im.scale(int(w * r), int(h * r))
        png = os.path.join(tmp, "%d.png" % k)
        im.filepath_raw = png
        im.file_format = "PNG"
        im.save()
        bpy.data.images.remove(im)
        with open(png, "rb") as f:
            out[t] = (f.read(), "image/png", decoupe)
        os.remove(png)
    os.rmdir(tmp)
    return out


def lire(chemin):
    with open(chemin, "rb") as f:
        o = f.read()
    magique, version, _ = struct.unpack("<III", o[:12])
    n, ty = struct.unpack("<II", o[12:20])
    if magique != 0x46546C67 or version != 2 or ty != 0x4E4F534A:
        raise ValueError("not a GLB")
    doc = json.loads(o[20:20 + n].decode("utf-8"))
    return doc, o[28 + n:]


def valider(chemin):
    """Structure checks: returns the list of defects (empty if all is well)."""
    doc, b = lire(chemin)
    defauts = []
    ext = doc.get("extensions", {}).get("VRMC_vrm")
    if not ext:
        return ["VRMC_vrm extension missing"]
    if ext.get("specVersion") != "1.0":
        defauts.append("specVersion %s" % ext.get("specVersion"))
    hb = ext["humanoid"]["humanBones"]
    for o in OBLIGATOIRES:
        if o not in hb:
            defauts.append("required bone missing: %s" % o)
    noeuds = doc["nodes"]
    parent = {}
    for i, n in enumerate(noeuds):
        for c in n.get("children", []):
            if c in parent:
                defauts.append("node %d has two parents" % c)
            parent[c] = i
    if any(not (0 <= h["node"] < len(noeuds)) for h in hb.values()):
        defauts.append("humanoid bone outside the nodes")
    n_chaine = {"spine": "hips", "head": "spine", "leftUpperArm": "spine", "rightUpperArm": "spine", "leftLowerArm": "leftUpperArm", "rightLowerArm": "rightUpperArm",
                "leftHand": "leftLowerArm", "rightHand": "rightLowerArm", "leftUpperLeg": "hips", "rightUpperLeg": "hips", "leftLowerLeg": "leftUpperLeg", "rightLowerLeg": "rightUpperLeg",
                "leftFoot": "leftLowerLeg", "rightFoot": "rightLowerLeg"}

    def descend(a, b_):
        i, vus = hb[a]["node"], set()
        cible = hb[b_]["node"]
        while i in parent and i not in vus:
            vus.add(i)
            i = parent[i]
            if i == cible:
                return True
        return False
    for a, b_ in n_chaine.items():
        if a in hb and b_ in hb and not descend(a, b_):
            defauts.append("%s does not descend from %s" % (a, b_))
    for i, n in enumerate(noeuds):
        if any(k in n for k in ("rotation", "scale", "matrix")):
            defauts.append("node %d with rotation or scale (T-pose expected)" % i)
    acc = doc["accessors"]
    vues = doc["bufferViews"]
    for a in acc:
        v = vues[a["bufferView"]]
        if v.get("byteOffset", 0) + v["byteLength"] > len(b):
            defauts.append("accessor outside the buffer")
    n_morph = {}
    for mi, m in enumerate(doc["meshes"]):
        for pr in m["primitives"]:
            at = pr["attributes"]
            n = acc[at["POSITION"]]["count"]
            for k in ("NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"):
                if acc[at[k]]["count"] != n:
                    defauts.append("%s: different count" % k)
            jt = acc[at["JOINTS_0"]]
            d = np.frombuffer(b, "<u2", jt["count"] * 4, vues[jt["bufferView"]].get("byteOffset", 0))
            if d.size and int(d.max()) >= len(doc["skins"][0]["joints"]):
                defauts.append("joint index outside the skeleton")
            wt = acc[at["WEIGHTS_0"]]
            w = np.frombuffer(b, "<f4", wt["count"] * 4, vues[wt["bufferView"]].get("byteOffset", 0)).reshape(-1, 4).sum(1)
            if np.abs(w - 1).max() > 1e-3:
                defauts.append("weights not summing to 1 (deviation %.4f)" % float(np.abs(w - 1).max()))
            if pr.get("targets"):
                n_morph[mi] = len(pr["targets"])
    for nom, e in ext.get("expressions", {}).get("preset", {}).items():
        for mb in e.get("morphTargetBinds", []):
            ni = doc["nodes"][mb["node"]].get("mesh")
            if ni is None or mb["index"] >= n_morph.get(ni, 0):
                defauts.append("expression %s: target not found" % nom)
    for m in doc["materials"]:
        if "VRMC_materials_mtoon" in m.get("extensions", {}) and "KHR_materials_unlit" not in m.get("extensions", {}):
            defauts.append("material %s without unlit fallback" % m.get("name"))
    return defauts
