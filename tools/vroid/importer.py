"""VRoid importer: a VRoid character (.vrm, VRM 0.x from VRoid Studio or VRM 1.0 from VRoid Hub) written as PMX in
the common language.

    python3 -I importer.py <file.vrm>... [--output <folder>]
        writes <folder>/<Name>/<Name>.pmx and its textures (default: the `mmd_extrait` source of the "vroid" game)
    python3 -I importer.py show <file.vrm>
        says what the file contains, without writing anything

What the conversion does (tables in tools/games/vroid/game.json):
  - frame: metres -> MMD units, the character's left at +x, front at -z;
  - bones: the humanoid bones that the file declares take their MMD name (`os_mmd`), the others keep their own;
    全ての親 and センター are added, 上半身 and 下半身 become siblings under センター as in MMD;
  - pose: VRoid outputs arms horizontal (T-pose), Chimera works with arms down (A-pose): the arms
    are rotated by `pose.bras` degrees around the shoulder, through the model's weights;
  - neck: the body skin, in one piece in VRoid, is cut at the bottom of the neck (`pose.cou`): the core takes as
    head the skin welded to the face;
  - merged bones (`os_fondus`): the chest bones, which Chimera's body does not have, give their weight back to their
    parent: otherwise the core takes what they hold for a free flap and leaves the skin underneath;
  - materials: translated by `materiaux` (颜 face skin, 肌 body skin, 目 iris, 髪 hair…), the original
    name kept as the English name; the primitives of the same material are merged;
  - expressions: the face shapes (Fcl_*) become vertex morphs, with the MMD names of `morphes_mmd`
    when there is one, otherwise under their own name;
  - textures: the base colour of each material, taken from the file as it is.
  - erased faces: a material cut out by its texture (alphaMode MASK) loses the faces that the texture makes
    entirely transparent (`invisibles`): the legs of shorts, the sleeves of a tank top.
Spring bones (hair, skirt) are not carried over: PMX has no physics.

VRM 0.x and VRM 1.0 (VRoid Studio 2.x, VRoid Hub) differ in what the file declares, not in the character:
  - extension: "VRM" (0.x) or "VRMC_vrm" (1.0); humanoid bones as a list (0.x) or a dict (1.0), where the thumb
    is renamed one segment down (1.0 thumbMetacarpal = 0.x thumbProximal...): read in the 0.x vocabulary;
  - frame: 0.x faces -z, 1.0 faces +z; the frame is measured on the hands and toes either way;
  - skeleton: 0.x bakes the bones without rotation; 1.0 keeps a rotation (and sometimes a scale) per bone, and a
    re-exported model (UniVRM) places its meshes on their own nodes. Bones are placed with their full transform,
    and the vertices with the skin as the file shows it (joint pose x inverse bind matrix);
  - names: same VRoid material names (with " (Instance)"), shapes without the mesh prefix ("Fcl_MTH_A");
  - meta: title/author/licenseName (0.x), name/authors/licenseUrl and usage fields (1.0): both reported.
A VRM that does not carry VRoid Studio's material names (face skin "..._Face_00_SKIN") is refused: the face, hair
and outfit are found through those names.
"""
import json
import os
import re
import struct
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ICI, ".."), os.path.join(ICI, "..", "atlas"), os.path.join(ICI, "..", "head")]
import pmx as PMX             # noqa: E402
import pmx_write as pmx_ecrire             # noqa: E402

ECH = 100.0 / 8.0             # metres -> MMD units
TYPES = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
NB = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
EXT = {"image/png": ".png", "image/jpeg": ".jpg"}
SEUIL_MORPHE = 1e-6           # metres: below this, a vertex does not move
SEUIL_LIAISON = 1e-5          # below this, the skin of a mesh is already in the pose of the skeleton
# VRM 1.0 humanoid bone -> VRM 0.x (the 1.0 thumb is named one segment down; the other bones keep their name)
HUMAIN_1_VERS_0 = {"%sThumb%s" % (c, a): "%sThumb%s" % (c, b) for c in ("left", "right")
                   for a, b in (("Metacarpal", "Proximal"), ("Proximal", "Intermediate"))}
COUPE = 0.002                 # MMD units: gap separating the neck skin from the body skin


def description():
    with open(os.path.join(ICI, "..", "games", "vroid", "game.json"), encoding="utf-8") as f:
        return json.load(f)


def lire_glb(chemin):
    with open(chemin, "rb") as f:
        d = f.read()
    if d[:4] != b"glTF":
        raise ValueError("%s is not a binary glTF file (.vrm, .glb)" % chemin)
    n = struct.unpack_from("<I", d, 12)[0]
    return json.loads(d[20:20 + n]), d[20 + n + 8:]


def _vue(j, b, vue, decalage, dt, k, n):
    v = j["bufferViews"][vue]
    deb = v.get("byteOffset", 0) + decalage
    pas = v.get("byteStride", 0)
    if pas and pas != dt.itemsize * k:
        brut = np.frombuffer(b, np.uint8, pas * n, deb).reshape(n, pas)[:, :dt.itemsize * k]
        return np.frombuffer(brut.tobytes(), dt).reshape(n, k)
    return np.frombuffer(b, dt, n * k, deb).reshape(n, k)


def acc(j, b, i):
    a = j["accessors"][i]
    dt = np.dtype(TYPES[a["componentType"]])
    k = NB[a["type"]]
    if "bufferView" in a:
        x = _vue(j, b, a["bufferView"], a.get("byteOffset", 0), dt, k, a["count"])
    else:
        x = np.zeros((a["count"], k), dt)
    sp = a.get("sparse")
    if sp:          # sparse form (a re-exported model's shapes): the listed values replace the base ones
        n = sp["count"]
        ix, va = sp["indices"], sp["values"]
        ou = _vue(j, b, ix["bufferView"], ix.get("byteOffset", 0), np.dtype(TYPES[ix["componentType"]]), 1, n)[:, 0]
        x = x.copy()
        x[ou.astype(np.int64)] = _vue(j, b, va["bufferView"], va.get("byteOffset", 0), dt, k, n)
    return x


SEMIS = 12      # points per side of the sampling grid that reads the transparency of a face


def lire_alpha(chemin):
    """Transparency of an image (h, w), 0..1, first row at the top; None if it has none or
    if OpenImageIO is missing (then nothing is removed)."""
    try:
        import OpenImageIO as OIIO
    except ImportError:
        return None
    r = OIIO.ImageInput.open(chemin)
    if not r:
        return None
    im = r.read_image(format=OIIO.FLOAT)
    r.close()
    if im is None or im.ndim != 3 or im.shape[2] < 4:
        return None
    return np.ascontiguousarray(im[:, :, 3])


def invisibles(a, uv, F, coupe=0.5):
    """Per face: the texture (transparency `a`) is below the cutoff everywhere on the face. In VRoid a
    garment is a whole template (legs, sleeves) whose texture draws only part of it: a pair of
    shorts is a pair of trousers with transparent legs. These faces are never seen; if kept, they
    would turn shorts into full trousers and hide the skin that must be seen."""
    h, w = a.shape
    B = np.array([(i, k, SEMIS - i - k) for i in range(SEMIS + 1) for k in range(SEMIS + 1 - i)], float) / SEMIS
    inv = np.ones(len(F), bool)
    for d in range(0, len(F), 4000):
        P = np.einsum("nk,fkc->fnc", B, uv[F[d:d + 4000]])
        x = (np.mod(P[..., 0], 1.0) * w).astype(int).clip(0, w - 1)
        y = (np.mod(P[..., 1], 1.0) * h).astype(int).clip(0, h - 1)
        inv[d:d + 4000] = (a[y, x] < coupe).all(1)
    return inv


def nom_court(nom):
    """"Face.M_F00_000_00_Fcl_MTH_A" -> "Fcl_MTH_A"."""
    m = re.search(r"(Fcl_.*)\Z", nom)
    return m.group(1) if m else nom.split(".")[-1]


def traduire(nom, table):
    """(common name, pattern number) of the first pattern that matches, otherwise (None, None)."""
    for k, (motif, commun) in enumerate(table):
        if re.search(motif, nom):
            return commun, k
    return None, None


class Refus(ValueError):
    """The file is readable but is not a character this importer can convert (said as is to the user)."""


def matrice(n):
    """Local transform of a glTF node (4x4)."""
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    x, y, z, w = n.get("rotation", (0.0, 0.0, 0.0, 1.0))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * np.array(n.get("scale", (1.0, 1.0, 1.0)), float)
    M[:3, 3] = n.get("translation", (0.0, 0.0, 0.0))
    return M


def meta_lisible(vrm, version):
    """(title, author, licence) declared by the file, in words; licence as the file states it."""
    m = vrm.get("meta") or {}
    if version == 0:
        return (m.get("title") or None, m.get("author") or "?",
                ("%s %s" % (m.get("licenseName") or "?", m.get("otherLicenseUrl") or "")).strip())
    usages = ["avatar: %s" % m.get("avatarPermission", "?"), "commercial: %s" % m.get("commercialUsage", "?"),
              "modification: %s" % m.get("modification", "?"),
              "redistribution: %s" % ("allowed" if m.get("allowRedistribution") else "not allowed"),
              "credit: %s" % m.get("creditNotation", "?")]
    autre = m.get("otherLicenseUrl")
    lic = "%s (%s)%s" % (m.get("licenseUrl") or "?", ", ".join(usages), (" + " + autre) if autre else "")
    return m.get("name") or None, ", ".join(m.get("authors") or []) or "?", lic


def rotation_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


class Modele:
    """The file as read: nodes, humanoid bones, orientation of the frame."""

    def __init__(self, chemin):
        self.chemin = chemin
        self.j, self.b = lire_glb(chemin)
        j = self.j
        ext = j.get("extensions", {})
        if "VRM" in ext:
            self.version, self.vrm = 0, ext["VRM"]
            self.humain = {h["bone"]: h["node"] for h in self.vrm["humanoid"]["humanBones"]}
        elif "VRMC_vrm" in ext:
            self.version, self.vrm = 1, ext["VRMC_vrm"]
            spec = str(self.vrm.get("specVersion", ""))
            if not spec.startswith("1."):
                raise Refus("%s declares VRM version \"%s\": only VRM 0.x and 1.x are read" % (chemin, spec))
            os1 = (self.vrm.get("humanoid") or {}).get("humanBones") or {}
            self.humain = {HUMAIN_1_VERS_0.get(r, r): h["node"] for r, h in os1.items()}
        else:
            raise Refus("%s has no VRM extension" % chemin)
        self.titre, self.auteur, self.licence = meta_lisible(self.vrm, self.version)
        self.noeuds = j["nodes"]
        self.parent = {c: i for i, n in enumerate(self.noeuds) for c in n.get("children", [])}
        self._monde = {}
        for r in ("hips", "spine", "head", "leftUpperArm", "leftLowerArm", "rightUpperArm", "leftHand", "leftFoot"):
            if r not in self.humain:
                raise Refus("missing humanoid bone: %s" % r)
        D = description()
        if not any(traduire(m.get("name", ""), D["materiaux"])[0] == "颜" for m in j.get("materials", [])):
            raise Refus("%s is not a VRoid character: no material has a VRoid Studio face name (\"..._Face_00_SKIN\"), "
                        "and the face, hair and outfit are found through those names" % chemin)
        g, d = self.monde0(self.humain["leftHand"]), self.monde0(self.humain["rightHand"])
        sx = 1.0 if g[0] > d[0] else -1.0
        orteil = self.humain.get("leftToes")
        pied = self.monde0(self.humain["leftFoot"])
        if orteil is not None:
            sz = 1.0 if self.monde0(orteil)[2] < pied[2] else -1.0
        else:                 # without toes: a right-handed frame turned by a half turn, or mirrored
            sz = sx
        self.S = np.array([sx, 1.0, sz]) * ECH
        self.retourne = sx * sz < 0

    def matrice_monde(self, i):
        """World transform of node i (4x4), in the frame of the file."""
        if i not in self._monde:
            p = self.parent.get(i)
            loc = matrice(self.noeuds[i])
            self._monde[i] = loc if p is None else self.matrice_monde(p) @ loc
        return self._monde[i]

    def monde0(self, i):
        return self.matrice_monde(i)[:3, 3].copy()

    def liaison(self, noeud, peau):
        """Per joint of the skin (or None for an unskinned mesh), the matrix that takes a vertex of the file to the
        pose of the skeleton: joint x inverse bind matrix (glTF skinning), or the node's own transform. None when it
        is the identity everywhere (the VRoid Studio 0.x export): the vertices are then taken as they are."""
        if peau is None:
            B = self.matrice_monde(noeud)[None]
        elif "inverseBindMatrices" not in peau:     # never seen in a VRM: the vertices are taken as they are
            return None
        else:
            n = len(peau["joints"])
            ibm = acc(self.j, self.b, peau["inverseBindMatrices"]).astype(float).reshape(n, 4, 4).transpose(0, 2, 1)
            B = np.array([self.matrice_monde(k) for k in peau["joints"]]) @ ibm
        return None if np.abs(B - np.eye(4)).max() < SEUIL_LIAISON else B

    def monde(self, i):
        return self.monde0(i) * self.S

    def maillages(self):
        """(node, mesh, skin) of each mesh placed on a node."""
        return [(i, self.j["meshes"][n["mesh"]], self.j["skins"][n["skin"]] if "skin" in n else None)
                for i, n in enumerate(self.noeuds) if "mesh" in n]


def squelette(M, D):
    """The bones of the PMX: (list of bones, index of each node). Parents first."""
    joints = set()
    for _, _, peau in M.maillages():
        if peau:
            joints |= set(peau["joints"])
    for n in list(joints):    # the ancestors too (root)
        p = M.parent.get(n)
        while p is not None:
            joints.add(p)
            p = M.parent.get(p)
    avec_maillage = {i for i, _, _ in M.maillages()}
    joints -= avec_maillage
    ordre = []

    def visite(i):
        if i in joints and i not in ordre:
            if M.parent.get(i) in joints:
                visite(M.parent[i])
            ordre.append(i)
    for i in sorted(joints):
        visite(i)
    role = {n: r for r, n in M.humain.items()}
    drap = PMX.OS_ROTATION | PMX.OS_VISIBLE | PMX.OS_ACTIF

    def os_(nom, nom_en, pos, parent, deplace=False):
        return {"nom": nom, "nom_en": nom_en, "pos": tuple(float(x) for x in pos), "parent": parent, "couche": 0,
                "drapeaux": drap | (PMX.OS_DEPLACEMENT if deplace else 0), "queue_os": None, "queue_dec": (0.0, 0.0, 0.0)}
    hanches = M.monde(M.humain["hips"])
    oss = [os_("全ての親", "root", (0, 0, 0), -1, True), os_("センター", "center", hanches, 0, True)]
    rang = {}
    for n in ordre:
        nom = M.noeuds[n].get("name", "os%d" % n)
        r = role.get(n)
        if M.parent.get(n) is None:           # the root of the file is our 全ての親
            rang[n] = 0
            continue
        p = rang[M.parent[n]]
        if any(re.search(x, nom) for x in D.get("os_fondus", ())):    # bone that Chimera's body does not have: its
            rang[n] = p                                               # weight goes back to its parent
            continue
        rang[n] = len(oss)
        if r in ("hips", "spine"):
            p = 1
        oss.append(os_(D["os_mmd"].get(r, nom), r or nom, M.monde(n), p))
    return oss, rang


# bones whose tail points to a given child (otherwise: the only child, or a tip extending the parent)
SUITE = {"spine": "chest", "chest": "upperChest", "upperChest": "neck", "neck": "head", "leftHand": "leftMiddleProximal",
         "rightHand": "rightMiddleProximal"}
BOUT = {"全ての親": (0.0, 0.0, -1.0), "センター": (0.0, -1.0, 0.0), "hips": (0.0, -1.0, 0.0), "head": (0.0, 1.5, 0.0),
        "leftEye": (0.0, 0.0, -0.5), "rightEye": (0.0, 0.0, -0.5)}


def queues(oss, enfants):
    """The tail of each bone (the core measures along the bones): the next bone of the chain, otherwise a tip."""
    rang = {o["nom_en"]: i for i, o in enumerate(oss)}
    for i, o in enumerate(oss):
        e = enfants.get(i, [])
        suite = rang.get(SUITE.get(o["nom_en"]))
        if o["nom_en"] in BOUT or o["nom"] in BOUT:
            o["queue_dec"] = BOUT.get(o["nom_en"]) or BOUT[o["nom"]]
            continue
        if suite is None and len(e) == 1:
            suite = e[0]
        if suite is not None and np.linalg.norm(np.array(oss[suite]["pos"]) - o["pos"]) > 1e-4:
            o["queue_os"], o["queue_dec"] = suite, None
            o["drapeaux"] |= PMX.OS_QUEUE_INDEX
        elif o["parent"] >= 0:
            d = np.array(o["pos"]) - oss[o["parent"]]["pos"]
            o["queue_dec"] = tuple(float(x) for x in 0.5 * d)


def convertir(chemin, dossier, D=None, journal=print, garder=()):
    """Writes <folder>/<Name>.pmx and its textures. Returns a short summary. `garder`: common names (肌…) whose
    faces stay even when their texture erases them (the VRoid base keeps all the skin and repaints it)."""
    D = D or description()
    M = Modele(chemin)
    j, b = M.j, M.b
    oss, rang = squelette(M, D)
    noms_os = [o["nom"] for o in oss]
    # --- vertices: one block per position accessor (the primitives of a VRoid mesh share it)
    blocs, pos, nor, uv, jo, po = {}, [], [], [], [], []
    faces_mat = {}            # glTF material -> [faces]
    morphes = {}              # name -> (indices, offsets); a shape split over several primitives is put back together
    n_som = 0
    for noeud, maillage, peau in M.maillages():
        vers_os = np.array([rang[n] for n in peau["joints"]]) if peau else None
        B = M.liaison(noeud, peau)
        vus, lier = set(), {}
        for pr in maillage["primitives"]:
            at = pr["attributes"]
            cle = at["POSITION"]
            if cle not in blocs:
                p = acc(j, b, cle).astype(float)
                n_ = acc(j, b, at["NORMAL"]).astype(float) if "NORMAL" in at else np.tile([0.0, 1.0, 0.0], (len(p), 1))
                blocs[cle] = n_som
                if vers_os is not None and "JOINTS_0" in at:
                    w = acc(j, b, at["WEIGHTS_0"]).astype(float)
                    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
                    jt = acc(j, b, at["JOINTS_0"]).astype(int)
                    o = vers_os[jt]
                    o[w == 0] = 0
                else:
                    w = np.tile([1.0, 0.0, 0.0, 0.0], (len(p), 1))
                    jt = np.zeros((len(p), 4), int)
                    o = np.zeros((len(p), 4), int)
                if B is not None:       # the skin as the file shows it: blend of the joints' matrices
                    L = np.einsum("nk,nkij->nij", w, B[np.minimum(jt, len(B) - 1)])
                    p = np.einsum("nij,nj->ni", L[:, :3, :3], p) + L[:, :3, 3]
                    n_ = np.einsum("nij,nj->ni", L[:, :3, :3], n_)
                    n_ /= np.maximum(np.linalg.norm(n_, axis=1, keepdims=True), 1e-12)
                    lier[cle] = L[:, :3, :3]
                pos.append(p * M.S)
                nor.append(n_ * np.sign(M.S))
                uv.append(acc(j, b, at["TEXCOORD_0"]).astype(float) if "TEXCOORD_0" in at else np.zeros((len(p), 2)))
                jo.append(o)
                po.append(w)
                n_som += len(p)
            dec = blocs[cle]
            f = acc(j, b, pr["indices"]).reshape(-1, 3).astype(np.int64) + dec
            faces_mat.setdefault(pr["material"], []).append(f[:, ::-1] if M.retourne else f)
            noms = pr.get("extras", {}).get("targetNames") or maillage.get("extras", {}).get("targetNames") or []
            for k, cible in enumerate(pr.get("targets", [])):
                if (cle, k) in vus or "POSITION" not in cible:
                    continue
                vus.add((cle, k))
                d = acc(j, b, cible["POSITION"]).astype(float)
                if cle in lier:
                    d = np.einsum("nij,nj->ni", lier[cle], d)
                ou = np.where(np.abs(d).max(1) > SEUIL_MORPHE)[0]
                if len(ou):
                    nm = noms[k] if k < len(noms) else "forme%d" % k
                    i0, d0 = morphes.get(nm, (np.zeros(0, np.int64), np.zeros((0, 3))))
                    morphes[nm] = (np.concatenate([i0, ou + dec]), np.concatenate([d0, d[ou] * M.S]))
    pos, nor, uv, jo, po = [np.concatenate(x) for x in (pos, nor, uv, jo, po)]
    # --- A-pose: the arms rotated around the shoulder, each vertex according to its weight on the arm
    angle = float(D["pose"]["bras"])
    enfants = {}
    for i, o in enumerate(oss):
        enfants.setdefault(o["parent"], []).append(i)
    lin = np.tile(np.eye(3), (len(pos), 1, 1))     # linear part of the transformation of each vertex
    for cote, signe in (("left", -1.0), ("right", 1.0)):
        i0 = noms_os.index(D["os_mmd"][cote + "UpperArm"])
        sous, pile = [], [i0]
        while pile:
            k = pile.pop()
            sous.append(k)
            pile += enfants.get(k, [])
        cote_x = 1.0 if oss[noms_os.index(D["os_mmd"][cote + "LowerArm"])]["pos"][0] > oss[i0]["pos"][0] else -1.0
        R = rotation_z(signe * angle * (1.0 if (cote == "left") == (cote_x > 0) else -1.0))
        c = np.array(oss[i0]["pos"])
        w = (po * np.isin(jo, sous)).sum(1)
        pos += w[:, None] * ((pos - c) @ R.T - (pos - c))
        lin += w[:, None, None] * (R - np.eye(3))
        for k in sous:
            oss[k]["pos"] = tuple(float(x) for x in c + R @ (np.array(oss[k]["pos"]) - c))
    queues(oss, enfants)
    nor = np.einsum("nij,nj->ni", lin, nor)
    nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)
    # --- neck: in VRoid the body skin is in one piece, welded to the face; the core expects the skin welded to the
    # face to stop at the bottom of the neck (the whole head: face, ears, neck). So the body skin is cut at the bottom of the
    # neck, along the edges: the faces below get their own vertices, lowered by a hair.
    y_cou, y_tete = oss[noms_os.index(D["os_mmd"]["neck"])]["pos"][1], oss[noms_os.index(D["os_mmd"]["head"])]["pos"][1]
    y0 = y_cou + float(D["pose"]["cou"]) * (y_tete - y_cou)
    coupees = 0
    for i, lf in faces_mat.items():
        if traduire(j["materials"][i].get("name", ""), D["materiaux"])[0] != "肌":
            continue
        f = np.concatenate(lf)
        haut = (pos[f][:, :, 1] >= y0).all(1)
        if not haut.any() or haut.all():
            continue
        cle = lambda P: [bytes(x) for x in np.round(P, 5).astype("<f4")]     # noqa: E731
        du_haut = set(cle(pos[np.unique(f[haut])]))
        bas = np.unique(f[~haut])
        bord = bas[[k in du_haut for k in cle(pos[bas])]]
        # edge brought back into the plane of the cut (it follows the edges, in a sawtooth): the core reads a flat ring
        du_bord = set(cle(pos[bord]))
        tous = np.unique(f)
        pos[tous[[k in du_bord for k in cle(pos[tous])]], 1] = y0
        nouveau = np.arange(len(bord)) + len(pos)
        pos = np.concatenate([pos, pos[bord] - [0.0, COUPE, 0.0]])
        nor, uv, jo, po, lin = [np.concatenate([x, x[bord]]) for x in (nor, uv, jo, po, lin)]
        vers = np.arange(len(pos))
        vers[bord] = nouveau
        f[~haut] = vers[f[~haut]]
        faces_mat[i] = [f]
        coupees += len(bord)
    # --- materials, in the order of the table (skin, face, eyes, hair, outfit)
    os.makedirs(dossier, exist_ok=True)
    textures, sorties = [], {}

    def texture(mat):
        t = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if t is None:
            return -1
        src = j["textures"][t["index"]]["source"]
        if src not in sorties:
            im = j["images"][src]
            v = j["bufferViews"][im["bufferView"]]
            nom = re.sub(r"[^\w.-]", "_", im.get("name") or "image%d" % src) + EXT.get(im.get("mimeType"), ".png")
            if nom in textures:         # two images of the same name: the second one must not overwrite the first
                nom = "%s_%d%s" % (os.path.splitext(nom)[0], src, os.path.splitext(nom)[1])
            with open(os.path.join(dossier, nom), "wb") as f:
                f.write(b[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]])
            sorties[src] = len(textures)
            textures.append(nom)
        return sorties[src]
    lignes, pris = [], {}
    for i, f in faces_mat.items():
        mat = j["materials"][i]
        commun, k = traduire(mat.get("name", ""), D["materiaux"])
        if commun is None:
            commun, k = mat.get("name", "materiau%d" % i), len(D["materiaux"])
        lignes.append((k, i, commun, mat, np.concatenate(f)))
    lignes.sort(key=lambda l: (l[0], l[1]))
    mats, faces, invisible = [], [], 0
    for k, i, commun, mat, f in lignes:
        if mat.get("alphaMode") == "MASK" and len(f) and commun not in garder:   # cut out by the texture: what it erases does not exist
            t = texture(mat)
            alpha = lire_alpha(os.path.join(dossier, textures[t])) if t >= 0 else None
            if alpha is not None:
                inv = invisibles(alpha, uv, f, float(mat.get("alphaCutoff", 0.5)))
                invisible += int(inv.sum())
                f = f[~inv]
                if not len(f):
                    continue
        pris[commun] = pris.get(commun, 0) + 1
        nom = commun if pris[commun] == 1 else "%s%d" % (commun, pris[commun])
        couleur = tuple(float(x) for x in mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1, 1, 1, 1]))
        trait = mat.get("alphaMode", "OPAQUE") != "BLEND"       # no outline on a transparent layer
        mats.append({"nom": nom, "nom_en": mat.get("name", ""), "diffus": couleur, "speculaire": (0.0, 0.0, 0.0), "brillance": 5.0,
                     "ambiant": (0.5, 0.5, 0.5), "drapeaux": 0x0E | (0x01 if mat.get("doubleSided") else 0) | (0x10 if trait else 0),
                     "contour_couleur": (0.2, 0.14, 0.12, 1.0), "contour_taille": 0.6, "texture": texture(mat), "sphere": -1,
                     "sphere_mode": 0, "toon_partage": 1, "toon": 0, "memo": "", "n_indices": 3 * len(f)})
        faces.append(f)
    faces = np.concatenate(faces)
    util = np.unique(faces)
    nv = -np.ones(len(pos), np.int64)
    nv[util] = np.arange(len(util))
    faces = nv[faces]
    n = len(util)
    som = {"pos": [tuple(p) for p in pos[util].tolist()], "nor": [tuple(p) for p in nor[util].tolist()],
           "uv": [tuple(p) for p in uv[util].tolist()], "uv_add": [[] for _ in range(n)], "type": [2] * n,
           "os": [tuple(int(x) for x in r) for r in jo[util]], "poids": [tuple(float(x) for x in r) for r in po[util]],
           "sdef": [None] * n, "contour": [1.0] * n}
    # --- morphs
    panneaux = D["panneaux"]
    out_m, sans_nom = [], []
    for nom, (ou, d) in morphes.items():
        garde = nv[ou] >= 0
        if not garde.any():
            continue
        ou, d = ou[garde], np.einsum("nij,nj->ni", lin[ou[garde]], d[garde])
        court = nom_court(nom)
        mmd = D["morphes_mmd"].get(court)
        if mmd is None:
            sans_nom.append(court)
        pan = next((p for motif, p in panneaux if re.search(motif, court)), 4)
        out_m.append({"nom": mmd or court, "nom_en": court, "panneau": pan, "type": 1,
                      "decalages": [(int(a), tuple(float(x) for x in v)) for a, v in zip(nv[ou], d)]})
    nom = os.path.splitext(os.path.basename(chemin))[0]
    sortie = os.path.join(dossier, nom + ".pmx")
    pmx_ecrire.ecrire({"version": 2.0, "encodage": "UTF-16", "nom": M.titre or nom, "nom_en": nom,
                       "commentaire": "Converted from a VRM %s file by tools/vroid/importer.py. Author: %s. Declared licence: %s"
                                      % ("0.x" if M.version == 0 else "1.0", M.auteur, M.licence),
                       "commentaire_en": "", "n_uv_add": 0, "sommets": som, "faces": [tuple(int(x) for x in f) for f in faces],
                       "textures": textures, "materiaux": mats, "os": oss, "morphes": out_m,
                       "cadres": [{"nom": "Root", "nom_en": "Root", "special": 1, "elements": [(0, 0)]},
                                  {"nom": "表情", "nom_en": "Exp", "special": 1, "elements": [(1, k) for k in range(len(out_m))]}],
                       "rigides": [], "joints": []}, sortie)
    bilan = {"pmx": sortie, "sommets": n, "faces": len(faces), "os": len(oss),
             "os_mmd": sum(o["nom"] != o["nom_en"] for o in oss), "materiaux": [m["nom"] for m in mats],
             "morphes": len(out_m), "morphes_mmd": len(out_m) - len(sans_nom), "textures": len(textures),
             "taille_cm": round(float(pos[util][:, 1].max() - pos[util][:, 1].min()) * 8, 1),
             "vrm": "0.x" if M.version == 0 else "1.0", "titre": M.titre, "auteur": M.auteur, "licence": M.licence}
    journal("wrote %s: %d vertices, %d faces, %d bones (%d with MMD names), %d morphs (%d with MMD names), %d textures, %.0f cm"
            % (sortie, n, len(faces), bilan["os"], bilan["os_mmd"], bilan["morphes"], bilan["morphes_mmd"], len(textures), bilan["taille_cm"]))
    journal("materials: " + " ".join(bilan["materiaux"]))
    journal("VRM %s, author %s, declared licence: %s" % (bilan["vrm"], M.auteur, M.licence))
    if invisible:
        bilan["faces_invisibles"] = invisible
        journal("%d faces erased by the transparency of their texture, removed" % invisible)
    return bilan


def voir(chemin):
    M = Modele(chemin)
    D = description()
    print("%s (VRM %s): \"%s\", author %s, licence %s" % (os.path.basename(chemin), "0.x" if M.version == 0 else "1.0",
                                                         M.titre, M.auteur, M.licence))
    print("humanoid bones: %d, of which without an MMD name: %s" % (len(M.humain), ", ".join(sorted(set(M.humain) - set(D["os_mmd"]))) or "none"))
    for _, m, _ in M.maillages():
        print("mesh %s: %d primitives" % (m.get("name"), len(m["primitives"])))
        for i in sorted({p["material"] for p in m["primitives"]}):
            nom = M.j["materials"][i].get("name", "")
            print("    %-44s -> %s" % (nom, traduire(nom, D["materiaux"])[0] or "(name kept: outfit)"))
        noms = m["primitives"][0].get("extras", {}).get("targetNames") or m.get("extras", {}).get("targetNames") or []
        if noms:
            court = [nom_court(x) for x in noms]
            print("    shapes: %d, of which with MMD names: %s" % (len(court), " ".join("%s=%s" % (c, D["morphes_mmd"][c]) for c in court if c in D["morphes_mmd"])))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "show":
        for c in argv[1:]:
            voir(c)
        return 0
    sortie = None
    if "--output" in argv:
        k = argv.index("--output")
        sortie = argv[k + 1]
        del argv[k:k + 2]
    if sortie is None:
        import games as jeux
        sortie = jeux.charger("vroid").chemin("mmd_extrait")
    for c in argv:
        nom = os.path.splitext(os.path.basename(c))[0]
        try:
            convertir(c, os.path.join(sortie, nom))
        except Refus as e:      # said as is, on one line that the workshop relays (tools/workshop/inputs.py)
            print("refused: %s" % str(e).replace(c, "the file"), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
