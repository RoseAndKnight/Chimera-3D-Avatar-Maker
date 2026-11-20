"""VRoid importer: a VRoid character (.vrm, VRM 0.x from VRoid Studio or VRM 1.0 from VRoid Hub) written as PMX in
the common language.

    python3 -I importer.py <file.vrm>... [--output <folder>] [--no-perfect-sync | --perfect-sync-donor female|male]
        writes <folder>/<Name>/<Name>.pmx and its textures (default: the `mmd_extracted` source of the "vroid" game),
        and perfect_sync.json (the Perfect Sync report)
    python3 -I importer.py show <file.vrm>
        says what the file contains, without writing anything

What the conversion does (tables in tools/games/vroid/game.json):
  - frame: metres -> MMD units, the character's left at +x, front at -z;
  - bones: the humanoid bones that the file declares take their MMD name (`os_mmd`), the others keep their own;
    全ての親 and センター are added, 上半身 and 下半身 become siblings under センター as in MMD;
  - pose: VRoid outputs arms horizontal (T-pose), Chimera works with arms down (A-pose): the arms
    are rotated by `pose.bras` degrees around the shoulder, through the model's weights;
  - neck: the body skin, in one piece in VRoid, is cut at the bottom of the neck (`pose.neck`): the core takes as
    head the skin welded to the face;
  - merged bones (`os_fondus`): the chest bones, which Chimera's body does not have, give their weight back to their
    parent: otherwise the core takes what they hold for a free flap and leaves the skin underneath;
  - materials: translated by `materials` (颜 face skin, 肌 body skin, 目 iris, 髪 hair…), the original
    name kept as the English name; the primitives of the same material are merged;
  - expressions: the face shapes (Fcl_*) become vertex morphs, with the MMD names of `morphs_mmd`
    when there is one, otherwise under their own name; the face also receives the 52 Perfect Sync (ARKit) shapes
    and the extra shapes of Hinzka's free models, under their own names (perfect_sync.py: matched by material and
    UV, eyelids and lips calibrated, checked);
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "atlas"), os.path.join(HERE, "..", "head")]
import pmx as PMX             # noqa: E402
import pmx_write as pmx_write             # noqa: E402

SAMP = 100.0 / 8.0             # metres -> MMD units
TYPES = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
NB = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
EXT = {"image/png": ".png", "image/jpeg": ".jpg"}
THRESHOLD_MORPH = 1e-6           # metres: below this, a vertex does not move
THRESHOLD_BOND = 1e-5          # below this, the skin of a mesh is already in the pose of the skeleton
# VRM 1.0 humanoid bone -> VRM 0.x (the 1.0 thumb is named one segment down; the other bones keep their name)
HUMAN_1_TO_0 = {"%sThumb%s" % (c, a): "%sThumb%s" % (c, b) for c in ("left", "right")
                   for a, b in (("Metacarpal", "Proximal"), ("Proximal", "Intermediate"))}
CUT = 0.002                 # MMD units: gap separating the neck skin from the body skin


def description():
    with open(os.path.join(HERE, "..", "games", "vroid", "game.json"), encoding="utf-8") as f:
        return json.load(f)


def read_glb(path):
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != b"glTF":
        raise ValueError("%s is not a binary glTF file (.vrm, .glb)" % path)
    n = struct.unpack_from("<I", d, 12)[0]
    return json.loads(d[20:20 + n]), d[20 + n + 8:]


def _view(j, b, view, offset, dt, k, n):
    v = j["bufferViews"][view]
    deb = v.get("byteOffset", 0) + offset
    stage = v.get("byteStride", 0)
    if stage and stage != dt.itemsize * k:
        raw = np.frombuffer(b, np.uint8, stage * n, deb).reshape(n, stage)[:, :dt.itemsize * k]
        return np.frombuffer(raw.tobytes(), dt).reshape(n, k)
    return np.frombuffer(b, dt, n * k, deb).reshape(n, k)


def accessory(j, b, i):
    a = j["accessors"][i]
    dt = np.dtype(TYPES[a["componentType"]])
    k = NB[a["type"]]
    if "bufferView" in a:
        x = _view(j, b, a["bufferView"], a.get("byteOffset", 0), dt, k, a["count"])
    else:
        x = np.zeros((a["count"], k), dt)
    sp = a.get("sparse")
    if sp:          # sparse form (a re-exported model's shapes): the listed values replace the base ones
        n = sp["count"]
        ix, va = sp["indices"], sp["values"]
        idx = _view(j, b, ix["bufferView"], ix.get("byteOffset", 0), np.dtype(TYPES[ix["componentType"]]), 1, n)[:, 0]
        x = x.copy()
        x[idx.astype(np.int64)] = _view(j, b, va["bufferView"], va.get("byteOffset", 0), dt, k, n)
    return x


SEEDS = 12      # points per side of the sampling grid that reads the transparency of a face


def read_alpha(path):
    """Transparency of an image (h, w), 0..1, first row at the top; None if it has none or
    if OpenImageIO is missing (then nothing is removed)."""
    try:
        import OpenImageIO as OIIO
    except ImportError:
        return None
    r = OIIO.ImageInput.open(path)
    if not r:
        return None
    im = r.read_image(format=OIIO.FLOAT)
    r.close()
    if im is None or im.ndim != 3 or im.shape[2] < 4:
        return None
    return np.ascontiguousarray(im[:, :, 3])


def invisible_ones(a, uv, F, cutting=0.5):
    """Per face: the texture (transparency `a`) is below the cutoff everywhere on the face. In VRoid a
    garment is a whole template (legs, sleeves) whose texture draws only part of it: a pair of
    shorts is a pair of trousers with transparent legs. These faces are never seen; if kept, they
    would turn shorts into full trousers and hide the skin that must be seen."""
    h, w = a.shape
    B = np.array([(i, k, SEEDS - i - k) for i in range(SEEDS + 1) for k in range(SEEDS + 1 - i)], float) / SEEDS
    inv = np.ones(len(F), bool)
    for d in range(0, len(F), 4000):
        P = np.einsum("nk,fkc->fnc", B, uv[F[d:d + 4000]])
        x = (np.mod(P[..., 0], 1.0) * w).astype(int).clip(0, w - 1)
        y = (np.mod(P[..., 1], 1.0) * h).astype(int).clip(0, h - 1)
        inv[d:d + 4000] = (a[y, x] < cutting).all(1)
    return inv


def name_court(name):
    """"Face.M_F00_000_00_Fcl_MTH_A" -> "Fcl_MTH_A"."""
    m = re.search(r"(Fcl_.*)\Z", name)
    return m.group(1) if m else name.split(".")[-1]


def translate(name, table):
    """(common name, pattern number) of the first pattern that matches, otherwise (None, None)."""
    for k, (pattern, common) in enumerate(table):
        if re.search(pattern, name):
            return common, k
    return None, None


class Refusals(ValueError):
    """The file is readable but is not a character this importer can convert (said as is to the user)."""


def matrix(n):
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


def meta_readable(vrm, version):
    """(title, author, licence) declared by the file, in words; licence as the file states it."""
    m = vrm.get("meta") or {}
    if version == 0:
        return (m.get("title") or None, m.get("author") or "?",
                ("%s %s" % (m.get("licenseName") or "?", m.get("otherLicenseUrl") or "")).strip())
    usages = ["avatar: %s" % m.get("avatarPermission", "?"), "commercial: %s" % m.get("commercialUsage", "?"),
              "modification: %s" % m.get("modification", "?"),
              "redistribution: %s" % ("allowed" if m.get("allowRedistribution") else "not allowed"),
              "credit: %s" % m.get("creditNotation", "?")]
    other = m.get("otherLicenseUrl")
    license_ = "%s (%s)%s" % (m.get("licenseUrl") or "?", ", ".join(usages), (" + " + other) if other else "")
    return m.get("name") or None, ", ".join(m.get("authors") or []) or "?", license_


def rotation_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


class Model:
    """The file as read: nodes, humanoid bones, orientation of the frame."""

    def __init__(self, path):
        self.path = path
        self.j, self.b = read_glb(path)
        j = self.j
        ext = j.get("extensions", {})
        if "VRM" in ext:
            self.version, self.vrm = 0, ext["VRM"]
            self.human = {h["bone"]: h["node"] for h in self.vrm["humanoid"]["humanBones"]}
        elif "VRMC_vrm" in ext:
            self.version, self.vrm = 1, ext["VRMC_vrm"]
            spec = str(self.vrm.get("specVersion", ""))
            if not spec.startswith("1."):
                raise Refusals("%s declares VRM version \"%s\": only VRM 0.x and 1.x are read" % (path, spec))
            os1 = (self.vrm.get("humanoid") or {}).get("humanBones") or {}
            self.human = {HUMAN_1_TO_0.get(r, r): h["node"] for r, h in os1.items()}
        else:
            raise Refusals("%s has no VRM extension" % path)
        self.title, self.author, self.license_ = meta_readable(self.vrm, self.version)
        self.nodes = j["nodes"]
        self.parent = {c: i for i, n in enumerate(self.nodes) for c in n.get("children", [])}
        self._world = {}
        for r in ("hips", "spine", "head", "leftUpperArm", "leftLowerArm", "rightUpperArm", "leftHand", "leftFoot"):
            if r not in self.human:
                raise Refusals("missing humanoid bone: %s" % r)
        D = description()
        if not any(translate(m.get("name", ""), D["materials"])[0] == "颜" for m in j.get("materials", [])):
            raise Refusals("%s is not a VRoid character: no material has a VRoid Studio face name (\"..._Face_00_SKIN\"), "
                        "and the face, hair and outfit are found through those names" % path)
        g, d = self.world0(self.human["leftHand"]), self.world0(self.human["rightHand"])
        sx = 1.0 if g[0] > d[0] else -1.0
        toe = self.human.get("leftToes")
        foot = self.world0(self.human["leftFoot"])
        if toe is not None:
            sz = 1.0 if self.world0(toe)[2] < foot[2] else -1.0
        else:                 # without toes: a right-handed frame turned by a half turn, or mirrored
            sz = sx
        self.S = np.array([sx, 1.0, sz]) * SAMP
        self.flips = sx * sz < 0

    def matrix_world(self, i):
        """World transform of node i (4x4), in the frame of the file."""
        if i not in self._world:
            p = self.parent.get(i)
            loc = matrix(self.nodes[i])
            self._world[i] = loc if p is None else self.matrix_world(p) @ loc
        return self._world[i]

    def world0(self, i):
        return self.matrix_world(i)[:3, 3].copy()

    def bond(self, node, skin):
        """Per joint of the skin (or None for an unskinned mesh), the matrix that takes a vertex of the file to the
        pose of the skeleton: joint x inverse bind matrix (glTF skinning), or the node's own transform. None when it
        is the identity everywhere (the VRoid Studio 0.x export): the vertices are then taken as they are."""
        if skin is None:
            B = self.matrix_world(node)[None]
        elif "inverseBindMatrices" not in skin:     # never seen in a VRM: the vertices are taken as they are
            return None
        else:
            n = len(skin["joints"])
            ibm = accessory(self.j, self.b, skin["inverseBindMatrices"]).astype(float).reshape(n, 4, 4).transpose(0, 2, 1)
            B = np.array([self.matrix_world(k) for k in skin["joints"]]) @ ibm
        return None if np.abs(B - np.eye(4)).max() < THRESHOLD_BOND else B

    def world(self, i):
        return self.world0(i) * self.S

    def meshes(self):
        """(node, mesh, skin) of each mesh placed on a node."""
        return [(i, self.j["meshes"][n["mesh"]], self.j["skins"][n["skin"]] if "skin" in n else None)
                for i, n in enumerate(self.nodes) if "mesh" in n]


def skeleton(M, D):
    """The bones of the PMX: (list of bones, index of each node). Parents first."""
    joints = set()
    for _, _, skin in M.meshes():
        if skin:
            joints |= set(skin["joints"])
    for n in list(joints):    # the ancestors too (root)
        p = M.parent.get(n)
        while p is not None:
            joints.add(p)
            p = M.parent.get(p)
    with_mesh = {i for i, _, _ in M.meshes()}
    joints -= with_mesh
    order = []

    def visited(i):
        if i in joints and i not in order:
            if M.parent.get(i) in joints:
                visited(M.parent[i])
            order.append(i)
    for i in sorted(joints):
        visited(i)
    role = {n: r for r, n in M.human.items()}
    sheet = PMX.OS_ROTATION | PMX.OS_VISIBLE | PMX.OS_ACTIVE

    def os_(name, name_en, pos, parent, moved=False):
        return {"name": name, "name_en": name_en, "pos": tuple(float(x) for x in pos), "parent": parent, "layer": 0,
                "flags": sheet | (PMX.OS_DISPLACEMENT if moved else 0), "tail_bone": None, "tail_offset": (0.0, 0.0, 0.0)}
    hips = M.world(M.human["hips"])
    bones = [os_("全ての親", "root", (0, 0, 0), -1, True), os_("センター", "center", hips, 0, True)]
    rank = {}
    for n in order:
        name = M.nodes[n].get("name", "os%d" % n)
        r = role.get(n)
        if M.parent.get(n) is None:           # the root of the file is our 全ての親
            rank[n] = 0
            continue
        p = rank[M.parent[n]]
        if any(re.search(x, name) for x in D.get("os_fondus", ())):    # bone that Chimera's body does not have: its
            rank[n] = p                                               # weight goes back to its parent
            continue
        rank[n] = len(bones)
        if r in ("hips", "spine"):
            p = 1
        bones.append(os_(D["os_mmd"].get(r, name), r or name, M.world(n), p))
    return bones, rank


# bones whose tail points to a given child (otherwise: the only child, or a tip extending the parent)
NEXT = {"spine": "chest", "chest": "upperChest", "upperChest": "neck", "neck": "head", "leftHand": "leftMiddleProximal",
         "rightHand": "rightMiddleProximal"}
TIP = {"全ての親": (0.0, 0.0, -1.0), "センター": (0.0, -1.0, 0.0), "hips": (0.0, -1.0, 0.0), "head": (0.0, 1.5, 0.0),
        "leftEye": (0.0, 0.0, -0.5), "rightEye": (0.0, 0.0, -0.5)}


def tails(bones, children):
    """The tail of each bone (the core measures along the bones): the next bone of the chain, otherwise a tip."""
    rank = {o["name_en"]: i for i, o in enumerate(bones)}
    for i, o in enumerate(bones):
        e = children.get(i, [])
        suite = rank.get(NEXT.get(o["name_en"]))
        if o["name_en"] in TIP or o["name"] in TIP:
            o["tail_offset"] = TIP.get(o["name_en"]) or TIP[o["name"]]
            continue
        if suite is None and len(e) == 1:
            suite = e[0]
        if suite is not None and np.linalg.norm(np.array(bones[suite]["pos"]) - o["pos"]) > 1e-4:
            o["tail_bone"], o["tail_offset"] = suite, None
            o["flags"] |= PMX.OS_QUEUE_INDEX
        elif o["parent"] >= 0:
            d = np.array(o["pos"]) - bones[o["parent"]]["pos"]
            o["tail_offset"] = tuple(float(x) for x in 0.5 * d)


def convert(path, folder, D=None, log=print, keep=(), perfect_sync="auto"):
    """Writes <folder>/<Name>.pmx and its textures. Returns a short summary. `keep`: common names (肌…) whose
    faces stay even when their texture erases them (the VRoid base keeps all the skin and repaints it).
    `perfect_sync`: donor of the Perfect Sync shapes added to the face (perfect_sync.py: "auto", "female", "male"),
    None for none."""
    D = D or description()
    M = Model(path)
    j, b = M.j, M.b
    bones, rank = skeleton(M, D)
    names_os = [o["name"] for o in bones]
    # --- vertices: one block per position accessor (the primitives of a VRoid mesh share it)
    blocks, pos, nor, uv, jo, po = {}, [], [], [], [], []
    faces_mat = {}            # glTF material -> [faces]
    morphs = {}              # name -> (indices, offsets); a shape split over several primitives is put back together
    n_vert = 0
    face_rows, face_faces = [], {}     # the face mesh (the one carrying the face skin): its vertices, its faces per material
    for node, mesh, skin in M.meshes():
        keys = [translate(j["materials"][p["material"]].get("name", ""), D["materials"])[0] for p in mesh["primitives"]]
        is_face = "颜" in keys
        to_os = np.array([rank[n] for n in skin["joints"]]) if skin else None
        B = M.bond(node, skin)
        seen, link = set(), {}
        for pr, common in zip(mesh["primitives"], keys):
            at = pr["attributes"]
            the_key = at["POSITION"]
            if the_key not in blocks:
                p = accessory(j, b, the_key).astype(float)
                n_ = accessory(j, b, at["NORMAL"]).astype(float) if "NORMAL" in at else np.tile([0.0, 1.0, 0.0], (len(p), 1))
                blocks[the_key] = n_vert
                if to_os is not None and "JOINTS_0" in at:
                    w = accessory(j, b, at["WEIGHTS_0"]).astype(float)
                    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
                    jt = accessory(j, b, at["JOINTS_0"]).astype(int)
                    o = to_os[jt]
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
                    link[the_key] = L[:, :3, :3]
                pos.append(p * M.S)
                nor.append(n_ * np.sign(M.S))
                uv.append(accessory(j, b, at["TEXCOORD_0"]).astype(float) if "TEXCOORD_0" in at else np.zeros((len(p), 2)))
                jo.append(o)
                po.append(w)
                if is_face:
                    face_rows.append(np.arange(n_vert, n_vert + len(p)))
                n_vert += len(p)
            offset = blocks[the_key]
            f = accessory(j, b, pr["indices"]).reshape(-1, 3).astype(np.int64) + offset
            if is_face:
                face_faces.setdefault(common or "?", []).append(f)
            faces_mat.setdefault(pr["material"], []).append(f[:, ::-1] if M.flips else f)
            names = pr.get("extras", {}).get("targetNames") or mesh.get("extras", {}).get("targetNames") or []
            for k, target in enumerate(pr.get("targets", [])):
                if (the_key, k) in seen or "POSITION" not in target:
                    continue
                seen.add((the_key, k))
                d = accessory(j, b, target["POSITION"]).astype(float)
                if the_key in link:
                    d = np.einsum("nij,nj->ni", link[the_key], d)
                idx = np.where(np.abs(d).max(1) > THRESHOLD_MORPH)[0]
                if len(idx):
                    nm = names[k] if k < len(names) else "shape%d" % k
                    i0, d0 = morphs.get(nm, (np.zeros(0, np.int64), np.zeros((0, 3))))
                    morphs[nm] = (np.concatenate([i0, idx + offset]), np.concatenate([d0, d[idx] * M.S]))
    pos, nor, uv, jo, po = [np.concatenate(x) for x in (pos, nor, uv, jo, po)]
    # --- Perfect Sync: the shapes of Hinzka's face that this face lacks (perfect_sync.py), offsets in the PMX frame
    sync = None
    if perfect_sync and face_rows:
        import perfect_sync as PS
        rows = np.concatenate(face_rows)
        key = np.full(len(pos), "", object)
        for k, lf in face_faces.items():
            key[np.concatenate(lf).ravel()] = k
        added, sync = PS.add(pos, uv, rows, key[rows], np.concatenate(face_faces.get("颜", [np.zeros((0, 3), np.int64)])),
                             {name_court(x) for x in morphs}, perfect_sync, log=log)
        morphs.update(added)
    # --- A-pose: the arms rotated around the shoulder, each vertex according to its weight on the arm
    angle = float(D["pose"]["arm"])
    children = {}
    for i, o in enumerate(bones):
        children.setdefault(o["parent"], []).append(i)
    lin = np.tile(np.eye(3), (len(pos), 1, 1))     # linear part of the transformation of each vertex
    for side, sign in (("left", -1.0), ("right", 1.0)):
        i0 = names_os.index(D["os_mmd"][side + "UpperArm"])
        under, pile = [], [i0]
        while pile:
            k = pile.pop()
            under.append(k)
            pile += children.get(k, [])
        side_x = 1.0 if bones[names_os.index(D["os_mmd"][side + "LowerArm"])]["pos"][0] > bones[i0]["pos"][0] else -1.0
        R = rotation_z(sign * angle * (1.0 if (side == "left") == (side_x > 0) else -1.0))
        c = np.array(bones[i0]["pos"])
        w = (po * np.isin(jo, under)).sum(1)
        pos += w[:, None] * ((pos - c) @ R.T - (pos - c))
        lin += w[:, None, None] * (R - np.eye(3))
        for k in under:
            bones[k]["pos"] = tuple(float(x) for x in c + R @ (np.array(bones[k]["pos"]) - c))
    tails(bones, children)
    nor = np.einsum("nij,nj->ni", lin, nor)
    nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)
    # --- neck: in VRoid the body skin is in one piece, welded to the face; the core expects the skin welded to the
    # face to stop at the bottom of the neck (the whole head: face, ears, neck). So the body skin is cut at the bottom of the
    # neck, along the edges: the faces below get their own vertices, lowered by a hair.
    y_neck, y_head = bones[names_os.index(D["os_mmd"]["neck"])]["pos"][1], bones[names_os.index(D["os_mmd"]["head"])]["pos"][1]
    y0 = y_neck + float(D["pose"]["neck"]) * (y_head - y_neck)
    cut = 0
    for i, lf in faces_mat.items():
        if translate(j["materials"][i].get("name", ""), D["materials"])[0] != "肌":
            continue
        f = np.concatenate(lf)
        top = (pos[f][:, :, 1] >= y0).all(1)
        if not top.any() or top.all():
            continue
        the_key = lambda P: [bytes(x) for x in np.round(P, 5).astype("<f4")]     # noqa: E731
        of_top = set(the_key(pos[np.unique(f[top])]))
        bottom = np.unique(f[~top])
        border = bottom[[k in of_top for k in the_key(pos[bottom])]]
        # edge brought back into the plane of the cut (it follows the edges, in a sawtooth): the core reads a flat ring
        of_border = set(the_key(pos[border]))
        all_ = np.unique(f)
        pos[all_[[k in of_border for k in the_key(pos[all_])]], 1] = y0
        new = np.arange(len(border)) + len(pos)
        pos = np.concatenate([pos, pos[border] - [0.0, CUT, 0.0]])
        nor, uv, jo, po, lin = [np.concatenate([x, x[border]]) for x in (nor, uv, jo, po, lin)]
        to = np.arange(len(pos))
        to[border] = new
        f[~top] = to[f[~top]]
        faces_mat[i] = [f]
        cut += len(border)
    # --- materials, in the order of the table (skin, face, eyes, hair, outfit)
    os.makedirs(folder, exist_ok=True)
    textures, outputs = [], {}

    def texture(mat):
        t = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if t is None:
            return -1
        src = j["textures"][t["index"]]["source"]
        if src not in outputs:
            im = j["images"][src]
            v = j["bufferViews"][im["bufferView"]]
            name = re.sub(r"[^\w.-]", "_", im.get("name") or "image%d" % src) + EXT.get(im.get("mimeType"), ".png")
            if name in textures:         # two images of the same name: the second one must not overwrite the first
                name = "%s_%d%s" % (os.path.splitext(name)[0], src, os.path.splitext(name)[1])
            with open(os.path.join(folder, name), "wb") as f:
                f.write(b[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]])
            outputs[src] = len(textures)
            textures.append(name)
        return outputs[src]
    lines, taken = [], {}
    for i, f in faces_mat.items():
        mat = j["materials"][i]
        common, k = translate(mat.get("name", ""), D["materials"])
        if common is None:
            common, k = mat.get("name", "material%d" % i), len(D["materials"])
        lines.append((k, i, common, mat, np.concatenate(f)))
    lines.sort(key=lambda l: (l[0], l[1]))
    mats, faces, invisible = [], [], 0
    for k, i, common, mat, f in lines:
        if mat.get("alphaMode") == "MASK" and len(f) and common not in keep:   # cut out by the texture: what it erases does not exist
            t = texture(mat)
            alpha = read_alpha(os.path.join(folder, textures[t])) if t >= 0 else None
            if alpha is not None:
                inv = invisible_ones(alpha, uv, f, float(mat.get("alphaCutoff", 0.5)))
                invisible += int(inv.sum())
                f = f[~inv]
                if not len(f):
                    continue
        taken[common] = taken.get(common, 0) + 1
        name = common if taken[common] == 1 else "%s%d" % (common, taken[common])
        color = tuple(float(x) for x in mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1, 1, 1, 1]))
        trait = mat.get("alphaMode", "OPAQUE") != "BLEND"       # no outline on a transparent layer
        mats.append({"name": name, "name_en": mat.get("name", ""), "diffuse": color, "specular": (0.0, 0.0, 0.0), "brightness": 5.0,
                     "ambient": (0.5, 0.5, 0.5), "flags": 0x0E | (0x01 if mat.get("doubleSided") else 0) | (0x10 if trait else 0),
                     "outline_color": (0.2, 0.14, 0.12, 1.0), "outline_size": 0.6, "texture": texture(mat), "sphere": -1,
                     "sphere_mode": 0, "toon_sharing": 1, "toon": 0, "memo": "", "n_indices": 3 * len(f)})
        faces.append(f)
    faces = np.concatenate(faces)
    util = np.unique(faces)
    nv = -np.ones(len(pos), np.int64)
    nv[util] = np.arange(len(util))
    faces = nv[faces]
    n = len(util)
    vert = {"pos": [tuple(p) for p in pos[util].tolist()], "nor": [tuple(p) for p in nor[util].tolist()],
           "uv": [tuple(p) for p in uv[util].tolist()], "uv_add": [[] for _ in range(n)], "type": [2] * n,
           "bones": [tuple(int(x) for x in r) for r in jo[util]], "weights": [tuple(float(x) for x in r) for r in po[util]],
           "sdef": [None] * n, "contour": [1.0] * n}
    # --- morphs
    panels = D["panels"]
    out_m, without_name = [], []
    for name, (idx, d) in morphs.items():
        guard = nv[idx] >= 0
        if not guard.any():
            continue
        idx, d = idx[guard], np.einsum("nij,nj->ni", lin[idx[guard]], d[guard])
        court = name_court(name)
        mmd = D["morphs_mmd"].get(court)
        if mmd is None:
            without_name.append(court)
        pan = next((p for pattern, p in panels if re.search(pattern, court)), 4)
        out_m.append({"name": mmd or court, "name_en": court, "panel": pan, "type": 1,
                      "offsets": [(int(a), tuple(float(x) for x in v)) for a, v in zip(nv[idx], d)]})
    name = os.path.splitext(os.path.basename(path))[0]
    output = os.path.join(folder, name + ".pmx")
    pmx_write.write({"version": 2.0, "encoding": "UTF-16", "name": M.title or name, "name_en": name,
                       "comment": "Converted from a VRM %s file by tools/vroid/importer.py. Author: %s. Declared licence: %s"
                                      % ("0.x" if M.version == 0 else "1.0", M.author, M.license_),
                       "comment_en": "", "n_uv_add": 0, "vertices": vert, "faces": [tuple(int(x) for x in f) for f in faces],
                       "textures": textures, "materials": mats, "bones": bones, "morphs": out_m,
                       "frames": [{"name": "Root", "name_en": "Root", "special": 1, "elements": [(0, 0)]},
                                  {"name": "表情", "name_en": "Exp", "special": 1, "elements": [(1, k) for k in range(len(out_m))]}],
                       "rigid_bodies": [], "joints": []}, output)
    assessment = {"pmx": output, "vertices": n, "faces": len(faces), "bones": len(bones),
             "os_mmd": sum(o["name"] != o["name_en"] for o in bones), "materials": [m["name"] for m in mats],
             "morphs": len(out_m), "morphs_mmd": len(out_m) - len(without_name), "textures": len(textures),
             "height_cm": round(float(pos[util][:, 1].max() - pos[util][:, 1].min()) * 8, 1),
             "vrm": "0.x" if M.version == 0 else "1.0", "title": M.title, "author": M.author, "license": M.license_}
    log("wrote %s: %d vertices, %d faces, %d bones (%d with MMD names), %d morphs (%d with MMD names), %d textures, %.0f cm"
            % (output, n, len(faces), assessment["bones"], assessment["os_mmd"], assessment["morphs"], assessment["morphs_mmd"], len(textures), assessment["height_cm"]))
    log("materials: " + " ".join(assessment["materials"]))
    log("VRM %s, author %s, declared licence: %s" % (assessment["vrm"], M.author, M.license_))
    if invisible:
        assessment["faces_invisible"] = invisible
        log("%d faces erased by the transparency of their texture, removed" % invisible)
    if sync is not None:
        if sync.get("names"):       # the check of morphs.py on the added shapes: frozen parts, parts piercing the skin, empty
            import perfect_sync as PS
            bad, n_checked = PS.checked(output, sync["names"])
            sync["check"] = {"checked": n_checked, "defects": bad}
            log("perfect sync check: %d shapes, %d with a defect%s" % (n_checked, len(bad), (": " + ", ".join(
                b["morph"] + (" (expected)" if b.get("expected") else "") for b in bad)) if bad else ""))
        with open(os.path.join(folder, "perfect_sync.json"), "w", encoding="utf-8") as f:
            json.dump(sync, f, ensure_ascii=False, indent=1)
        assessment["perfect_sync"] = {k: sync[k] for k in ("donor", "vertices", "unmatched", "added", "skipped", "calibration", "check")
                                      if k in sync}
    return assessment


def see(path):
    M = Model(path)
    D = description()
    print("%s (VRM %s): \"%s\", author %s, licence %s" % (os.path.basename(path), "0.x" if M.version == 0 else "1.0",
                                                         M.title, M.author, M.license_))
    print("humanoid bones: %d, of which without an MMD name: %s" % (len(M.human), ", ".join(sorted(set(M.human) - set(D["os_mmd"]))) or "none"))
    for _, m, _ in M.meshes():
        print("mesh %s: %d primitives" % (m.get("name"), len(m["primitives"])))
        for i in sorted({p["material"] for p in m["primitives"]}):
            name = M.j["materials"][i].get("name", "")
            print("    %-44s -> %s" % (name, translate(name, D["materials"])[0] or "(name kept: outfit)"))
        names = m["primitives"][0].get("extras", {}).get("targetNames") or m.get("extras", {}).get("targetNames") or []
        if names:
            court = [name_court(x) for x in names]
            print("    shapes: %d, of which with MMD names: %s" % (len(court), " ".join("%s=%s" % (c, D["morphs_mmd"][c]) for c in court if c in D["morphs_mmd"])))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "show":
        for c in argv[1:]:
            see(c)
        return 0
    sync = "auto"
    if "--no-perfect-sync" in argv:
        argv.remove("--no-perfect-sync")
        sync = None
    if "--perfect-sync-donor" in argv:
        k = argv.index("--perfect-sync-donor")
        sync = argv[k + 1]
        del argv[k:k + 2]
    output = None
    if "--output" in argv:
        k = argv.index("--output")
        output = argv[k + 1]
        del argv[k:k + 2]
    if output is None:
        import games as games
        output = games.load("vroid").path("mmd_extracted")
    for c in argv:
        name = os.path.splitext(os.path.basename(c))[0]
        try:
            convert(c, os.path.join(output, name), perfect_sync=sync)
        except Refusals as e:      # said as is, on one line that the workshop relays (tools/workshop/inputs.py)
            print("refused: %s" % str(e).replace(c, "the file"), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
