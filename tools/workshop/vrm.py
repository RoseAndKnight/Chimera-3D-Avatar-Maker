"""VRM 1.0 export of an assembled model (PMX read by pmx.py). Numpy only, except for the texture
conversion (`images`, supplied by Blender).

    write_vrm(P, "sortie.vrm", images=dict(file -> (bytes, mimetype)), name="Outfit", author="Chimera")

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

METER = 0.08
ALPHA_MASK = 0.4
PART_CUTOUT = 0.001  # share of pixels below the threshold from which the texture has holes

# VRM humanoid bone -> MMD bones, from the most precise to the fallback
HUMANOIDS = {
    "hips": ["下半身"], "spine": ["上半身"], "chest": ["上半身1"], "upperChest": ["上半身2", "上半身3"],
    "neck": ["首"], "head": ["頭"], "leftEye": ["左目"], "rightEye": ["右目"],
    "leftShoulder": ["左肩"], "leftUpperArm": ["左腕"], "leftLowerArm": ["左ひじ"], "leftHand": ["左手首"],
    "rightShoulder": ["右肩"], "rightUpperArm": ["右腕"], "rightLowerArm": ["右ひじ"], "rightHand": ["右手首"],
    "leftUpperLeg": ["左足D", "左足"], "leftLowerLeg": ["左ひざD", "左ひざ"], "leftFoot": ["左足首D", "左足首"], "leftToes": ["左足先EX"],
    "rightUpperLeg": ["右足D", "右足"], "rightLowerLeg": ["右ひざD", "右ひざ"], "rightFoot": ["右足首D", "右足首"], "rightToes": ["右足先EX"],
}
for side, vrm in (("左", "left"), ("右", "right")):
    for finger, mmd, trunk in (("Thumb", "親指", ("Metacarpal", "Proximal", "Distal")), ("Index", "人指", None), ("Middle", "中指", None),
                              ("Ring", "薬指", None), ("Little", "小指", None)):
        if trunk:  # the MMD thumb goes from ０ to ２
            for k, name in zip("０１２", trunk):
                HUMANOIDS["%s%s%s" % (vrm, finger, name)] = ["%s%s%s" % (side, mmd, k)]
        else:
            for k, name in zip("１２３", ("Proximal", "Intermediate", "Distal")):
                HUMANOIDS["%s%s%s" % (vrm, finger, name)] = ["%s%s%s" % (side, mmd, k)]
REQUIRED = ["hips", "spine", "head", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand",
                "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]

# Weights placed on the control chain go to the deforming chain
DUPLICATES = {"左足": "左足D", "左ひざ": "左ひざD", "左足首": "左足首D", "右足": "右足D", "右ひざ": "右ひざD", "右足首": "右足首D"}

# VRM expression -> MMD morphs (the first one present)
EXPRESSIONS = {
    "happy": ["笑い", "にこり"], "angry": ["怒り", "怒り目"], "sad": ["悲しむ", "困る"], "surprised": ["びっくり"], "relaxed": ["じと目"],
    "blink": ["まばたき"], "blinkLeft": ["ウィンク", "ウィンク２"], "blinkRight": ["ウィンク右", "ウィンク２右"],
    "aa": ["あ"], "ih": ["い"], "ou": ["う"], "ee": ["え"], "oh": ["お"],   # fr-ok: VRM preset names
}


def _to_gltf(v):
    v = np.asarray(v, float) * METER
    return v * np.array([1.0, 1.0, -1.0])


def _rotation(a, b):
    """Smallest rotation bringing direction a onto b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.9999:
        return np.diag([-1.0, 1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


class Skeleton:
    """Kept bones (PMX indices), parents corrected for VRM, and T-pose."""

    def __init__(self, P):
        self.P = P
        self.names = [o["name"] for o in P["bones"]]
        self.per_name = {n: i for i, n in enumerate(self.names)}
        self.pos = _to_gltf([o["pos"] for o in P["bones"]])
        self.parent = [o["parent"] for o in P["bones"]]
        I = np.array(P["vertices"]["bones"], np.int64)
        W = np.array(P["vertices"]["weights"], float)
        self.I, self.W = self._without_duplicates(I, W)
        self.human = {}
        # house skeleton (custom_body/plug.py): each bone carries its VRM role as its English name
        roles = {o.get("name_en"): i for i, o in enumerate(P["bones"]) if o.get("name_en") in HUMANOIDS}
        self.own_rig = all(b in roles for b in REQUIRED)
        for vrm, candidates in HUMANOIDS.items():
            if self.own_rig and vrm in roles:
                self.human[vrm] = roles[vrm]
                continue
            for n in candidates:
                if n in self.per_name:
                    self.human[vrm] = self.per_name[n]
                    break
        missing = [b for b in REQUIRED if b not in self.human]
        if missing:
            raise ValueError("missing humanoid bones: %s" % ", ".join(missing))
        # in MMD, upper and lower body are siblings under 腰: VRM wants the spine under the hips
        self.parent[self.human["spine"]] = self.human["hips"]
        self._resume_chain()
        self.guard = self._kept()

    def _without_duplicates(self, I, W):
        map_ = np.arange(len(self.names))
        for a, b in DUPLICATES.items():
            if a in self.per_name and b in self.per_name:
                map_[self.per_name[a]] = self.per_name[b]
        J = I.copy()
        J[J >= 0] = map_[J[J >= 0]]
        return J, W

    def _resume_chain(self):
        """Each humanoid bone descends from the humanoid bone before it: the parent is moved up if
        a control branch sits in between without going through it."""
        def ancestors(i):
            out, seen = [], set()
            while i >= 0 and i not in seen:
                seen.add(i)
                out.append(i)
                i = self.parent[i]
            return out
        chain = {"chest": "spine", "upperChest": "chest", "neck": "upperChest", "head": "neck",
                  "leftUpperArm": "leftShoulder", "rightUpperArm": "rightShoulder", "leftShoulder": "upperChest",
                  "rightShoulder": "upperChest", "leftLowerArm": "leftUpperArm", "rightLowerArm": "rightUpperArm",
                  "leftHand": "leftLowerArm", "rightHand": "rightLowerArm", "leftLowerLeg": "leftUpperLeg", "rightLowerLeg": "rightUpperLeg",
                  "leftFoot": "leftLowerLeg", "rightFoot": "rightLowerLeg", "leftToes": "leftFoot", "rightToes": "rightFoot",
                  "leftUpperLeg": "hips", "rightUpperLeg": "hips"}
        for b, p in chain.items():
            if b in self.human and p in self.human and self.human[p] not in ancestors(self.human[b]):
                self.parent[self.human[b]] = self.human[p]

    def _kept(self):
        useful = set(int(x) for x in np.unique(self.I[(self.I >= 0) & (self.W > 0)]))
        useful |= set(self.human.values())
        for i in list(useful):
            while i >= 0:
                useful.add(i)
                i = self.parent[i]
        return sorted(useful)

    def tpose(self, legs=False):
        """Returns (bone positions in T-pose, rotation per bone, deformation per vertex)."""
        n = len(self.names)
        target = {}
        for side, x in (("left", 1.0), ("right", -1.0)):
            for bone, suite in (("UpperArm", "LowerArm"), ("LowerArm", "Hand"), ("Hand", "MiddleProximal")):
                a, b = side + bone, side + suite
                if a in self.human and b in self.human:
                    target[self.human[a]] = (self.human[b], np.array([x, 0.0, 0.0]))
            a, b = side + "UpperLeg", side + "LowerLeg"
            if legs and self.own_rig:   # the thigh straight down: its twist is then expressed along a frame axis
                target[self.human[a]] = (self.human[b], np.array([0.0, -1.0, 0.0]))
        order, seen = [], set()

        def visit(i):
            if i < 0 or i in seen:
                return
            seen.add(i)
            visit(self.parent[i])
            order.append(i)
        for i in range(n):
            visit(i)
        Rw = np.tile(np.eye(3), (n, 1, 1))
        Pw = self.pos.copy()
        for i in order:
            p = self.parent[i]
            Rp = Rw[p] if p >= 0 else np.eye(3)
            Pw[i] = (Pw[p] + Rp @ (self.pos[i] - self.pos[p])) if p >= 0 else self.pos[i]
            R = Rp
            if i in target:
                c, t = target[i]
                R = _rotation(Rp @ (self._to(i, c)), t) @ Rp
            Rw[i] = R
        return Pw, Rw

    def _to(self, a, b):
        return self.pos[b] - self.pos[a]


class Bin:
    def __init__(self):
        self.o = bytearray()
        self.views, self.accessory = [], []

    def view(self, bytes_, target=None):
        while len(self.o) % 4:
            self.o.append(0)
        v = {"buffer": 0, "byteOffset": len(self.o), "byteLength": len(bytes_)}
        if target:
            v["target"] = target
        self.o += bytes_
        self.views.append(v)
        return len(self.views) - 1

    def access(self, a, component, type_, target=None, bounds=False, norm=False):
        a = np.asarray(a)
        v = self.view(a.tobytes(), target)
        d = {"bufferView": v, "componentType": component, "count": int(len(a)), "type": type_}
        if norm:
            d["normalized"] = True
        if bounds:
            d["min"] = [float(x) for x in a.min(0)]
            d["max"] = [float(x) for x in a.max(0)]
        self.accessory.append(d)
        return len(self.accessory) - 1


def _mtoon(m, tex, double, name, cutout=True):
    """`cutout`: the texture has real holes. Without them, a MASK material gains nothing and
    breaks in viewers that plug the alpha channel in as is: it stays opaque."""
    d = list(m["diffuse"])
    base = {"baseColorFactor": [float(d[0]), float(d[1]), float(d[2]), float(d[3])], "metallicFactor": 0.0, "roughnessFactor": 0.9}
    if tex is not None:
        base["baseColorTexture"] = {"index": tex}
    a = {"name": name, "pbrMetallicRoughness": base, "doubleSided": double,
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
    if m["flags"] & 0x10 and m["outline_size"] > 0:
        mt["outlineWidthMode"] = "worldCoordinates"
        mt["outlineWidthFactor"] = float(m["outline_size"]) * 0.0012
        mt["outlineColorFactor"] = [float(x) for x in m["outline_color"][:3]]
    if "+" in name:
        a["alphaMode"] = "BLEND"
    elif tex is not None and cutout:
        a["alphaMode"] = "MASK"
        a["alphaCutoff"] = ALPHA_MASK
    return a


def write_vrm(P, output, images, name="Character", author="Chimera", twist=None, sights=False):
    """Writes `sortie`. `images`: PMX texture (path as written in P) -> (bytes, mimetype).
    `twist`: twist bones of the house skeleton, name -> {sort, source, fraction} (skeleton.json). A bone that "follows"
    becomes a VRM 1.0 roll constraint, changing nothing for a viewer that ignores constraints. A bone that
    "holds" would need a negative weight, which VRM lacks: its rule is in `extras`. With `sights`, it
    goes under an aimer (aim constraint: its source's swing without its roll) and takes back its share of the
    roll: exact in a VRM viewer (three-vrm), but the bone no longer follows its arm in a viewer that ignores
    constraints (Blender, Godot). Hence the default: without."""
    S = Skeleton(P)
    Pw, Rw = S.tpose(legs=sights)
    # ---- vertices in T-pose
    X = _to_gltf(P["vertices"]["pos"])
    N = np.array(P["vertices"]["nor"], float) * np.array([1.0, 1.0, -1.0])
    UV = np.array(P["vertices"]["uv"], float)
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
    joints = S.guard
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
    for m in P["materials"]:
        n = m["n_indices"] // 3
        mats.append(F[k:k + n])
        k += n
    # ---- selected morphs and the vertices they touch
    morphs = {m["name"]: m for m in P["morphs"] if m["type"] == 1}
    chosen = {}
    for expr, candidates in EXPRESSIONS.items():
        for c in candidates:
            if c in morphs:
                chosen[expr] = morphs[c]
                break
    with_morph = set()
    for m in chosen.values():
        with_morph.update(i for i, _ in m["offsets"])
    head = [i for i, f in enumerate(mats) if len(f) and any(int(x) in with_morph for x in np.unique(f))]
    others = [i for i, f in enumerate(mats) if len(f) and i not in head]
    # ---- writing
    B = Bin()
    meshes, images_doc, textures, materials, tex_of = [], [], [], [], {}

    def tex(i):
        if i < 0 or i >= len(P["textures"]):
            return None
        the_key = P["textures"][i]
        if the_key not in images:
            return None
        if the_key not in tex_of:
            bytes_, mime = images[the_key][:2]
            images_doc.append({"bufferView": B.view(bytes_), "mimeType": mime, "name": os.path.basename(the_key.replace("\\", "/"))})
            textures.append({"source": len(images_doc) - 1, "sampler": 0})
            tex_of[the_key] = len(textures) - 1
        return tex_of[the_key]

    def mesh(items, name_m, targets):
        vs = np.unique(np.concatenate([mats[i].ravel() for i in items]))
        map_ = -np.ones(len(X), np.int64)
        map_[vs] = np.arange(len(vs))
        at = {"POSITION": B.access(Y[vs].astype("<f4"), 5126, "VEC3", 34962, bounds=True),
              "NORMAL": B.access(M[vs].astype("<f4"), 5126, "VEC3", 34962),
              "TEXCOORD_0": B.access(UV[vs].astype("<f4"), 5126, "VEC2", 34962),
              "JOINTS_0": B.access(J[vs].astype("<u2"), 5123, "VEC4", 34962),
              "WEIGHTS_0": B.access(Wn[vs].astype("<f4"), 5126, "VEC4", 34962)}
        target = []
        for m in targets:
            d = np.zeros((len(vs), 3), np.float32)
            for i, v in m["offsets"]:
                if map_[i] >= 0:
                    d[map_[i]] = (float(v[0]) * METER, float(v[1]) * METER, -float(v[2]) * METER)
            target.append({"POSITION": B.access(d.astype("<f4"), 5126, "VEC3", 34962, bounds=True)})
        prims = []
        for i in items:
            m = P["materials"][i]
            g = map_[mats[i]]
            if not len(materials) > i or materials[i] is None:
                while len(materials) <= i:
                    materials.append(None)
                input_ = images.get(P["textures"][m["texture"]]) if 0 <= m["texture"] < len(P["textures"]) else None
                cutout = input_ is None or len(input_) < 3 or bool(input_[2])
                materials[i] = _mtoon(m, tex(m["texture"]), bool(m["flags"] & 1), m["name"], cutout)
            pr = {"attributes": at, "indices": B.access(g[:, ::-1].astype("<u4").ravel(), 5125, "SCALAR", 34963),
                  "material": i, "mode": 4}
            if target:
                pr["targets"] = target
            prims.append(pr)
        d = {"name": name_m, "primitives": prims}
        if target:
            d["extras"] = {"targetNames": [m["name"] for m in targets]}
        meshes.append(d)
        return len(meshes) - 1

    targets = list(dict.fromkeys(id(m) for m in chosen.values()))
    per_id = {id(m): m for m in chosen.values()}
    targets_m = [per_id[c] for c in targets]
    if head:
        i_head = mesh(head, "head", targets_m)
    i_others = mesh(others, "body", []) if others else None
    while None in materials:   # materials without faces: an empty material keeps the numbering
        materials[materials.index(None)] = {"name": "empty"}

    # ---- bones -> nodes
    n_joints = len(joints)
    nodes = []
    for b in joints:
        p = S.parent[b]
        while p >= 0 and p not in idx_joint:
            p = S.parent[p]
        t = Pw[b] - (Pw[p] if p >= 0 else np.zeros(3))
        nodes.append({"name": S.names[b], "translation": [float(x) for x in t]})
    for k, b in enumerate(joints):
        p = S.parent[b]
        while p >= 0 and p not in idx_joint:
            p = S.parent[p]
        if p >= 0:
            nodes[idx_joint[p]].setdefault("children", []).append(k)
    # ---- twist of the house skeleton
    twist = twist or {}
    constraints = 0
    pos_n = [Pw[b] for b in joints]
    node_of = {S.names[b]: k for k, b in enumerate(joints)}
    parent = {c: k for k, n in enumerate(nodes) for c in n.get("children", [])}
    vis = {}

    def axe_of(src):
        """The bone extending src and the frame axis it follows in T-pose: (node, "X", "PositiveX"), or None."""
        suite = [c for c in nodes[src].get("children", []) if nodes[c]["name"] not in twist and c < len(joints)]
        if not suite:
            return None
        c = max(suite, key=lambda c: np.linalg.norm(pos_n[c] - pos_n[src]))
        d = pos_n[c] - pos_n[src]
        d = d / max(np.linalg.norm(d), 1e-9)
        a = int(np.abs(d).argmax())
        if abs(d[a]) < 0.999:
            return None
        return c, "XYZ"[a], ("Positive" if d[a] > 0 else "Negative") + "XYZ"[a]

    def constraint(sort, **kw):
        return {"VRMC_node_constraint": {"specVersion": "1.0", "constraint": {sort: dict(kw, weight=float(kw.get("weight", 1.0)))}}}
    for n, t in twist.items():
        if n not in node_of or t["source"] not in node_of:
            continue
        k, s = node_of[n], node_of[t["source"]]
        nodes[k]["extras"] = {"twist": {"sort": t["sort"], "source": t["source"], "fraction": t["fraction"]}}
        if t["sort"] == "follows":
            a = axe_of(parent[k])
            if a and t["fraction"] > 0:
                nodes[k]["extensions"] = constraint("roll", source=s, rollAxis=a[1], weight=t["fraction"])
                constraints += 1
        elif sights and s in parent and axe_of(s):
            # "keeps" (holds back): the bone keeps a share of its source's roll. It leaves its source for an aimer
            # placed at the same spot, aiming at the next bone (the swing only, no roll), and takes back a share of the roll.
            a = axe_of(s)
            if s not in vis:
                nodes.append({"name": t["source"] + "_viseur", "translation": list(nodes[s]["translation"]),
                               "extensions": constraint("aim", source=a[0], aimAxis=a[2])})
                vis[s] = len(nodes) - 1
                pos_n.append(pos_n[s])
                nodes[parent[s]]["children"].append(vis[s])
                constraints += 1
            nodes[s]["children"].remove(k)
            nodes[vis[s]].setdefault("children", []).append(k)
            if t["fraction"] > 0:
                nodes[k]["extensions"] = constraint("roll", source=s, rollAxis=a[1], weight=t["fraction"])
                constraints += 1
    n_joints = len(nodes)
    roots = [k for k in range(n_joints) if not any(k in n.get("children", []) for n in nodes)]
    ibm = np.tile(np.eye(4), (n_joints, 1, 1))
    ibm[:, :3, 3] = -np.array(pos_n)
    ibm_accessory = B.access(np.transpose(ibm, (0, 2, 1)).astype("<f4"), 5126, "MAT4")
    skin = {"joints": list(range(n_joints)), "inverseBindMatrices": ibm_accessory, "skeleton": roots[0]}
    i_node_head = None
    if head:
        nodes.append({"name": "head", "mesh": i_head, "skin": 0})
        i_node_head = len(nodes) - 1
    if i_others is not None:
        nodes.append({"name": "body", "mesh": i_others, "skin": 0})
    scene = roots + [k for k in (i_node_head, len(nodes) - 1 if i_others is not None else None) if k is not None]
    # ---- VRM extension
    human = {v: {"node": idx_joint[b]} for v, b in S.human.items() if b in idx_joint}
    preset = {}
    if head:
        for expr, m in chosen.items():
            preset[expr] = {"morphTargetBinds": [{"node": i_node_head, "index": targets.index(id(m)), "weight": 1.0}], "isBinary": expr.startswith("blink"),
                            "overrideBlink": "none", "overrideLookAt": "none", "overrideMouth": "none"}
    ext = {"specVersion": "1.0", "meta": {
        "name": name, "version": "1.0", "authors": [author], "licenseUrl": "https://vrm.dev/licenses/1.0/", "avatarPermission": "onlyAuthor",
        "allowExcessivelyViolentUsage": False, "allowExcessivelySexualUsage": False, "commercialUsage": "personalNonProfit",
        "allowPoliticalOrReligiousUsage": False, "allowAntisocialOrHateUsage": False, "creditNotation": "required",
        "allowRedistribution": False, "modification": "prohibited"},
        "humanoid": {"humanBones": human}, "expressions": {"preset": preset}}
    doc = {"asset": {"version": "2.0", "generator": "Chimera"}, "scene": 0, "scenes": [{"nodes": scene}], "nodes": nodes,
           "meshes": meshes, "skins": [skin], "materials": materials, "accessors": B.accessory, "bufferViews": B.views,
           "buffers": [{"byteLength": 0}], "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}],
           "extensionsUsed": ["KHR_materials_unlit", "VRMC_materials_mtoon", "VRMC_vrm"], "extensions": {"VRMC_vrm": ext}}
    if images_doc:
        doc["images"], doc["textures"] = images_doc, textures
    if constraints:
        doc["extensionsUsed"].append("VRMC_node_constraint")
    while len(B.o) % 4:
        B.o.append(0)
    doc["buffers"][0]["byteLength"] = len(B.o)
    j = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    j += b" " * ((4 - len(j) % 4) % 4)
    total = 12 + 8 + len(j) + 8 + len(B.o)
    tmp = output + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(j), 0x4E4F534A) + j)
        f.write(struct.pack("<II", len(B.o), 0x004E4942) + bytes(B.o))
    os.replace(tmp, output)
    return {"vertices": int(len(X)), "bones": n_joints, "humanoids": len(human), "constraints": constraints, "expressions": len(preset), "materials": len(materials), "bytes": total}


def images_blender(P, folder, cap=2048):
    """PMX textures as PNG (in Blender): path written in P -> (bytes, mimetype, has holes). Maximum side length."""
    import tempfile
    import bpy
    out = {}
    tmp = tempfile.mkdtemp(prefix="vrm-", dir=os.environ.get("TMPDIR") or None)
    for k, t in enumerate(P["textures"]):
        ch = os.path.join(folder, t.replace("\\", "/"))
        if not os.path.isfile(ch):
            continue
        im = bpy.data.images.load(ch, check_existing=False)
        w, h = im.size
        a = np.empty(w * h * im.channels, np.float32)
        im.pixels.foreach_get(a)
        cutout = im.channels == 4 and float((a[3::4] < ALPHA_MASK).mean()) > PART_CUTOUT
        if max(w, h) > cap:
            r = cap / max(w, h)
            im.scale(int(w * r), int(h * r))
        png = os.path.join(tmp, "%d.png" % k)
        im.filepath_raw = png
        im.file_format = "PNG"
        im.save()
        bpy.data.images.remove(im)
        with open(png, "rb") as f:
            out[t] = (f.read(), "image/png", cutout)
        os.remove(png)
    os.rmdir(tmp)
    return out


def read(path):
    with open(path, "rb") as f:
        o = f.read()
    magic, version, _ = struct.unpack("<III", o[:12])
    n, ty = struct.unpack("<II", o[12:20])
    if magic != 0x46546C67 or version != 2 or ty != 0x4E4F534A:
        raise ValueError("not a GLB")
    doc = json.loads(o[20:20 + n].decode("utf-8"))
    return doc, o[28 + n:]


def validate(path):
    """Structure checks: returns the list of defects (empty if all is well)."""
    doc, b = read(path)
    defaults = []
    ext = doc.get("extensions", {}).get("VRMC_vrm")
    if not ext:
        return ["VRMC_vrm extension missing"]
    if ext.get("specVersion") != "1.0":
        defaults.append("specVersion %s" % ext.get("specVersion"))
    hb = ext["humanoid"]["humanBones"]
    for o in REQUIRED:
        if o not in hb:
            defaults.append("required bone missing: %s" % o)
    nodes = doc["nodes"]
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            if c in parent:
                defaults.append("node %d has two parents" % c)
            parent[c] = i
    if any(not (0 <= h["node"] < len(nodes)) for h in hb.values()):
        defaults.append("humanoid bone outside the nodes")
    n_chain = {"spine": "hips", "head": "spine", "leftUpperArm": "spine", "rightUpperArm": "spine", "leftLowerArm": "leftUpperArm", "rightLowerArm": "rightUpperArm",
                "leftHand": "leftLowerArm", "rightHand": "rightLowerArm", "leftUpperLeg": "hips", "rightUpperLeg": "hips", "leftLowerLeg": "leftUpperLeg", "rightLowerLeg": "rightUpperLeg",
                "leftFoot": "leftLowerLeg", "rightFoot": "rightLowerLeg"}

    def descends(a, b_):
        i, seen = hb[a]["node"], set()
        target = hb[b_]["node"]
        while i in parent and i not in seen:
            seen.add(i)
            i = parent[i]
            if i == target:
                return True
        return False
    for a, b_ in n_chain.items():
        if a in hb and b_ in hb and not descends(a, b_):
            defaults.append("%s does not descend from %s" % (a, b_))
    for i, n in enumerate(nodes):
        if any(k in n for k in ("rotation", "scale", "matrix")):
            defaults.append("node %d with rotation or scale (T-pose expected)" % i)
    accessory = doc["accessors"]
    views = doc["bufferViews"]
    for a in accessory:
        v = views[a["bufferView"]]
        if v.get("byteOffset", 0) + v["byteLength"] > len(b):
            defaults.append("accessor outside the buffer")
    n_morph = {}
    for mi, m in enumerate(doc["meshes"]):
        for pr in m["primitives"]:
            at = pr["attributes"]
            n = accessory[at["POSITION"]]["count"]
            for k in ("NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"):
                if accessory[at[k]]["count"] != n:
                    defaults.append("%s: different count" % k)
            jt = accessory[at["JOINTS_0"]]
            d = np.frombuffer(b, "<u2", jt["count"] * 4, views[jt["bufferView"]].get("byteOffset", 0))
            if d.size and int(d.max()) >= len(doc["skins"][0]["joints"]):
                defaults.append("joint index outside the skeleton")
            wt = accessory[at["WEIGHTS_0"]]
            w = np.frombuffer(b, "<f4", wt["count"] * 4, views[wt["bufferView"]].get("byteOffset", 0)).reshape(-1, 4).sum(1)
            if np.abs(w - 1).max() > 1e-3:
                defaults.append("weights not summing to 1 (deviation %.4f)" % float(np.abs(w - 1).max()))
            if pr.get("targets"):
                n_morph[mi] = len(pr["targets"])
    for name, e in ext.get("expressions", {}).get("preset", {}).items():
        for mb in e.get("morphTargetBinds", []):
            ni = doc["nodes"][mb["node"]].get("mesh")
            if ni is None or mb["index"] >= n_morph.get(ni, 0):
                defaults.append("expression %s: target not found" % name)
    for m in doc["materials"]:
        if "VRMC_materials_mtoon" in m.get("extensions", {}) and "KHR_materials_unlit" not in m.get("extensions", {}):
            defaults.append("material %s without unlit fallback" % m.get("name"))
    return defaults
