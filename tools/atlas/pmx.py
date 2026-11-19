"""PMX 2.0 / 2.1 reader in pure Python (struct), without mmd_tools.

`read(path)` returns a dict of Python lists (no dependency).
`to_arrays(pmx)` converts it into numpy arrays (Blender's numpy).

Material names: `read` is the only place where the core translates a game's names into the common language
(tools/games/README.md): a PMX stored in a source folder of a game whose description has a
"materials" table gets its names translated (the original name stays in "name_game"). Elsewhere, and for ZZZ, nothing changes.

Frame: positions stay in MMD units (left-handed, Y up, facing -Z).
The conversion to Blender is in `landmark.py`.
"""
import os
import struct
import sys

# Bone flags (PMX 2.0 specification)
OS_QUEUE_INDEX = 0x0001
OS_ROTATION = 0x0002
OS_DISPLACEMENT = 0x0004
OS_VISIBLE = 0x0008
OS_ACTIVE = 0x0010
OS_IK = 0x0020
OS_ADDITION_LOCAL = 0x0080
OS_ADDITION_ROT = 0x0100
OS_ADDITION_DISP = 0x0200
OS_AXE_FIXED = 0x0400
OS_AXES_LOCAL = 0x0800
OS_AFTER_PHYSICS = 0x1000
OS_PARENT_EXTERNAL = 0x2000

# Material flags
MAT_DOUBLE_FACE = 0x01
MAT_SHADE_GROUND = 0x02
MAT_PROJECTED = 0x04
MAT_RECEIVES = 0x08
MAT_OUTLINE = 0x10
MAT_COLOR_VERTEX = 0x20  # 2.1
MAT_POINTS = 0x40  # 2.1
MAT_LINES = 0x80  # 2.1

TYPES_WEIGHT = {0: "BDEF1", 1: "BDEF2", 2: "BDEF4", 3: "SDEF", 4: "QDEF"}
TYPES_MORPH = {0: "group", 1: "vertex", 2: "bones", 3: "uv", 4: "uv1", 5: "uv2",
                6: "uv3", 7: "uv4", 8: "material", 9: "flip", 10: "impulsion"}
PANELS = {0: "system", 1: "eyebrows", 2: "eyes", 3: "mouth", 4: "other"}
MODES_PHYSICS = {0: "suit_os", 1: "physics", 2: "physics_position_os"}
SHAPES_RIGID = {0: "sphere", 1: "box", 2: "capsule"}


class Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def lit(self, fmt):
        t = struct.calcsize("<" + fmt)
        v = struct.unpack_from("<" + fmt, self.d, self.p)
        self.p += t
        return v

    def un(self, fmt):
        return self.lit(fmt)[0]

    def bytes_(self, n):
        v = self.d[self.p:self.p + n]
        self.p += n
        return v


def _fmt_index(size, vertex=False):
    if vertex:
        return {1: "B", 2: "H", 4: "i"}[size]
    return {1: "b", 2: "h", 4: "i"}[size]


def _game_of_file(path):
    """The game (tools/games) whose "materials" table translates this file, otherwise None."""
    tools = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if tools not in sys.path:
        sys.path.append(tools)
    import games as games
    return games.game_of_file(path)


def read(path):
    with open(path, "rb") as f:
        data = f.read()
    r = Reader(data)
    if r.bytes_(4) != b"PMX ":
        raise ValueError("not a PMX file: %s" % path)
    version = round(r.un("f"), 2)
    n_glob = r.un("B")
    glob = list(r.lit("%dB" % n_glob))
    encoding, n_uv_add, t_vert, t_tex, t_mat, t_os, t_morph, t_rig = glob[:8]
    enc = "utf-16-le" if encoding == 0 else "utf-8"

    def text():
        n = r.un("i")
        return r.bytes_(n).decode(enc, errors="replace")

    fs = _fmt_index(t_vert, True)
    ft = _fmt_index(t_tex)
    fm = _fmt_index(t_mat)
    fo = _fmt_index(t_os)
    fmo = _fmt_index(t_morph)
    fr = _fmt_index(t_rig)

    pmx = {
        "version": version,
        "encoding": "UTF-16" if encoding == 0 else "UTF-8",
        "n_uv_add": n_uv_add,
        "sizes_index": {"vertex": t_vert, "texture": t_tex, "material": t_mat,
                          "bones": t_os, "morph": t_morph, "rigid": t_rig},
    }
    pmx["name"] = text()
    pmx["name_en"] = text()
    pmx["comment"] = text()
    pmx["comment_en"] = text()

    # --- Vertices
    n = r.un("i")
    pos, nor, uv, uva, typ, idx, weight, sdef, outline = [], [], [], [], [], [], [], [], []
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
            weight.append((1.0, 0.0, 0.0, 0.0))
            sdef.append(None)
        elif t == 1:
            b1, b2 = r.lit(fo * 2)
            w = r.un("f")
            idx.append((b1, b2, -1, -1))
            weight.append((w, 1.0 - w, 0.0, 0.0))
            sdef.append(None)
        elif t in (2, 4):
            idx.append(r.lit(fo * 4))
            weight.append(r.lit("4f"))
            sdef.append(None)
        elif t == 3:
            b1, b2 = r.lit(fo * 2)
            w = r.un("f")
            c = r.lit("9f")
            idx.append((b1, b2, -1, -1))
            weight.append((w, 1.0 - w, 0.0, 0.0))
            sdef.append(c)
        else:
            raise ValueError("unknown weight type %d" % t)
        outline.append(r.un("f"))
    pmx["vertices"] = {"pos": pos, "nor": nor, "uv": uv, "uv_add": uva, "type": typ,
                      "bones": idx, "weights": weight, "sdef": sdef, "contour": outline}

    # --- Faces
    n = r.un("i")
    ind = r.lit(fs * n)
    pmx["faces"] = [ind[i:i + 3] for i in range(0, n, 3)]

    # --- Textures
    n = r.un("i")
    pmx["textures"] = [text() for _ in range(n)]

    # --- Materials
    n = r.un("i")
    mats = []
    for _ in range(n):
        m = {"name": text(), "name_en": text()}
        m["diffuse"] = r.lit("4f")
        m["specular"] = r.lit("3f")
        m["brightness"] = r.un("f")
        m["ambient"] = r.lit("3f")
        m["flags"] = r.un("B")
        m["outline_color"] = r.lit("4f")
        m["outline_size"] = r.un("f")
        m["texture"] = r.un(ft)
        m["sphere"] = r.un(ft)
        m["sphere_mode"] = r.un("B")
        m["toon_sharing"] = r.un("B")
        m["toon"] = r.un("B") if m["toon_sharing"] == 1 else r.un(ft)
        m["memo"] = text()
        m["n_indices"] = r.un("i")
        mats.append(m)
    J = _game_of_file(path)
    if J is not None:
        for m in mats:
            t = J.translate(m["name"])
            if t != m["name"]:
                m["name_game"], m["name"] = m["name"], t
    pmx["materials"] = mats

    # --- Bones
    n = r.un("i")
    bones = []
    for _ in range(n):
        o = {"name": text(), "name_en": text()}
        o["pos"] = r.lit("3f")
        o["parent"] = r.un(fo)
        o["layer"] = r.un("i")
        fl = r.un("H")
        o["flags"] = fl
        if fl & OS_QUEUE_INDEX:
            o["tail_bone"] = r.un(fo)
            o["tail_offset"] = None
        else:
            o["tail_bone"] = None
            o["tail_offset"] = r.lit("3f")
        o["addition"] = None
        if fl & (OS_ADDITION_ROT | OS_ADDITION_DISP):
            src = r.un(fo)
            w = r.un("f")
            o["addition"] = {"source": src, "factor": w,
                          "rotation": bool(fl & OS_ADDITION_ROT),
                          "displacement": bool(fl & OS_ADDITION_DISP),
                          "local": bool(fl & OS_ADDITION_LOCAL)}
        o["fixed_axis"] = r.lit("3f") if fl & OS_AXE_FIXED else None
        o["local_axes"] = (r.lit("3f"), r.lit("3f")) if fl & OS_AXES_LOCAL else None
        o["external_parent"] = r.un("i") if fl & OS_PARENT_EXTERNAL else None
        o["ik"] = None
        if fl & OS_IK:
            target = r.un(fo)
            loops = r.un("i")
            limit = r.un("f")
            nl = r.un("i")
            links = []
            for _ in range(nl):
                b = r.un(fo)
                lim = r.un("B")
                if lim:
                    links.append({"bones": b, "min": r.lit("3f"), "max": r.lit("3f")})
                else:
                    links.append({"bones": b, "min": None, "max": None})
            o["ik"] = {"target": target, "loops": loops, "angle_limit": limit, "links": links}
        o["after_physics"] = bool(fl & OS_AFTER_PHYSICS)
        bones.append(o)
    pmx["bones"] = bones

    # --- Morphs
    n = r.un("i")
    morphs = []
    for _ in range(n):
        m = {"name": text(), "name_en": text()}
        m["panel"] = r.un("B")
        t = r.un("B")
        m["type"] = t
        k = r.un("i")
        offset = []
        for _ in range(k):
            if t in (0, 9):
                offset.append((r.un(fmo), r.un("f")))
            elif t == 1:
                offset.append((r.un(fs), r.lit("3f")))
            elif t == 2:
                offset.append((r.un(fo), r.lit("3f"), r.lit("4f")))
            elif 3 <= t <= 7:
                offset.append((r.un(fs), r.lit("4f")))
            elif t == 8:
                mi = r.un(fm)
                op = r.un("B")
                offset.append({"material": mi, "operation": op, "values": r.lit("28f")})
            elif t == 10:
                offset.append((r.un(fr), r.un("B"), r.lit("3f"), r.lit("3f")))
            else:
                raise ValueError("unknown morph type %d" % t)
        m["offsets"] = offset
        morphs.append(m)
    pmx["morphs"] = morphs

    # --- Display frames
    n = r.un("i")
    frames = []
    for _ in range(n):
        c = {"name": text(), "name_en": text(), "special": r.un("B")}
        k = r.un("i")
        el = []
        for _ in range(k):
            ty = r.un("B")
            el.append((ty, r.un(fo) if ty == 0 else r.un(fmo)))
        c["elements"] = el
        frames.append(c)
    pmx["frames"] = frames

    # --- Rigid bodies
    n = r.un("i")
    rig = []
    for _ in range(n):
        c = {"name": text(), "name_en": text()}
        c["bones"] = r.un(fo)
        c["group"] = r.un("B")
        c["mask"] = r.un("H")
        c["shape"] = r.un("B")
        c["size"] = r.lit("3f")
        c["pos"] = r.lit("3f")
        c["rot"] = r.lit("3f")
        c["mass"] = r.un("f")
        c["damping_disp"] = r.un("f")
        c["damping_rot"] = r.un("f")
        c["restitution"] = r.un("f")
        c["friction"] = r.un("f")
        c["mode"] = r.un("B")
        rig.append(c)
    pmx["rigid_bodies"] = rig

    # --- Joints
    n = r.un("i")
    jts = []
    for _ in range(n):
        j = {"name": text(), "name_en": text()}
        j["type"] = r.un("B")
        j["a"] = r.un(fr)
        j["b"] = r.un(fr)
        j["pos"] = r.lit("3f")
        j["rot"] = r.lit("3f")
        j["disp_min"] = r.lit("3f")
        j["disp_max"] = r.lit("3f")
        j["rot_min"] = r.lit("3f")
        j["rot_max"] = r.lit("3f")
        j["spring_disp"] = r.lit("3f")
        j["spring_rot"] = r.lit("3f")
        jts.append(j)
    pmx["joints"] = jts
    pmx["bytes_remaining"] = len(data) - r.p
    return pmx


def to_arrays(pmx):
    """Numpy arrays of the vertices and faces (MMD units and frame)."""
    import numpy as np
    s = pmx["vertices"]
    t = {
        "pos": np.asarray(s["pos"], dtype=np.float64),
        "nor": np.asarray(s["nor"], dtype=np.float64),
        "uv": np.asarray(s["uv"], dtype=np.float64),
        "weight_type": np.asarray(s["type"], dtype=np.int8),
        "bones": np.asarray(s["bones"], dtype=np.int32),
        "weights": np.asarray(s["weights"], dtype=np.float64),
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
    for i, m in enumerate(pmx["materials"]):
        k = m["n_indices"] // 3
        mat[deb:deb + k] = i
        deb += k
    t["mat_face"] = mat
    return t


def resume_json(pmx):
    """Everything except the heavy geometry, serializable to JSON."""
    out = {k: v for k, v in pmx.items() if k not in ("vertices", "faces")}
    out["n_vertices"] = len(pmx["vertices"]["pos"])
    out["n_faces"] = len(pmx["faces"])
    # vertex morph offsets are heavy: they are kept in the npz
    morphs = []
    for m in pmx["morphs"]:
        mm = dict(m)
        if m["type"] in (1, 3, 4, 5, 6, 7):
            mm["offsets"] = None
            mm["n_offsets"] = len(m["offsets"])
        else:
            mm["n_offsets"] = len(m["offsets"])
        morphs.append(mm)
    out["morphs"] = morphs
    return out
