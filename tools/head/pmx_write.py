"""PMX 2.0 / 2.1 writer, exact mirror of `tools/atlas/pmx.lire`.

`write(pmx, path)` takes the dict returned by `pmx.lire` (possibly modified:
vertices, faces, materials, morphs...) and writes it. Index sizes are recomputed
from the content; the encoding is kept. A round trip lire -> write -> lire returns
the same values (up to 32-bit floats).
"""
import os
import struct
import tempfile

import pmx as PMX


def _size(n, vertex=False):
    if vertex:
        return 1 if n < 256 else 2 if n < 65536 else 4
    return 1 if n < 128 else 2 if n < 32768 else 4


def _fmt(size, vertex=False):
    return PMX._fmt_index(size, vertex)


def write(pmx, path):
    enc = "utf-16-le" if pmx.get("encoding", "UTF-16") == "UTF-16" else "utf-8"   # fr-ok: codec name
    s = pmx["vertices"]
    n_uv_add = pmx["n_uv_add"]
    t_vert = _size(len(s["pos"]), True)
    t_tex = _size(len(pmx["textures"]))
    t_mat = _size(len(pmx["materials"]))
    t_os = _size(len(pmx["bones"]))
    t_mor = _size(len(pmx["morphs"]))
    t_rig = _size(len(pmx["rigid_bodies"]))
    fs, ft, fm, fo, fmo, fr = (_fmt(t_vert, True), _fmt(t_tex), _fmt(t_mat), _fmt(t_os),
                               _fmt(t_mor), _fmt(t_rig))
    out = bytearray()

    def w(fmt, *v):
        out.extend(struct.pack("<" + fmt, *v))

    def text(x):
        b = (x or "").encode(enc)
        w("i", len(b))
        out.extend(b)

    out.extend(b"PMX ")
    w("f", float(pmx.get("version", 2.0)))
    w("B", 8)
    w("8B", 0 if enc == "utf-16-le" else 1, n_uv_add, t_vert, t_tex, t_mat, t_os, t_mor, t_rig)   # fr-ok: codec name
    text(pmx.get("name", ""))
    text(pmx.get("name_en", ""))
    text(pmx.get("comment", ""))
    text(pmx.get("comment_en", ""))

    # --- vertices
    n = len(s["pos"])
    w("i", n)
    for i in range(n):
        w("8f", *s["pos"][i], *s["nor"][i], *s["uv"][i])
        for k in range(n_uv_add):
            w("4f", *s["uv_add"][i][k])
        t = int(s["type"][i])
        w("B", t)
        o, p = s["bones"][i], s["weights"][i]
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
        text(x)

    # --- materials
    w("i", len(pmx["materials"]))
    for m in pmx["materials"]:
        text(m["name"])
        text(m["name_en"])
        w("4f", *m["diffuse"])
        w("3f", *m["specular"])
        w("f", m["brightness"])
        w("3f", *m["ambient"])
        w("B", m["flags"])
        w("4f", *m["outline_color"])
        w("f", m["outline_size"])
        w(ft, m["texture"])
        w(ft, m["sphere"])
        w("B", m["sphere_mode"])
        w("B", m["toon_sharing"])
        if m["toon_sharing"] == 1:
            w("B", m["toon"])
        else:
            w(ft, m["toon"])
        text(m["memo"])
        w("i", m["n_indices"])

    # --- bones
    w("i", len(pmx["bones"]))
    for o in pmx["bones"]:
        text(o["name"])
        text(o["name_en"])
        w("3f", *o["pos"])
        w(fo, o["parent"])
        w("i", o["layer"])
        f = o["flags"]
        w("H", f)
        if f & PMX.OS_QUEUE_INDEX:
            w(fo, o["tail_bone"])
        else:
            w("3f", *o["tail_offset"])
        if f & (PMX.OS_ADDITION_ROT | PMX.OS_ADDITION_DISP):
            w(fo, o["addition"]["source"])
            w("f", o["addition"]["factor"])
        if f & PMX.OS_AXE_FIXED:
            w("3f", *o["fixed_axis"])
        if f & PMX.OS_AXES_LOCAL:
            w("3f", *o["local_axes"][0])
            w("3f", *o["local_axes"][1])
        if f & PMX.OS_PARENT_EXTERNAL:
            w("i", o["external_parent"])
        if f & PMX.OS_IK:
            ik = o["ik"]
            w(fo, ik["target"])
            w("i", ik["loops"])
            w("f", ik["angle_limit"])
            w("i", len(ik["links"]))
            for l in ik["links"]:
                w(fo, l["bones"])
                if l["min"] is not None:
                    w("B", 1)
                    w("3f", *l["min"])
                    w("3f", *l["max"])
                else:
                    w("B", 0)

    # --- morphs
    w("i", len(pmx["morphs"]))
    for m in pmx["morphs"]:
        text(m["name"])
        text(m["name_en"])
        w("B", m["panel"])
        t = m["type"]
        w("B", t)
        w("i", len(m["offsets"]))
        for d in m["offsets"]:
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
                w(fm, d["material"])
                w("B", d["operation"])
                w("28f", *d["values"])
            elif t == 10:
                w(fr, d[0])
                w("B", d[1])
                w("3f", *d[2])
                w("3f", *d[3])

    # --- display frames
    w("i", len(pmx["frames"]))
    for c in pmx["frames"]:
        text(c["name"])
        text(c["name_en"])
        w("B", c["special"])
        w("i", len(c["elements"]))
        for ty, ix in c["elements"]:
            w("B", ty)
            w(fo if ty == 0 else fmo, ix)

    # --- rigid bodies
    w("i", len(pmx["rigid_bodies"]))
    for c in pmx["rigid_bodies"]:
        text(c["name"])
        text(c["name_en"])
        w(fo, c["bones"])
        w("B", c["group"])
        w("H", c["mask"])
        w("B", c["shape"])
        w("3f", *c["size"])
        w("3f", *c["pos"])
        w("3f", *c["rot"])
        for k in ("mass", "damping_disp", "damping_rot", "restitution", "friction"):
            w("f", c[k])
        w("B", c["mode"])

    # --- joints
    w("i", len(pmx["joints"]))
    for j in pmx["joints"]:
        text(j["name"])
        text(j["name_en"])
        w("B", j["type"])
        w(fr, j["a"])
        w(fr, j["b"])
        for k in ("pos", "rot", "disp_min", "disp_max", "rot_min", "rot_max", "spring_disp",
                  "spring_rot"):
            w("3f", *j[k])

    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(out)
    os.replace(tmp, path)
