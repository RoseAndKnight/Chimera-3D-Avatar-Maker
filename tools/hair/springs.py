"""Godot bone springs (SpringBoneSimulator3D) converted into quantities comparable with the
physics rules of ZZZ hairstyles (CONSTRUCTION.md, P1-P16), without simulating.

Input: Godot scene and resource files containing SpringBoneSimulator3D nodes and their
SpringBoneCollision{Sphere,Capsule,Plane}3D children, or a JSON in the format of `lire()`:

    {"simulators": [{"name": "...", "settings": [{"root_bone_name": "...", "end_bone_name": "...",
       "radius": 0.02, "stiffness": 1.0, "drag": 0.4, "gravity": 0.0,
       "gravity_direction": [0, -1, 0], "extend_end_bone": false, "end_bone_length": 0.0,
       "enable_all_child_collisions": true, "exclude_collisions": [], "collisions": [],
       "joints": [{"bone_name": "...", "radius": ..., "stiffness": ..., ...}]}],
       "collisions": [{"name": "...", "type": "sphere|capsule|plane", "bone_name": "...",
       "position_offset": [x, y, z], "rotation_offset": [x, y, z, w], "radius": 0.1,
       "height": 0.5, "inside": false}]}]}

Godot 4.5+ model (spring_bone_simulator_3d.cpp), per time step dt:
    tail' = tail + (tail - previous_tail)*(1 - drag) + rot*rest_axis*(stiffness*dt)
            + gravity_direction*(gravity*dt)
then the tail is brought back to the bone length, then pushed out of the collisions (sphere of
radius `radius` around the tail). No mass, no angle limits, no collision between strands.
`drag` is removed at every step (not multiplied by dt); stiffness and gravity are
displacements per second, in world units.

Conversions (all in the MMD frame of measure.py, lengths in H):
- damping: velocity kept per second at 60 steps/s = (1 - drag)^60, hence an equivalent Bullet
  damping of 1 - (1 - drag)^60 (Bullet keeps (1 - d) per second). ZZZ 0.999 <-> drag 0.109.
- rest sag of a horizontal bone: atan(gravity / stiffness) (balance between the restoring
  push and the gravity push, both per second).
- restoring rate: stiffness / bone length, in rad/s (angular return speed for a small offset).
- radius: joint radius / half-width of the strand at that bone (same measure as P7).
- collisions: shapes, radii in H, attached to the head or not (P10, P14), overlap at rest
  (P16), hair inside the head collisions at rest ("nothing inside the head" check).
Assumption: the skeleton is at scale 1 in the scene (Godot units = GLB units)."""
import json
import math
import os
import re

import numpy as np

DEFAULTS = {"radius": 0.02, "stiffness": 1.0, "drag": 0.4, "gravity": 0.0,
           "gravity_direction": [0.0, -1.0, 0.0], "extend_end_bone": False, "end_bone_length": 0.0,
           "enable_all_child_collisions": True}
DEFAULTS_COLLISION = {"radius": 0.1, "height": 0.5, "inside": False,
                     "position_offset": [0.0, 0.0, 0.0], "rotation_offset": [0.0, 0.0, 0.0, 1.0]}
STEP_PER_SECOND = 60
TYPES_COLLISION = {"SpringBoneCollisionSphere3D": "sphere", "SpringBoneCollisionCapsule3D": "capsule",
                   "SpringBoneCollisionPlane3D": "plane"}

NON_COMPARABLE = {
    "P8": "masses: the Godot spring has no mass (gravity is a displacement per second, "
          "the same for every link).",
    "P11": "joint limits: Godot has no angle limit; only collision holds the strand back.",
    "P12": "joint springs: ZZZ has zero springs and Bullet carries the strand; Godot pulls "
           "each bone back to its rest pose (stiffness). Different model: we give the sag "
           "atan(g/k) and the restoring rate k/L, without a verdict.",
    "P13": "錘 weights: no equivalent (no mass).",
    "P15": "worst pose within the limits: with no limits, collision with the head is required "
           "for every chain (P10).",
    "gravity": "Bullet applies an acceleration (MMD, 98 u/s²) to bodies with mass; Godot a "
               "displacement per second. Only the rest sag can be compared, by hand.",
    "lock-lock collision": "absent in Godot (ZZZ: a hair group that ignores itself, "
                             "so absent too: P10 holds by construction).",
}


# ============================================================ reading Godot scenes

def _value(txt):
    t = txt.strip()
    if t in ("true", "false"):
        return t == "true"
    if t == "null":
        return None
    m = re.fullmatch(r'&?"(.*)"', t, re.S)
    if m:
        return m.group(1)
    try:
        return float(t) if any(ch in t for ch in ".eE") or t in ("inf", "-inf", "nan") else int(t)
    except ValueError:
        pass
    m = re.fullmatch(r"(\w+)\((.*)\)", t, re.S)
    if m:
        f, a = m.groups()
        if f in ("SubResource", "ExtResource", "NodePath", "StringName"):
            return {"_" + f: _value(a)}
        parts = [p.strip() for p in _slice(a)] if a.strip() else []
        try:
            return [float(p) for p in parts]
        except ValueError:
            return [_value(p) for p in parts]
    if t.startswith("["):
        return [_value(p) for p in _slice(t[1:-1])] if t[1:-1].strip() else []
    return t


def _slice(a):
    """Split "a, B(c, d), [e, f]" at the top level."""
    out, depth, cur, chain = [], 0, "", False
    for ch in a:
        if ch == '"':
            chain = not chain
        if not chain:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                out.append(cur)
                cur = ""
                continue
        cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _sections(text):
    """Sections of a .tscn/.tres: [(header, header attributes, {key: value})]."""
    secs, cur, buffer = [], None, ""
    for line in text.splitlines():
        if buffer:
            buffer += "\n" + line
        elif re.match(r"^\[(\w+)(.*)\]\s*$", line):
            m = re.match(r"^\[(\w+)(.*)\]\s*$", line)
            att = {k: v for k, v in re.findall(r'(\w+)=(?:&?"([^"]*)"|[^\s\]]+)', m.group(2))}
            att.update({k: v for k, v in re.findall(r'(\w+)="([^"]*)"', m.group(2))})
            cur = (m.group(1), att, {})
            secs.append(cur)
            continue
        elif "=" in line and cur is not None:
            buffer = line
        else:
            continue
        k, v = buffer.split("=", 1)
        if v.count("(") + v.count("[") + v.count("{") > v.count(")") + v.count("]") + v.count("}"):
            continue          # value spanning several lines
        cur[2][k.strip()] = _value(v)
        buffer = ""
    return secs


def _curve(sr):
    """(x, y) points of a Curve resource (tangents ignored: linear interpolation)."""
    d = sr.get("_data", [])
    pts = []
    for k in range(0, len(d), 5):
        v = d[k]
        if isinstance(v, list) and len(v) == 2:
            pts.append((float(v[0]), float(v[1])))
    return sorted(pts) or None


def _path_node(att):
    p = att.get("parent")
    if p is None:
        return "."
    return att["name"] if p == "." else p + "/" + att["name"]


def read_tscn(path):
    with open(path, encoding="utf-8") as fh:
        secs = _sections(fh.read())
    under = {a.get("id"): v for t, a, v in secs if t == "sub_resource"}
    sims, cols = [], []
    for t, a, v in secs:
        if t != "node":
            continue
        typ = a.get("type", "")
        if typ == "SpringBoneSimulator3D":
            sims.append((_path_node(a), a["name"], v))
        elif typ in TYPES_COLLISION:
            cols.append((a.get("parent", "."), a["name"], typ, v))
    out = []
    for pth, name, v in sims:
        n = int(v.get("setting_count", 0))
        settings = []
        for i in range(n):
            pre = "settings/%d/" % i
            s = {k[len(pre):]: x for k, x in v.items() if k.startswith(pre)}
            r = {"root_bone_name": s.get("root_bone_name"), "end_bone_name": s.get("end_bone_name"),
                 "defaults": []}
            for the_key, key_g in (("radius", "radius/value"), ("stiffness", "stiffness/value"),
                               ("drag", "drag/value"), ("gravity", "gravity/value"),
                               ("gravity_direction", "gravity/direction"),
                               ("extend_end_bone", "extend_end_bone"), ("end_bone_length", "end_bone/length"),
                               ("enable_all_child_collisions", "enable_all_child_collisions")):
                if key_g in s:
                    r[the_key] = s[key_g]
                else:
                    r[the_key] = DEFAULTS[the_key]
                    r["defaults"].append(the_key)
            for the_key in ("radius", "stiffness", "drag", "gravity"):
                c_ = s.get(the_key + "/damping_curve")
                if isinstance(c_, dict) and "_SubResource" in c_:
                    r[the_key + "_curve"] = _curve(under.get(c_["_SubResource"], {}))
            r["individual_config"] = bool(s.get("individual_config", False))
            r["rotation_axis"] = s.get("rotation_axis")
            nj = int(s.get("joint_count", 0))
            r["joints"] = []
            for j in range(nj):
                pj = "joints/%d/" % j
                jd = {k[len(pj):]: x for k, x in s.items() if k.startswith(pj)}
                r["joints"].append({"bone_name": jd.get("bone_name"), "radius": jd.get("radius"),
                                    "stiffness": jd.get("stiffness"), "drag": jd.get("drag"),
                                    "gravity": jd.get("gravity"), "gravity_direction": jd.get("gravity_direction")})
            r["exclude_collisions"] = [x.get("_NodePath") if isinstance(x, dict) else x
                                       for k, x in sorted(s.items()) if k.startswith("exclude_collisions/")]
            r["collisions"] = [x.get("_NodePath") if isinstance(x, dict) else x
                               for k, x in sorted(s.items()) if k.startswith("collisions/")]
            settings.append(r)
        collisions = []
        for per, nc, typ, cv in cols:
            if per not in (pth, name) and not per.endswith("/" + name):
                continue
            collar = {"name": nc, "type": TYPES_COLLISION[typ], "bone_name": cv.get("bone_name"),
                   "defaults": [k for k in DEFAULTS_COLLISION if k not in cv]}
            for k, d in DEFAULTS_COLLISION.items():
                collar[k] = cv.get(k, d)
            if "transform" in cv and not collar["bone_name"]:
                collar["transform"] = cv["transform"]
            collisions.append(collar)
        out.append({"file": path, "name": name, "path": pth, "settings": settings,
                    "collisions": collisions})
    return out


def read(paths):
    sims = []
    for ch in paths:
        if ch.endswith(".json"):
            with open(ch, encoding="utf-8") as fh:
                d = json.load(fh)
            for s in d.get("simulators", []):
                s.setdefault("file", ch)
                for r in s.get("settings", []):
                    r.setdefault("defaults", [k for k in DEFAULTS if k not in r])
                    for k, v in DEFAULTS.items():
                        r.setdefault(k, v)
                    r.setdefault("joints", [])
                    r.setdefault("exclude_collisions", [])
                    r.setdefault("collisions", [])
                for collar in s.get("collisions", []):
                    collar.setdefault("defaults", [k for k in DEFAULTS_COLLISION if k not in collar])
                    for k, v in DEFAULTS_COLLISION.items():
                        collar.setdefault(k, v)
                sims.append(s)
        else:
            sims += read_tscn(ch)
    return sims


# ============================================================ geometry

def _norm(n):
    return re.sub(r"[^0-9a-z]", "", (n or "").lower())


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _sample(curve, t):
    if not curve:
        return 1.0
    xs, ys = zip(*curve)
    return float(np.interp(t, xs, ys))


def _shape(c, collar, os_idx):
    """Shape of a collision in the measurement frame: (type, a, b, radius, normal, inside)."""
    O, u = c.p["bones"], c.unit
    if os_idx is not None:
        o = O[os_idx]
        base, samp, org = np.asarray(o["base"]), o["scale_os"], np.asarray(o["pos"])
    else:   # no bone: node transform (skeleton frame assumed)
        T = collar.get("transform") or [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        B = np.array(T[:9]).reshape(3, 3).T
        base, samp = c.rotation @ B, 1.0
        org = (np.asarray(T[9:12]) - c.origin) @ c.rotation.T * u
    rot = base @ _quat_mat(collar["rotation_offset"])
    center = org + base @ np.asarray(collar["position_offset"], float) * samp * u
    r = float(collar["radius"]) * samp * u
    if collar["type"] == "sphere":
        return "sphere", center, center, r, None, bool(collar["inside"])
    if collar["type"] == "capsule":
        half = max(float(collar["height"]) / 2 - float(collar["radius"]), 0.0) * samp * u
        ax = rot[:, 1]
        return "capsule", center - ax * half, center + ax * half, r, None, bool(collar["inside"])
    return "plane", center, center, 0.0, rot[:, 1], False


def _distance(shape, P):
    """Signed distance from points P to the surface (negative = inside, forbidden side)."""
    typ, a, b, r, n, inside = shape
    P = np.atleast_2d(P)
    if typ == "plane":
        return (P - a) @ n
    ab = b - a
    L2 = float(ab @ ab)
    t = np.clip(((P - a) @ ab) / L2, 0, 1) if L2 > 0 else np.zeros(len(P))
    d = np.linalg.norm(P - (a + np.outer(t, ab)), axis=1) - r
    return -d if inside else d


# ============================================================ measurement

def _find_os(c, name, index_norm):
    if name is None:
        return None
    if name in c.names_os:
        return c.names_os.index(name)
    return index_norm.get(_norm(name))


def _q(v, n=3):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None
    return {"n": len(v), "min": round(float(np.min(v)), n), "p50": round(float(np.median(v)), n),
            "max": round(float(np.max(v)), n)}


def measure(c, paths):
    """Godot bone springs of a GLB hairstyle (c: an already measured measure.CoiffureGLB)."""
    sims = read(paths)
    H, u = c.H, c.unit
    O = c.p["bones"]
    index_norm = {}
    for k, n in enumerate(c.names_os):
        index_norm.setdefault(_norm(n), k)
    children = {}
    for i, o in enumerate(O):
        children.setdefault(o["parent"], []).append(i)
    lest = np.array([("錘" in n or "锤" in n) for n in c.names_os])
    oc = c.mask_os & ~lest
    Ph = c.pos[c.vh]
    dom = c.dom
    top = Ph[:, 1] > c.y_eyes
    i_head = c.i_head
    head_desc = set()
    if i_head >= 0:     # head bones: the head and its descendants outside the hairstyle
        pile = [i_head]
        while pile:
            i = pile.pop()
            if not oc[i]:
                head_desc.add(i)
                pile += children.get(i, [])

    def half_width(i, j):
        vv = Ph[dom == i]
        if j is None or len(vv) < 6:
            return None
        a, b = np.asarray(O[i]["pos"]), np.asarray(O[j]["pos"])
        ax = b - a
        if np.linalg.norm(ax) < 1e-9:
            return None
        ax /= np.linalg.norm(ax)
        rel = vv - a
        perp = rel - np.outer(rel @ ax, ax)
        return float(np.percentile(np.linalg.norm(perp, axis=1), 90))

    settings_out, covered, not_found = [], set(), []
    all_collisions = []
    for s in sims:
        shapes = {}
        for collar in s.get("collisions", []):
            io = _find_os(c, collar.get("bone_name"), index_norm)
            if collar.get("bone_name") and io is None:
                not_found.append(collar["bone_name"])
            f = _shape(c, collar, io)
            on_head = (io is not None and io in head_desc) or (
                io is None and f[0] != "plane" and np.linalg.norm(f[1] - c.C) < c.R)
            shapes[collar["name"]] = (f, on_head, collar, io)
            all_collisions.append({
                "simulator": s["name"], "name": collar["name"], "shape": collar["type"],
                "bones": c.names_os[io] if io is not None else collar.get("bone_name"),
                "on_head": bool(on_head), "rayon_H": round(f[3] / H, 3),
                "length_H": round(float(np.linalg.norm(f[2] - f[1])) / H, 3),
                "inside": f[5], "center_to_skull_H": round(float(np.linalg.norm(0.5 * (f[1] + f[2]) - c.C)) / H, 3),
                "defaults": collar.get("defaults", [])})
        for r in s.get("settings", []):
            ir = _find_os(c, r.get("root_bone_name"), index_norm)
            ie = _find_os(c, r.get("end_bone_name"), index_norm)
            if ir is None or ie is None:
                not_found += [n for n, i in ((r.get("root_bone_name"), ir), (r.get("end_bone_name"), ie)) if i is None]
                settings_out.append({"simulator": s["name"], "racine": r.get("root_bone_name"),
                                     "done": r.get("end_bone_name"), "error": "bone not found in the GLB"})
                continue
            ch, k = [ie], ie
            while k != ir and c.per[k] >= 0:
                k = c.per[k]
                ch.append(k)
            ch = ch[::-1]
            if ch[0] != ir:
                settings_out.append({"simulator": s["name"], "racine": c.names_os[ir], "done": c.names_os[ie],
                                     "error": "the end bone does not descend from the root"})
                continue
            covered.update(ch[1:] if len(ch) > 1 else ch)
            pos = np.array([O[i]["pos"] for i in ch])
            tails = list(pos[1:])
            if r.get("extend_end_bone") and float(r.get("end_bone_length") or 0) > 0:
                ax = np.asarray(O[ie]["base"])[:, 1]       # bone +Y axis, as in Godot (default end_bone/direction)
                tails.append(pos[-1] + ax * float(r["end_bone_length"]) * O[ie]["scale_os"] * u)
            seg = np.linalg.norm(np.diff(np.vstack([pos[:1], np.array(tails)]) if tails else pos, axis=0), axis=1)
            L = float(seg.sum())
            t_long = np.cumsum(seg) / L if L > 0 else np.zeros(len(seg))
            # per-joint values (individual_config, or value x curve along the chain)
            per_j = {}
            for the_key in ("radius", "stiffness", "drag", "gravity"):
                if r.get("individual_config") and r.get("joints"):
                    vals = [j.get(the_key) if j.get(the_key) is not None else r[the_key] for j in r["joints"]]
                    vals = (vals + [vals[-1]] * len(seg))[:len(seg)] if vals else [r[the_key]] * len(seg)
                else:
                    vals = [float(r[the_key]) * _sample(r.get(the_key + "_curve"), t) for t in t_long]
                per_j[the_key] = [float(x) for x in vals]
            rad = np.array(per_j["radius"]) * u          # joint radius, measurement units
            # radius / strand half-width (P7)
            reports = []
            for n_, i in enumerate(ch[:-1]):
                dl = half_width(i, ch[n_ + 1])
                if dl and dl > 0:
                    reports.append(rad[min(n_, len(rad) - 1)] / dl)
            # collisions active for this setting
            if r.get("enable_all_child_collisions", True):
                actives = [n for n in shapes if n not in {str(x).split("/")[-1] for x in r.get("exclude_collisions", [])}]
            else:
                actives = [n for n in shapes if n in {str(x).split("/")[-1] for x in r.get("collisions", [])}]
            head_act = [n for n in actives if shapes[n][1]]
            # overlap at rest: sphere of each tail against each collision (P16)
            rec, rec_head = [], []
            Q = np.array(tails) if tails else pos[-1:]
            for n in actives:
                f = shapes[n][0]
                d = (_distance(f, Q) - rad[:len(Q)]) / H
                rec.append(float(d.min()))
                if shapes[n][1]:
                    rec_head.append(float(d.min()))
            k_ = np.array(per_j["stiffness"])
            g_ = np.array(per_j["gravity"])
            dr = np.array(per_j["drag"])
            seg_godot = seg / u                       # bone length in Godot units
            # carried vertices: dominant bone in the chain or below it
            under, kk = list(ch), 0
            while kk < len(under):
                under += [x for x in children.get(under[kk], []) if x not in under]
                kk += 1
            settings_out.append({
                "simulator": s["name"], "racine": c.names_os[ir], "done": c.names_os[ie],
                "bones": [c.names_os[i] for i in ch], "chain_links": len(seg), "length_H": round(L / H, 3),
                "spacing_H": [round(float(x) / H, 3) for x in seg],
                "root_dist_skull_H": round(float((np.linalg.norm(pos[0] - c.C) - c.R) / H), 3),
                "vertices_worn": int(np.isin(dom, under).sum()),
                "rayon_H": _q(list(rad / H)), "radius_on_half_width": _q(reports, 2),
                "stiffness": _q(list(k_)), "drag": _q(list(dr)), "gravity": _q(list(g_)),
                "gravity_direction": r.get("gravity_direction"),
                "damping_bullet_equiv": _q(list(1 - (1 - np.clip(dr, 0, 1)) ** STEP_PER_SECOND), 4),
                "speed_kept_per_s": _q(list((1 - np.clip(dr, 0, 1)) ** STEP_PER_SECOND), 4),
                "sag_rest_deg": _q([math.degrees(math.atan2(g, k)) for g, k in zip(g_, k_)], 1),
                "recall_rad_s": _q(list(k_ / np.maximum(seg_godot, 1e-9)), 2),
                "collisions": actives, "collisions_head": head_act, "touched_head": bool(head_act),
                "overlap_rest_min_H": round(min(rec), 3) if rec else None,
                "overlap_rest_head_min_H": round(min(rec_head), 3) if rec_head else None,
                "curves": [k for k in ("radius", "stiffness", "drag", "gravity") if r.get(k + "_curve")],
                "individual_config": bool(r.get("individual_config")), "defaults": r.get("defaults", [])})
    # hair inside the head collisions at rest (union of the shapes attached to the head)
    shapes_head = []
    for s in sims:
        for collar in s.get("collisions", []):
            io = _find_os(c, collar.get("bone_name"), index_norm)
            f = _shape(c, collar, io)
            if f[0] != "plane" and ((io is not None and io in head_desc) or
                                    (io is None and np.linalg.norm(f[1] - c.C) < c.R)):
                shapes_head.append(f)
    if shapes_head:
        dmin = np.min([_distance(f, Ph) for f in shapes_head], axis=0)
        inside = round(float((dmin < 0).mean()), 3)
        gap = {k: round(float(v), 3) for k, v in zip(("p10", "p50", "p90"), np.percentile(dmin[top] / H, [10, 50, 90]))} if top.any() else None
    else:
        inside, gap = None, None
    # coverage: hairstyle chain bones driven by a spring, and the vertices they carry
    os_ch = np.where(oc)[0]
    cov_os = float(np.isin(os_ch, list(covered)).mean()) if len(os_ch) else None
    dyn_v = np.isin(dom, list(covered))
    ok = [x for x in settings_out if "error" not in x]
    return {
        "files": sorted({s["file"] for s in sims}),
        "simulators": [s["name"] for s in sims], "settings": settings_out, "count_settings": len(ok),
        "collisions": all_collisions, "os_not_found": sorted(set(not_found)),
        "hairstyle_bones_covered_part": round(cov_os, 3) if cov_os is not None else None,
        "vertices_on_spring_part": round(float(dyn_v.mean()), 3),
        "settings_touching_head_part": round(float(np.mean([x["touched_head"] for x in ok])), 3) if ok else None,
        "hair_in_collision_head_part": inside, "cap_gap_collision_head_H": gap,
        "step_per_second_suppose": STEP_PER_SECOND,
        "scale_skeleton_assumed": 1.0,
        "non_comparable": NON_COMPARABLE}
