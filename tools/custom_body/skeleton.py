"""The house skeleton: bones of our own, placed by measurement on our mesh, with their twist.

    python3 -I skeleton.py [--output skeleton.json]      # places the bones, compares them with those of ZZZ

What sets it apart from VRoid and ZZZ:
 - the joints are placed by measurements on the rings of the grid (measurements.py) and by
   anatomy rules written here, not taken from another model;
 - the twist is in the skeleton: the forearm follows the roll of the hand, the upper arm and the thigh
   hold back the roll of their bone (VRoid has no twist bones; those of ZZZ are turned by hand);
 - one bone per toe, under a common bone "toes" (toes) that keeps the usual role;
 - readable names: pelvis, spine, arm_L, forearm_L, index1_L...

Frame: cm, y up, front = -z, character's left = +x. As in MMD, the rest frame of each
bone is aligned with the world: a local rotation is a rotation about the world axes, under the parent.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "atlas")):
    if d not in sys.path:
        sys.path.insert(0, d)
import kinematics as KIN     # noqa: E402
import measurements as M           # noqa: E402

FINGERS = ("index", "middle", "ring", "little")
# anatomy rules (fractions of length), each one stated where it is used
PART_ELBOW = 0.54          # shoulder -> elbow, over shoulder -> wrist (humerus 0.186 H; forearm 0.146 H... rounded for the posed arm)
PART_PHALANGES = (0.45, 0.28, 0.27)   # the three phalanges, fingertip included
PART_FINGER_FREE = 0.75    # the web is in the middle of the first phalanx: the free finger is 3/4 of it
PART_THUMB = 0.52          # first phalanx of the thumb over its free part
PART_SPINE_BACK = 0.35   # the spine is in the back third of the trunk, not in the middle
PART_CAGE = 0.6            # the trunk is a cage, not a line: crossbars up to 60 % of the half-width
TWIST = (1 / 3, 2 / 3)   # twist steps between the two ends of a bone

# roles (VRM 1.0, as in the skeletons tool) and equivalent MMD name, to translate poses and animations
ROLES = {"root": ("x.root", "全ての親"), "pelvis": ("hips", "下半身"), "spine": ("spine", "上半身"),
         "chest": ("chest", "上半身1"), "upper_chest": ("upperChest", "上半身2"), "neck": ("neck", "首"),
         "head": ("head", "頭")}
for _c, _C, _m in (("L", "left", "左"), ("R", "right", "右")):
    for _n, _r, _j in (("shoulder", "Shoulder", "肩"), ("arm", "UpperArm", "腕"), ("forearm", "LowerArm", "ひじ"),
                       ("hand", "Hand", "手首"), ("thigh", "UpperLeg", "足"), ("leg", "LowerLeg", "ひざ"),
                       ("foot", "Foot", "足首"), ("toes", "Toes", "つま先")):
        ROLES["%s_%s" % (_n, _c)] = (_C + _r, _m + _j)
    for _n, _r, _j in (("thumb", "Thumb", "親指"), ("index", "Index", "人指"), ("middle", "Middle", "中指"),
                       ("ring", "Ring", "薬指"), ("little", "Little", "小指")):
        for _k, _p in enumerate(("Metacarpal", "Proximal", "Distal") if _n == "thumb" else ("Proximal", "Intermediate", "Distal")):
            ROLES["%s%d_%s" % (_n, _k + 1, _c)] = (_C + _r + _p, _m + _j + "０１２"[_k] if _n == "thumb" else _m + _j + "１２３"[_k])


def _long(C, s):
    """Point at abscissa s (cm) along a line of centres; before the start: the first slope is extended."""
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    if s <= 0:
        d = (C[1] - C[0]) / np.linalg.norm(C[1] - C[0])
        return C[0] + s * d
    s = min(s, L[-1])
    k = min(int(np.searchsorted(L, s, "right")) - 1, len(C) - 2)
    return C[k] + (C[k + 1] - C[k]) * (s - L[k]) / (L[k + 1] - L[k])


def put(g):
    """Positions of the joints of the left side and the middle, with the measurement that gives them."""
    P = g["P"]
    J, said = {}, {}

    def pose(name, p, comment):
        J[name], said[name] = np.asarray(p, float), comment

    # ---- leg: ankle, knee, hip
    r, C, L = M.line(g, "leg")
    ground = float(P[:, 1].min())
    end = int(np.argmin(L[40:56])) + 40                      # the narrowest ring of the lower leg
    che = next(v for v in range(end, len(L)) if L[v] > 1.15 * L[end])
    pose("foot_L", C[che], "first ring below the narrowest of the lower leg (rank %d) that widens by 15 per cent: the malleoli" % end)
    y_crotch = float(P[M.loop(g, "band", 4)][:, 1].min())
    wid = float(np.abs(P[M.loop(g, "trunk", 0)][:, 0]).max())
    axis = (C[6] - C[20]) / np.linalg.norm(C[6] - C[20])
    y_h = y_crotch + 0.10 * (y_crotch - ground)
    h = C[6] + axis * (y_h - C[6][1]) / axis[1]
    pose("thigh_L", [0.45 * wid, y_h, h[2]],
         "height: crotch + 10 % of the leg; spread: 45 % of the half-width of the hips; depth: axis of the thigh")
    hollow = int(np.argmin(L[20:34])) + 20                    # the hollow under the knee
    y_g = C[hollow][1] + 0.04 * (y_h - C[che][1])
    k = int(np.argmin(np.abs(C[:, 1] - y_g)))
    pose("leg_L", [C[k][0], y_g, C[k][2]], "4 per cent of the leg above the hollow under the knee (narrowest ring, rank %d)" % hollow)
    pose("toes_L", C[-2], "centre of the second-to-last ring of the foot: the line where the toes attach")
    heel = P[M.loop(g, "leg", che + 2)]
    J["_heel_L"] = heel[heel[:, 2].argmax()] * [1, 1, 0.8] + [0, 0, 0]
    for k in range(5):
        _, Cd, _ = M.line(g, "digit_toe%d" % k)
        lng = np.linalg.norm(np.diff(Cd, axis=0), axis=1).sum()
        pose("toe%d_L" % (k + 1), _long(Cd, -min(1.0, 0.35 * lng)), "a little behind the base of the toe")
        J["_tip_toe%d_L" % (k + 1)] = Cd[-1]

    # ---- trunk
    rt = M.ranks(g, "trunk")
    per = np.array([M.perimeter(g, "trunk", v) for v in rt])
    v_t = int(np.argmin(per))

    def back(map_, v):
        """Point of the spine at this ring: in the back third, on the middle plane."""
        p = P[M.loop(g, map_, v)]
        m = p[np.abs(p[:, 0]) < 3.0]
        za, zb = m[:, 2].min(), m[:, 2].max()
        return np.array([0.0, p[:, 1].mean(), zb - PART_SPINE_BACK * (zb - za)])
    size = back("trunk", v_t)
    y_armpit = float(P[M.loop(g, "arm", 0)][:, 1].min())
    pose("pelvis", [0, y_h + 0.3 * (size[1] - y_h), size[2]], "30 % of the way from the hips to the waist")
    pose("spine", [0, y_h + 0.4 * (size[1] - y_h), size[2]], "40 % of the way from the hips to the waist: the bottom of the lumbar spine")
    pose("chest", size, "the waist: the narrowest ring of the trunk (rank %d)" % v_t)
    v_s = int(np.argmin(np.abs(np.array([P[M.loop(g, "trunk", v)][:, 1].mean() for v in rt]) - (size[1] + y_armpit) / 2)))
    pose("upper_chest", back("trunk", v_s), "halfway from the waist to the armpit: under the chest")
    rc = M.ranks(g, "neck")
    pose("neck", back("neck", rc[0]), "the base of the neck")
    pose("head", back("neck", rc[int(round(0.6 * (len(rc) - 1)))]), "at 60 % of the neck")
    J["_head_top"] = J["head"] + [0, 5.0, 0]

    # ---- arm: wrist, shoulder, elbow
    r, C, L = M.line(g, "arm")
    poi = int(np.argmin(L[20:40])) + 20
    pose("hand_L", C[poi], "the narrowest ring of the arm (rank %d)" % poi)
    axis = (C[4] - C[12]) / np.linalg.norm(C[4] - C[12])
    ray = L[4] / (2 * np.pi)
    stage = np.arange(0.0, 20.0, 0.25)
    free = np.array([np.linalg.norm(P - (C[4] + x * axis), axis=1).min() for x in stage])
    end = next((k for k in range(len(stage)) if free[k] < 0.5 * ray), len(stage))      # the axis exits through the top of the shoulder
    s = stage[int(np.argmax(free[:end]))]
    pose("arm_L", C[4] + s * axis, "on the axis of the arm extended into the shoulder: the point where the largest ball fits under the skin (%.1f cm)" % free[:end].max())
    d = J["hand_L"] - J["arm_L"]
    y_c = J["arm_L"] + PART_ELBOW * d
    k = int(np.argmin(np.linalg.norm(C[:poi] - y_c, axis=1)))
    pose("forearm_L", C[k], "%d per cent from the shoulder to the wrist, brought back to the centre of the ring" % round(100 * PART_ELBOW))
    pose("shoulder_L", [0.18 * J["arm_L"][0], J["arm_L"][1] + 0.25 * (J["neck"][1] - J["arm_L"][1]), J["neck"][2]],
         "near the sternum: 18 % of the shoulder spread, a quarter of the way to the base of the neck")

    # ---- hand
    for k, name in enumerate(FINGERS):
        _, Cd, _ = M.line(g, "digit_finger%d" % k)
        lng = np.linalg.norm(np.diff(Cd, axis=0), axis=1).sum()
        all_ = lng / PART_FINGER_FREE
        s0 = lng - all_
        for i, part in enumerate((0.0, PART_PHALANGES[0], PART_PHALANGES[0] + PART_PHALANGES[1])):
            pose("%s%d_L" % (name, i + 1), _long(Cd, s0 + part * all_),
                 "phalanges at 45, 28 and 27 % of the finger, 3/4 of which is free")
        J["_tip_%s_L" % name] = Cd[-1]
    _, Cd, _ = M.line(g, "digit_thumb")
    lng = np.linalg.norm(np.diff(Cd, axis=0), axis=1).sum()
    pose("thumb1_L", J["hand_L"] + 0.35 * (Cd[0] - J["hand_L"]), "35 % from the wrist to the free base of the thumb")
    pose("thumb2_L", Cd[0], "the free base of the thumb")
    pose("thumb3_L", _long(Cd, PART_THUMB * lng), "52 % of the free part of the thumb")
    J["_tip_thumb_L"] = Cd[-1]
    J["_ground"] = np.array([0.0, ground, 0.0])
    return J, said


def cage(g, y0, y1, x_max):
    """Crossbars through the trunk between two heights (left-right and front-back, every other ring): the
    skin of the trunk is far from the spine; without them, the chest and the flanks take the heat of the arm."""
    P = g["P"]
    segs = []
    for v in M.ranks(g, "trunk")[::2]:
        p = P[M.loop(g, "trunk", v)]
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


def build(g, place=None):
    """The whole skeleton: list of bones {name, parent, pos, segments, role, mmd, torsion}.
    place: (joints, measurements) already placed (free_base/body.py); otherwise placer(g)."""
    J, said = place or put(g)
    os_ = []

    def os(name, parent, pos, segs, twist=None):
        role, mmd = ROLES.get(name, (None, None))
        os_.append({"name": name, "parent": parent, "pos": [float(x) for x in pos],
                    "segments": [[[float(x) for x in a], [float(x) for x in b]] for a, b in segs],
                    "role": role, "mmd": mmd, "twist": twist, "measure": said.get(name.replace("_R", "_L"))})

    mir = np.array([-1.0, 1.0, 1.0])
    os("root", None, [0, J["_ground"][1], 0], [])
    hips = J["thigh_L"] * [0, 1, 1]
    large = 1e9
    os("pelvis", "root", J["pelvis"], [(J["pelvis"], hips), (J["thigh_L"], J["thigh_L"] * mir)]
       + cage(g, J["thigh_L"][1], J["spine"][1], large))
    for a, b in (("spine", "chest"), ("chest", "upper_chest"), ("upper_chest", "neck"), ("neck", "head")):
        top = min(J[b][1], J["shoulder_L"][1] - 2.0)          # above, the shoulder holds the skin
        bars = cage(g, J[a][1], top, J["arm_L"][0]) if a != "neck" else []
        os(a, "pelvis" if a == "spine" else os_[-1]["name"], J[a], [(J[a], J[b])] + bars)
    os("head", "neck", J["head"], [(J["head"], J["_head_top"])])
    for c, m in (("L", np.ones(3)), ("R", mir)):
        def j(n):
            return J[n + "_L"] * m

        def tors(name, parent, a, b, sort, source, tiers):
            """Twist bones along a -> b. kind "keeps" (holds back): child of the bone, undoes a share of its
            roll (the top stays in place); "suit" (follows): child of the bone, takes a share of the roll of the source."""
            axis = (b - a) / np.linalg.norm(b - a)
            for i, t in enumerate(tiers):
                os("%s_twist%d_%s" % (name, i + 1, c), parent, a + t * (b - a), [],
                   {"sort": sort, "source": source, "axis": [float(x) for x in axis], "fraction": float(t)})
        os("shoulder_" + c, "upper_chest", j("shoulder"), [(j("shoulder"), j("arm"))])
        os("arm_" + c, "shoulder_" + c, j("arm"), [(j("arm"), j("forearm"))])
        tors("arm", "arm_" + c, j("arm"), j("forearm"), "holds", "arm_" + c, (0.0,) + TWIST)
        os("forearm_" + c, "arm_" + c, j("forearm"), [(j("forearm"), j("hand"))])
        tors("forearm", "forearm_" + c, j("forearm"), j("hand"), "follows", "hand_" + c, TWIST + (1.0,))
        os("hand_" + c, "forearm_" + c, j("hand"), [(j("hand"), j(d + "1")) for d in FINGERS])
        for d in ("thumb",) + FINGERS:
            pts = [j(d + "1"), j(d + "2"), j(d + "3"), J["_tip_%s_L" % d] * m]
            for i in range(3):
                os("%s%d_%s" % (d, i + 1, c), "hand_" + c if i == 0 else "%s%d_%s" % (d, i, c), pts[i], [(pts[i], pts[i + 1])])
        os("thigh_" + c, "pelvis", j("thigh"), [(j("thigh"), j("leg"))])
        tors("thigh", "thigh_" + c, j("thigh"), j("leg"), "holds", "thigh_" + c, (0.0,) + TWIST)
        os("leg_" + c, "thigh_" + c, j("leg"), [(j("leg"), j("foot"))])
        os("foot_" + c, "leg_" + c, j("foot"), [(j("foot"), j("toes")), (j("foot"), J["_heel_L"] * m)])
        os("toes_" + c, "foot_" + c, j("toes"), [])
        for k in range(1, 6):
            os("toe%d_%s" % (k, c), "toes_" + c, j("toe%d" % k), [(j("toe%d" % k), J["_tip_toe%d_L" % k] * m)])
    return os_


# ------------------------------------------------------------------------------------------ kinematics
def _roll(q, axis):
    """Part of a rotation that turns about an axis (swing / twist decomposition)."""
    p = np.dot(q[1:], axis) * np.asarray(axis)
    r = np.r_[q[0], p]
    n = np.linalg.norm(r)
    return r / n if n > 1e-9 else KIN.IDENTITY.copy()


class Skeleton:
    def __init__(self, os_):
        self.os = os_
        self.names = [o["name"] for o in os_]
        self.i = {n: k for k, n in enumerate(self.names)}
        self.heads = np.array([o["pos"] for o in os_])
        self.parents = np.array([-1 if o["parent"] is None else self.i[o["parent"]] for o in os_])
        self.of_mmd = {o["mmd"]: o["name"] for o in os_ if o["mmd"]}
        n = len(os_)
        self._nothing = (np.full(n, -1), np.zeros(n), np.zeros(n, bool), np.zeros(n, bool))

    def mirror(self):
        """Index of the opposite bone."""
        return np.array([self.i.get(n[:-2] + ("_R" if n.endswith("_L") else "_L"), k) if n[-2:] in ("_L", "_R") else k
                         for k, n in enumerate(self.names)])

    def place(self, rotations):
        """World matrices (R, t) for local rotations given by bone name; the twist follows on its own."""
        rot = {self.i[n]: np.asarray(q, float) for n, q in rotations.items()}
        for k, o in enumerate(self.os):
            t = o["twist"]
            if not t:
                continue
            r = _roll(rot.get(self.i[t["source"]], KIN.IDENTITY), t["axis"])
            rot[k] = KIN.q_power(r, t["fraction"] - 1.0 if t["sort"] == "holds" else t["fraction"])
        return KIN.kinematics(self.heads, self.parents, *self._nothing, rotations=rot)


def write(os_, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"unit": "cm", "frame": "y up, front -z, character's left +x",
                   "twist": "holds: the bone undoes (1 - fraction) of the roll of its source about the axis; "
                              "follows: the bone takes the fraction of the roll of its source about the axis",
                   "bones": os_}, f, ensure_ascii=False, indent=1)


def read(path):
    with open(path, encoding="utf-8") as f:
        return Skeleton(json.load(f)["bones"])


def main():
    import pmx as PMX
    g = M.load()
    os_ = build(g)
    a = sys.argv[1:]
    if "--output" in a:
        write(os_, a[a.index("--output") + 1])
    Z = PMX.read(os.path.join(os.path.dirname(M.SHAPE), "body.pmx"))
    z = {o["name"]: np.array(o["pos"]) * 8 for o in Z["bones"]}
    print("%d bones. Gap to the ZZZ bone with the same role (cm):" % len(os_))
    for o in os_:
        if o["mmd"] in z and not o["name"].endswith("_R"):
            p = np.array(o["pos"])
            print("  %-16s %-7s (%6.1f %6.1f %6.1f)  ZZZ (%6.1f %6.1f %6.1f)  gap %4.1f   %s" % (
                o["name"], o["mmd"], *p, *z[o["mmd"]], np.linalg.norm(p - z[o["mmd"]]), o["measure"] or ""))


if __name__ == "__main__":
    main()
