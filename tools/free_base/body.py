"""The free base body: skeleton, rest pose and weights, on the grid drawn from MakeHuman (CC0).

    python3 -I body.py <MakeHuman base.obj> <output folder>

Reads <folder>/shape-makehuman.pkl (trial.py: our grid laid on MakeHuman's neutral skin), makes it
follow the chosen silhouette (silhouette.json: MakeHuman shape targets), then:
 1. places the bones: the limbs on the joints that MakeHuman ships with its mesh ("joint-*"
    cubes, CC0 like it), the trunk from our measurements on the rings (custom_body/skeleton.py);
 2. computes the weights by bone heat (custom_body/weights.py);
 3. straightens the pose: MakeHuman has the forearm bent forwards and the legs apart; game parts
    come with straight arms and nearly vertical legs (A-pose of MMD models). So the elbow is
    unbent, the hand is put in line with the arm, palm down, and the legs are brought together;
 4. places the bones again and recomputes the weights on the straightened pose.
Writes <folder>/shape.pkl (chosen silhouette, straightened pose), own-rig/skeleton.json,
own-rig/weights.npz.
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(HERE, "..", d) for d in ("custom_body", "rebuilt_body", "atlas", "head")]
import kinematics as KIN     # noqa: E402
import geom as G              # noqa: E402
import measurements as M           # noqa: E402
import weights as PO            # noqa: E402
import reference as RL        # noqa: E402
import skeleton as SQ        # noqa: E402
import free_head as TE           # noqa: E402

# house bone -> MakeHuman joint (left side and middle)
ART = {"arm_L": "l-shoulder", "forearm_L": "l-elbow", "hand_L": "l-hand", "thigh_L": "l-upper-leg",
       "leg_L": "l-knee", "foot_L": "l-ankle", "_head_top": "head-2"}
# The neck and head follow the convention of MMD models, not MakeHuman: an imported head is placed by its bone
# 頭, and its face must land on our skull. Measured on these models: the bone is inside the head, at mouth
# height, 17.5 cm below the top of the skull and 9.5 cm in front of the nape; the neck bone 6 to 8 cm below.
HEAD_UNDER_VERTEX = 17.5      # cm
HEAD_FRONT_NAPE = 9.5      # cm
NECK_UNDER_HEAD = (6.5, 0.5)   # cm: the neck bone below the head bone, and behind it
FINGERS = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "little": 5}
# MakeHuman's neck leans forwards (head carried forwards) and is thick; that of game heads is
# straight and thin. Measured 4 cm below the neck cut of these heads: 6 cm front to back, centred under the head
# bone, 5.6 cm wide; MakeHuman: 7.6 cm, 3 cm too far back, 8 cm wide.
NECK_HEIGHT = 7.0            # cm below the neck ring: the part of the body that is straightened
NECK_NAPE = 5.0              # cm above the ring: the nape thins up to there
NECK_SETBACK = 3.0              # cm: the head moves back by this much above its neck
NECK_END = (0.75, 0.72)       # width and depth kept at the top of the neck (the depth is taken off the nape)
SLOPE_LEG = 0.045          # outward offset per cm of descent: nearly vertical legs, feet apart


def put(g, R):
    """Joints: our measurements, then MakeHuman's where it provides them."""
    J, said = SQ.put(g)
    A = R.joints
    for n, a in ART.items():
        J[n], said[n] = A[a].copy(), "MakeHuman joint \"%s\"" % a
    for d, k in FINGERS.items():
        for i in (1, 2, 3):
            J["%s%d_L" % (d, i)] = A["l-finger-%d-%d" % (k, i)].copy()
            said["%s%d_L" % (d, i)] = "MakeHuman joint \"l-finger-%d-%d\"" % (k, i)
        J["_tip_%s_L" % d] = A["l-finger-%d-4" % k].copy()
    bases = {k: A["l-toe-%d-1" % k] for k in range(1, 6)}
    for o in range(1, 6):                                   # our toes are not ordered like its toes
        k = min(bases, key=lambda k: np.linalg.norm(bases[k] - J["toe%d_L" % o]))
        J["toe%d_L" % o], said["toe%d_L" % o] = bases.pop(k).copy(), "MakeHuman joint \"l-toe-%d-1\"" % k
        J["_tip_toe%d_L" % o] = A["l-toe-%d-%d" % (k, 3 if k == 1 else 4)].copy()
    J["toes_L"] = np.mean([J["toe%d_L" % o] for o in range(1, 6)], 0)
    said["toes_L"] = "middle of MakeHuman's five toe base joints"
    neck = A["neck"]
    J["shoulder_L"] = np.array([0.18 * J["arm_L"][0], J["arm_L"][1] + 0.25 * (neck[1] - J["arm_L"][1]), neck[2]])
    an = g["P"][TE.loop(g["P"], PO.triangles(g["quads"]), True)]
    skull = R.V[R.V[:, 1] > an[:, 1].max() + 4.0]
    J["head"] = np.array([0.0, skull[:, 1].max() - HEAD_UNDER_VERTEX, skull[:, 2].max() - HEAD_FRONT_NAPE])
    J["neck"] = J["head"] + [0.0, -NECK_UNDER_HEAD[0], NECK_UNDER_HEAD[1]]
    said["head"] = "%.1f cm below the top of the skull, %.1f cm in front of the nape (convention of MMD models)" % (
        HEAD_UNDER_VERTEX, HEAD_FRONT_NAPE)
    said["neck"] = "%.1f cm below the head bone (convention of MMD models)" % NECK_UNDER_HEAD[0]
    return J, said


def q_between(a, b):
    """The smallest rotation that takes direction a onto direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return KIN.IDENTITY.copy()
    return KIN.q_axe_angle(axis / s, np.degrees(np.arctan2(s, a @ b)))


def straighten(sq):
    """Local rotations (by bone name) that lead from the MakeHuman pose to the A-pose, arms straight."""
    T, i = sq.heads, sq.i

    def to(a, b):
        v = T[i[b]] - T[i[a]]
        return v / np.linalg.norm(v)
    u = to("arm_L", "forearm_L")
    q_ab = q_between(to("forearm_L", "hand_L"), u)                       # the elbow unbends
    q_m = KIN.q_mul(q_between(KIN.q_mat(q_ab) @ to("hand_L", "middle1_L"), u), q_ab)   # the hand in line with the arm
    c = KIN.q_mat(q_m) @ to("index1_L", "little1_L")                 # palm down: index finger in front
    c -= (c @ u) * u
    z = np.array([0.0, 0.0, 1.0]) - u[2] * u
    q_m = KIN.q_mul(KIN.q_axe_angle(u, np.degrees(np.arctan2(np.cross(c, z) @ u, c @ z))), q_m)
    dc, dj = to("thigh_L", "leg_L"), to("leg_L", "foot_L")
    q_c = q_between(dc, np.array([-SLOPE_LEG * dc[1], dc[1], dc[2]]))
    q_j = q_between(dj, np.array([-SLOPE_LEG * dj[1], dj[1], dj[2]]))
    loc = {"forearm": q_ab, "hand": KIN.q_mul(KIN.q_conj(q_ab), q_m), "thigh": q_c,
           "leg": KIN.q_mul(KIN.q_conj(q_c), q_j), "foot": KIN.q_conj(q_j)}      # the foot stays flat
    rot = {}
    for n, q in loc.items():
        rot[n + "_L"], rot[n + "_R"] = q, KIN.q_mirror_x(q)
    print("straightening: elbow unbent by %.0f°, thigh brought in by %.0f°" % (
        2 * np.degrees(np.arccos(min(1, abs(q_ab[0])))), 2 * np.degrees(np.arccos(min(1, abs(q_c[0]))))))
    return rot


def carrier(name):
    """Bone that carries an auxiliary joint (fingertip, heel...)."""
    if name.startswith("_tip_toe"):
        return name[5:]
    if name.startswith("_tip_"):
        return name[5:-2] + "3_L"
    return {"_heel_L": "foot_L", "_head_top": "head"}.get(name)


def follow(g, R0, R1):
    """The grid, laid on the neutral skin R0, follows the silhouette R1: each vertex keeps its place on the
    MakeHuman triangle it touches (and its small offset from the skin)."""
    q, f, b, d = G.plus_near(g["P"], R0.V, R0.F)
    P = (R1.V[R1.F[f]] * b[:, :, None]).sum(1) + (g["P"] - q)
    P = 0.5 * (P + P[g["sym"]] * [-1, 1, 1])
    P[np.abs(g["P"][:, 0]) < 1e-6, 0] = 0.0
    print("silhouette: grid offset from the neutral skin %.2f cm at worst; displacement %.1f cm at worst" % (
        d.max(), np.linalg.norm(P - g["P"], axis=1).max()))
    return dict(g, P=P)


def neck_right(g, R):
    """Straightens and thins the neck: the head (grid, MakeHuman skin and joints) moves back above
    it, and the neck grid thins towards its ring."""
    P = g["P"]
    an = P[TE.loop(P, PO.triangles(g["quads"]), True)]
    y1 = float(an[:, 1].min())
    y0 = y1 - NECK_HEIGHT

    def t(X):
        u = np.clip((X[:, 1] - y0) / (y1 - y0), 0, 1)
        return u * u * (3 - 2 * u)
    ahead = float(an[:, 2].min())                       # the front of the neck stays where it is, the nape moves forwards

    def smooth(u):
        u = np.clip(u, 0, 1)
        return u * u * (3 - 2 * u)

    def end(X):
        """Thinning share: rises along the neck, full at the ring, then fades out under the occiput;
        above the ring it only takes the nape (in front, it is the jaw)."""
        over = X[:, 1] - y1
        w = t(X) * (1 - smooth(over / NECK_NAPE)) * (np.abs(X[:, 0]) < 9.0)
        return w * (1 - smooth(over / 1.5) * (1 - smooth((X[:, 2] - ahead) / 2.0)))

    def shape(X):
        w, Q = end(X), X.copy()
        Q[:, 0] *= 1 - (1 - NECK_END[0]) * w
        Q[:, 2] = ahead + (X[:, 2] - ahead) * (1 - (1 - NECK_END[1]) * w) + NECK_SETBACK * t(X)
        return Q
    Q = shape(P)
    R.V = shape(R.V)
    for k, p in R.joints.items():
        p[2] += NECK_SETBACK * float(t(p[None])[0])
    print("neck: straightened over %.0f cm, the head moves back by %.1f cm" % (NECK_HEIGHT, NECK_SETBACK))
    return dict(g, P=Q)


def make(g, R):
    J, said = put(g, R)
    sq = SQ.Skeleton(SQ.build(g, (J, said)))
    print("skeleton: %d bones; weights on the MakeHuman pose" % len(sq.os))
    idx, pw = PO.four(PO.compute(g, sq))
    Rm, tm = sq.place(straighten(sq))
    P = KIN.skinning(g["P"], idx, pw, sq.heads, Rm, tm, True)
    P = 0.5 * (P + P[g["sym"]] * [-1, 1, 1])
    P[np.abs(g["P"][:, 0]) < 1e-6, 0] = 0.0
    dy = float(g["P"][:, 1].min() - P[:, 1].min())
    P[:, 1] += dy
    J2 = {}
    for n, p in J.items():
        b = sq.i.get(n) if n in sq.i else (sq.i[carrier(n)] if carrier(n) else None)
        J2[n] = p.copy() if b is None else Rm[b] @ (p - sq.heads[b]) + tm[b] + [0, dy, 0]
    g2 = dict(g, P=P)
    g2["head"] = TE.make(g2, R, dy)
    sq2 = SQ.Skeleton(SQ.build(g2, (J2, said)))
    print("weights on the straightened pose")
    w = PO.compute(g2, sq2)
    return g2, sq2, w


def main():
    obj, d = sys.argv[1], sys.argv[2]
    R = RL.Reference(obj)
    source = os.path.join(d, "shape-makehuman.pkl")          # the grid on the neutral skin (trial.py)
    g = follow(M.load(source), RL.Reference(obj, {}), R)
    g2, sq, w = make(neck_right(g, R), R)
    g2 = {k: v for k, v in g2.items() if k != "grids"}
    with open(os.path.join(d, "shape.pkl"), "wb") as f:
        pickle.dump(g2, f)
    os.makedirs(os.path.join(d, "own-rig"), exist_ok=True)
    SQ.write(sq.os, os.path.join(d, "own-rig", "skeleton.json"))
    np.savez_compressed(os.path.join(d, "own-rig", "weights.npz"), weights=w.astype(np.float32), bones=np.array(sq.names))
    print("wrote", d)


if __name__ == "__main__":
    main()
