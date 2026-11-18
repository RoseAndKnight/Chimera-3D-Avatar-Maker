"""The free base body: skeleton, rest pose and weights, on the grid drawn from MakeHuman (CC0).

    python3 -I body.py <MakeHuman base.obj> <output folder>

Reads <folder>/forme-makehuman.pkl (trial.py: our grid laid on MakeHuman's neutral skin), makes it
follow the chosen silhouette (silhouette.json: MakeHuman shape targets), then:
 1. places the bones: the limbs on the joints that MakeHuman ships with its mesh ("joint-*"
    cubes, CC0 like it), the trunk from our measurements on the rings (custom_body/skeleton.py);
 2. computes the weights by bone heat (custom_body/weights.py);
 3. straightens the pose: MakeHuman has the forearm bent forwards and the legs apart; game parts
    come with straight arms and nearly vertical legs (A-pose of MMD models). So the elbow is
    unbent, the hand is put in line with the arm, palm down, and the legs are brought together;
 4. places the bones again and recomputes the weights on the straightened pose.
Writes <folder>/forme.pkl (chosen silhouette, straightened pose), maison/squelette.json,
maison/poids.npz.
"""
import os
import pickle
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI] + [os.path.join(ICI, "..", d) for d in ("custom_body", "rebuilt_body", "atlas", "head")]
import kinematics as CIN     # noqa: E402
import geom as G              # noqa: E402
import measurements as M           # noqa: E402
import weights as PO            # noqa: E402
import reference as RL        # noqa: E402
import skeleton as SQ        # noqa: E402
import free_head as TE           # noqa: E402

# house bone -> MakeHuman joint (left side and middle)
ART = {"bras_G": "l-shoulder", "avant_bras_G": "l-elbow", "main_G": "l-hand", "cuisse_G": "l-upper-leg",
       "jambe_G": "l-knee", "pied_G": "l-ankle", "_haut_tete": "head-2"}
# The neck and head follow the convention of MMD models, not MakeHuman: an imported head is placed by its bone
# 頭, and its face must land on our skull. Measured on these models: the bone is inside the head, at mouth
# height, 17.5 cm below the top of the skull and 9.5 cm in front of the nape; the neck bone 6 to 8 cm below.
TETE_SOUS_SOMMET = 17.5      # cm
TETE_DEVANT_NUQUE = 9.5      # cm
COU_SOUS_TETE = (6.5, 0.5)   # cm: the neck bone below the head bone, and behind it
DOIGTS = {"pouce": 1, "index": 2, "majeur": 3, "annulaire": 4, "petit_doigt": 5}
# MakeHuman's neck leans forwards (head carried forwards) and is thick; that of game heads is
# straight and thin. Measured 4 cm below the neck cut of these heads: 6 cm front to back, centred under the head
# bone, 5.6 cm wide; MakeHuman: 7.6 cm, 3 cm too far back, 8 cm wide.
COU_HAUTEUR = 7.0            # cm below the neck ring: the part of the body that is straightened
COU_NUQUE = 5.0              # cm above the ring: the nape thins up to there
COU_RECUL = 3.0              # cm: the head moves back by this much above its neck
COU_FIN = (0.75, 0.72)       # width and depth kept at the top of the neck (the depth is taken off the nape)
PENTE_JAMBE = 0.045          # outward offset per cm of descent: nearly vertical legs, feet apart


def placer(g, R):
    """Joints: our measurements, then MakeHuman's where it provides them."""
    J, dit = SQ.placer(g)
    A = R.articulations
    for n, a in ART.items():
        J[n], dit[n] = A[a].copy(), "MakeHuman joint \"%s\"" % a
    for d, k in DOIGTS.items():
        for i in (1, 2, 3):
            J["%s%d_G" % (d, i)] = A["l-finger-%d-%d" % (k, i)].copy()
            dit["%s%d_G" % (d, i)] = "MakeHuman joint \"l-finger-%d-%d\"" % (k, i)
        J["_bout_%s_G" % d] = A["l-finger-%d-4" % k].copy()
    bases = {k: A["l-toe-%d-1" % k] for k in range(1, 6)}
    for o in range(1, 6):                                   # our toes are not ordered like its toes
        k = min(bases, key=lambda k: np.linalg.norm(bases[k] - J["orteil%d_G" % o]))
        J["orteil%d_G" % o], dit["orteil%d_G" % o] = bases.pop(k).copy(), "MakeHuman joint \"l-toe-%d-1\"" % k
        J["_bout_orteil%d_G" % o] = A["l-toe-%d-%d" % (k, 3 if k == 1 else 4)].copy()
    J["orteils_G"] = np.mean([J["orteil%d_G" % o] for o in range(1, 6)], 0)
    dit["orteils_G"] = "middle of MakeHuman's five toe base joints"
    cou = A["neck"]
    J["epaule_G"] = np.array([0.18 * J["bras_G"][0], J["bras_G"][1] + 0.25 * (cou[1] - J["bras_G"][1]), cou[2]])
    an = g["P"][TE.anneau(g["P"], PO.triangles(g["quads"]), True)]
    crane = R.V[R.V[:, 1] > an[:, 1].max() + 4.0]
    J["tete"] = np.array([0.0, crane[:, 1].max() - TETE_SOUS_SOMMET, crane[:, 2].max() - TETE_DEVANT_NUQUE])
    J["cou"] = J["tete"] + [0.0, -COU_SOUS_TETE[0], COU_SOUS_TETE[1]]
    dit["tete"] = "%.1f cm below the top of the skull, %.1f cm in front of the nape (convention of MMD models)" % (
        TETE_SOUS_SOMMET, TETE_DEVANT_NUQUE)
    dit["cou"] = "%.1f cm below the head bone (convention of MMD models)" % COU_SOUS_TETE[0]
    return J, dit


def q_entre(a, b):
    """The smallest rotation that takes direction a onto direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axe = np.cross(a, b)
    s = np.linalg.norm(axe)
    if s < 1e-9:
        return CIN.IDENTITE.copy()
    return CIN.q_axe_angle(axe / s, np.degrees(np.arctan2(s, a @ b)))


def redresser(sq):
    """Local rotations (by bone name) that lead from the MakeHuman pose to the A-pose, arms straight."""
    T, i = sq.tetes, sq.i

    def vers(a, b):
        v = T[i[b]] - T[i[a]]
        return v / np.linalg.norm(v)
    u = vers("bras_G", "avant_bras_G")
    q_ab = q_entre(vers("avant_bras_G", "main_G"), u)                       # the elbow unbends
    q_m = CIN.q_mul(q_entre(CIN.q_mat(q_ab) @ vers("main_G", "majeur1_G"), u), q_ab)   # the hand in line with the arm
    c = CIN.q_mat(q_m) @ vers("index1_G", "petit_doigt1_G")                 # palm down: index finger in front
    c -= (c @ u) * u
    z = np.array([0.0, 0.0, 1.0]) - u[2] * u
    q_m = CIN.q_mul(CIN.q_axe_angle(u, np.degrees(np.arctan2(np.cross(c, z) @ u, c @ z))), q_m)
    dc, dj = vers("cuisse_G", "jambe_G"), vers("jambe_G", "pied_G")
    q_c = q_entre(dc, np.array([-PENTE_JAMBE * dc[1], dc[1], dc[2]]))
    q_j = q_entre(dj, np.array([-PENTE_JAMBE * dj[1], dj[1], dj[2]]))
    loc = {"avant_bras": q_ab, "main": CIN.q_mul(CIN.q_conj(q_ab), q_m), "cuisse": q_c,
           "jambe": CIN.q_mul(CIN.q_conj(q_c), q_j), "pied": CIN.q_conj(q_j)}      # the foot stays flat
    rot = {}
    for n, q in loc.items():
        rot[n + "_G"], rot[n + "_D"] = q, CIN.q_miroir_x(q)
    print("straightening: elbow unbent by %.0f°, thigh brought in by %.0f°" % (
        2 * np.degrees(np.arccos(min(1, abs(q_ab[0])))), 2 * np.degrees(np.arccos(min(1, abs(q_c[0]))))))
    return rot


def porteur(nom):
    """Bone that carries an auxiliary joint (fingertip, heel...)."""
    if nom.startswith("_bout_orteil"):
        return nom[6:]
    if nom.startswith("_bout_"):
        return nom[6:-2] + "3_G"
    return {"_talon_G": "pied_G", "_haut_tete": "tete"}.get(nom)


def suivre(g, R0, R1):
    """The grid, laid on the neutral skin R0, follows the silhouette R1: each vertex keeps its place on the
    MakeHuman triangle it touches (and its small offset from the skin)."""
    q, f, b, d = G.plus_proche(g["P"], R0.V, R0.F)
    P = (R1.V[R1.F[f]] * b[:, :, None]).sum(1) + (g["P"] - q)
    P = 0.5 * (P + P[g["sym"]] * [-1, 1, 1])
    P[np.abs(g["P"][:, 0]) < 1e-6, 0] = 0.0
    print("silhouette: grid offset from the neutral skin %.2f cm at worst; displacement %.1f cm at worst" % (
        d.max(), np.linalg.norm(P - g["P"], axis=1).max()))
    return dict(g, P=P)


def cou_droit(g, R):
    """Straightens and thins the neck: the head (grid, MakeHuman skin and joints) moves back above
    it, and the neck grid thins towards its ring."""
    P = g["P"]
    an = P[TE.anneau(P, PO.triangles(g["quads"]), True)]
    y1 = float(an[:, 1].min())
    y0 = y1 - COU_HAUTEUR

    def t(X):
        u = np.clip((X[:, 1] - y0) / (y1 - y0), 0, 1)
        return u * u * (3 - 2 * u)
    devant = float(an[:, 2].min())                       # the front of the neck stays where it is, the nape moves forwards

    def lisse(u):
        u = np.clip(u, 0, 1)
        return u * u * (3 - 2 * u)

    def fin(X):
        """Thinning share: rises along the neck, full at the ring, then fades out under the occiput;
        above the ring it only takes the nape (in front, it is the jaw)."""
        dessus = X[:, 1] - y1
        w = t(X) * (1 - lisse(dessus / COU_NUQUE)) * (np.abs(X[:, 0]) < 9.0)
        return w * (1 - lisse(dessus / 1.5) * (1 - lisse((X[:, 2] - devant) / 2.0)))

    def forme(X):
        w, Q = fin(X), X.copy()
        Q[:, 0] *= 1 - (1 - COU_FIN[0]) * w
        Q[:, 2] = devant + (X[:, 2] - devant) * (1 - (1 - COU_FIN[1]) * w) + COU_RECUL * t(X)
        return Q
    Q = forme(P)
    R.V = forme(R.V)
    for k, p in R.articulations.items():
        p[2] += COU_RECUL * float(t(p[None])[0])
    print("neck: straightened over %.0f cm, the head moves back by %.1f cm" % (COU_HAUTEUR, COU_RECUL))
    return dict(g, P=Q)


def faire(g, R):
    J, dit = placer(g, R)
    sq = SQ.Squelette(SQ.construire(g, (J, dit)))
    print("skeleton: %d bones; weights on the MakeHuman pose" % len(sq.os))
    idx, pw = PO.quatre(PO.calculer(g, sq))
    Rm, tm = sq.poser(redresser(sq))
    P = CIN.skinning(g["P"], idx, pw, sq.tetes, Rm, tm, True)
    P = 0.5 * (P + P[g["sym"]] * [-1, 1, 1])
    P[np.abs(g["P"][:, 0]) < 1e-6, 0] = 0.0
    dy = float(g["P"][:, 1].min() - P[:, 1].min())
    P[:, 1] += dy
    J2 = {}
    for n, p in J.items():
        b = sq.i.get(n) if n in sq.i else (sq.i[porteur(n)] if porteur(n) else None)
        J2[n] = p.copy() if b is None else Rm[b] @ (p - sq.tetes[b]) + tm[b] + [0, dy, 0]
    g2 = dict(g, P=P)
    g2["tete"] = TE.faire(g2, R, dy)
    sq2 = SQ.Squelette(SQ.construire(g2, (J2, dit)))
    print("weights on the straightened pose")
    w = PO.calculer(g2, sq2)
    return g2, sq2, w


def main():
    obj, d = sys.argv[1], sys.argv[2]
    R = RL.Reference(obj)
    source = os.path.join(d, "forme-makehuman.pkl")          # the grid on the neutral skin (trial.py)
    g = suivre(M.charger(source), RL.Reference(obj, {}), R)
    g2, sq, w = faire(cou_droit(g, R), R)
    g2 = {k: v for k, v in g2.items() if k != "grilles"}
    with open(os.path.join(d, "forme.pkl"), "wb") as f:
        pickle.dump(g2, f)
    os.makedirs(os.path.join(d, "maison"), exist_ok=True)
    SQ.ecrire(sq.os, os.path.join(d, "maison", "squelette.json"))
    np.savez_compressed(os.path.join(d, "maison", "poids.npz"), poids=w.astype(np.float32), os=np.array(sq.noms))
    print("wrote", d)


if __name__ == "__main__":
    main()
