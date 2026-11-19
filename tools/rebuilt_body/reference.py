"""The reference: Chimera's current body (PMX), read in centimetres, welded, cut into parts, and
rid of its skin folds.

The current body is a mod body: its 肌 skin is made of four parts (body from neck to feet, two
closed hands with a collar, skull). Under each buttock and in the cleft, the skin is folded over itself
(angles between faces of 60 to 156°, edges of 1.3 cm): `unfold` replaces these folds with a taut skin
(the vertices of the fold and their neighbours take the average of their neighbours, boundary fixed).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ("atlas", "head", "rebuilt_head"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import geom as G  # noqa: E402
import pmx as PMX  # noqa: E402

CM = 8.0   # 1 MMD unit = 8 cm


def components(n, F):
    per = np.arange(n)

    def r(x):
        while per[x] != x:
            per[x] = per[per[x]]
            x = per[x]
        return x
    for a, b, c in F:
        ra = r(a)
        per[r(b)] = ra
        per[r(c)] = ra
    return np.array([r(x) for x in range(n)])


def angles_edges(V, F):
    """Edges with two faces and the angle between the two faces, in degrees."""
    a, u = G.dihedrals(V, F)
    return u, a


class Reference:
    def __init__(self, path):
        self.pmx = P = PMX.read(path)
        t = PMX.to_arrays(P)
        self.t = t
        names = [m["name"] for m in P["materials"]]
        self.i_skin = names.index("肌")
        self.skin_faces = np.where(t["mat_face"] == self.i_skin)[0]
        Fb = t["faces"][self.skin_faces]
        pos = t["pos"] * CM
        rep, inv = G.weld(pos, 1e-3)
        self.rep, self.inv = rep, inv          # welded -> one raw vertex; raw -> welded
        V = pos[rep]
        F = inv[Fb]
        good = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F = F[good]
        self.V, self.F = V, F
        self.faces_raw = self.skin_faces[good]     # file face of each row of F
        self.os = {o["name"]: np.array(o["pos"]) * CM for o in P["bones"]}
        comp = components(len(V), F)
        cf = comp[F[:, 0]]
        ids = np.unique(cf)
        boxes = {k: (V[np.unique(F[cf == k])].min(0), V[np.unique(F[cf == k])].max(0)) for k in ids}
        # body: the tallest part (feet to neck); hands: parts at the wrists; skull: above the neck
        top = {k: b[1][1] - b[0][1] for k, b in boxes.items()}
        k_body = max(top, key=top.get)
        self.f_body = np.where(cf == k_body)[0]
        self.f_main = {}
        self.f_skull = np.zeros(0, int)
        for k in ids:
            if k == k_body:
                continue
            c = 0.5 * (boxes[k][0] + boxes[k][1])
            if abs(c[0]) > 25:
                self.f_main["G" if c[0] > 0 else "D"] = np.where(cf == k)[0]
            else:
                self.f_skull = np.concatenate([self.f_skull, np.where(cf == k)[0]])
        # weights per welded vertex (those of the representative raw vertex)
        self.os_s = t["bones"][rep]
        self.weight_s = t["weights"][rep]

    def weight_os(self, names):
        """Share of the weights of each welded vertex carried by the named bones."""
        idx = [i for i, o in enumerate(self.pmx["bones"]) if o["name"] in names]
        return (np.isin(self.os_s, idx) * self.weight_s).sum(1)

    def unfold(self, threshold=40.0, ranks=2, iterations=300, zone=None, log=print):
        """Removes the folds of the body (not of the hands or toes): returns the number of vertices moved."""
        F = self.F[self.f_body]
        ar, ang = angles_edges(self.V, F)
        vif = ar[ang > threshold]
        if zone is not None:
            vif = vif[zone(self.V[vif[:, 0]]) & zone(self.V[vif[:, 1]])]
        S = np.zeros(len(self.V), bool)
        S[vif.ravel()] = True
        adj = G.neighbors(len(self.V), F)
        for _ in range(ranks):
            S2 = S.copy()
            for i in np.where(S)[0]:
                S2[adj[i]] = True
            S = S2
        idx = np.where(S)[0]
        self.folds = idx
        V = self.V.copy()
        for _ in range(iterations):
            V[idx] = np.array([V[adj[i]].mean(0) for i in idx])
        disp = np.linalg.norm(V - self.V, axis=1)
        log("folds: %d sharp edges, %d vertices stretched taut, max displacement %.1f mm" % (len(vif), len(idx), 10 * disp.max()))
        self.V_brut = self.V
        self.V = V
        return len(idx)


if __name__ == "__main__":
    import view as view
    R = Reference(sys.argv[1])
    print("body", len(R.f_body), "hands", {k: len(v) for k, v in R.f_main.items()}, "skull", len(R.f_skull))
    F = R.F[R.f_body]
    ar, ang = angles_edges(R.V, F)
    m = R.V[ar].mean(1)
    for s in (40, 60):
        print("edges >", s, (ang > s).sum())
    vif = m[ang > 40]
    # where are they: by height slices
    for y0 in range(0, 160, 10):
        k = (vif[:, 1] >= y0) & (vif[:, 1] < y0 + 10)
        if k.sum():
            print("y %3d-%3d : %4d   |x| %.0f-%.0f  z %.0f..%.0f" % (y0, y0 + 10, k.sum(), np.abs(vif[k, 0]).min(), np.abs(vif[k, 0]).max(), vif[k, 2].min(), vif[k, 2].max()))
