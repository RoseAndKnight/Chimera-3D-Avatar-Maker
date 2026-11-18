"""The reference: Chimera's current body (PMX), read in centimetres, welded, cut into parts, and
rid of its skin folds.

The current body is a mod body: its 肌 skin is made of four parts (body from neck to feet, two
closed hands with a collar, skull). Under each buttock and in the cleft, the skin is folded over itself
(angles between faces of 60 to 156°, edges of 1.3 cm): `deplier` replaces these folds with a taut skin
(the vertices of the fold and their neighbours take the average of their neighbours, boundary fixed).
"""
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
for d in ("atlas", "head", "rebuilt_head"):
    sys.path.insert(0, os.path.join(ICI, "..", d))
import geom as G  # noqa: E402
import pmx as PMX  # noqa: E402

CM = 8.0   # 1 MMD unit = 8 cm


def composantes(n, F):
    par = np.arange(n)

    def r(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b, c in F:
        ra = r(a)
        par[r(b)] = ra
        par[r(c)] = ra
    return np.array([r(x) for x in range(n)])


def angles_aretes(V, F):
    """Edges with two faces and the angle between the two faces, in degrees."""
    a, u = G.diedres(V, F)
    return u, a


class Reference:
    def __init__(self, chemin):
        self.pmx = P = PMX.lire(chemin)
        t = PMX.vers_tableaux(P)
        self.t = t
        noms = [m["nom"] for m in P["materiaux"]]
        self.i_peau = noms.index("肌")
        self.faces_peau = np.where(t["mat_face"] == self.i_peau)[0]
        Fb = t["faces"][self.faces_peau]
        pos = t["pos"] * CM
        rep, inv = G.souder(pos, 1e-3)
        self.rep, self.inv = rep, inv          # welded -> one raw vertex; raw -> welded
        V = pos[rep]
        F = inv[Fb]
        bon = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F = F[bon]
        self.V, self.F = V, F
        self.faces_brut = self.faces_peau[bon]     # file face of each row of F
        self.os = {o["nom"]: np.array(o["pos"]) * CM for o in P["os"]}
        comp = composantes(len(V), F)
        cf = comp[F[:, 0]]
        ids = np.unique(cf)
        boites = {k: (V[np.unique(F[cf == k])].min(0), V[np.unique(F[cf == k])].max(0)) for k in ids}
        # body: the tallest part (feet to neck); hands: parts at the wrists; skull: above the neck
        haut = {k: b[1][1] - b[0][1] for k, b in boites.items()}
        k_corps = max(haut, key=haut.get)
        self.f_corps = np.where(cf == k_corps)[0]
        self.f_main = {}
        self.f_crane = np.zeros(0, int)
        for k in ids:
            if k == k_corps:
                continue
            c = 0.5 * (boites[k][0] + boites[k][1])
            if abs(c[0]) > 25:
                self.f_main["G" if c[0] > 0 else "D"] = np.where(cf == k)[0]
            else:
                self.f_crane = np.concatenate([self.f_crane, np.where(cf == k)[0]])
        # weights per welded vertex (those of the representative raw vertex)
        self.os_s = t["os"][rep]
        self.poids_s = t["poids"][rep]

    def poids_os(self, noms):
        """Share of the weights of each welded vertex carried by the named bones."""
        idx = [i for i, o in enumerate(self.pmx["os"]) if o["nom"] in noms]
        return (np.isin(self.os_s, idx) * self.poids_s).sum(1)

    def deplier(self, seuil=40.0, rangs=2, iterations=300, zone=None, journal=print):
        """Removes the folds of the body (not of the hands or toes): returns the number of vertices moved."""
        F = self.F[self.f_corps]
        ar, ang = angles_aretes(self.V, F)
        vif = ar[ang > seuil]
        if zone is not None:
            vif = vif[zone(self.V[vif[:, 0]]) & zone(self.V[vif[:, 1]])]
        S = np.zeros(len(self.V), bool)
        S[vif.ravel()] = True
        adj = G.voisins(len(self.V), F)
        for _ in range(rangs):
            S2 = S.copy()
            for i in np.where(S)[0]:
                S2[adj[i]] = True
            S = S2
        idx = np.where(S)[0]
        self.replis = idx
        V = self.V.copy()
        for _ in range(iterations):
            V[idx] = np.array([V[adj[i]].mean(0) for i in idx])
        dep = np.linalg.norm(V - self.V, axis=1)
        journal("folds: %d sharp edges, %d vertices stretched taut, max displacement %.1f mm" % (len(vif), len(idx), 10 * dep.max()))
        self.V_brut = self.V
        self.V = V
        return len(idx)


if __name__ == "__main__":
    import view as vue
    R = Reference(sys.argv[1])
    print("body", len(R.f_corps), "hands", {k: len(v) for k, v in R.f_main.items()}, "skull", len(R.f_crane))
    F = R.F[R.f_corps]
    ar, ang = angles_aretes(R.V, F)
    m = R.V[ar].mean(1)
    for s in (40, 60):
        print("edges >", s, (ang > s).sum())
    vif = m[ang > 40]
    # where are they: by height slices
    for y0 in range(0, 160, 10):
        k = (vif[:, 1] >= y0) & (vif[:, 1] < y0 + 10)
        if k.sum():
            print("y %3d-%3d : %4d   |x| %.0f-%.0f  z %.0f..%.0f" % (y0, y0 + 10, k.sum(), np.abs(vif[k, 0]).min(), np.abs(vif[k, 0]).max(), vif[k, 2].min(), vif[k, 2].max()))
