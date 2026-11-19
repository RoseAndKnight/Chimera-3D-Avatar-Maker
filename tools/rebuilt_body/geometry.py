"""The shape of the remade body: grid placed on the reference, junctions relaxed, hands and feet stitched,
sharp hollows softened. Returns a dictionary of arrays (positions in cm).

  P        (n, 3)  positions
  quads    (q, 4)  the grid
  tris     (t, 3)  triangles of the hands, the feet and the seam strips
  sym      (n,)    the mirror vertex of each one
  source   (n,)    reference vertex taken as is (hands, feet), -1 for the grid
  mirror   (n,)    true if the vertex taken is the mirror of a left vertex of the reference
  map_    per quad: map name and grid coordinates of the four corners (for the UVs)
"""
import numpy as np

import fit as AJ
import geom as G
import grid as QD

MIR = np.array([-1.0, 1.0, 1.0])

def build(R, log=print):
    C = QD.Site_(R, log)
    C.trunk()
    C.arm()
    C.leg()
    n_safe = len(C.M.quads)
    C.main()
    C.foot()
    C.orient(n_safe)
    C.reflect()
    P, former, Qd, zone = C.arrays()
    n0 = len(P)
    neigh = AJ.neighbors_quads(n0, Qd)
    hands = np.concatenate([R.F[R.f_main["G"]], R.F[R.f_main["D"]]])
    target = [((zone == "body") | (zone == "toe"), R.V, C.Fc), (zone == "hand", R.V, hands)]
    P = AJ.interpolate(P, ~former, neigh)
    P = AJ.relax(P, ~former, neigh, C.sym, target, perimeters=25)
    # around the junctions, the anchored rows also slide on the surface: the cells even out
    border = np.zeros(n0, bool)
    for b in G.border_loops(AJ.triangles(Qd)):
        border[b] = True
    around = AJ.dilate(~former, neigh, 4) & ~AJ.dilate(border, neigh, 1)
    # the chest too: horizontal rows do not follow the underside of a breast, they must slide
    around |= (P[:, 1] > 114) & (P[:, 1] < C.info["y_armpit"] + 1.5) & (np.abs(P[:, 0]) < 13) & (P[:, 2] < 2)
    # fingers and toes keep their rings (only the first one slides): otherwise they move back towards the palm
    fixes = np.array([k[0] == "D" and (k[3] == "tip" or k[3] >= 2) for k in C.M.keys])
    around &= ~fixes
    P = AJ.relax(P, around, neigh, C.sym, target, perimeters=30, stage=0.5)
    log(f"grid: {n0} vertices, {len(Qd)} quads, of which {int((~former).sum())} junction vertices")
    maps = [c for _, c in C.M.quads]
    neigh = [sorted(a) for a in _neighbors(n0, AJ.triangles(Qd), Qd)]
    # sharp hollows are softened everywhere except at the hands and feet (the folds between the fingers are real, and
    # the heel is a real corner: softening it trims 2 cm off it)
    free = (zone == "body") & (P[:, 1] > R.os["左足首"][1])
    P_before = P.copy()
    P = AJ.soften(P, AJ.triangles(Qd), free, neigh, C.sym, ranks=0, perimeters=1, passes=40, log=log)
    return dict(P_before=P_before, neighbors=neigh, P=P, quads=Qd, tris=np.zeros((0, 3), int), sym=C.sym,
                source=-np.ones(n0, int), mirror=np.zeros(n0, bool), maps=maps, free=free, zone=zone, info=C.info)


def _neighbors(n, T, quads):
    A = [set() for _ in range(n)]
    for q in quads:
        for a, b in zip(q, np.roll(q, -1)):
            A[a].add(int(b))
            A[b].add(int(a))
    in_ = set(map(int, np.unique(quads)))
    for t in T[2 * len(quads):]:
        for a, b in zip(t, np.roll(t, -1)):
            A[a].add(int(b))
            A[b].add(int(a))
    return A
