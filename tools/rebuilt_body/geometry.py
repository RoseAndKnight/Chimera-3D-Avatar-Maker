"""The shape of the remade body: grid placed on the reference, junctions relaxed, hands and feet stitched,
sharp hollows softened. Returns a dictionary of arrays (positions in cm).

  P        (n, 3)  positions
  quads    (q, 4)  the grid
  tris     (t, 3)  triangles of the hands, the feet and the seam strips
  sym      (n,)    the mirror vertex of each one
  source   (n,)    reference vertex taken as is (hands, feet), -1 for the grid
  miroir   (n,)    true if the vertex taken is the mirror of a left vertex of the reference
  carte    per quad: map name and grid coordinates of the four corners (for the UVs)
"""
import numpy as np

import fit as AJ
import geom as G
import grid as QD

MIR = np.array([-1.0, 1.0, 1.0])

def construire(R, journal=print):
    C = QD.Chantier(R, journal)
    C.tronc()
    C.bras()
    C.jambe()
    n_surs = len(C.M.quads)
    C.main()
    C.pied()
    C.orienter(n_surs)
    C.refleter()
    P, anc, Qd, zone = C.tableaux()
    n0 = len(P)
    vois = AJ.voisins_quads(n0, Qd)
    mains = np.concatenate([R.F[R.f_main["G"]], R.F[R.f_main["D"]]])
    cible = [((zone == "corps") | (zone == "orteil"), R.V, C.Fc), (zone == "main", R.V, mains)]
    P = AJ.interpoler(P, ~anc, vois)
    P = AJ.detendre(P, ~anc, vois, C.sym, cible, tours=25)
    # around the junctions, the anchored rows also slide on the surface: the cells even out
    bord = np.zeros(n0, bool)
    for b in G.boucles_bord(AJ.triangles(Qd)):
        bord[b] = True
    autour = AJ.dilater(~anc, vois, 4) & ~AJ.dilater(bord, vois, 1)
    # the chest too: horizontal rows do not follow the underside of a breast, they must slide
    autour |= (P[:, 1] > 114) & (P[:, 1] < C.info["y_aisselle"] + 1.5) & (np.abs(P[:, 0]) < 13) & (P[:, 2] < 2)
    # fingers and toes keep their rings (only the first one slides): otherwise they move back towards the palm
    fixes = np.array([k[0] == "D" and (k[3] == "bout" or k[3] >= 2) for k in C.M.cles])
    autour &= ~fixes
    P = AJ.detendre(P, autour, vois, C.sym, cible, tours=30, pas=0.5)
    journal(f"grid: {n0} vertices, {len(Qd)} quads, of which {int((~anc).sum())} junction vertices")
    cartes = [c for _, c in C.M.quads]
    vois = [sorted(a) for a in _voisins(n0, AJ.triangles(Qd), Qd)]
    # sharp hollows are softened everywhere except at the hands and feet (the folds between the fingers are real, and
    # the heel is a real corner: softening it trims 2 cm off it)
    libre = (zone == "corps") & (P[:, 1] > R.os["左足首"][1])
    P_avant = P.copy()
    P = AJ.adoucir(P, AJ.triangles(Qd), libre, vois, C.sym, rangs=0, tours=1, passes=40, journal=journal)
    return dict(P_avant=P_avant, vois=vois, P=P, quads=Qd, tris=np.zeros((0, 3), int), sym=C.sym,
                source=-np.ones(n0, int), miroir=np.zeros(n0, bool), cartes=cartes, libre=libre, zone=zone, info=C.info)


def _voisins(n, T, quads):
    A = [set() for _ in range(n)]
    for q in quads:
        for a, b in zip(q, np.roll(q, -1)):
            A[a].add(int(b))
            A[b].add(int(a))
    dans = set(map(int, np.unique(quads)))
    for t in T[2 * len(quads):]:
        for a, b in zip(t, np.roll(t, -1)):
            A[a].add(int(b))
            A[b].add(int(a))
    return A
