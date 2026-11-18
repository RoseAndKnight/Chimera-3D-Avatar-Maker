"""Layers of an outfit (numpy only), the same computation for the page preview and for the final PMX.

Each part was fitted alone on the skin. An outfit is adjusted as a whole, in a common frame:
each garment vertex has its **line** (the direction moving away from the skin at that spot) and its
**height** above the skin. It moves only along its line. The rule to keep fits in one
sentence: along a line, parts meet in the order of their ranks, at least
`MARGE_RENTREE` apart. It is measured by exact rays (`rays.py`),
both ways: from the vertices underneath to the surface on top, and from the vertices on top
to the surface underneath (this catches wide faces, none of whose vertices pokes through).
A vertex's margin grows with the length of its edges (`fleche`): between its vertices, a
wide face drifts away from the curved surface it follows.

1. remove: an undergarment the bottom fully covers is not worn (underwear under shorts);
2. order: from inside to outside, by rank. A top whose hem goes over
   the bottom stays outside; boots and shoes are always over the trousers;
3. tuck in: from top to bottom, each part goes under the ones above. It is pressed towards the skin
   but never goes below `PLANCHER`: a garment does not enter the body (except under a
   boot, which replaces the leg: the floor goes down with it);
4. push out: from bottom to top, where the floor left no room (two tight parts one
   on the other), the upper part moves out, by what is missing and by `SORTIE_MAX` at
   most: a jacket does not puff up. A belt never squeezes the fabric: it is always pushed out;
5. hide: faces fully hidden by an upper part held to the body are removed. None is
   left to poke through when moving;
6. remove what pokes through: faces underneath left in front of an opaque face on top (flap, bow,
   ornament nothing could tuck away) are removed, except at the edge of the upper part where the part shows again;
7. placed accessories (necklace, bag…) move out above what they touch.

Two deliberate limits on the movement: a free flap (frill, skirt hem) more than
`LIBRE_MAX` in front of the upper part is not pushed in, nor is a volume too large to go
underneath (bow on the back of a dress under a jacket). What pokes through the upper part is
removed (6): the upper part hides the rest.
What still breaks the rule after adjustment is counted by the same rays and reported; the independent
check (`check_layers.py`) counts edge and face crossings instead.

Everything is in metres in the cache's glTF frame.
"""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import glb as GLB  # noqa: E402

import geom as G  # noqa: E402
import rays as R  # noqa: E402

# Adjustable thresholds: section couches.seuils of layers.default.json (repository root), in millimetres, which
# settings.json overrides key by key; their meaning is written next to each one ("_help") and in the README.
_S = C.COUCHES["thresholds"]
MARGE = _S["accessory_gap_mm"] / 1000             # gap of an accessory above what it touches
MARGE_RENTREE = _S["layer_gap_mm"] / 1000  # gap between two layers, along a line
PLANCHER = _S["floor_mm"] / 1000                  # height above the skin below which a part is not pressed
PORTEE_RAYON = _S["ray_reach_mm"] / 1000          # beyond this, two parts are no longer one on the other
LIBRE_MAX = _S["free_flap_mm"] / 1000                # a free flap (skirt hem, frill) further than this from the upper part flares over it: it stays
SORTIE_MAX = _S["max_outward_mm"] / 1000              # an upper part never moves out more than this; the rest is reported
RAYON_MEMBRE = _S["limb_radius_mm"] / 1000          # curvature of a limb: an edge of length e drifts from it by e² / (8 × radius)
FLECHE_MAX = _S["max_sag_mm"] / 1000
PORTEE_MASQUE = _S["hide_reach_mm"] / 1000        # beyond this, the upper part floats: one sees underneath through the opening
TOLERANCE_MASQUE = _S["hide_tolerance_mm"] / 1000
PASSES = int(_S["passes"])
SEUIL = _S["flag_depth_mm"] / 1000            # a point further than this in front of the upper part goes through it
PART_TRAVERSE = _S["flag_share"]                    # share of points going through from which a pair is reported
ORDRE_DEHORS = _S["top_over_mm"] / 1000          # median outward offset from which a top is worn over the bottom

# Algorithm details, not settings: search ranges, geometric criteria, grids.
PORTEE_RENTREE = 0.05   # range of the search for a top's hem over the bottom (order)
PORTEE_PEAU = 0.06      # beyond this, the height above the skin is unknown
ACCORD = 0.3            # two lines point the same way: otherwise it is the other side of a limb
FIN = 0.0002            # gap to the rule taken as zero
EMPREINTE = 0.8         # |distance along the normal| / distance: the point is "under" the surface, not beside it
CONTACT = 0.005         # closer than this to the surface, the point's direction says nothing: it touches it
PORTEE_PAN = 0.3        # a free flap underneath that poked through the upper part is searched this far in front of it
PORTEE_CHEVEUX = 0.06   # a strand deeper than this under a part stays there (it is elsewhere: under a flap, behind a bag)
PASSES_CHEVEUX = 4
MARGE_CHEVEUX = 0.004
RANG_TETE = 82          # from this rank on, an accessory is worn on the head or over everything
SUR_CHEVEUX = ("couvre_chef", "capuche", "casque_audio", "oreilles", "lunettes", "masque")
SOUS_PEAU = 0.02        # a vertex lower than this under the skin is there on purpose
CREUX_MAX = 0.008       # the floor of a part with large faces does not rise higher than this
PORTEE = 0.025          # boxes further apart than this: the pair is not measured by traverse()
CELLULE = 0.08
VOXEL = 0.04
VOXEL_CORPS = 0.012
VOISINS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])

# couvre.bin, one byte per face
TENUE, OPAQUE, EXTERIEUR = 1, 2, 4
COUVRANTE = TENUE | OPAQUE

# Outfit rules: section couches.tenue of the settings.
_T = C.COUCHES["outfit"]
RANGS = dict(_T["ranks_in_outfit"])   # a belt is worn over the bottom and under the open jacket, whatever its accessory rank
BAS_FERMES = tuple(_T["closed_bottoms"])      # they cover the underwear: it is not worn underneath
HAUTS = tuple(_T["tops"])
BAS = tuple(_T["bottoms"])

_pieces, _verrou = {}, threading.Lock()
_combos = {}


def _lire_couvre(chemin, n):
    try:
        with open(chemin, "rb") as f:
            o = np.frombuffer(f.read(), np.uint8)
    except FileNotFoundError:
        return None
    return o if len(o) == n else None


class Piece:
    def __init__(self, ident):
        self.id = ident
        d = os.path.join(C.CACHE, "pieces", ident)
        self.meta = C.ranger(C.lire_json(os.path.join(d, "meta.json")))   # rank and tab: those of the settings
        doc, b = GLB.lire(os.path.join(d, "piece.glb"))
        acc = doc["accessors"]
        vues = doc["bufferViews"]

        def tab(i, ty, n):
            a, v = acc[i], vues[acc[i]["bufferView"]]
            compte = a["count"] * (1 if a.get("type") == "SCALAR" else n)  # indices are counted one by one
            return np.frombuffer(b, ty, compte, v.get("byteOffset", 0)).reshape(-1, n).copy()
        pr = doc["meshes"][0]["primitives"]
        self.pos = tab(pr[0]["attributes"]["POSITION"], "<f4", 3).astype(float)
        self.nor = tab(pr[0]["attributes"]["NORMAL"], "<f4", 3).astype(float)
        self.F = np.concatenate([tab(p["indices"], "<u4", 3).astype(np.int64) for p in pr])
        self.rep, self.inv = G.souder(self.pos, 1e-5)
        self.nom = self.meta.get("type_fr") or ident
        self.drapeaux = _lire_couvre(os.path.join(d, "couvre.bin"), len(self.F))


def _date(ident):
    """Date of the part's file: a re-fitted part is no longer the one held in memory."""
    try:
        return os.stat(os.path.join(C.CACHE, "pieces", ident, "piece.glb")).st_mtime_ns
    except OSError:
        return 0


def piece(ident):
    d = _date(ident)
    with _verrou:
        if ident not in _pieces or _pieces[ident][0] != d:
            if len(_pieces) > 40:
                _pieces.clear()
            _pieces[ident] = (d, Piece(ident))
        return _pieces[ident][1]


class Groupe:
    """A part of the final model, like a cache `Piece` (metres, local faces)."""

    def __init__(self, pos, nor, F, meta=None, ident="", drapeaux=None):
        self.pos, self.nor, self.F = pos, nor, F
        self.rep, self.inv = G.souder(pos, 1e-5)
        self.meta, self.id, self.nom = meta or {}, ident, (meta or {}).get("type_fr") or ident
        self.drapeaux = drapeaux if drapeaux is not None and len(drapeaux) == len(F) else None


# ------------------------------------------------------------------ neighbourhood search
def _cles(v):
    return v[:, 0] * 1000003 + v[:, 1] * 1009 + v[:, 2]


def voisinage(Q, P):
    """Mask of the points of Q close (a few cm) to a point of P."""
    v = np.unique(np.floor(P / VOXEL).astype(np.int64), axis=0)
    occ = np.unique(_cles((v[:, None] + VOISINS[None]).reshape(-1, 3)))
    return np.isin(_cles(np.floor(Q / VOXEL).astype(np.int64)), occ)


def proche(Q, V, F, N, portee, avec_face=False):
    """Surface (V, F) seen from each point of Q: signed distance along the normal, normal,
    distance, point. With no surface within `portee`, the distance is inf. Faces are sorted
    into cells: only those of neighbouring cells are examined."""
    n = len(Q)
    s, d, nor, pts = np.zeros(n), np.full(n, np.inf), np.zeros((n, 3)), np.zeros((n, 3))
    face = np.full(n, -1, np.int64)
    if not n or not len(F):
        return (s, nor, d, pts, face) if avec_face else (s, nor, d, pts)
    T = V[F]
    cen = T.mean(1)
    grosses = np.where(np.linalg.norm(T - cen[:, None], axis=2).max(1) > CELLULE - portee)[0]
    k_f = _cles(np.floor(cen / CELLULE).astype(np.int64))
    ordre = np.argsort(k_f)
    k_f = k_f[ordre]
    cq = np.floor(Q / CELLULE).astype(np.int64)
    ucq, inv = np.unique(cq, axis=0, return_inverse=True)
    inv = inv.ravel()
    tri = np.argsort(inv, kind="stable")
    bornes = np.searchsorted(inv[tri], np.arange(len(ucq) + 1))
    for i, c in enumerate(ucq):
        cles = np.unique(_cles(c[None] + VOISINS))
        g, h = np.searchsorted(k_f, cles), np.searchsorted(k_f, cles, side="right")
        morceaux = [ordre[a:b] for a, b in zip(g, h) if b > a]
        if len(grosses):
            morceaux.append(grosses)
        if not morceaux:
            continue
        cand = np.unique(np.concatenate(morceaux))
        idx = tri[bornes[i]:bornes[i + 1]]
        q, f, b, dd = G.plus_proche(Q[idx], V, F[cand], k=12, bloc=512)
        ff = F[cand][f]
        m = (b[:, :, None] * N[ff]).sum(1)
        m /= np.maximum(np.linalg.norm(m, axis=1), 1e-12)[:, None]
        s[idx], nor[idx], d[idx], pts[idx] = ((Q[idx] - q) * m).sum(1), m, dd, q
        face[idx] = cand[f]
    d[d > portee] = np.inf
    return (s, nor, d, pts, face) if avec_face else (s, nor, d, pts)


_corps = None


def corps():
    """(points, normals) of the bare body, one point per voxel; None outside a complete cache."""
    global _corps
    if _corps is None:
        chemin = os.path.join(C.CACHE, "corps", "corps.glb")
        if not os.path.isfile(chemin):
            return None
        doc, b = GLB.lire(chemin)
        acc, vues = doc["accessors"], doc["bufferViews"]
        at = doc["meshes"][0]["primitives"][0]["attributes"]

        def tab(i):
            return np.frombuffer(b, "<f4", acc[i]["count"] * 3, vues[acc[i]["bufferView"]].get("byteOffset", 0)).reshape(-1, 3).astype(float)
        pos, nor = tab(at["POSITION"]), tab(at["NORMAL"])
        _, i = np.unique(np.floor(pos / VOXEL_CORPS).astype(np.int64), axis=0, return_index=True)
        _corps = (pos[i], nor[i])
    return _corps


_peau = None


def peau():
    """(vertices, faces, normals, faces at the edge of an opening) of the bare body; None outside a complete cache."""
    global _peau
    if _peau is None:
        chemin = os.path.join(C.CACHE, "corps", "corps.glb")
        if not os.path.isfile(chemin):
            return None
        doc, b = GLB.lire(chemin)
        acc, vues = doc["accessors"], doc["bufferViews"]

        def tab(i, ty, n):
            a, v = acc[i], vues[acc[i]["bufferView"]]
            compte = a["count"] * (1 if a.get("type") == "SCALAR" else n)
            return np.frombuffer(b, ty, compte, v.get("byteOffset", 0)).reshape(-1, n)
        pr = doc["meshes"][0]["primitives"]
        V = tab(pr[0]["attributes"]["POSITION"], "<f4", 3).astype(float)
        N = tab(pr[0]["attributes"]["NORMAL"], "<f4", 3).astype(float)
        F = np.concatenate([tab(p["indices"], "<u4", 3).astype(np.int64) for p in pr])
        _, inv = G.souder(V, 1e-5)
        Fw = inv[F]
        E = np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1)
        u, nb = np.unique(E, axis=0, return_counts=True)
        au_bord = np.zeros(int(Fw.max()) + 1, bool)
        au_bord[u[nb == 1].ravel()] = True   # there, "inside" and "outside" no longer mean anything
        _peau = (V, F, N, au_bord[Fw].any(1))
    return _peau


def lignes(W, repli):
    """Line and height of each point of W above the skin: (unit directions, heights).
    The line is the direction from the closest skin point to the point (the gradient of the
    distance to the skin); the height is inf where the skin is too far or open, and the line
    is then `repli`."""
    n, h = repli.copy(), np.full(len(W), np.inf)
    pe = peau()
    if pe is None or not len(W):
        return n, h
    V, F, N, bord = pe
    s, nor, d, q, f = proche(W, V, F, N, PORTEE_PEAU, avec_face=True)
    ok = np.isfinite(d)
    ok[ok] &= ~bord[f[ok]]
    loin = ok & (d > 0.002)
    signe = np.where(s[loin] >= 0, 1.0, -1.0)
    n[loin] = (W[loin] - q[loin]) / d[loin, None] * signe[:, None]
    pres = ok & ~loin
    n[pres] = nor[pres]
    h[ok] = np.where(s[ok] >= 0, d[ok], -d[ok])
    return n, h


def _pied_corps(Q, cp):
    """Closest body point to each point of Q, and its normal."""
    q, n = np.zeros((len(Q), 3)), np.zeros((len(Q), 3))
    if not len(Q):
        return q, n
    # wide cells (a body point is always close: garments follow it), fallback on the whole body
    cellule = 0.1
    cq = np.floor(Q / cellule).astype(np.int64)
    ucq, inv = np.unique(cq, axis=0, return_inverse=True)
    inv = inv.ravel()
    tri = np.argsort(inv, kind="stable")
    bornes = np.searchsorted(inv[tri], np.arange(len(ucq) + 1))
    k_c = _cles(np.floor(cp[0] / cellule).astype(np.int64))
    for i, c in enumerate(ucq):
        idx = tri[bornes[i]:bornes[i + 1]]
        cand = np.where(np.isin(k_c, _cles(c[None] + VOISINS)))[0]
        if not len(cand):
            cand = np.arange(len(cp[0]))
        for a in range(0, len(idx), 2048):
            b = idx[a:a + 2048]
            k = cand[((Q[b, None] - cp[0][cand][None]) ** 2).sum(-1).argmin(1)]
            q[b], n[b] = cp[0][k], cp[1][k]
    return q, n


def _normale_corps(Q, cp):
    """Normal of the closest body point to each point of Q."""
    return _pied_corps(Q, cp)[1]


def au_dessus(P, q, s, d, cp):
    """Is point P straight above q, seen from the body? The line q -> P must follow the body's
    normal: otherwise P is not under the part but beside it (a lower leg higher than the hem
    of a boot, an undergarment lower than the hem of a top). A point in contact always
    counts: that is where wide faces poke through the neighbouring surface."""
    if cp is None:
        return (np.abs(s) > EMPREINTE * d) | (d < CONTACT)
    n = _normale_corps(P, cp)
    v = P - q
    return (np.abs((v * n).sum(1)) > EMPREINTE * d) | (d < CONTACT)


def _boite(Z, Q, ecart):
    return (Z.max(0) >= Q.min(0) - ecart).all() and (Z.min(0) <= Q.max(0) + ecart).all()


# ------------------------------------------------------------------ a layer being adjusted
class Couche:
    def __init__(self, p):
        self.p, self.id, self.meta, self.nom = p, p.id, p.meta, p.nom
        self.F, self.nor, self.rep, self.inv = p.F, p.nor, p.rep, p.inv
        self.Z = p.pos.copy()
        self.masque = np.zeros(len(p.F), bool)
        self.type = p.meta.get("type")
        self.enveloppe = C.enveloppant(p.meta)
        self.rang = RANGS.get(self.type, p.meta.get("rang", 50))
        if p.drapeaux is not None:
            self.fl = p.drapeaux
        else:
            T = p.pos[p.F]
            cen = T.mean(1)
            rad = cen - cen.mean(0)
            rad[:, 1] = 0.0
            dehors = (np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]) * rad).sum(1) > 0
            self.fl = np.where(dehors, COUVRANTE | EXTERIEUR, COUVRANTE).astype(np.uint8)
        self.Fext = p.F[(self.fl & EXTERIEUR) > 0]
        self.Fcouvre = p.F[((self.fl & COUVRANTE) == COUVRANTE) & ((self.fl & EXTERIEUR) > 0)]
        self.tenue = (self.fl & TENUE) > 0
        Fw = p.inv[p.F]
        self.E = np.unique(np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1), axis=0)

        # adjustment frame, per welded vertex: line, rest height, push-in along the line
        self.Fw = Fw
        nr = p.nor[p.rep]
        self.n = nr / np.maximum(np.linalg.norm(nr, axis=1), 1e-12)[:, None]
        self.h0 = np.full(len(p.rep), np.inf)
        self.a = np.zeros(len(p.rep))
        self.tenue_s = np.zeros(len(p.rep), bool)
        self.tenue_s[Fw[self.tenue].ravel()] = True
        self.pose = self.type in C.CEINTURES   # laid over the fabrics, never squeezed underneath
        # sag: between its vertices, a wide face drifts away from the curved surface it follows.
        # Two parts whose vertices all keep the rule still cross by this amount.
        W = p.pos[p.rep]
        e = np.zeros(len(p.rep))
        if len(self.E):
            lon = np.linalg.norm(W[self.E[:, 0]] - W[self.E[:, 1]], axis=1)
            np.maximum.at(e, self.E[:, 0], lon)
            np.maximum.at(e, self.E[:, 1], lon)
        self.fleche = np.minimum(e * e / (8 * RAYON_MEMBRE), FLECHE_MAX)
        self.creux = np.minimum(e * e / (8 * RAYON_MEMBRE), CREUX_MAX)   # how far a face dips below its vertices, on a limb
        self.version, self._grille = 0, None

    def asseoir(self, cp):
        """Takes the skin as the frame: lines and heights. Without skin, the part's normals.
        The frame depends only on the part at rest: it is kept with it."""
        if cp is None:
            return
        garde = getattr(self.p, "_lignes", None)
        if garde is None:
            W = self.p.pos[self.rep]
            garde = lignes(W, _normale_corps(W, cp))
            try:
                self.p._lignes = garde
            except AttributeError:
                pass
        self.n, self.h0 = garde

    def sortir_de_la_peau(self):
        """What does not hide the skin must go in front of it: a transparent fabric, a free
        flap. Their vertices left under the skin (down to SOUS_PEAU) rise to the floor;
        lower, it is a part tucked in on purpose (a boot shaft in the leg), left alone."""
        montre = np.zeros(len(self.p.rep), bool)
        montre[self.Fw[(self.fl & COUVRANTE) != COUVRANTE].ravel()] = True
        besoin = PLANCHER + self.creux - self.h0
        ou = montre & np.isfinite(self.h0) & (besoin > 0) & (self.h0 > -SOUS_PEAU)
        if ou.any():
            self.enfoncer(-self.etaler(np.where(ou, besoin, 0.0)))

    @property
    def hauteur(self):
        return self.h0 - self.a

    def grille(self):
        if self._grille is None or self._grille[0] != self.version:
            self._grille = (self.version, R.Grille(self.Z, self.F), self.n[self.Fw].sum(1))
        return self._grille[1], self._grille[2]

    def etaler(self, a, anneaux=8, pente=0.75):
        """Spreads a push-in along the edges: the part moves as a whole, with no step or tear."""
        a = a.copy()
        if len(self.E):
            for _ in range(anneaux):
                for i, j in ((0, 1), (1, 0)):
                    ou = np.where(pente * a[self.E[:, i]] > a[self.E[:, j]])[0]
                    a[self.E[ou, j]] = pente * a[self.E[ou, i]]
        return a

    def enfoncer(self, a):
        """Moves each vertex by `a` along its line, towards the skin (or outwards if a < 0)."""
        self.a = self.a + a
        self.Z = self.p.pos - (self.a[:, None] * self.n)[self.inv]
        self.version += 1


def _sondes(Z, Zu, portee):
    return (Z[:, 1] > Zu[:, 1].min() - portee) & (Z[:, 1] < Zu[:, 1].max() + portee) & voisinage(Z, Zu)


def ecarts(L, U):
    """What is missing for L to be under U: (push-in owed per vertex of L, push-out owed per vertex
    of U, ground of L), in metres, zero where the rule holds. Each gap is measured by a ray
    along a line and applies to both parts: either one can close it. The ground is the
    height below which L may go at that spot: under a part that is itself under the skin
    (a boot replaces the leg), it goes down with it; elsewhere it is inf (the floor)."""
    bL, bU, sol = np.zeros(len(L.rep)), np.zeros(len(U.rep)), np.full(len(L.rep), np.inf)
    if not len(L.F) or not len(U.F) or not _boite(U.Z, L.Z, PORTEE_RAYON):
        return bL, bU, sol
    WL, WU = L.Z[L.rep], U.Z[U.rep]
    hU = U.hauteur

    def noter(b_soi, soi, b_autre, faces_autre, f, besoin):
        np.maximum.at(b_soi, soi, besoin)
        np.maximum.at(b_autre, faces_autre[f].ravel(), np.repeat(besoin, 3))

    # 1. vertices of L: in front of U (ray towards the skin), or underneath but too close (ray outwards)
    sel = np.where(_sondes(WL, U.Z, PORTEE_RAYON))[0]
    if len(sel):
        g, nf = U.grille()
        h = L.hauteur[sel]
        marge = MARGE_RENTREE + L.fleche[sel]
        lon = np.minimum(np.where(np.isfinite(h), np.maximum(h, 0.0) + 0.003, PORTEE_RAYON), PORTEE_RAYON)
        for sens, longueur, dernier in ((-1.0, lon, True), (1.0, marge, False)):
            t, f = g.tirer(WL[sel], sens * L.n[sel], longueur, dernier)
            ok = np.isfinite(t)
            ok[ok] &= (L.n[sel][ok] * nf[f[ok]]).sum(1) > ACCORD
            besoin = (t + marge) if sens < 0 else (marge - t)
            ok &= L.tenue_s[sel] | (besoin < LIBRE_MAX)
            noter(bL, sel[ok], bU, U.Fw, f[ok], besoin[ok])
            hf = hU[U.Fw[f[ok]]].min(1)
            bas = np.isfinite(hf) & (hf - marge[ok] < PLANCHER)
            np.minimum.at(sol, sel[ok][bas], (hf - marge[ok])[bas])
    # 2. vertices of U: behind a face of L (ray outwards), or in front but too close.
    #    Further than U can move out, it is no longer U's job to fix.
    sel = np.where(_sondes(WU, L.Z, PORTEE_RAYON))[0]
    if len(sel):
        g, nf = L.grille()
        for sens, longueur in ((1.0, LIBRE_MAX), (-1.0, MARGE_RENTREE + FLECHE_MAX)):
            t, f = g.tirer(WU[sel], sens * U.n[sel], longueur)
            ok = np.isfinite(t)
            ok[ok] &= (U.n[sel][ok] * nf[f[ok]]).sum(1) > ACCORD
            marge = np.zeros(len(sel))
            marge[ok] = MARGE_RENTREE + L.fleche[L.Fw[f[ok]]].max(1)
            besoin = (t + marge) if sens > 0 else (marge - t)
            ok &= besoin > 0
            noter(bU, sel[ok], bL, L.Fw, f[ok], besoin[ok])
            hs = hU[sel[ok]]
            bas = np.isfinite(hs) & (hs - marge[ok] < PLANCHER)
            for k in range(3):
                np.minimum.at(sol, L.Fw[f[ok]][bas, k], (hs - marge[ok])[bas])
    return bL, bU, sol


class _Paires:
    """Gaps of the pairs, recomputed only when one of the two parts has moved."""

    def __init__(self):
        self.vus = {}

    def __call__(self, L, U):
        cle, etat = (id(L), id(U)), (L.version, U.version)
        if cle not in self.vus or self.vus[cle][0] != etat:
            self.vus[cle] = (etat, ecarts(L, U))
        return self.vus[cle][1]


def passer_dessous(L, dessus, paires=None):
    """Pushes L under the parts `dessus`, without going below the floor. Where its height
    is unknown (flap far from the body), it is not pushed in by more than `LIBRE_MAX` in all.
    Returns the largest push-in made (metres)."""
    paires = paires or _Paires()
    fait = 0.0
    for _ in range(PASSES):
        # the floor rises with edge length: between two vertices lying flush with the skin,
        # the face of a part with large faces cuts under the curved skin (it showed in front)
        b, sol = np.zeros(len(L.rep)), PLANCHER + L.creux
        for U in dessus:
            e = paires(L, U)
            b, sol = np.maximum(b, e[0]), np.minimum(sol, e[2])
        h = L.hauteur
        place = np.where(np.isfinite(h), np.maximum(h - sol, 0.0), np.maximum(LIBRE_MAX - L.a, 0.0))
        a = np.minimum(b, place)
        if a.max(initial=0.0) < FIN:
            break
        L.enfoncer(np.minimum(L.etaler(a), np.maximum(place, 0.0)))
        fait = max(fait, float(a.max()))
    return fait


def passer_dessus(U, dessous, paires=None):
    """Pushes U out by the gap left with the parts `dessous`, by `SORTIE_MAX` at most in all
    (`LIBRE_MAX` for a belt). Returns the largest push-out made."""
    paires = paires or _Paires()
    fait = 0.0
    borne = LIBRE_MAX if U.pose else SORTIE_MAX   # a belt always moves out: it does not squeeze the fabric
    for _ in range(PASSES):
        b = np.zeros(len(U.rep))
        for L in dessous:
            b = np.maximum(b, paires(L, U)[1])
        b = np.minimum(U.etaler(b), np.maximum(borne + np.minimum(U.a, 0.0), 0.0))
        if b.max(initial=0.0) < FIN:
            break
        U.enfoncer(-b)
        fait = max(fait, float(b.max()))
    return fait


def rentrer(L, U, cp=None):
    """Puts L under U. Returns the number of vertices moved."""
    avant = L.a.copy()
    passer_dessous(L, [U])
    return int((np.abs(L.a - avant) > FIN).sum())


def masquer(L, U, cp=None):
    """Marks the faces of L fully under a covering, body-held area of U."""
    if not len(U.Fcouvre) or not _boite(U.Z, L.Z, PORTEE_MASQUE):
        return
    a_voir = np.where(L.tenue & ~L.masque)[0]
    if not len(a_voir):
        return
    T = L.Z[L.F[a_voir]]
    pts = np.concatenate([T, T.mean(1)[:, None]], axis=1).reshape(-1, 3)
    pres = _sondes(pts, U.Z, PORTEE_MASQUE)
    ok = np.zeros(len(pts), bool)
    if pres.any():
        s, _, d, q = proche(pts[pres], U.Z, U.Fcouvre, U.nor, PORTEE_MASQUE)
        v = np.isfinite(d) & (s < TOLERANCE_MASQUE)
        v[v] &= au_dessus(pts[pres][v], q[v], s[v], d[v], cp)
        ok[pres] = v
    k = np.where(ok.reshape(-1, 4).all(1))[0]
    opq = ((U.fl & OPAQUE) > 0) & ~U.masque
    if len(k) and opq.any():
        # at the edge of U the closest point is indeed on top, but from the front nothing is in front:
        # the face goes only if, along its normal, an opaque face of U comes before or after it
        g = R.Grille(U.Z, U.F[opq])
        n = np.cross(T[k, 1] - T[k, 0], T[k, 2] - T[k, 0])
        n = np.repeat(n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None], 4, axis=0)
        P = pts.reshape(-1, 4, 3)[k]
        P = (0.9 * P + 0.1 * P[:, 3:4]).reshape(-1, 3)   # slightly inwards: a vertex right under the edge of U would miss it
        vu = np.isfinite(g.tirer(P, n, PORTEE_MASQUE)[0]) | np.isfinite(g.tirer(P, -n, PORTEE_MASQUE)[0])
        k = k[vu.reshape(-1, 4).all(1)]
    L.masque[a_voir[k]] = True


def retirer_devant(L, U):
    """Marks the faces of L left in front of U after adjustment: what neither the push-in (floor, free
    flap) nor U's push-out could tuck away, flaps and ornaments underneath poking through the top.
    A face goes if one of its vertices is in front of an opaque face of U (ray towards the skin) and
    the others are in front or underneath: at the edge of U, where the lower part comes out into the
    open, nothing is removed. What remains of it under U is not seen."""
    opq = ((U.fl & OPAQUE) > 0) & ((U.fl & EXTERIEUR) > 0) & ~U.masque
    if not opq.any() or not len(L.F) or not _boite(U.Z, L.Z, PORTEE):
        return
    W = L.Z[L.rep]
    V = np.unique(L.Fw[~L.masque])
    if len(V):   # everything inside U's box: a flap that poked through may be far in front
        V = V[((W[V] >= U.Z.min(0) - PORTEE_PAN) & (W[V] <= U.Z.max(0) + PORTEE_PAN)).all(1)]
    if not len(V):
        return
    g, nf = R.Grille(U.Z, U.F[opq]), U.n[U.Fw[opq]].sum(1)
    h = L.hauteur[V]
    lon = np.minimum(np.where(np.isfinite(h), np.maximum(h, 0.0) + 0.003, PORTEE_RAYON), PORTEE_RAYON)
    devant = np.zeros(len(V), bool)
    # near the body: the last surface before the skin; far (free flap): the first one met
    for longueur, dernier in ((lon, True), (np.where(np.isfinite(h), lon, PORTEE_PAN), False)):
        t, f = g.tirer(W[V], -L.n[V], longueur, dernier)
        d = np.isfinite(t) & (t > SEUIL)
        d[d] &= (L.n[V][d] * nf[f[d]]).sum(1) > ACCORD
        devant |= d
    if not devant.any():
        return
    t2, f2 = g.tirer(W[V], L.n[V], PORTEE_RAYON)
    dessous = np.isfinite(t2)
    dessous[dessous] &= (L.n[V][dessous] * nf[f2[dessous]]).sum(1) > ACCORD
    dv, ds = np.zeros(len(L.rep), bool), np.zeros(len(L.rep), bool)
    dv[V[devant]], ds[V[devant | dessous]] = True, True
    L.masque |= dv[L.Fw].any(1) & ds[L.Fw].all(1)


def pousser(p, Z, dessous):
    """Offset (all vertices of p) that moves p out to MARGE from the surfaces underneath:
    `dessous` = [(vertices, faces, normals)]."""
    W = Z[p.rep]
    d = np.zeros_like(W)
    for V, F, N in dessous:
        if not len(F):
            continue
        sel = np.where(voisinage(W, V[F].reshape(-1, 3)))[0]
        if not len(sel):
            continue
        s, nor, dist, _ = proche(W[sel] + d[sel], V, F, N, 0.03)
        a = np.isfinite(dist) & (s < MARGE)
        extra = np.zeros_like(d)
        extra[sel[a]] = (MARGE - s[a])[:, None] * nor[a]
        d += extra
    Fw = p.inv[p.F]
    E = np.unique(np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1), axis=0)
    if len(E):
        for _ in range(6):
            m = np.linalg.norm(d, axis=1)
            for a, b in ((0, 1), (1, 0)):
                ou = np.where(0.72 * m[E[:, a]] > m[E[:, b]])[0]
                d[E[ou, b]] = 0.72 * d[E[ou, a]]
    return d[p.inv]


def coiffure(nom, tete):
    """Cached hair placed on the head `tete`: (vertices, normals, faces), in metres."""
    d = os.path.join(C.CACHE, "coiffures", nom)
    doc, b = GLB.lire(os.path.join(d, "coiffure.glb"))
    acc, vues = doc["accessors"], doc["bufferViews"]

    def tab(i, ty, n):
        a, v = acc[i], vues[acc[i]["bufferView"]]
        return np.frombuffer(b, ty, a["count"] * (1 if a.get("type") == "SCALAR" else n), v.get("byteOffset", 0)).reshape(-1, n)
    pr = doc["meshes"][0]["primitives"]
    V = tab(pr[0]["attributes"]["POSITION"], "<f4", 3).astype(float)
    N = tab(pr[0]["attributes"]["NORMAL"], "<f4", 3).astype(float)
    F = np.concatenate([tab(q["indices"], "<u4", 3).astype(np.int64) for q in pr])
    pose = os.path.join(d, "%s.bin" % tete)
    if os.path.exists(pose):
        Z = np.fromfile(pose, "<f4").reshape(-1, 3).astype(float)
        if len(Z) == len(V):
            V = Z
    return V, N, F


def sous_cheveux(c):
    """The hair falls over this part (not a hat, glasses, a hood)."""
    return c.type not in SUR_CHEVEUX and (c.enveloppe or c.rang < RANG_TETE)


def coiffer(couches, V, N, F):
    """Offset of the hairstyle's vertices (V, N, F: placed on the head, metres) that makes it
    fall over the adjusted outfit: a vertex under the outer face of a part (straight under it,
    not beside it) moves out to MARGE_CHEVEUX along the part's normal, by the shortest way; the
    strand follows as a whole. A strand deeper than PORTEE_CHEVEUX stays where it is."""
    sur = [c for c in couches if sous_cheveux(c) and _boite(V, c.Z, 0.0)]
    if not sur or not len(F):
        return np.zeros_like(V)
    g = Groupe(V, N, F)
    W = V[g.rep]
    d = np.zeros_like(W)
    Fw = g.inv[F]
    E = np.unique(np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1), axis=0)
    for _ in range(PASSES_CHEVEUX):
        bouge = False
        for c in sur:
            Fc = c.F[((c.fl & EXTERIEUR) > 0) & ~c.masque] if len(c.Fext) else c.F[~c.masque]
            if not len(Fc):
                continue
            P = W + d
            sel = np.where(voisinage(P, c.Z[Fc].reshape(-1, 3)))[0]
            if not len(sel):
                continue
            s, nor, dist, _ = proche(P[sel], c.Z, Fc, c.nor, PORTEE_CHEVEUX)
            a = np.isfinite(dist) & (s < MARGE_CHEVEUX) & ((np.abs(s) > EMPREINTE * dist) | (dist < CONTACT))
            if a.any():
                d[sel[a]] += (MARGE_CHEVEUX - s[a])[:, None] * nor[a]
                bouge = True
        if not bouge:
            break
        for _ in range(10):                      # the strand follows: no step between two neighbouring vertices
            m = np.linalg.norm(d, axis=1)
            for i, j in ((0, 1), (1, 0)):
                ou = np.where(0.85 * m[E[:, i]] > m[E[:, j]])[0]
                d[E[ou, j]] = 0.85 * d[E[ou, i]]
    return d[g.inv]


def traverse(L, U, cp=None):
    """Share of the visible vertices of L that are in front of U, among those U covers, and depth
    of the worst. Measured by rays along the lines. Returns None if the parts do not overlap."""
    V = np.unique(L.Fw[~L.masque])
    if not len(V) or not len(U.F) or not _boite(U.Z, L.Z, PORTEE):
        return None
    W = L.Z[L.rep]
    V = V[_sondes(W[V], U.Z, PORTEE_RAYON)]
    if not len(V):
        return None
    g, nf = U.grille()
    h = L.hauteur[V]
    lon = np.minimum(np.where(np.isfinite(h), np.maximum(h, 0.0) + 0.003, PORTEE_RAYON), PORTEE_RAYON)
    t, f = g.tirer(W[V], -L.n[V], lon, True)
    dehors = np.isfinite(t) & (t > SEUIL)
    dehors[dehors] &= (L.n[V][dehors] * nf[f[dehors]]).sum(1) > ACCORD
    dehors &= L.tenue_s[V] | (t < LIBRE_MAX)
    t2, f2 = g.tirer(W[V], L.n[V], PORTEE_RAYON)
    couvert = np.isfinite(t2)
    couvert[couvert] &= (L.n[V][couvert] * nf[f2[couvert]]).sum(1) > ACCORD
    pres = dehors | couvert
    if pres.sum() < 20:
        return None
    return {"sommets": int(pres.sum()), "part": round(float(dehors.sum() / pres.sum()), 4),
            "pire_mm": round(float(t[dehors].max() * 1000), 1) if dehors.any() else 0.0}


# ------------------------------------------------------------------ the outfit
def retirees(metas):
    """{id: why} of the parts not worn in this outfit."""
    types = {m.get("type"): m for m in metas.values()}
    out = {}
    for i, m in metas.items():
        if m.get("type") == "culotte":
            for t in BAS_FERMES:
                if t in types:
                    out[i] = "%s removed: the %s covers it" % (m.get("type_fr") or "culotte", types[t].get("type_fr") or t)
    return out


def sortie_mediane(H, B, cp=None):
    """Median outward offset of H from B where they overlap (metres), None without overlap."""
    if not len(B.Fext):
        return None
    W = H.Z[H.rep]
    sel = np.where(_sondes(W, B.Z, PORTEE_RENTREE))[0]
    if not len(sel):
        return None
    s, _, d, q = proche(W[sel], B.Z, B.Fext, B.nor, PORTEE_RENTREE)
    pres = np.isfinite(d)
    pres[pres] &= au_dessus(W[sel][pres], q[pres], s[pres], d[pres], cp)
    return float(np.median(s[pres])) if pres.sum() >= 20 else None


def ordonner(couches, cp=None):
    """Sorts from inside to outside; a top whose hem goes over the bottom is worn outside. A
    sock that stops below the underwear's waist (stocking, not tights) goes underneath."""
    notes = []
    for C_ in couches:
        if C_.type == "culotte":
            for B in couches:
                if B.type == "chaussettes" and B.Z[:, 1].max() < C_.Z[:, 1].max():
                    B.rang = min(B.rang, C_.rang - 0.5)
    for H in couches:
        if H.type not in HAUTS:
            continue
        for B in couches:
            if B.type in BAS:
                m = sortie_mediane(H, B, cp)
                if m is not None and m > ORDRE_DEHORS:
                    H.rang = max(H.rang, B.rang + 0.5)
                    notes.append("%s worn over %s" % (H.nom, B.nom))
    return sorted(couches, key=lambda c: c.rang), notes


def regler(env):
    """Puts the wrapping parts (sorted from inside to outside) in order along each line."""
    paires = _Paires()
    for i in range(len(env) - 2, -1, -1):      # top to bottom: each one is pressed under the ones above
        if not env[i].pose:
            passer_dessous(env[i], [U for U in env[i + 1:] if not U.pose], paires)
    for i in range(1, len(env)):               # bottom to top: what found no room moves out
        passer_dessus(env[i], env[:i], paires)


def resoudre(pieces):
    """Adjusts a list of Piece/Groupe objects. Returns (sorted layers, notes)."""
    cp = corps()
    couches, notes = ordonner([Couche(p) for p in pieces], cp)
    env = [c for c in couches if c.enveloppe]
    for c in env:
        c.asseoir(cp)
        c.sortir_de_la_peau()
    regler(env)
    for i, L in enumerate(env):
        for U in env[i + 1:]:
            masquer(L, U, cp)
    for i, L in enumerate(env):
        for U in env[i + 1:]:
            retirer_devant(L, U)
    dessous = []
    for c in couches:
        if not c.enveloppe and dessous:
            d = pousser(c, c.Z, dessous)
            c.Z = c.Z + d
        dessous.append((c.Z, c.Fext if len(c.Fext) else c.F, c.nor))
    return couches, notes


def signaler(couches):
    out = []
    env = [c for c in couches if c.enveloppe]
    for i, L in enumerate(env):
        for U in env[i + 1:]:
            r = traverse(L, U)
            if r:
                out.append({"dessous": L.id, "dessus": U.id, "nom_dessous": L.nom, "nom_dessus": U.nom, **r})
    return out


def part_rentree(c):
    """Share of a part's vertices moved under another (by more than 1 mm): they are no longer seen."""
    if not c.enveloppe or c.type in C.CEINTURES:
        return 0.0
    return float((np.linalg.norm(c.Z - c.p.pos, axis=1) > 0.001).mean())


def traversee_visible(valeur, part_cachee, seuil=0.031):
    """Skin poking through a part where another part covers it is not seen."""
    return valeur > seuil + part_cachee


def resoudre_pmx(P, infos, cheveux=()):
    """Applies the preview's adjustment to the PMX. `infos`: per part, id, meta and materials (names);
    the flags come from the cache. Moves the vertices and removes the hidden faces; the
    `cheveux` materials fall over the adjusted outfit.
    Returns (checked layers, notes, number of faces removed)."""
    pos = np.array(P["sommets"]["pos"], float) * C.METRE
    nor = np.array(P["sommets"]["nor"], float)
    faces = np.array(P["faces"], np.int64)
    debut, par_nom, plages = 0, {}, {}
    for m in P["materiaux"]:
        n = m["n_indices"] // 3
        par_nom[m["nom"]] = faces[debut:debut + n]
        plages[m["nom"]] = (debut, debut + n)
        debut += n
    pieces, sommets, locales = [], {}, {}
    for info in infos:
        mats = [n for n in info["mats"] if n in par_nom and len(par_nom[n])]
        if not mats:
            continue
        F = np.concatenate([par_nom[n] for n in mats])
        vs, loc = np.unique(F, return_inverse=True)
        Fl = loc.reshape(-1, 3)
        fl = _lire_couvre(os.path.join(C.CACHE, "pieces", info["id"], "couvre.bin"), len(Fl))
        pieces.append(Groupe(pos[vs], nor[vs], Fl, info.get("meta"), info["id"], fl))
        sommets[info["id"]], locales[info["id"]] = vs, mats
    couches, notes = resoudre(pieces)
    garde = np.ones(len(faces), bool)
    for c in couches:
        pos[sommets[c.id]] = c.Z
        k = 0
        for n in locales[c.id]:
            a, b = plages[n]
            garde[a:b] = ~c.masque[k:k + b - a]
            k += b - a
    Fch = [par_nom[n] for n in cheveux if n in par_nom and len(par_nom[n])]
    if Fch and couches:
        Fch = np.concatenate(Fch)
        vs, loc = np.unique(Fch, return_inverse=True)
        pos[vs] += coiffer(couches, pos[vs], nor[vs], loc.reshape(-1, 3))
    faces_gardees, retire = [], 0
    for m in P["materiaux"]:
        a, b = plages[m["nom"]]
        reste = int(garde[a:b].sum())
        retire += b - a - reste
        m["n_indices"] = reste * 3
        faces_gardees.append(faces[a:b][garde[a:b]])
    P["faces"] = [tuple(int(x) for x in f) for f in (np.concatenate(faces_gardees) if faces_gardees else faces[:0])]
    P["sommets"]["pos"] = [tuple(float(x) for x in v) for v in pos / C.METRE]
    return couches, notes, retire


# ---------------------------------------------------------------------- feet posed in a shoe
# A shoe with a heel was fitted on a body whose feet take its pose (clothes/foot.py) and is stored
# in the rest frame. The outfit is adjusted in the rest frame, then the body and every part are posed:
# the shoe with its own weights (pied.bin), the other parts with those of the closest skin.
BAS_POSE = 0.6      # metres: above this, nothing follows the feet
_pied_cache = None


def _pied():
    """(chains, wa, wt, side) of the bare body, per vertex of corps.glb; None without a foot chain."""
    global _pied_cache
    if _pied_cache is None or _pied_cache[0] != C.CORPS:
        import foot as PD
        import pmx as PMX
        Pc = PMX.lire(C.CORPS)
        ch = PD.chaines(Pc["os"])
        wa, wt, cote = PD.poids_pmx(Pc, ch)
        T = PMX.vers_tableaux(Pc)
        vs = np.unique(T["faces"][T["mat_face"] == [m["nom"] for m in Pc["materiaux"]].index("肌")])
        _pied_cache = (C.CORPS, (ch, wa[vs], wt[vs], cote[vs]) if ch else None)
    return _pied_cache[1]


def poser_gltf(V, wa, wt, cote, ch, pose):
    """Points of the cache's glTF frame (metres, z flipped) posed by foot.py, which works in the PMX frame."""
    import foot as PD
    X = V / C.METRE
    X[:, 2] *= -1
    Y = PD.poser(X, wa, wt, cote, PD.transfos(ch, pose)) * C.METRE
    Y[:, 2] *= -1
    return Y


def _plus_proche(Q, P):
    """Index of the closest point of P to each point of Q (by blocks)."""
    out = np.zeros(len(Q), np.int64)
    for a in range(0, len(Q), 2048):
        d = ((Q[a:a + 2048, None, :] - P[None]) ** 2).sum(2)
        out[a:a + 2048] = d.argmin(1)
    return out


def pose_tenue(couches):
    """The feet's pose of an outfit: that of its shoe, None without one."""
    return next((c.meta["pied"] for c in couches if c.meta.get("pied")), None)


def poser_tenue(couches):
    """(body offset, {id: posed vertices}) of an outfit worn with a posed shoe; (None, {}) otherwise."""
    pose, pd, pe = pose_tenue(couches), _pied(), peau()
    if not pose or pd is None or pe is None:
        return None, {}
    ch, wa, wt, cote = pd
    Vb = pe[0]
    corps_d = poser_gltf(Vb, wa, wt, cote, ch, pose) - Vb
    bas = np.where(Vb[:, 1] < BAS_POSE)[0]
    poses = {}
    for c in couches:
        Z = c.Z
        q = None
        try:
            q = np.fromfile(os.path.join(C.CACHE, "pieces", c.id, "pied.bin"), "<f4").reshape(-1, 3)
        except (OSError, ValueError):
            pass
        if q is not None and len(q) == len(Z):
            a, t, k = q[:, 0].astype(float), q[:, 1].astype(float), q[:, 2].astype(np.int64)
        else:
            m = np.where(Z[:, 1] < BAS_POSE)[0]
            if not len(m) or not len(bas):
                continue
            a, t, k = np.zeros(len(Z)), np.zeros(len(Z)), np.full(len(Z), -1)
            j = bas[_plus_proche(Z[m], Vb[bas])]
            a[m], t[m], k[m] = wa[j], wt[j], cote[j]
        if not (a + t > 0).any():
            continue
        poses[c.id] = poser_gltf(Z, a, t, k, ch, pose)
    return corps_d, poses


def combiner(ids, coif=None, tete=None):
    """ids (in any order) -> {"ordre", "decalages" {id: Float32 bytes},
    "masques" {id: uint8 bytes per face}, "retirees" {id: why}, "notes", "signal",
    "coiffure": Float32 bytes of the offset of the hair `coif` placed on `tete`, or None}."""
    cle = "|".join("%s@%d" % (i, _date(i)) for i in sorted(ids))   # once re-fitted, a part changes key
    cle += "|cheveux:%s:%s" % (coif, tete) if coif else ""
    with _verrou:
        if cle in _combos:
            return _combos[cle]
    pieces = {i: piece(i) for i in ids}
    ret = retirees({i: p.meta for i, p in pieces.items()})
    couches, notes = resoudre([p for i, p in pieces.items() if i not in ret])
    decal, masques = {}, {}
    corps_d, poses = poser_tenue(couches)
    for c in couches:
        d = poses.get(c.id, c.Z) - c.p.pos
        if np.abs(d).max() > 1e-6:
            decal[c.id] = d.astype("<f4").tobytes()
        if c.masque.any():
            masques[c.id] = c.masque.astype(np.uint8).tobytes()
    signal = [s for s in signaler(couches) if s["part"] >= PART_TRAVERSE]
    cheveux = None
    if coif and os.path.isfile(os.path.join(C.CACHE, "coiffures", coif, "coiffure.glb")):
        d = coiffer(couches, *coiffure(coif, tete or coif))
        if np.abs(d).max() > 1e-6:
            cheveux = d.astype("<f4").tobytes()
    res = {"ordre": [c.id for c in couches], "decalages": decal, "masques": masques, "retirees": ret,
           "notes": list(ret.values()) + notes, "signal": signal, "coiffure": cheveux,
           "corps": corps_d.astype("<f4").tobytes() if corps_d is not None else None}
    with _verrou:
        if len(_combos) > 200:
            _combos.clear()
        _combos[cle] = res
    return res
