"""Godot bone springs (SpringBoneSimulator3D) converted into quantities comparable with the
physics rules of ZZZ hairstyles (CONSTRUCTION.md, P1-P16), without simulating.

Input: Godot scenes (.tscn, .tres) containing SpringBoneSimulator3D nodes and their
SpringBoneCollision{Sphere,Capsule,Plane}3D children, or a JSON in the format of `lire()`:

    {"simulateurs": [{"nom": "...", "reglages": [{"root_bone_name": "...", "end_bone_name": "...",
       "radius": 0.02, "stiffness": 1.0, "drag": 0.4, "gravity": 0.0,
       "gravity_direction": [0, -1, 0], "extend_end_bone": false, "end_bone_length": 0.0,
       "enable_all_child_collisions": true, "exclude_collisions": [], "collisions": [],
       "joints": [{"bone_name": "...", "radius": ..., "stiffness": ..., ...}]}],
       "collisions": [{"nom": "...", "type": "sphere|capsule|plane", "bone_name": "...",
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

DEFAUTS = {"radius": 0.02, "stiffness": 1.0, "drag": 0.4, "gravity": 0.0,
           "gravity_direction": [0.0, -1.0, 0.0], "extend_end_bone": False, "end_bone_length": 0.0,
           "enable_all_child_collisions": True}
DEFAUTS_COLLISION = {"radius": 0.1, "height": 0.5, "inside": False,
                     "position_offset": [0.0, 0.0, 0.0], "rotation_offset": [0.0, 0.0, 0.0, 1.0]}
PAS_PAR_SECONDE = 60
TYPES_COLLISION = {"SpringBoneCollisionSphere3D": "sphere", "SpringBoneCollisionCapsule3D": "capsule",
                   "SpringBoneCollisionPlane3D": "plane"}

NON_COMPARABLES = {
    "P8": "masses: the Godot spring has no mass (gravity is a displacement per second, "
          "the same for every link).",
    "P11": "joint limits: Godot has no angle limit; only collision holds the strand back.",
    "P12": "joint springs: ZZZ has zero springs and Bullet carries the strand; Godot pulls "
           "each bone back to its rest pose (stiffness). Different model: we give the sag "
           "atan(g/k) and the restoring rate k/L, without a verdict.",
    "P13": "錘 weights: no equivalent (no mass).",
    "P15": "worst pose within the limits: with no limits, collision with the head is required "
           "for every chain (P10).",
    "gravité": "Bullet applies an acceleration (MMD, 98 u/s²) to bodies with mass; Godot a "
               "displacement per second. Only the rest sag can be compared, by hand.",
    "collision mèche-mèche": "absent in Godot (ZZZ: a hair group that ignores itself, "
                             "so absent too: P10 holds by construction).",
}


# ============================================================ reading Godot scenes

def _valeur(txt):
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
            return {"_" + f: _valeur(a)}
        parts = [p.strip() for p in _decouper(a)] if a.strip() else []
        try:
            return [float(p) for p in parts]
        except ValueError:
            return [_valeur(p) for p in parts]
    if t.startswith("["):
        return [_valeur(p) for p in _decouper(t[1:-1])] if t[1:-1].strip() else []
    return t


def _decouper(a):
    """Split "a, B(c, d), [e, f]" at the top level."""
    out, prof, cur, chaine = [], 0, "", False
    for ch in a:
        if ch == '"':
            chaine = not chaine
        if not chaine:
            if ch in "([{":
                prof += 1
            elif ch in ")]}":
                prof -= 1
            elif ch == "," and prof == 0:
                out.append(cur)
                cur = ""
                continue
        cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _sections(texte):
    """Sections of a .tscn/.tres: [(header, header attributes, {key: value})]."""
    secs, cur, tampon = [], None, ""
    for ligne in texte.splitlines():
        if tampon:
            tampon += "\n" + ligne
        elif re.match(r"^\[(\w+)(.*)\]\s*$", ligne):
            m = re.match(r"^\[(\w+)(.*)\]\s*$", ligne)
            att = {k: v for k, v in re.findall(r'(\w+)=(?:&?"([^"]*)"|[^\s\]]+)', m.group(2))}
            att.update({k: v for k, v in re.findall(r'(\w+)="([^"]*)"', m.group(2))})
            cur = (m.group(1), att, {})
            secs.append(cur)
            continue
        elif "=" in ligne and cur is not None:
            tampon = ligne
        else:
            continue
        k, v = tampon.split("=", 1)
        if v.count("(") + v.count("[") + v.count("{") > v.count(")") + v.count("]") + v.count("}"):
            continue          # value spanning several lines
        cur[2][k.strip()] = _valeur(v)
        tampon = ""
    return secs


def _courbe(sr):
    """(x, y) points of a Curve resource (tangents ignored: linear interpolation)."""
    d = sr.get("_data", [])
    pts = []
    for k in range(0, len(d), 5):
        v = d[k]
        if isinstance(v, list) and len(v) == 2:
            pts.append((float(v[0]), float(v[1])))
    return sorted(pts) or None


def _chemin_noeud(att):
    p = att.get("parent")
    if p is None:
        return "."
    return att["name"] if p == "." else p + "/" + att["name"]


def lire_tscn(chemin):
    with open(chemin, encoding="utf-8") as fh:
        secs = _sections(fh.read())
    sous = {a.get("id"): v for t, a, v in secs if t == "sub_resource"}
    sims, cols = [], []
    for t, a, v in secs:
        if t != "node":
            continue
        typ = a.get("type", "")
        if typ == "SpringBoneSimulator3D":
            sims.append((_chemin_noeud(a), a["name"], v))
        elif typ in TYPES_COLLISION:
            cols.append((a.get("parent", "."), a["name"], typ, v))
    out = []
    for chem, nom, v in sims:
        n = int(v.get("setting_count", 0))
        reglages = []
        for i in range(n):
            pre = "settings/%d/" % i
            s = {k[len(pre):]: x for k, x in v.items() if k.startswith(pre)}
            r = {"root_bone_name": s.get("root_bone_name"), "end_bone_name": s.get("end_bone_name"),
                 "defauts": []}
            for cle, cle_g in (("radius", "radius/value"), ("stiffness", "stiffness/value"),
                               ("drag", "drag/value"), ("gravity", "gravity/value"),
                               ("gravity_direction", "gravity/direction"),
                               ("extend_end_bone", "extend_end_bone"), ("end_bone_length", "end_bone/length"),
                               ("enable_all_child_collisions", "enable_all_child_collisions")):
                if cle_g in s:
                    r[cle] = s[cle_g]
                else:
                    r[cle] = DEFAUTS[cle]
                    r["defauts"].append(cle)
            for cle in ("radius", "stiffness", "drag", "gravity"):
                c_ = s.get(cle + "/damping_curve")
                if isinstance(c_, dict) and "_SubResource" in c_:
                    r[cle + "_courbe"] = _courbe(sous.get(c_["_SubResource"], {}))
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
            reglages.append(r)
        collisions = []
        for par, nc, typ, cv in cols:
            if par not in (chem, nom) and not par.endswith("/" + nom):
                continue
            col = {"nom": nc, "type": TYPES_COLLISION[typ], "bone_name": cv.get("bone_name"),
                   "defauts": [k for k in DEFAUTS_COLLISION if k not in cv]}
            for k, d in DEFAUTS_COLLISION.items():
                col[k] = cv.get(k, d)
            if "transform" in cv and not col["bone_name"]:
                col["transform"] = cv["transform"]
            collisions.append(col)
        out.append({"fichier": chemin, "nom": nom, "chemin": chem, "reglages": reglages,
                    "collisions": collisions})
    return out


def lire(chemins):
    sims = []
    for ch in chemins:
        if ch.endswith(".json"):
            with open(ch, encoding="utf-8") as fh:
                d = json.load(fh)
            for s in d.get("simulateurs", []):
                s.setdefault("fichier", ch)
                for r in s.get("reglages", []):
                    r.setdefault("defauts", [k for k in DEFAUTS if k not in r])
                    for k, v in DEFAUTS.items():
                        r.setdefault(k, v)
                    r.setdefault("joints", [])
                    r.setdefault("exclude_collisions", [])
                    r.setdefault("collisions", [])
                for col in s.get("collisions", []):
                    col.setdefault("defauts", [k for k in DEFAUTS_COLLISION if k not in col])
                    for k, v in DEFAUTS_COLLISION.items():
                        col.setdefault(k, v)
                sims.append(s)
        else:
            sims += lire_tscn(ch)
    return sims


# ============================================================ geometry

def _norm(n):
    return re.sub(r"[^0-9a-z]", "", (n or "").lower())


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _echant(courbe, t):
    if not courbe:
        return 1.0
    xs, ys = zip(*courbe)
    return float(np.interp(t, xs, ys))


def _forme(c, col, os_idx):
    """Shape of a collision in the measurement frame: (type, a, b, radius, normal, inside)."""
    O, u = c.p["os"], c.unite
    if os_idx is not None:
        o = O[os_idx]
        base, ech, org = np.asarray(o["base"]), o["echelle_os"], np.asarray(o["pos"])
    else:   # no bone: node transform (skeleton frame assumed)
        T = col.get("transform") or [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        B = np.array(T[:9]).reshape(3, 3).T
        base, ech = c.rotation @ B, 1.0
        org = (np.asarray(T[9:12]) - c.origine) @ c.rotation.T * u
    rot = base @ _quat_mat(col["rotation_offset"])
    centre = org + base @ np.asarray(col["position_offset"], float) * ech * u
    r = float(col["radius"]) * ech * u
    if col["type"] == "sphere":
        return "sphere", centre, centre, r, None, bool(col["inside"])
    if col["type"] == "capsule":
        demi = max(float(col["height"]) / 2 - float(col["radius"]), 0.0) * ech * u
        ax = rot[:, 1]
        return "capsule", centre - ax * demi, centre + ax * demi, r, None, bool(col["inside"])
    return "plane", centre, centre, 0.0, rot[:, 1], False


def _distance(forme, P):
    """Signed distance from points P to the surface (negative = inside, forbidden side)."""
    typ, a, b, r, n, dedans = forme
    P = np.atleast_2d(P)
    if typ == "plane":
        return (P - a) @ n
    ab = b - a
    L2 = float(ab @ ab)
    t = np.clip(((P - a) @ ab) / L2, 0, 1) if L2 > 0 else np.zeros(len(P))
    d = np.linalg.norm(P - (a + np.outer(t, ab)), axis=1) - r
    return -d if dedans else d


# ============================================================ measurement

def _trouver_os(c, nom, index_norm):
    if nom is None:
        return None
    if nom in c.noms_os:
        return c.noms_os.index(nom)
    return index_norm.get(_norm(nom))


def _q(v, n=3):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None
    return {"n": len(v), "min": round(float(np.min(v)), n), "p50": round(float(np.median(v)), n),
            "max": round(float(np.max(v)), n)}


def mesurer(c, chemins):
    """Godot bone springs of a GLB hairstyle (c: an already measured mesure.CoiffureGLB)."""
    sims = lire(chemins)
    H, u = c.H, c.unite
    O = c.p["os"]
    index_norm = {}
    for k, n in enumerate(c.noms_os):
        index_norm.setdefault(_norm(n), k)
    enfants = {}
    for i, o in enumerate(O):
        enfants.setdefault(o["parent"], []).append(i)
    lest = np.array([("錘" in n or "锤" in n) for n in c.noms_os])
    oc = c.masque_os & ~lest
    Ph = c.pos[c.vh]
    dom = c.dom
    haut = Ph[:, 1] > c.y_yeux
    i_tete = c.i_tete
    tete_desc = set()
    if i_tete >= 0:     # head bones: the head and its descendants outside the hairstyle
        pile = [i_tete]
        while pile:
            i = pile.pop()
            if not oc[i]:
                tete_desc.add(i)
                pile += enfants.get(i, [])

    def demi_largeur(i, j):
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

    reglages_out, couverts, introuvables = [], set(), []
    toutes_collisions = []
    for s in sims:
        formes = {}
        for col in s.get("collisions", []):
            io = _trouver_os(c, col.get("bone_name"), index_norm)
            if col.get("bone_name") and io is None:
                introuvables.append(col["bone_name"])
            f = _forme(c, col, io)
            sur_tete = (io is not None and io in tete_desc) or (
                io is None and f[0] != "plane" and np.linalg.norm(f[1] - c.C) < c.R)
            formes[col["nom"]] = (f, sur_tete, col, io)
            toutes_collisions.append({
                "simulateur": s["nom"], "nom": col["nom"], "forme": col["type"],
                "os": c.noms_os[io] if io is not None else col.get("bone_name"),
                "sur_tete": bool(sur_tete), "rayon_H": round(f[3] / H, 3),
                "longueur_H": round(float(np.linalg.norm(f[2] - f[1])) / H, 3),
                "dedans": f[5], "centre_au_crane_H": round(float(np.linalg.norm(0.5 * (f[1] + f[2]) - c.C)) / H, 3),
                "defauts": col.get("defauts", [])})
        for r in s.get("reglages", []):
            ir = _trouver_os(c, r.get("root_bone_name"), index_norm)
            ie = _trouver_os(c, r.get("end_bone_name"), index_norm)
            if ir is None or ie is None:
                introuvables += [n for n, i in ((r.get("root_bone_name"), ir), (r.get("end_bone_name"), ie)) if i is None]
                reglages_out.append({"simulateur": s["nom"], "racine": r.get("root_bone_name"),
                                     "fin": r.get("end_bone_name"), "erreur": "bone not found in the GLB"})
                continue
            ch, k = [ie], ie
            while k != ir and c.par[k] >= 0:
                k = c.par[k]
                ch.append(k)
            ch = ch[::-1]
            if ch[0] != ir:
                reglages_out.append({"simulateur": s["nom"], "racine": c.noms_os[ir], "fin": c.noms_os[ie],
                                     "erreur": "the end bone does not descend from the root"})
                continue
            couverts.update(ch[1:] if len(ch) > 1 else ch)
            pos = np.array([O[i]["pos"] for i in ch])
            queues = list(pos[1:])
            if r.get("extend_end_bone") and float(r.get("end_bone_length") or 0) > 0:
                ax = np.asarray(O[ie]["base"])[:, 1]       # bone +Y axis, as in Godot (default end_bone/direction)
                queues.append(pos[-1] + ax * float(r["end_bone_length"]) * O[ie]["echelle_os"] * u)
            seg = np.linalg.norm(np.diff(np.vstack([pos[:1], np.array(queues)]) if queues else pos, axis=0), axis=1)
            L = float(seg.sum())
            t_long = np.cumsum(seg) / L if L > 0 else np.zeros(len(seg))
            # per-joint values (individual_config, or value x curve along the chain)
            par_j = {}
            for cle in ("radius", "stiffness", "drag", "gravity"):
                if r.get("individual_config") and r.get("joints"):
                    vals = [j.get(cle) if j.get(cle) is not None else r[cle] for j in r["joints"]]
                    vals = (vals + [vals[-1]] * len(seg))[:len(seg)] if vals else [r[cle]] * len(seg)
                else:
                    vals = [float(r[cle]) * _echant(r.get(cle + "_courbe"), t) for t in t_long]
                par_j[cle] = [float(x) for x in vals]
            rad = np.array(par_j["radius"]) * u          # joint radius, measurement units
            # radius / strand half-width (P7)
            rapports = []
            for n_, i in enumerate(ch[:-1]):
                dl = demi_largeur(i, ch[n_ + 1])
                if dl and dl > 0:
                    rapports.append(rad[min(n_, len(rad) - 1)] / dl)
            # collisions active for this setting
            if r.get("enable_all_child_collisions", True):
                actives = [n for n in formes if n not in {str(x).split("/")[-1] for x in r.get("exclude_collisions", [])}]
            else:
                actives = [n for n in formes if n in {str(x).split("/")[-1] for x in r.get("collisions", [])}]
            tete_act = [n for n in actives if formes[n][1]]
            # overlap at rest: sphere of each tail against each collision (P16)
            rec, rec_tete = [], []
            Q = np.array(queues) if queues else pos[-1:]
            for n in actives:
                f = formes[n][0]
                d = (_distance(f, Q) - rad[:len(Q)]) / H
                rec.append(float(d.min()))
                if formes[n][1]:
                    rec_tete.append(float(d.min()))
            k_ = np.array(par_j["stiffness"])
            g_ = np.array(par_j["gravity"])
            dr = np.array(par_j["drag"])
            seg_godot = seg / u                       # bone length in Godot units
            # carried vertices: dominant bone in the chain or below it
            sous, kk = list(ch), 0
            while kk < len(sous):
                sous += [x for x in enfants.get(sous[kk], []) if x not in sous]
                kk += 1
            reglages_out.append({
                "simulateur": s["nom"], "racine": c.noms_os[ir], "fin": c.noms_os[ie],
                "os": [c.noms_os[i] for i in ch], "maillons": len(seg), "longueur_H": round(L / H, 3),
                "espacement_H": [round(float(x) / H, 3) for x in seg],
                "racine_dist_crane_H": round(float((np.linalg.norm(pos[0] - c.C) - c.R) / H), 3),
                "sommets_portes": int(np.isin(dom, sous).sum()),
                "rayon_H": _q(list(rad / H)), "rayon_sur_demi_largeur": _q(rapports, 2),
                "stiffness": _q(list(k_)), "drag": _q(list(dr)), "gravity": _q(list(g_)),
                "gravity_direction": r.get("gravity_direction"),
                "amort_bullet_equiv": _q(list(1 - (1 - np.clip(dr, 0, 1)) ** PAS_PAR_SECONDE), 4),
                "vitesse_gardee_par_s": _q(list((1 - np.clip(dr, 0, 1)) ** PAS_PAR_SECONDE), 4),
                "affaissement_repos_deg": _q([math.degrees(math.atan2(g, k)) for g, k in zip(g_, k_)], 1),
                "rappel_rad_s": _q(list(k_ / np.maximum(seg_godot, 1e-9)), 2),
                "collisions": actives, "collisions_tete": tete_act, "touche_tete": bool(tete_act),
                "recouvrement_repos_min_H": round(min(rec), 3) if rec else None,
                "recouvrement_repos_tete_min_H": round(min(rec_tete), 3) if rec_tete else None,
                "courbes": [k for k in ("radius", "stiffness", "drag", "gravity") if r.get(k + "_courbe")],
                "individual_config": bool(r.get("individual_config")), "defauts": r.get("defauts", [])})
    # hair inside the head collisions at rest (union of the shapes attached to the head)
    formes_tete = []
    for s in sims:
        for col in s.get("collisions", []):
            io = _trouver_os(c, col.get("bone_name"), index_norm)
            f = _forme(c, col, io)
            if f[0] != "plane" and ((io is not None and io in tete_desc) or
                                    (io is None and np.linalg.norm(f[1] - c.C) < c.R)):
                formes_tete.append(f)
    if formes_tete:
        dmin = np.min([_distance(f, Ph) for f in formes_tete], axis=0)
        dedans = round(float((dmin < 0).mean()), 3)
        ecart = {k: round(float(v), 3) for k, v in zip(("p10", "p50", "p90"), np.percentile(dmin[haut] / H, [10, 50, 90]))} if haut.any() else None
    else:
        dedans, ecart = None, None
    # coverage: hairstyle chain bones driven by a spring, and the vertices they carry
    os_ch = np.where(oc)[0]
    cov_os = float(np.isin(os_ch, list(couverts)).mean()) if len(os_ch) else None
    dyn_v = np.isin(dom, list(couverts))
    ok = [x for x in reglages_out if "erreur" not in x]
    return {
        "fichiers": sorted({s["fichier"] for s in sims}),
        "simulateurs": [s["nom"] for s in sims], "reglages": reglages_out, "nombre_reglages": len(ok),
        "collisions": toutes_collisions, "os_introuvables": sorted(set(introuvables)),
        "os_coiffure_couverts_part": round(cov_os, 3) if cov_os is not None else None,
        "sommets_sur_ressort_part": round(float(dyn_v.mean()), 3),
        "reglages_touchant_tete_part": round(float(np.mean([x["touche_tete"] for x in ok])), 3) if ok else None,
        "cheveux_dans_collision_tete_part": dedans, "ecart_calotte_collision_tete_H": ecart,
        "pas_par_seconde_suppose": PAS_PAR_SECONDE,
        "echelle_squelette_supposee": 1.0,
        "non_comparables": NON_COMPARABLES}
