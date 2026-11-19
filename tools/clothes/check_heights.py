"""Height check of fitted garments: does each garment sit on the target body's anatomy as it sat on its source?

    bin/blender-queue -b --factory-startup --python tools/clothes/check_heights.py -- \
        --body BODY.pmx --output DIR <piece folder> [<piece folder> ...]
    python3 -I tools/clothes/check_heights.py --summary DIR [DIR_BEFORE]   # table (and before/after)

For each part (library folder), the part is fitted onto BODY with `porter.porter` (as the workshop does,
skin removed under it), then landmarks are measured with the same functions on both sides:
- body landmarks (`landmarks.py`): crotch C, waist W, underbust U, bust apex B, shoulders S; on the target
  its skin, on the source the innermost surface (skin, or the garment that stands in for it);
- garment landmarks, on the part alone: its bust apex (most forward point of its front, near the body's
  chest), its underbust fold (where its front recedes, when it follows the breast), its top edge (bottoms),
  its hem (tops), its crotch (ray cast upwards at x = 0).
A garment landmark at height y on the source is expected at map_(y) on the target, map_ being the
piecewise-linear map through the two bodies' landmarks (crotch on crotch, waist on waist...). The error is
the distance (cm) between where it lands and where it is expected; |error| > THRESHOLD_CM is a defect.
Also: whether the target's bust apex is covered by the part when the source's was. A top open in front on
its source (apex not covered) or without a fold under its most forward point (a puffer) has no bust
landmarks; above the shoulders, the neck joint is the landmark.

Writes DIR/<character>__<part>.json, DIR/<character>__<part>.png (row 1: source, row 2: target; front, side, close-ups
of the chest from the front and the side, close-up of the pelvis; thin lines = body landmarks, thick ticks at the edges = garment landmarks, same colours:
C blue, W green, U orange, B red, top edge/hem magenta, white tick = expected height on the target).
"""
import glob
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
for d in ("rebuilt_head", "rebuilt_body", "head", "atlas"):
    sys.path.insert(0, os.path.join(TOOLS, d))
sys.path.insert(0, HERE)

CM = 8.0
THRESHOLD_CM = 2.5
TOPS = ("top", "jacket", "waistcoat", "coat", "jumpsuit", "dress")
BOTTOM = ("shorts", "trousers", "skirt", "underwear", "dress", "jumpsuit")
CROTCH = ("shorts", "trousers", "underwear", "jumpsuit")
COLOR = {"C": (60, 110, 255), "W": (40, 200, 70), "U": (255, 150, 30), "B": (240, 40, 40),
        "top": (230, 60, 230), "bottom": (230, 60, 230)}


# ------------------------------------------------------------------ garment landmarks
def front_part(bvh, bones, ys, half_width):
    """Forward extent of the part's front per height (rays cast from in front, first hit), relative to the
    trunk axis; -inf where the part is absent."""
    import landmarks as RP
    zc = RP.axe_z(bones)(ys)
    xs = np.arange(-half_width, half_width + 1e-9, 0.5 / CM)
    out = np.full(len(ys), -np.inf)
    for j, y in enumerate(ys):
        for x in xs:
            loc = bvh.ray_cast((float(x), float(y), -10.0), (0.0, 0.0, 1.0), 20.0)[0]
            if loc is not None:
                out[j] = max(out[j], -(loc[2] - zc[j]))
    return out


def part_trunk(pmx, chains=(0,)):
    """Share of each vertex's weights held by the given chains (transport.CHAINS; 0 = trunk)."""
    import transport as TR
    seg = TR._segments_os(pmx["bones"])
    trunk = [list(TR.SEGMENTS).index(n) for c in chains for n in TR.CHAINS[c]]
    I = np.asarray(pmx["vertices"]["bones"], np.int64).reshape(len(pmx["vertices"]["pos"]), -1)
    W = np.asarray(pmx["vertices"]["weights"], float).reshape(I.shape)
    W = np.where(I >= 0, W, 0.0)
    t = np.isin(seg[np.clip(I, 0, len(seg) - 1)], trunk) & (I >= 0)
    return (W * t).sum(1) / np.maximum(W.sum(1), 1e-9)


def borders(V, bones, top=True):
    """Median over 24 sectors around the trunk axis of the highest (top) or lowest vertex."""
    import landmarks as RP
    zc = RP.axe_z(bones)(V[:, 1])
    a = np.arctan2(V[:, 0], -(V[:, 2] - zc))
    s = ((a + np.pi) / (2 * np.pi) * 24).astype(int) % 24
    v = [V[s == k, 1].max() if top else V[s == k, 1].min() for k in range(24) if (s == k).any()]
    return float(np.median(v)) if len(v) >= 12 else np.nan


def landmarks_part(P, F, pmx_os, body_r, typ, trunk):
    """Garment landmarks (MMD units) of the part (P, F) on a body with landmarks corps_r."""
    import landmarks as RP
    out = {}
    b = RP.tree(P, F)
    L = body_r["S"] - body_r["hip"]
    if typ in TOPS:
        ys = np.arange(body_r["hip"] + 0.45 * L, body_r["S"] - 0.05 * L + 1e-9, RP.STEP_Y)
        fr = front_part(b, pmx_os, ys, 0.3 * L)
        if np.isfinite(fr).mean() >= 0.5:
            j = int(np.argmax(fr))
            out["B"] = float(ys[j])
            # the fold, read over the whole front below the apex (down to the hips)
            ys2 = np.arange(body_r["hip"], ys[j] + 1e-9, RP.STEP_Y)
            fr2 = front_part(b, pmx_os, ys2, 0.3 * L)
            fr2 = np.where(np.isfinite(fr2), fr2, np.nan)
            yU, R = RP.fold(ys2, fr2, len(ys2) - 1, 0.3 * L)
            if R >= 1.0 / CM:
                out["U"] = yU
            out["projection_cm"] = round(R * CM, 1)
    V = P[np.unique(F)]
    tv = trunk[np.unique(F)] >= 0.5
    if typ in BOTTOM and tv.sum() >= 12:
        out["top"] = borders(V[tv], pmx_os, True)
    if typ in ("top", "jacket", "waistcoat", "coat") and tv.sum() >= 12:
        out["bottom"] = borders(V[tv], pmx_os, False)
    if typ in CROTCH:
        y = RP.crotch(b, pmx_os)
        if np.isfinite(y):
            out["C"] = float(y)
    return out


def covered(bvh_part, Pc, Fc, bones, yB):
    """Is the body's bust apex (most forward point at height yB, either side) under the part?"""
    import landmarks as RP
    b = RP.tree(Pc, Fc)
    ys, ang, pts = RP.sections(b, bones, yB, yB)
    zc = RP.axe_z(bones)(ys)
    p = pts[0]
    m = np.abs(((ang + np.pi) % (2 * np.pi)) - np.pi) <= RP.FRONT
    ok = m & np.isfinite(p[:, 0])
    if not ok.any():
        return None
    k = np.where(ok)[0][np.argmin(p[ok, 2])]
    o = p[k] + np.array([0, 0, 2e-3])        # just behind the surface: on a source, it may be the part itself
    return bvh_part.ray_cast(tuple(o), (0.0, 0.0, -1.0), 1.0)[0] is not None


# ------------------------------------------------------------------ drawing
POLICE = {  # 3x5 glyphs, rows top to bottom
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001010010010",
    "8": "111101111101111", "9": "111101111001111", ".": "000000000000010", "-": "000000111000000",
    "+": "000010111010000", " ": "000000000000000", ":": "000010000010000", "=": "000111000111000",
    "_": "000000000000111", "/": "001001010100100", "%": "101001010100101",
}
for _c, _g in {"A": "010101111101101", "B": "110101110101110", "C": "111100100100111", "D": "110101101101110",
               "E": "111100110100111", "F": "111100110100100", "G": "111100101101111", "H": "101101111101101",
               "I": "111010010010111", "J": "001001001101111", "K": "101101110101101", "L": "100100100100111",
               "M": "101111111101101", "N": "110101101101101", "O": "111101101101111", "P": "111101111100100",
               "Q": "111101101111001", "R": "110101110101101", "S": "111100111001111", "T": "111010010010010",
               "U": "101101101101111", "V": "101101101101010", "W": "101101111111101", "X": "101101010101101",
               "Y": "101101010010010", "Z": "111001010100111"}.items():
    POLICE[_c] = _g


def write_text(img, x, y, text, color=(255, 255, 255), k=3):
    for c in text.upper():
        g = POLICE.get(c, POLICE[" "])
        for i in range(5):
            for j in range(3):
                if g[i * 3 + j] == "1":
                    img[y + i * k:y + (i + 1) * k, x + j * k:x + (j + 1) * k] = color
        x += 4 * k
    return img


def view(V, F, color, az, box, T, lines, ticks, expected=(), title=""):
    """Grey render (rebuilt_body/view.py) with horizontal lines: lines {name: y (cm)} thin and full width,
    ticks {name: y} thick at both edges, expected [y] white ticks."""
    import view as VU
    img = VU.render(V, F, (az, 0), T, box, color=color).copy()
    d, h, o = VU.base(az, 0)
    sel = np.all((V >= np.asarray(box[:3])) & (V <= np.asarray(box[3:])), 1)
    u = np.unique(F[sel[F].all(1)])
    P = np.stack([V @ d, V @ h], 1)
    lo, hi = P[u].min(0), P[u].max(0)
    c, r = 0.5 * (lo + hi), 0.54 * float((hi - lo).max())

    def py(y):
        return int(round(0.5 * T - (y - c[1]) / r * 0.5 * T))
    for n, y in lines.items():
        if np.isfinite(y) and 1 <= py(y) < T - 1:
            img[py(y) - 1:py(y) + 1, :] = COLOR[n]
    for n, y in ticks.items():
        if np.isfinite(y) and 2 <= py(y) < T - 2:
            img[py(y) - 2:py(y) + 3, :T // 6] = COLOR[n]
            img[py(y) - 2:py(y) + 3, -T // 6:] = COLOR[n]
    for y in expected:
        if np.isfinite(y) and 1 <= py(y) < T - 1:
            img[py(y) - 1:py(y) + 2, T // 6:T // 6 + T // 12] = 255
            img[py(y) - 1:py(y) + 2, -T // 6 - T // 12:-T // 6] = 255
    if title:
        write_text(img, 6, 6, title, k=2)
    return img


def sheet(side, T=330):
    """One row: front, side, close-ups of the chest (front and side) and of the pelvis (close-ups and side
    views without the arms, which hide the chest in an A pose). cote: dict with V (cm), F, color, bras (faces held by an arm), corps_r, piece_r, expected (cm)."""
    V, F, color = side["V"], side["F"], side["color"]
    Fp, cp = F[~side["arm"]], color[~side["arm"]]
    r = {k: side["body_r"][k] * CM for k in ("C", "W", "U", "B", "S")}
    lines = {k: r[k] for k in ("C", "W", "U", "B")}
    ticks = {k: v * CM for k, v in side["part_r"].items() if k in COLOR and isinstance(v, float)}
    att = [v * CM for v in side.get("expected", {}).values()]
    L = r["S"] - r["C"]
    big = [-60, r["C"] - 0.35 * L, -60, 60, r["S"] + 0.2 * L, 60]
    near = [-60, r["U"] - 0.35 * L, -60, 60, r["S"] + 0.05 * L, 60]
    pelvis = [-60, r["C"] - 0.3 * L, -60, 60, r["W"] + 0.1 * L, 60]
    return np.concatenate([view(V, F, color, 0, big, T, lines, ticks, att, side["title"]),
                           view(V, Fp, cp, 90, big, T, lines, ticks, att, side.get("text", "")),
                           view(V, Fp, cp, 0, near, T, lines, ticks, att, "BUST FRONT"),
                           view(V, Fp, cp, 90, near, T, lines, ticks, att, "BUST PROFILE"),
                           view(V, Fp, cp, 0, pelvis, T, lines, ticks, att, "PELVIS")], 1)


# ------------------------------------------------------------------ one part
def check(part_dir, body, output, cache_src={}):
    import pmx as PMX
    import model as MD
    import wear as PO
    import landmarks as RP
    card, path_part = PO.load_part(part_dir)
    name = "%s__%s" % (card["character"], card["id"])
    typ = card.get("type")
    # ---- target body
    if body not in cache_src:
        C = PMX.read(body)
        T = PMX.to_arrays(C)
        names = [m["name"] for m in C["materials"]]
        skin = np.array([bool(MD.SKIN.match(n)) for n in names])[T["mat_face"]]
        cache_src[body] = (C, T, skin, RP.landmarks(T["pos"], T["faces"][skin], C["bones"]))
    C, Tc, skin_c, rc = cache_src[body]
    # ---- source body: innermost surface (skin and garments, no face or hair)
    src = card["source"]["pmx"]
    if src not in cache_src:
        S = PMX.read(src)
        T = PMX.to_arrays(S)
        names = [m["name"] for m in S["materials"]]
        guard = np.array([not MD.NON_GARMENT.match(n) for n in names])[T["mat_face"]]
        cache_src[src] = (S, T, guard, RP.landmarks(T["pos"], T["faces"][guard], S["bones"]))
    S, Ts, keep_s, rs = cache_src[src]
    # ---- the part on its source
    Pp = PMX.read(path_part)
    Tp = PMX.to_arrays(Pp)
    gs = landmarks_part(Tp["pos"], Tp["faces"], S["bones"], rs, typ, part_trunk(Pp))
    # ---- the part fitted on the target
    tmp = tempfile.mkdtemp(prefix="heights-", dir=output)
    res = PO.wear(part_dir, body, tmp, name="part", margin_mm=3.0, skin="remove")
    O = PMX.read(res["pmx"])
    To = PMX.to_arrays(O)
    names_o = [m["name"] for m in O["materials"]]
    fp = np.isin(To["mat_face"], [i for i, n in enumerate(names_o) if n in res["mats"]])
    gt = landmarks_part(To["pos"], To["faces"][fp], C["bones"], rc, typ, part_trunk(O))
    # above the shoulders (collars), the neck joint is the landmark: the joint the head turns about
    for r_, o_ in ((rs, S["bones"]), (rc, C["bones"])):
        j = RP._joint(o_, ("首",))
        r_["首"] = float(j[1]) if j is not None else np.nan
    to_t, _ = RP.map_(rs, rc, RP.NAMES + ("首",))
    cov = {}
    if typ in TOPS:
        # the target's whole skin (as before the fitting removed what the part hides)
        cov["target"] = covered(RP.tree(To["pos"], To["faces"][fp]), Tc["pos"], Tc["faces"][skin_c], C["bones"], rc["B"])
        cov["source"] = covered(RP.tree(Tp["pos"], Tp["faces"]), Ts["pos"], Ts["faces"][keep_s], S["bones"], rs["B"])
    lines, errors = {}, []
    for k in ("C", "W", "U", "B", "top", "bottom"):
        if k in ("U", "B") and cov.get("source") is False:
            continue             # open in front on its source: its apex and fold are not the bust's
        if k == "B" and "U" not in gs:
            continue             # no fold under its most forward point: not a bust shape (a puffer's belly)
        if k in gs and k in gt and np.isfinite(gs[k]) and np.isfinite(gt[k]):
            att = float(to_t(gs[k]))
            e = (gt[k] - att) * CM
            lines[k] = {"source_cm": round(gs[k] * CM, 1), "target_cm": round(gt[k] * CM, 1),
                         "expected_cm": round(att * CM, 1), "error_cm": round(e, 1),
                         # relative to the body's own landmark of the same name (C, U, B), source and target
                         **({"gap_source_cm": round((gs[k] - rs[k]) * CM, 1),
                             "gap_target_cm": round((gt[k] - rc[k]) * CM, 1)} if k in rs else {})}
            errors.append(abs(e))
    assessment = {"part_name": name, "type": typ, "body": os.path.abspath(body), "source": src,
             "body_source_cm": {k: round(rs[k] * CM, 1) for k in ("C", "hip", "W", "U", "B", "S")},
             "body_target_cm": {k: round(rc[k] * CM, 1) for k in ("C", "hip", "W", "U", "B", "S")},
             "landmarks": lines, "bust_covered": cov, "worst_cm": round(max(errors), 1) if errors else None,
             "default": bool(errors and max(errors) > THRESHOLD_CM) or (cov.get("source") is True and cov.get("target") is False),
             "fitting": {k: res.get(k) for k in ("under_skin", "stays_inside", "elongation", "transport")}}
    with open(os.path.join(output, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(assessment, f, ensure_ascii=False, indent=1)
    # ---- board
    gray = np.array([0.86, 0.80, 0.78])
    blue = np.array([0.35, 0.55, 0.95])
    others = np.array([0.62, 0.62, 0.66])
    Fs = Ts["faces"][keep_s]
    the_key = {tuple(v) for v in np.round(Tp["pos"], 4)}
    in_ = np.array([tuple(v) in the_key for v in np.round(Ts["pos"], 4)])
    pf = in_[Fs].all(1)
    skin_s = np.array([bool(MD.SKIN.match(m["name"])) for m in S["materials"]])[Ts["mat_face"]][keep_s]
    color_s = np.where(pf[:, None], blue, np.where(skin_s[:, None], gray, others))
    keep_o = np.array([not MD.NON_GARMENT.match(n) for n in names_o])[To["mat_face"]]
    Fo = To["faces"][keep_o]
    color_o = np.where(fp[keep_o][:, None], blue, gray)
    att = {k: to_t(gs[k]) for k in gs if k in ("C", "U", "B", "top", "bottom") and np.isfinite(gs[k])}
    arm_s = part_trunk(S, (0, 3, 4))[Fs].mean(1) < 0.5       # arms and hands
    arm_o = part_trunk(O, (0, 3, 4))[Fo].mean(1) < 0.5
    text = " ".join("%s%+.0f" % ({"top": "T", "bottom": "H"}.get(k, k), v["error_cm"]) for k, v in lines.items())
    tops = sheet({"V": Ts["pos"] * CM, "F": Fs, "color": color_s, "arm": arm_s, "body_r": rs, "part_r": gs,
                     "title": "SOURCE " + name.replace("__", " ")})
    bottom = sheet({"V": To["pos"] * CM, "F": Fo, "color": color_o, "arm": arm_o, "text": text, "body_r": rc, "part_r": gt, "expected": att,
                   "title": "TARGET WORST %s CM" % (assessment["worst_cm"] if assessment["worst_cm"] is not None else "-")})
    import png
    png.write(os.path.join(output, name + ".png"), np.concatenate([tops, bottom], 0))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return assessment


# ------------------------------------------------------------------ table
def judge(b):
    """(worst error, defect) of a check, with the rules of this version (older JSON files are judged again)."""
    rp = b["landmarks"]
    e = [abs(v["error_cm"]) for k, v in rp.items() if not (k == "B" and "U" not in rp)
         and not (k in ("U", "B") and b["bust_covered"].get("source") is False)]
    cv = b["bust_covered"]
    worst = round(max(e), 1) if e else None
    return worst, bool(e and worst > THRESHOLD_CM) or (cv.get("source") is True and cv.get("target") is False)


def assessment(folder, before=None):
    def read(d):
        return {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(d, "*.json")))}
    after = read(folder)
    av = read(before) if before else {}
    lines = ["piece                 | " + " | ".join("%-15s" % k for k in ("B apex", "U fold", "C crotch", "top edge", "hem"))
              + " | covered | defect" + (" | before (worst)" if av else "")]
    for name, b in after.items():
        cel = []
        b["worst_cm"], b["default"] = judge(b)
        for k in ("B", "U", "C", "top", "bottom"):
            x = b["landmarks"].get(k)
            if k == "B" and "U" not in b["landmarks"]:
                x = None
            cel.append("%-15s" % ("" if x is None else "%+5.1f (%5.1f)" % (x["error_cm"], x["target_cm"])))
        cv = b["bust_covered"]
        cov = "" if not cv else "%s>%s" % ("y" if cv.get("source") else "n", "y" if cv.get("target") else "n")
        s = "%-21s | %s | %-7s | %s" % (name, " | ".join(cel), cov, "DEFECT" if b["default"] else "ok")
        if name in av:
            s += " | %s (%s)" % (lambda pd: ("DEFECT" if pd[1] else "ok", pd[0]))(judge(av[name]))
        lines.append(s)
    lines.append("cells: error in cm (target height in cm); error = landed height - height expected from the source's "
                  "anatomy; defect if |error| > %.1f cm or bust apex no longer covered" % THRESHOLD_CM)
    return "\n".join(lines)


def main(argv):
    if "--summary" in argv:
        i = argv.index("--summary")
        print(assessment(argv[i + 1], argv[i + 2] if len(argv) > i + 2 else None))
        return
    val = lambda o: argv[argv.index(o) + 1]  # noqa: E731
    body, output = val("--body"), val("--output")
    os.makedirs(output, exist_ok=True)
    parts = [a for k, a in enumerate(argv) if not a.startswith("--") and (k == 0 or argv[k - 1] not in ("--body", "--output"))]
    for p in parts:
        try:
            b = check(p, body, output)
            print("HEIGHTS", b["part_name"], "worst %s cm" % b["worst_cm"], "DEFECT" if b["default"] else "ok", flush=True)
        except Exception as e:                    # noqa: BLE001  (one part failing does not stop the others)
            import traceback
            traceback.print_exc()
            print("HEIGHTS", p, "FAILED", e, flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
