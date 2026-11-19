"""Real assembly of a combination (a single Blender process) with the project's tools:
place() on the base head, the hairstyle fitted on the chosen face's model (already sewn onto the
common body, see prepare.make_head), wear part by part, recolouring, layers, measurement, render.

    blender -b --factory-startup --python assemble_blender.py -- --dossier D
D/selection.json describes the choice; D/state.json tracks progress (read by the page).
"""
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import pmx as PMX  # noqa: E402
import pmx_write as pmx_write  # noqa: E402

BLOCKING = ("rest_crossing", "rest_crossing_p90_mm", "body_only_poses")
STEPS = ["hairstyle", "graft", "parts", "color", "checks", "skeleton", "vrm", "render"]
D, STATE = None, {}


def state(**kw):
    STATE.update(kw)
    C.write_json(os.path.join(D, "state.json"), STATE)


def recolor(path, sel, hair):
    """Hair tint: same computation as the page's shader (tools/workshop/page/viewer.js)."""
    import bpy
    P = PMX.read(path)
    hair = set(hair)
    t = sel["tint"]
    tr, tg, tb = t["r"], t["g"], t["b"]
    vcible = max(tr, tg, tb)
    scible = 0.0 if vcible <= 0 else 1 - min(tr, tg, tb) / vcible
    # target tint as saturated RGB, recomposed on each pixel
    pure = np.array([tr, tg, tb]) / max(vcible, 1e-6)
    ref = max(float(t.get("ref") or 0.5), 1e-3)
    made = {}
    for m in P["materials"]:
        if m["name"] not in hair or m["texture"] < 0:
            continue
        i = m["texture"]
        if i not in made:
            rel = P["textures"][i].replace("\\", "/")
            src = os.path.join(os.path.dirname(path), rel)
            im = bpy.data.images.load(src, check_existing=False)
            w, h = im.size
            px = np.empty(w * h * 4, np.float32)
            im.pixels.foreach_get(px)
            px = px.reshape(h, w, 4)
            v = px[..., :3].max(2)
            v2 = np.minimum(v * vcible / ref, 1.0)
            rgb = (v2[..., None] * pure) * scible + v2[..., None] * (1 - scible)
            px[..., :3] = rgb
            name, _ = os.path.splitext(rel)
            new = name + "_tint.png"
            out = bpy.data.images.new("t", w, h, alpha=True)
            out.pixels.foreach_set(px.ravel())
            tmp = os.path.join(os.path.dirname(path), new)
            out.filepath_raw = tmp
            out.file_format = "PNG"
            out.save()
            bpy.data.images.remove(out)
            bpy.data.images.remove(im)
            P["textures"].append(new.replace("/", "\\"))
            made[i] = len(P["textures"]) - 1
        m["texture"] = made[i]
    if made:
        pmx_write.write(P, path + ".tmp.pmx")
        os.replace(path + ".tmp.pmx", path)
    return len(made)


def keep_morphs(model, path):
    """head_graft keeps only the morphs of what it brings (here, hair): those of the face's model
    (expressions) come back as they are. Its vertices, materials and bones stayed first, in order."""
    M, P = PMX.read(model), PMX.read(path)
    n = len(M["vertices"]["pos"])
    if not np.array_equal(np.asarray(P["vertices"]["pos"][:n]), np.asarray(M["vertices"]["pos"])) or \
       [m["name"] for m in P["materials"][:len(M["materials"])]] != [m["name"] for m in M["materials"]] or \
       [o["name"] for o in P["bones"][:len(M["bones"])]] != [o["name"] for o in M["bones"]]:
        raise RuntimeError("the face's model was renumbered by fitting the hair: expressions lost")
    P["morphs"] = list(M["morphs"]) + list(P["morphs"])
    pmx_write.write(P, path + ".tmp.pmx")
    os.replace(path + ".tmp.pmx", path)
    return len(M["morphs"])


def main():
    global D
    a = sys.argv[sys.argv.index("--") + 1:]
    D = a[a.index("--dossier") + 1]
    sel = C.read_json(os.path.join(D, "selection.json"))
    state(step="hairstyle", steps=STEPS, start=time.time(), done=False)
    cards = {i: C.arrange(C.read_json(os.path.join(C.CACHE, "parts", i, "meta.json"))) for i in sel["parts"]}
    unknown = [i for i, m in cards.items() if not m]
    if unknown:   # the server only sends catalogue ids; a hand-written selection may not (refused before any work)
        raise ValueError("unknown pieces (not in the cache): %s" % ", ".join(repr(i) for i in unknown))
    lib = os.path.join(C.CACHE, "library")
    import hairstyle as CO
    tm = C.read_json(os.path.join(C.CACHE, "heads", sel["head"], "meta.json"))
    d_co = os.path.join(D, "1-hairstyle")
    cm = C.read_json(os.path.join(C.CACHE, "hairstyles", sel["hairstyle"], "meta.json")) or {}
    if cm.get("free"):      # free hairstyle: made for the free head, already in place; nothing to pose
        hair = cm["pmx"]
        mats_hair = [m["name"] for m in PMX.read(hair)["materials"] if m["n_indices"] > 0]
    else:
        CO.place(lib, sel["hairstyle"], tm["library"], d_co)
        hair = os.path.join(d_co, "result.pmx")
        Pc = PMX.read(hair)
        # the materials posed by place() are those the original head did not have
        # (their suffix comes from the source's name, not from the catalogue id)
        import hairstyles as CF
        origin = {m["name"] for m in PMX.read(CF.BI.read_card(lib, tm["library"])["pmx"])["materials"]}
        mats_hair = [m["name"] for m in Pc["materials"] if m["n_indices"] > 0 and m["name"] not in origin]
    state(step="graft")
    d_gr = os.path.join(D, "2-graft")
    # only the hair goes onto the face's model: the head, its neck seam and the skin are not touched
    C.graft(["--body", C.absolute(tm["model"]), "--head", hair, "--output", d_gr, "--take", ",".join(mats_hair),
               "--remove", "", "--keep-skull", "--no-tint", "--max-neck-gap", "10",
               # the hair was posed on the game's base head (its skeleton), it is worn on our body (longer neck,
               # made by the house shape tool): strands that hang on the neck or shoulders follow the body's bones
               "--follow-bones"])
    current = os.path.join(d_gr, "result.pmx")
    print("face expressions:", keep_morphs(C.absolute(tm["model"]), current), flush=True)
    import wear as PO
    infos = []
    order = sorted(sel["parts"], key=lambda i: cards[i].get("rank", 50))
    state(step="parts", parts_total=len(order), current_part=0)
    # each part is fitted on the skin alone, whole; the outfit is then settled in one block, like the preview
    import outfit as CB
    discarded = CB.removed({i: cards[i] for i in order})
    order = [i for i in order if i not in discarded]
    state(parts_total=len(order))
    hidden = set()
    for k, ident in enumerate(order):
        m = cards[ident]
        d_p = os.path.join(D, "3-parts", "%02d" % k)
        # a shoe with a heel is worn on feet posed as in the preview (its pose found when it was prepared)
        res = PO.wear(C.absolute(m["folder"]), current, d_p, name="e", margin_mm=3.0, skin="note", surface="skin",
                        foot=m.get("foot"))
        current = os.path.join(d_p, "e.pmx")
        hidden.update(res["hidden_skin"])
        infos.append({"id": ident, "mats": res["mats"], "measure_type": res["measure_type"], "name": "%s · %s" % (m["character"], m["type_name"]),
                      "type": m["type"], "enveloping": C.enveloping(m), "meta": m,
                      "foot": res.get("foot")})
        state(current_part=k + 1)
    # the final folder holds the last state (PMX + textures)
    final = os.path.join(D, "model")
    if os.path.isdir(final):
        shutil.rmtree(final)
    shutil.copytree(os.path.dirname(current), final)
    path = os.path.join(final, "result.pmx")
    os.replace(os.path.join(final, os.path.basename(current)), path)
    for n in ("fitting.json", "graft.json"):
        if os.path.exists(os.path.join(final, n)):
            os.remove(os.path.join(final, n))
    P = PMX.read(path)
    remove_faces(P, hidden)
    layers, notes, removed = CB.resolve_pmx(P, infos, mats_hair)
    # the outfit was settled in the rest frame; with a posed shoe, the feet now take its pose (body and
    # every part, by their weights) and the whole model is raised onto its sole, as in the preview
    # (the height is the one of this fit: the shoe was fitted again over what is already worn)
    pose = next((i["foot"] for i in infos if i.get("foot")), None)
    if pose:
        import foot as PD
        PD.place_pmx(P, {k: v for k, v in pose.items() if isinstance(v, dict)})
        PD.lift_pmx(P, float(pose.get("lift", 0.0)) / PO.UNIT)
        print("feet posed in the shoe: %s, raised %.1f cm" % (
            ", ".join("%s %.0f/%.0f deg" % (k, v["ankle"], v["toes"]) for k, v in pose.items() if isinstance(v, dict)),
            100 * float(pose.get("lift", 0.0))), flush=True)
    for m in P["materials"]:   # no black outline (the author's choice): not in MMD, not on the contact sheet, not in the VRM
        m["flags"] &= ~0x10
    pmx_write.write(P, path + ".tmp.pmx")
    os.replace(path + ".tmp.pmx", path)
    notes = list(discarded.values()) + notes
    print("layers: %d faces hidden; %s" % (removed, "; ".join(notes)), flush=True)
    state(step="color")
    if sel.get("tint"):
        recolor(path, sel, mats_hair)
    state(step="checks")
    report = checks(model_check(path, os.path.join(d_gr, "result.pmx")), infos, CB.flag(layers),
                        {c.id: CB.part_tucked(c) for c in layers})
    report["notes"] = notes
    C.write_json(os.path.join(D, "checks.json"), report)
    state(step="skeleton")
    skeleton = skeleton_own_rig(path)
    state(step="vrm")
    vrm_ok, vrm_note = leave_vrm(path, sel, skeleton)
    state(step="render")
    sheet_note = None
    try:                       # the model is already written: an image failure does not make it fail
        import render as rendered
        sys.argv = ["x", "--", "--pmx", model_rendered(path, mats_hair), "--output", os.path.join(D, "renders"), "--views", "face,three_quarters,back",
                    "--samples", "16", "--size", "720", "--sheet", "sheet.png", "--mat", "肌", "--frame", "body"]
        rendered.main()
        C.link(os.path.join(D, "renders", "sheet.png"), os.path.join(D, "sheet.png"))
    except (Exception, SystemExit) as e:
        sheet_note = repr(e)[:300]
    state(step="done", sheet=sheet_note is None, sheet_note=sheet_note, done=True, ok=report["ok"], duration=round(time.time() - STATE["start"], 1),
         summary=report["summary"], failures=report["failures"], layers=report["layers"], crosses=report["crosses"], notes=notes,
         vrm=vrm_ok, vrm_note=vrm_note, skeleton=skeleton)


def skeleton_own_rig(path):
    """Replaces in result.pmx the ZZZ body's bones and weights with the house skeleton (tools/custom_body);
    the version on the ZZZ bones stays next to it (result-zzz.pmx). Runs after the checks, which read the ZZZ bones.
    Only on the remade body, the one this skeleton weights; workshop.skeleton = zzz (CHIMERA_SKELETON) keeps the ZZZ bones.
    Returns "own_rig" or "zzz". A failure here leaves the model on the ZZZ bones."""
    if C.R.value("workshop.skeleton", "custom") not in ("custom", "own_rig"):   # "own_rig": former name
        return "zzz"
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_body"))
        import plug as BR
        import measurements as ME
        if not os.path.exists(os.path.join(BR.OWN_RIG, "skeleton.json")):
            print("house skeleton missing (%s): ZZZ bones kept" % BR.OWN_RIG, flush=True)
            return "zzz"
        if not os.path.samefile(C.BODY, os.path.join(os.path.dirname(ME.SHAPE), "body.pmx")):
            print("house skeleton: made for the remade body, not for %s; ZZZ bones kept" % C.BODY, flush=True)
            return "zzz"
        Pm = BR.convert(PMX.read(path), log=lambda *a: print(*a, flush=True))
        zzz = os.path.join(os.path.dirname(path), "result-zzz.pmx")
        shutil.copyfile(path, zzz)
        pmx_write.write(Pm, path + ".tmp.pmx")
        os.replace(path + ".tmp.pmx", path)
        return "own_rig"
    except Exception:
        traceback.print_exc()
        return "zzz"


def leave_vrm(path, sel, skeleton="zzz"):
    """Writes model/result.vrm next to the PMX. A failure here does not prevent delivering the PMX."""
    import traceback
    import vrm as VR
    output = os.path.join(os.path.dirname(path), "result.vrm")
    try:
        P = PMX.read(path)
        twist = None
        if skeleton == "own_rig":
            import plug as BR
            with open(os.path.join(BR.OWN_RIG, "skeleton.json"), encoding="utf-8") as f:
                twist = {o["name"]: o["twist"] for o in json.load(f)["bones"] if o["twist"]}
        r = VR.write_vrm(P, output, VR.images_blender(P, os.path.dirname(path)), name="%s · %s" % (sel["head"], sel["hairstyle"]),
                          twist=twist)
        defaults = VR.validate(output)
        print("vrm:", r, "defects:", defaults, flush=True)
        return not defaults, "; ".join(defaults)
    except Exception as e:
        traceback.print_exc()
        return False, "%s: %s" % (type(e).__name__, e)


def remove_faces(P, indices):
    """Removes from the PMX the faces with the given indices (numbering of P["faces"])."""
    if not indices:
        return
    k, faces = 0, []
    for m in P["materials"]:
        n = m["n_indices"] // 3
        rest = [f for j, f in enumerate(P["faces"][k:k + n], k) if j not in indices]
        m["n_indices"] = len(rest) * 3
        faces += rest
        k += n
    P["faces"] = faces


def model_check(path, graft):
    """Copy of the model with all its skin: porter removes the hidden skin, and measure.py would
    otherwise read the holes as clipping. Same vertices, only the faces of 肌 come back."""
    P, G = PMX.read(path), PMX.read(graft)
    if P["materials"][0]["name"] != "肌" or G["materials"][0]["name"] != "肌":
        return path
    na, ng = P["materials"][0]["n_indices"] // 3, G["materials"][0]["n_indices"] // 3
    P["faces"] = list(G["faces"][:ng]) + list(P["faces"][na:])
    P["materials"][0]["n_indices"] = ng * 3
    return copy_neighbor(P, path, "check")


def copy_neighbor(P, path, name):
    """Writes P to D/<name>/result.pmx, with a link to the model's textures."""
    d = os.path.join(D, name)
    os.makedirs(d, exist_ok=True)
    link = os.path.join(d, "tex")
    if not os.path.lexists(link):
        os.symlink(os.path.join(os.path.dirname(path), "tex"), link)
    output = os.path.join(d, "result.pmx")
    pmx_write.write(P, output + ".tmp.pmx")
    os.replace(output + ".tmp.pmx", output)
    return output


def model_rendered(path, hair):
    """Copy for the contact sheet, without the "髮+" highlight layers: they repeat exactly
    the hair's faces, and render.py's outline (Solidify) then covers the whole hairstyle
    in black. The delivered PMX keeps these layers; MMD shows them normally."""
    P = PMX.read(path)
    reflections = {m["name"] for m in P["materials"] if m["name"] in set(hair) and "+" in m["name"] and m["n_indices"] > 0}
    if not reflections:
        return path
    faces, k = [], 0
    for m in P["materials"]:
        n = m["n_indices"] // 3
        if m["name"] in reflections:
            m["n_indices"] = 0
        else:
            faces += list(P["faces"][k:k + n])
        k += n
    P["faces"] = faces
    return copy_neighbor(P, path, "render")


def checks(path, infos, signal, hidden=None):
    """Layers between parts (same computation as the page), measure.py checks on each part."""
    import outfit as CB
    import model as MD
    import measure as measure
    M = None
    failures, details = [], []
    cps = [{"under": r["name_under"], "over": r["name_over"], "vertices": r["vertices"], "left_out": r["fraction"], "worst_mm": r["worst_mm"]}
           for r in signal]
    for inf in infos:
        spec = {"id": inf["id"], "character": inf["id"], "type": {"underwear": "shorts"}.get(inf["type"], inf["measure_type"]), "mats": inf["mats"],
                "texture_layers": [], "why": "workshop"}
        M = M or MD.Model("assembly", path=path)
        try:
            r = measure.measure(spec, M, True)
        except Exception as e:
            failures.append("%s: measurement impossible (%s)" % (inf["name"], str(e)[:80]))
            continue
        ctr = [{"name": x["name"], "value": x["value"], "threshold": x["threshold"], "ok": x["ok"]} for x in r["checks"]]
        details.append({"part_name": inf["name"], "checks": ctr})
        for x in ctr:
            if x["ok"] is False and x["name"] == "rest_crossing" and not CB.crossing_visible(x["value"], (hidden or {}).get(inf["id"], 0.0)):
                x["ok"] = None   # in excess only where another part covers this one
            if x["ok"] is False and x["name"] in BLOCKING:
                failures.append("%s: %s = %s (threshold %s)" % (inf["name"], x["name"], x["value"], x["threshold"]))
    reread = PMX.read(path)
    n_s, n_m = len(reread["vertices"]["pos"]), len(reread["materials"])
    crosses = [c for c in cps if c["left_out"] > 0.01 and c["left_out"] * c["vertices"] >= 6]
    return {"ok": not failures, "failures": failures, "layers": cps, "crosses": crosses, "detail": details,
            "summary": "%d vertices, %d materials, %d bones; %d parts measured, %d layer pairs, %d clippings" % (
                n_s, n_m, len(reread["bones"]), len(details), len(cps), len(crosses))}


if __name__ == "__main__":
    try:
        main()
    except (Exception, SystemExit) as e:   # the project's tools refuse by SystemExit (head_graft): the page reads why
        traceback.print_exc()
        state(step="error", done=True, ok=False,
             error=(str(e) if isinstance(e, SystemExit) else "%s: %s" % (type(e).__name__, e))[:400])
        sys.exit(1)
