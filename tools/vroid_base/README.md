# VRoid base

Chimera's base body, its face and its hairstyle, taken from a VRoid Studio character that you
provide (exported as .vrm, VRM 0.x). The repository contains no model.

    python3 -I tools/vroid_base/build.py <character.vrm>
    python3 -I tools/vroid_base/build.py <Name>       # <sources.vrm>/<Name>.vrm, otherwise the already imported PMX

A few seconds, without Blender. Output (by default `{data}/vroid-base`, the base body of the public defaults):

| File | Content |
|---|---|
| `body.pmx` | bare skin only (material 肌), all bones, MMD names, arms-down pose; texture `skin.png` |
| `head.pmx` | the same skin, vertices in the same order, plus the face, the eyes and the expressions |
| `hairstyle.pmx` | the character's hair (髪, 髪2…) |
| `base.json` | base head of the VRoid game (`base_head`: name, distance between the eyes and their midpoint measured by `head.Head`), source model and its title (`model`, `title`) |
| `checks.json` | checks: bones, pose, skin closed below the neck, opaque skin, share of original skin, height, eyes |
| `provenance.json`, `PROVENANCE.md` | source file (sha256 fingerprint) and terms of use read from the VRM (meta) |
| `body.png`, `head.png` | grey check contact sheets |
| `model/<Name>/` | the whole conversion, outfit included |

The character's outfit is not in the base. To wear it, import the same file like any character
(`tools/vroid/importer.py`, see `COMPATIBILITY.md`, VRoid section). A failed blocking check returns exit code 1:
the base must not be used.

## What VRoid does under the outfit (measured on 17/11/2026)

On the VRoid Studio sample models, the body **mesh** is whole under the clothes. Once the
seams are welded, it has only one opening, at the neck, where it joins the face. Its **texture** is not:

- VRoid paints on it the layer that sits on the body: tights, socks, tight top, underwear;
- on export, when it removes what is hidden, it makes the skin under the outfit transparent (the whole trunk
  of one model). The ordinary import deletes these faces: the body would have a hole.

`build.py` therefore keeps all the skin faces (`importer.convert(..., keep={"肌"})`). `skin.py`
then repaints the texture: it keeps the texels close to the tint of the face, trims their edge and removes the
islands. The rest is filled from the neighbouring skin (pyramid of averages). `checks.json` gives the share kept
as it is: about 43 % for a lightly covered model, 18 % for an almost fully clothed model.

## Limits

- The repainted areas are plain skin. Painted shadows, nails and skin details are lost
  there. An export without an outfit and without removal of hidden parts gives a faithful skin; it is the best file
  to provide.
- Only the VRoid base of VRM 0.x is read. The materials 肌 (body) and 颜 (face) are required.
- The base head follows the chosen character. Changing character requires a new cache (`workshop.cache`).
- The file's terms of use apply to every model assembled on the base. `build.py` reports a
  restricted use, a ban on redistribution or a ban on commercial use, but stops nothing.

## Checking

    python3 -I tools/vroid_base/tests/test_vroid_base.py
