# Free base: body, face, hairstyle

Enough to open Chimera without any game model: a base body, a face and a hairstyle. The shape
comes from MakeHuman (CC0); the rest is computed here. Provenance and licence: `PROVENANCE.md`.

    python3 -I tools/free_base/build.py [--sources <folder>] [--output <folder>] [--no-sheets]

Two minutes, without Blender. Output (by default `{data}/free-base`): `body.pmx` (95 bones with MMD names,
material `肌`), `skin.png`, `shape.pkl`, `own-rig/` (skeleton and weights), contact sheets `shape.png`, `arm.png`,
`hand.png`, `foot.png`, `head.png`; and, next to them, `head.pmx` (the body and its eyes: the free face of the
catalogue) and `hairstyle.pmx` (the free bob). This is the body Chimera reads by default (setting
`workshop.body`); then `python3 -I tools/workshop/catalog.py prepare`, then `bin/chimera`.

The catalogue switches to "free mode" when there is no base head from a game and `head.pmx` and
`hairstyle.pmx` are next to the body: the free head serves as the base head, the free hairstyle is taken as it
is (it is made for this skull). As soon as a game's models are in place, the game's base head is used, in a
cache of its own.

| file | role |
|---|---|
| `build.py` | the whole pipeline; downloads whatever is missing from MakeHuman, at a fixed version |
| `silhouette.json` | MakeHuman's shape targets and their dose: the silhouette is tuned here |
| `makehuman.py`, `reference.py` | reading `base.obj` (skin, `joint-*` joints), body landmarks |
| `trial.py` | our grid (`tools/rebuilt_body`) laid on the neutral skin |
| `body.py` | silhouette, straightened and thinned neck, skeleton, A-pose, weights; constants at the top of the file |
| `free_head.py` | MakeHuman's head cut at the neck and joined to the grid |
| `pmx_mmd.py` | writing the PMX with MMD bones |
| `free_face.py` | MakeHuman's eyes placed in the eye sockets (`head.pmx`), the bob trimmed from its hair volume (`hairstyle.pmx`); colour and lengths at the top of the file |
| `view_body.py`, `view_base.py` | contact sheets of the body, and of a game's base head stitched onto it |

The measurements that set the proportions (head bone 17.5 cm below the top of the skull, neck 6 cm
deep, shoulder at 139 cm) are those of the MMD models that Chimera can read: that is what makes an
imported face land on the skull and a garment land on the shoulders.

Known limits: skin and hair of a flat colour; a single face and a single hairstyle; no eyebrows
(the fringe covers them), no eyelashes, no facial expressions; faceted eyes; a few gaps between the
locks at the nape; no free clothes. It is a base for trying the tool and for wearing your own
parts, not a finished character.

    python3 -I tools/free_base/tests/test_free_base.py
