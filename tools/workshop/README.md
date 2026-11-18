# Character workshop

Choose a face, a hairstyle and an outfit in a page, see the character in 3D, then get
a single finished PMX.

## Running

    bin/chimera          # opens http://127.0.0.1:8767 in the browser
    bin/chimera 9000     # on another port

Installing on a new machine: `INSTALLATION.md` (repository root). The repository provides no model, not
even the base body: as long as the prepared body, a face or a hairstyle is missing, `/` serves a page that says
what to do (`serveur.accueil`, `catalogue.a_faire`), and `catalog.py prepare` stops without a body or a base head.

Left column: Face, Hair, Top, Bottom, Shoes, Extras. Click a
thumbnail and the view changes at once; drag with the mouse to turn, the wheel zooms. Clicking
a part again removes it. The right column sums up the choice, flags what clips through and
carries the **Assemble** button. The "Rejected parts" panel lists the parts set aside and why.

The view uses WebGL2. If the browser lacks it, the page draws the same scene in software and
shows "Simplified preview": slower, less smooth, but the choice stays correct.

**Assemble** goes through the real tools again (`coiffure.poser`, `porter`, `mesure`,
`rendu`) and writes into `assemblages/<date>/`: `modele/resultat.pmx` (with its textures),
`modele/resultat.vrm` (VRM 1.0: humanoid bones, 13 expressions, MToon materials, T-pose),
`planche.png` (front, three-quarter, back), `controles.json` and `journal.txt`. The contact sheet does not show
the "髮+" highlight layers of made hairstyles; the PMX keeps them. The VRM has no
hair or skirt physics (spring bones).

**Skeleton.** After the checks, the `squelette` step can replace, in `resultat.pmx`, the bones and weights
of the body with a skeleton with readable names (`tools/custom_body/plug.py`), when its data exist
(`<data>/corps-maison/squelette.json`); the version on the original bones then stays next to it
(`modele/resultat-zzz.pmx`). Without these data, or with `workshop.skeleton` = `zzz`, the model keeps the bones of
its base body, and the VRM comes out of it as is.

## Layers

The page and the assembly run the same computation (`outfit.py`), starting from parts that are all fitted
on the bare skin. The skin is the common frame: each garment vertex has its line (the direction
going away from the skin) and its height above it, and moves only along its line. The
rule: along a line, the parts are met in the order of their ranks, at least 1.5 mm apart
(more for coarse meshes). It is measured by exact rays (`rays.py`)
in both directions, from the vertices of the lower part to the faces of the upper part and the other way round.

1. **remove**: underwear under shorts or trousers is not worn (it stays under a skirt);
2. **order**: from inside to outside; a top whose hem goes over the bottom stays
   outside, shoes and boots are always above the trousers, a belt goes over the bottom;
3. **tuck in**, from top to bottom: each part is pressed under those above it, without going below
   1 mm above the skin (except under a boot, narrower than the leg). The thicknesses of a single part
   (lining of a skirt) are put back in their order by `porter` (`couches_internes`);
4. **push out**, from bottom to top: where room is lacking (two fitted parts), the upper part
   moves away by what is lacking, by 8 mm at most; a belt is always pushed out, never tightened;
5. **mask**: the faces entirely hidden by an opaque upper part held to the body are
   removed (`couvre.bin` says, per face, whether it is held, opaque, outer);
6. **push**: placed accessories (necklace, bag) come out above what they touch.

A free flap that passes more than 25 mm in front of the upper part (ruffle, flared skirt hem under
a jacket) is left alone: it flares over it. A volume too big to pass underneath (bow
at the back of a dress under a jacket) stays, and the page flags it.

The types, the ranks and these millimetres are default values: see "Adjusting the layers" below.

Check independent of the adjustment: `check_layers.py` counts, on the displayed triangles, the
edges of one part that cross a face of the other and the area of the lower part that shows in front of
the upper one, on six reference outfits (or `--pieces a,b,c`). To run again after any change to
`outfit.py`.

The hair falls over the adjusted outfit (`combinaison.coiffer`): a strand under the outer face
of a part comes out by the shortest way, except under a hat, a hood or glasses.


## Catalogue

    python3 -I tools/workshop/catalog.py prepare [--level small|full]   # only redoes what is missing
    python3 -I tools/workshop/catalog.py prepare --stale                  # also redoes the parts fitted by another method (V_PORTAGE) or on another body
    python3 -I tools/workshop/catalog.py add <part folder>               # one more part
    python3 -I tools/workshop/catalog.py summary                               # counts and rejection reasons

(`python3` = a Python with numpy, for example Blender's: `programs.python` in the settings, see below.)
A part in the library format (`fiche.json`, `<part>.pmx`, `tex/`) is fitted on the
common body, measured (`measure.py`) then cached. Rejected are those whose fitting or
measurement fails, or that fail `sous_la_peau` (more than 1 % of the vertices inside the body, counted by
the fitting without a depth limit: `traversee_repos` only sees down to 10 mm), `traversee_repos`,
`traversee_repos_p90_mm` or `poses_corps_seul`, as well as shoes without an upper (mere
soles, the foot would stay bare). A check without a value is not a passed check: the record
names it (`non_mesures`), and `summary` counts them. An accessory whose garment measurement
discards all the parts (wing, bag, rivets) is no longer rejected for that (`mesure_sans_objet`);
placed as one block, it is stuck into the body by nature and `sous_la_peau` does not apply to it.
The record keeps the fitting method (`v_portage`) and the body (`corps`): `summary` counts the
outdated parts, `prepare --stale` redoes them.
The other checks, irrelevant on a bare body or tied to the source, stay recorded in the record.

## Where the files are

The cache (light parts, assembled models) is in `workshop.cache` of the settings (`CHIMERA_CACHE`), by default
`<data>/atelier-base-vroid`, for the base body `<data>/base-vroid/corps.pmx`: bare body, face
(`tete.pmx`) and hairstyle (`coiffure.pmx`) of a VRoid Studio character, with `base.json` (base head of the VRoid
game and its measurements), `PROVENANCE.md` (terms of use read from the file) and `controles.json`, written by
`tools/vroid_base/build.py <modele.vrm>`. As long as that base does not exist, the free MakeHuman base
(`<data>/base-libre/corps.pmx`, `tools/free_base`) is used if it is built, with the cache
`<data>/atelier-base-libre`. Another base body is set with `workshop.body`, with its own cache
(`workshop.cache`): a cache is only valid for the body its parts were fitted on. The assembled models are in `<cache>/assemblages/<date>/`. `CHIMERA_LAUNCHER` prefixes Blender launches (for example
`systemd-run --user --scope -p MemoryMax=8G --`).

## Settings

No machine path is written in the code: everything goes through `tools/settings.py`. Three layers, each one
overriding the previous one key by key:

1. `settings.default.json` (repository root, tracked): public values, not to be changed for your machine;
2. `settings.json` (repository root, ignored by git; `CHIMERA_SETTINGS` names another file): your own,
   same shape, only the keys to change;
3. environment variables.

| Key | Public default | Variable | Role |
|---|---|---|---|
| `paths.data` | `donnees` (in the repository) | `CHIMERA_DATA` | data root (`<data>`: libraries, bodies, heads) |
| `paths.skeletons` | `{data}/squelettes` | | bone namings per game (`maison.json`, `biped.json`) |
| `workshop.cache` | `{data}/atelier-base-vroid` | `CHIMERA_CACHE` | Chimera's cache (fallback: `{data}/atelier-base-libre`) |
| `workshop.body` | `{data}/base-vroid/corps.pmx` | `CHIMERA_BODY` | base body (`tools/vroid_base`; fallback: `{data}/base-libre/corps.pmx`) |
| `workshop.skeleton` | `maison` | `CHIMERA_SKELETON` | `zzz` keeps the game's bones |
| `programs.blender` | `blender` (looked up in the PATH) | `CHIMERA_BLENDER` | Blender of the queue |
| `programs.python` | the running interpreter | `CHIMERA_PYTHON` | Python (numpy) of the sub-computations and of `bin/chimera` |
| `compute.slots` | one per physical core, 2.5 GB each, 3 at least | `CHIMERA_SLOTS` (`BLENDER_PLACES` for the queue alone) | heavy computations at once |
| `compute.launcher` | nothing | `CHIMERA_LAUNCHER` | prefix of Blender launches |
| `compute.queue_dir` | `$TMPDIR/blender-queue-<uid>` | | folder of the locks of `bin/blender-queue` |
| `games.active` | `vroid` | `CHIMERA_GAME` | game the models come from (its description: `tools/games/<game>/game.json`) |
| `games.<game>.<source>` | the one in the description (`sources`) | | source folders of a game for your machine (e.g. `jeux.zzz.mmd_extrait`, `jeux.outerplane.personnages`) |

What the code knows about a game (characters, base head, special cases) is in its
description, `tools/games/<game>/`; parts you made yourself are declared in `tools/games/homemade/`.
Adding a game: `tools/games/README.md`.

In a value, `{data}` (or any dotted key between braces) is replaced; a relative path starts from the
repository root; `~` is expanded. Example of `settings.json`:

    {"paths": {"data": "~/mes-donnees"}, "programs": {"blender": "~/blender-5.2/blender"}}

To check: `python3 -I tools/settings.py all` (or `path workshop.cache`, `program blender`). Blender
launches go through the queue `bin/blender-queue`, which reads the same settings.

### Adjusting the layers

The garment types, the accessory kinds, their ranks (what goes over what), the outfit rules
and the stacking thresholds are in `layers.default.json` (repository root, tracked), read by
`tools/settings.py` right after `settings.default.json`. The meaning of each key is written in the file itself
(`_help` of each section). To change something, do not touch that file: write in your
`settings.json`, under `"layers"`, only what changes. The merge is done key by key, down into an
entry; a list (`closed_bottoms`, `also_takes`…) is replaced as a whole; `null` removes an entry.

    {"layers": {
      "garments": {"jupe": {"rank": 33},
                    "kimono": {"tab": "haut", "slot": "dessus", "rank": 57, "name": "kimono"}},
      "accessories": {"broche": {"rank": 79, "exclusive": false}},
      "thresholds": {"max_outward_mm": 10}}}

| Section | An entry | Role |
|---|---|---|
| `garments` | `tab`, `slot`, `rank` (required), `name`, `also_takes` | `type` of `fiche.json`: tab of the page, exclusive slot (two parts of the same slot are not worn together), rank (the smallest underneath), displayed name, extra slots taken (a dress takes top, bottom and underneath) |
| `accessories` | `rank` (required), `exclusive`, `name`, `belt_like` | `sorte` of `fiche.json`; always in the extras tab; `belt_like`: goes around the body, adjusted with the garments, placed on the fabric without tightening it |
| `outfit` | `ranks_in_outfit`, `closed_bottoms`, `tops`, `bottoms` | rank imposed during the adjustment (belt 36: over the bottom, under the jacket); bottoms that cover the underwear (it is not worn); a top whose hem comes out of a bottom is worn over it |
| `thresholds` | millimetres (`_mm`), `flag_share`, `passes` | margins between layers and for accessories, floor above the skin, free flap, largest push-out, sag of coarse meshes, masking of hidden faces, flagging of a clipping |

What stays in the code of `outfit.py`, on purpose: the search ranges (`PORTEE_RENTREE`, `PORTEE_PEAU`,
`PORTEE`), the geometric criteria (`ACCORD`, `EMPREINTE`, `CONTACT`, `FIN`) and the grids (`CELLULE`, `VOXEL`,
`VOXEL_CORPS`). These are choices of the algorithm, not render preferences: changing them means rereading
the computation and running `check_layers.py` again. The same goes for the rule "a sock lower than the underwear goes
under it" and the flags of `couvre.bin`.

When the change takes effect:

- the settings are read at startup: **restart the server** (`bin/chimera`) after changing `settings.json`;
- a rank, a tab, a slot, `also_takes`: nothing to redo. Each part record (`meta.json`) keeps
  those of the day it was prepared, but the catalogue, the layer adjustment and the assembly read them again from
  today's settings (`common.ranger`). Only the displayed name of an already prepared part stays the one in
  its record;
- a threshold or an outfit rule: nothing to redo, no outfit adjustment is kept on disk (the server
  keeps them in memory while it runs). Models already assembled keep the settings of the
  day they were assembled;
- an added type or kind: its parts enter the catalogue at the next `catalog.py prepare` (which only
  does what is missing) or through `catalog.py add <folder>`. A part of a removed type stays in the cache
  and in the catalogue until it is deleted;
- the page knows only four part tabs (`haut`, `bas`, `chaussures`, `accessoires`, written in
  `page/workshop.js`): a type filed under another tab does not appear there.

To see the merged values: `python3 -I tools/settings.py value layers.garments.jupe`.

Tests: `python3 -I tools/workshop/tests/test_workshop.py`.


## Security

The server listens only on `127.0.0.1` and answers only its own page: a request that names it other
than by its local address, or that comes from another site open in the browser, is refused (403). Do not
expose it on a network: it launches Blender and writes into the cache at the page's request.
