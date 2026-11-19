# Compatibility with game models

> **Read this first.** This tool contains no game model and distributes none. The game names
> mentioned here only say which files the tool can work with. You bring your own files, and you
> alone are responsible for what you import and what you do with it.
> This document describes a tool: it is not legal advice.

For each game, this document says which files the tool can read, what it does with them and how well
proven that is. It also says where to put your files and which commands to run to get them
into Chimera. To add a game or a format: [CONTRIBUTING.md](CONTRIBUTING.md).

## The levels

- **proven**: done by the tool on real models of the game, from start to finish, with measurements that
  automatically sort what passes and what is rejected.
- **partial**: part of the pipeline works, or it worked on few models, or the result does not reach
  Chimera yet. The detail says which.
- **not supported**: the tool does not read it.

The levels and the numbers come from the code and from our own collection, on 15/11/2026. Each game
declares them in its description (`tools/games/<game>/game.json`, key `compatibility`); the per-game
sections below are drawn from them by the command `python3 -I tools/games write`.

## Where to put your files

Everything starts from a data folder: the setting `paths.data` (environment variable
`CHIMERA_DATA`). Your settings go in `settings.json` at the repository root: that file is yours,
git does not track it. You only write the keys to change, for instance:

```json
{
  "paths": {"data": "~/MyModels"},
  "games": {"zzz": {"mmd_extracted": "~/Archives/zzz-mmd"}}
}
```

Default layout under the data folder (`<data>`). Each source folder can be put
elsewhere with the key `games.<game>.<source>` given in the tables below.

```
<data>/
  sources/zzz/mmd-extracted/<Character>/…/<model>.pmx       your PMX files, textures next to them
  sources/outerplane/people/<id>_<Name>/controle.glb  + rendu.json and the textures   <!-- fr-ok: file names set by the third-party extraction -->
  vroid-base/body.pmx      base body (setting workshop.body), written by tools/vroid_base/build.py
  free-base/body.pmx      fallback without a .vrm, written by tools/free_base/build.py
  pivot-head/               written by tools/head/pivot.py
  face-library/      written by tools/head/library.py and hairstyle.py
  clothes-library/   written by tools/clothes/library.py
  workshop-vroid-base/       Chimera's cache (setting workshop.cache), written by tools/workshop/catalog.py
```

To see what the tool finds on your machine:

```
python3 -I tools/games
```

For each game: its level, each source folder, whether it is present, whether it comes from a setting or
from the default, and how many of the expected files it contains.

In the commands of the sections below:
- `<python>` is a Python with numpy, for instance Blender's (setting `programs.python`);
- `<blender>` is Blender (setting `programs.blender`);
- `<data>` is your data folder (written `{data}` in the descriptions and the tables).

The commands are run from the repository root.

## What is not provided

- **The base body** (`workshop.body`): every part is worn on it. It is not in the
  repository; `python3 -I tools/free_base/build.py` builds a free one (MakeHuman, CC0), with a free
  face and hairstyle that are enough to open the page without any game model (`INSTALLATION.md`).
- **The base head of a game**: a character of the active game (for ZZZ, Yuzuha), onto which all the faces
  of that game are stitched. It comes from your files; without it, the free head is used.

## Per game

<!-- compatibility:start (written by "python3 -I tools/games write": do not edit by hand) -->

| Game | Key | Format | Level | Faces | Hairstyles | Clothes |
|---|---|---|---|---|---|---|
| Zenless Zone Zero | `zzz` | PMX (MMD) | proven | proven | proven | proven |
| Outerplane | `outerplane` | GLB (binary glTF), rendu.json | partial | partial | partial | not supported | <!-- fr-ok: see _fr_ok in the game description -->
| VRoid | `vroid` | VRM 0.x (.vrm), VRM 1.0 (.vrm, VRoid Hub) | partial | partial | partial | partial |

### Zenless Zone Zero (`zzz`): proven

The game the pipeline was written for: faces, hairstyles, clothes and accessories of its MMD models all go through Chimera, measured and sorted automatically.

- **Expected format**: PMX (MMD)
- **Where these files come from**: The MMD model archives that the studio publishes for this game, one per character, unpacked as they are (one folder per character, the PMX and its textures). Each archive has its terms of use: read them.

| What the tool does | Level | Detail |
|---|---|---|
| Faces | proven | Each character's face measured on the pivot head, then stitched into the oval of the base head; 16 faces offered (list "catalog.heads"), 16 kept out of 16 on 15/11/2026. |
| Hairstyles | proven | Hair, headwear and animal ears measured for the 49 characters; 44 of the game's hairstyles kept out of 49 on 15/11/2026 (rejected: fits cleanly on no head, or fails on its own head). |
| Clothes | proven | Outfit cut into parts (top, bottom, shoes…), each part worn on the base body, then checked; 210 clothing parts kept on 15/11/2026; 25 of the game's parts rejected out of 484 (clothes and accessories; reasons in COMPATIBILITY.md). |
| Accessories | proven | Belts, gloves, jewellery, straps… cut with the outfit and checked the same way; 249 accessories kept on 15/11/2026. |
| Expressions | partial | Built from the model's MMD morphs (まばたき, あ, 笑い…), with a strength specific to some heads ("people"); computed on demand (tools/workshop/expressions.py), only for Jane in the cache of 15/11/2026. |

**Where to put the files**

| Source | Default folder | Setting key that moves it | What it must contain |
|---|---|---|---|
| `mmd_extracted` | `{data}/sources/zzz/mmd-extracted` | `games.zzz.mmd_extracted` | one folder per character, at the path given in "models" (e.g. `Anby/安比.pmx`), with the textures next to the PMX |

**From your files to Chimera**

1. Put the extracted PMX files in the `{data}/sources/zzz/mmd-extracted` folder (or write its path in `settings.json` under `games.zzz.mmd_extracted`). Check: `python3 -I tools/games` must say "49/49 expected files".
2. The base body (setting `workshop.body`) must exist: `<python> -I tools/free_base/build.py` builds a free one. If Chimera has already been used without a game model, give another cache (`workshop.cache`): the base head changes.
3. `<python> -I tools/head/pivot.py --output <data>/pivot-head`: the pivot head, through which all faces go.
4. `<python> -I tools/head/library.py extract --output <data>/face-library`: each character's face.
5. `<python> -I tools/head/hairstyle.py extract --library <data>/face-library`: hairstyles.
6. `<python> -I tools/clothes/library.py extract --output <data>/clothes-library`: outfits cut into parts (trims and index included).
7. `<python> -I tools/workshop/catalog.py prepare` (short trial: `--level small`): Chimera's cache; then `catalog.py summary` for the parts kept and rejected.
8. `bin/chimera`: the page.

**Known limits**

- The paths in "models" follow a specific layout of the archives (English folder name of the character, Chinese file name). Another layout requires correcting "models" in the description: no setting changes it.
- 49 human characters only. Not listed: non-human or helmeted heads (Ben, Billy, Lycaon, Pan Yinhu, Pyrois, Banyue), Trigger (eyes closed under the blindfold), Rina (painted eyebrows), Norma and Verina (material 脸 instead of 颜, which the core does not recognize).
- The first-pass head (Belle) is required by the pivot head, the base head (Yuzuha) by the catalogue: without them, nothing gets prepared.
- The measurement thresholds (eye radius, mouth, distance between the eyes) are tuned to the scale of these models (MMD units).
- Outfit variants of the same character are left out: same head, they would count twice.
- `clothes/library.py extract` without `--only` cuts every entry of the face library, Outerplane included if there are any (never tried on them).

### Outerplane (`outerplane`): partial

The heads are converted and stored in the face library, but Chimera offers none of them today.

- **Expected format**: GLB (binary glTF), rendu.json <!-- fr-ok: see _fr_ok in the game description -->
- **Where these files come from**: An extraction of the game files by a third-party tool, laid out in the format of our extraction tool: one folder per character with `controle.glb` (mesh, bones, morphs), `rendu.json` (materials and textures) and the textures. No tool in this repository produces this format. <!-- fr-ok: see _fr_ok in the game description -->

| What the tool does | Level | Detail |
|---|---|---|
| Faces | partial | Converted into a donor PMX (tools/head/outerplane.py), then measured like a ZZZ head: 60 heads stored out of 78 on 15/11/2026, 14 rejected (eyeball welded to the skin, eyebrows without geometry), 4 left out. Not in Chimera's catalogue: running them through Chimera has never been tried. 16 heads do not close their eyes (blink at 30-93 %). |
| Hairstyles | partial | 44 hairstyles measured in the face library, but not offered: Chimera only takes the hairstyles of a character whose outfit is cut. |
| Clothes | not supported | The donor PMX contains only the head: cutting the outfits has never been done on these models. |
| Accessories | not supported | Same reason as the clothes. |
| Expressions | partial | The EYE_* and MOUTH_* morphs are renamed to MMD names ("morphs_mmd") during the conversion; the others keep their names. In the donor PMX only. |

**Where to put the files**

| Source | Default folder | Setting key that moves it | What it must contain |
|---|---|---|---|
| `people` | `{data}/sources/outerplane/people` | `games.outerplane.people` | one folder per character (`<id>` or `<id>_<Name>`) with `controle.glb`, `rendu.json` and the textures named by `rendu.json` | <!-- fr-ok: see _fr_ok in the game description -->

**From your files to Chimera**

1. Put the character folders in `{data}/sources/outerplane/people` (or write its path under `games.outerplane.people` in the settings). Check: `python3 -I tools/games` counts the folders that have `controle.glb` and `rendu.json`. <!-- fr-ok: see _fr_ok in the game description -->
2. Heads: the ZZZ pipeline must have produced the pivot head (`<data>/pivot-head`), then `<python> -I tools/head/library.py extract --outerplane --output <data>/face-library`; a single head as a trial: `<python> -I tools/head/outerplane.py convert <character folder> <output.pmx>`.
3. Hairstyles: `<python> -I tools/head/hairstyle.py extract --library <data>/face-library --only <names>`.

**Known limits**

- Cannot be the active game (`games.active`): its description has neither a base head nor a list of models. Its heads go through the pivot head of the active game (ZZZ).
- The "materials" table (FACE, FACE_EYE, HAIR, BODY) assumes this game's M_<id>_… material names; cutting the face into eyes, mouth and eyebrows is done by geometry and failed on 14 heads.

### VRoid (`vroid`): partial

A character made with VRoid Studio or taken from VRoid Hub (.vrm, VRM 0.x or 1.0) gives Chimera its face, its hairstyle and its clothes, from import to assembly; tried on three sample models and eight other characters, without physics.

- **Expected format**: VRM 0.x (.vrm), VRM 1.0 (.vrm, VRoid Hub)
- **Where these files come from**: Your own characters, made with VRoid Studio and exported as .vrm (VRM 0.x or 1.0), or VRoid Hub models whose licence allows it (the licence declared in the file is written in the PMX comment). No model is provided with the tool.

| What the tool does | Level | Detail |
|---|---|---|
| Faces | partial | 3 heads out of 3 kept on 15/11/2026, 8 out of 8 on 18/11/2026 (seven VRoid Hub models in VRM 1.0, one VRM 0.x), checked up close in the page: facing forward, eyes in place. The base head is stitched onto Chimera's body (closed head: no skull seam); the other faces go through face transfer. The body skin takes the tint of the face, the neck keeps the shadow that VRoid paints on it: without hair, the back of a transferred face's skull keeps the base head's tint. VRoid's neck is thinner than the free body's: slight fold in profile at the neck seam. |
| Hairstyles | partial | 3 hairstyles out of 3 kept on 15/11/2026, 5 out of 8 on 18/11/2026, held by the head bone: the springs of the locks are not carried over. A hairstyle is measured on its own head, then carried onto the base head: two VRoid Hub hairstyles lose too much skull coverage on the way (skull_under_hair, one by 0.001), a third goes into the skin (hair_off_skin). |
| Clothes | partial | 9 parts kept out of 9 on 17/11/2026 (tops, bottoms, shoes); on 18/11/2026, 38 parts kept out of the 38 of a kind the workshop offers (24 others, straps, buckles, ornaments, are not offered on their own), one whole outfit per model checked in the page. VRoid cuts a garment from a whole template whose texture erases part of it (shorts are trousers with transparent legs): the import removes the faces that the texture erases entirely, and the part takes its real shape. The underwear that VRoid paints on the skin is not carried over: an open vest shows the skin. A loose top is filed as a coat. The pose check (body_only_poses) is not measured. |
| Accessories | partial | Necklace, harness, armband and a halo of VRoid Hub models kept on 18/11/2026 and seen worn with their outfit; straps, buckles and ornaments are not offered on their own. |
| Expressions | partial | The face shapes (Fcl_*) become morphs with MMD names ("morphs_mmd"); in the page, all 8 expressions on four VRoid Hub models and the VRM 0.x one (18/11/2026). A model whose face was reworked outside VRoid Studio may carry fewer shapes (14 instead of 57): only the expressions its morphs make are offered. |

**Where to put the files**

| Source | Default folder | Setting key that moves it | What it must contain |
|---|---|---|---|
| `vrm` | `{data}/sources/vroid/vrm` | `games.vroid.vrm` | your .vrm files; this folder is only for storage, the importer takes the files it is given |
| `mmd_extracted` | `{data}/sources/vroid/mmd-extracted` | `games.vroid.mmd_extracted` | what the importer writes: one `<Name>/` folder per character with `<Name>.pmx` and its textures |

**From your files to Chimera**

1. Convert: `python3 -I tools/vroid/importer.py <file.vrm>... --output <mmd_extracted>` (`show <file.vrm>` says what the file contains without writing anything).
2. Chimera's base (bare body, face and hairstyle): `python3 -I tools/vroid_base/build.py <file.vrm>` (output `{data}/vroid-base`, default `workshop.body` setting). The game's base head is this character's: name and eye measurements are read from `vroid-base/base.json`.
3. In `settings.json`: `games.active` = `vroid`, `games.vroid.mmd_extracted` = the output folder of step 1.
4. Libraries: `<python> -I tools/head/pivot.py --output <data>/pivot-head --no-sheet`, `<python> -I tools/head/library.py extract --output <data>/face-library`, `<python> -I tools/head/hairstyle.py extract --library <data>/face-library`, `<python> -I tools/clothes/library.py extract --output <data>/clothes-library`.
5. Chimera: `<python> -I tools/workshop/catalog.py prepare`, then `summary`.

**Known limits**

- A VRM that is not a VRoid character (no VRoid Studio face material, "..._Face_00_SKIN") is refused with that reason: the face, hair and outfit are found through VRoid's material names.
- Tried on three VRoid Studio sample models (15/11/2026) and on eight characters through the workshop page (18/11/2026); physics and spring bones are never read.
- No physics: hair and garment flaps do not move on their own.
- The underwear painted on VRoid's skin is not carried over.
- VRoid base: VRoid paints the layer stuck to the body (tights, socks, underwear) into the skin texture, or makes it transparent under the outfit; tools/vroid_base repaints those areas with a plain skin taken from the neighbouring skin (painted shadows and nails are lost). An export without an outfit gives a more faithful skin.
- VRoid's neck is thinner than the free body's: the body is narrowed at the neck seam, a slight fold remains visible in profile.
- The thumbnail of a face shows the whole model.

<!-- compatibility:end -->

## Measurement of 15/11/2026

These numbers come from Chimera's catalogue prepared on our own collection (what
`catalog.py summary` shows). They show what the tool keeps and rejects. It sorts on its own:
nothing is picked by hand.

- 561 items prepared, 531 kept, 30 rejected.
- Faces: 18 kept out of 18 (16 from ZZZ, 2 of our own).
- Hairstyles: 46 kept out of 51 (44 out of 49 from ZZZ, 2 of our own).
- Parts: 467 kept out of 492 (459 out of 484 from ZZZ, 8 of our own).

Reasons for the 30 rejections:

| Rejection | Count | What it means |
|---|---|---|
| `body_only_poses` | 14 | the part goes through the body when the body is posed |
| `under_skin` | 6 | more than 1 % of the part's vertices are inside the body |
| fits cleanly on only 0 heads out of 18 | 4 | hairstyle that holds on no head of the catalogue |
| sole without an upper | 3 | shoe that covers less than 65 % of the top of the foot |
| measurement impossible | 2 | measuring the part failed |
| fails on its own head | 1 | hairstyle that does not even hold on its own character's head |

Outside Chimera (face library, same date):
- 49 ZZZ faces measured;
- 60 Outerplane heads stored out of 78, 14 rejected, 4 left out;
- 44 Outerplane hairstyles measured.

## Model terms of use

Game models are often published with terms of use. Read those of each of your
models. For instance, the terms of the ZZZ MMD archives that we have read forbid redistributing
the model, taking parts of it to modify other models, and commercial use. Yet Chimera
puts the face, the hairstyle or the clothes of one model on another head or another body.

## Checking this document

```
python3 -I tools/games check     # does the document still say what the descriptions say?
python3 -I tools/games write       # if not, rewrites the per-game sections from the descriptions
```
