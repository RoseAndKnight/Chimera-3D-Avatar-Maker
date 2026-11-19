# Games: one description per game

The core (workshop, heads, clothes, hair, atlas) names no game and no character. What it knows about a game is
written here, once, in `tools/games/<game>/game.json`, and read by `tools/games/__init__.py` (`games.active()`,
`games.load(name)`, `games.own_rig()`, `games.case_(key)`). No model is shipped: the game files live in each
user's data (`{data}`), never in the repository.

| Folder | What it is |
|---|---|
| `zzz/` | Zenless Zone Zero (official PMX models) |
| `outerplane/` | Outerplane (GLB): donor heads |
| `vroid/` | VRoid Studio (.vrm, VRM 0.x): your own characters, converted by `tools/vroid/importer.py`; faces, hairstyles, clothes; the active game by default |
| `homemade/` | our own parts (Maelys, Orpheya, clothes built in the workshop), described the same way |

## Adding a game

1. Create `tools/games/<game>/game.json`. Keys that start with `_` are notes: never read.
2. Make it active: `"games": {"active": "<game>"}` in your `settings.json`, or `CHIMERA_GAME=<game>`.
3. If its folders are somewhere other than the default, `"games": {"<game>": {"<source>": "/path"}}` in `settings.json`.
4. Declare what the tool does with it: the `compatibility` key (below, "Declaring compatibility"), then
   `python3 -I tools/games write` to rewrite `COMPATIBILITY.md`. Full step by step: `CONTRIBUTING.md`.

Keys read (all optional except the ones used by the tool you run):

| Key | Read by | Content |
|---|---|---|
| `title`, `short_name`, `format` | comments, log | game name, abbreviation, `pmx` or `glb` |
| `sources.<source>` | `J.path(source)` | public default of a source folder (`{data}/…`); `mmd_extracted` = the game's PMX files |
| `models` | `head/pivot.py`, `head/library.py` | `{character: relative path of the PMX in mmd_extracted}`, or `"*"`: every `<Name>/<Name>.pmx` folder of `mmd_extracted` (a family where everyone brings their own models, like VRoid) |
| `base_head` | `workshop/common.py`, `head/pivot.py`, `head/outerplane.py` | Chimera's base head (`name`), the distance between its eyes and their midpoint, base of the pivot's first pass; `tint`: `face` when the neck of the head carries a painted shadow (the body skin then takes the tint of the face, not that of the neck ring) |
| `os_mmd`, `os_fondus`, `pose`, `morphs_mmd`, `panels` | the game's importer (`vroid/importer.py`) | conversion tables specific to an importer: bone names, bones merged into their parent, pose, morph names |
| `heads.typical_eye_gap` | `head/head_glb.py` | typical distance between the eyes (MMD units) to which a head from another studio is scaled |
| `catalog` | `workshop/catalog.py` | heads and hairstyles offered (`"all"` = every hairstyle of the library), `small` trial level |
| `people.<character>.expressions` | `workshop/expressions.py` | expression corrections for a head |
| `people.<character>.exception_garment` | `clothes/library.py` | garment that the mesh does not separate, cut by a study entry |
| `studies.clothes`, `studies.hair`, `studies.outfits_layers` | `clothes/measure.py`, `hair/measure.py`, `workshop/check_layers.py` | test sets of the studies |
| `families.json` (file next to it) | `atlas/primitives.py` | classification of materials into families; without it, no classification |
| `materials` | `atlas/pmx.py` (`read`), `head/outerplane.py` | the game's material names translated into the common language (below) |
| `compatibility` | `games/compatibility.py` only | what the tool does with the game's files, and at what level (below); changes nothing in what the other tools do |

Any value of a description can be overridden on a machine by the setting `games.<game>.<key>` in `settings.json`
(`games.vroid.base_head.name`: a base head other than the one of the built base). `base_head.since_base`:
name, `first_pass`, `eye_gap` and `eyes_center` are read from `base.json` next to the base body
(`tools/vroid_base`); the setting wins, then that file, then the description.

The house set also has `heads` (heads computed separately) and `built` (clothes made in the workshop).

## Declaring compatibility

The `compatibility` key says what the tool does with a game's files. `COMPATIBILITY.md` at the root is drawn from it.

```
python3 -I tools/games             # known games, level, source folders present or not on this machine
python3 -I tools/games check    # declarations complete and COMPATIBILITY.md up to date? (exit code 1 if not)
python3 -I tools/games write      # rewrites the per-game sections of COMPATIBILITY.md
```

| Key | Content |
|---|---|
| `level` | `tested`, `partial` or `non`: the overall level |
| `summary` | one sentence |
| `formats` | the file formats read, in plain words (`["PMX (MMD)"]`) |
| `origin` | where these files usually come from. Never a download link. |
| `sources.<source>.content` | what each folder of `sources` must contain (same source name) |
| `sources.<source>.check` | optional: `{"files_of": ["<dotted key>"]}` counts the files named by these keys (text, list or values of an object, relative to the folder); `{"folders_with": ["model.glb", …]}` counts the subfolders that have all these files |
| `support.<heading>` | `{"level": …, "detail": "…"}` for each of the headings `heads`, `hairstyles`, `clothes`, `accessories`, `expressions` |
| `steps` | the commands, in order, from "I have put my files in place" to "they are in Chimera" |
| `limits` | what does not work or has not been tried |

The levels:
- `tested` (proven): done on real models of the game, from start to finish, sorted by the measurements.
- `partial` (partial): only part of the pipeline, or few models, or not all the way to Chimera.
- `non` (not supported): not read.

Every claim comes from the code or from a real trial. In the `detail`, give the number of models and the date.

## What a game must follow

Bones and morphs are those of the MMD standard (センター, 上半身, 頭, まばたき, あ…). Material names are the
core's **common language**, inherited from ZZZ (below). A game whose models name them differently translates them
with its `materials` table.

### The `materials` table: translating a game's names

An ordered list of `[pattern, common name]` pairs. The pattern is a Python regular expression searched in the
material name; the first one that matches gives the common name (`\1`… reuses its groups; written in JSON,
`["(?i)^hair_?(\\d*)$", "髮\\1"]` turns `Hair_2` into `髮2`). A name that no pattern matches stays as it is. Without
a table, nothing is translated: that is the case of ZZZ, whose names are the common language itself.

The translation is done in **one single place of the core**: `atlas/pmx.read`, for any PMX stored in a
`sources.<source>` folder of a game that has a table (real paths, links resolved; the longest root wins). The
original name stays in the material's `name_game`. Everything the core reads or writes afterwards (libraries,
workshop, GLB of the page) is therefore in the common language. The source folders of a game that has a table must
contain only its files (a broad source such as `{data}` would get our own parts translated). An importer can
read the same table through `J.translate(name)`: `head/outerplane.py` derives the role of each primitive of its GLBs
from it.

The translation works name by name. Two materials translated to the same name keep that same name (the core then
often takes the first one: `names.index("肌")`); if 髮 and 髮2 are both needed, the pattern must tell them apart.

### The common language

What the core expects, by the exact name or its beginning ("髮…": any name that contains it or starts with it,
depending on the table that reads it). Read by `head/head.py`, `head/library.py` (`PER_MATERIAL`),
`head/hairstyle.py`, `hair/measure.py`, `clothes/model.py`, `clothes/library.py`, `workshop/prepare.py`,
`workshop/maps.py`, `atlas/primitives.py` (`GROUPS_MATERIAL`).

| Name | Meaning |
|---|---|
| `颜` | face skin (ears and neck included when the model does not separate them): the head is measured on it |
| `颜2` | detached piece of face skin (cheeks, blush), classified by position |
| `目` | eye plate (iris); `白目` white of the eye; `瞳` pupil; `目色` eye colour |
| `目光`, `目光2`, `瞳光` | eye highlights; `目影` shadow cast on the eye |
| `二重`, `二重影` | eyelid fold and its shadow |
| `眉`, `眉毛` | eyebrows; `睫` eyelashes; `眉睫`, `睫眉`, `右睫眉` eyelashes and eyebrows mixed (separated by position); `眉睫影` their shadow |
| `齿` | teeth; `口舌` inside of the mouth and tongue; `口`, `舌` the same, separated; `口线` mouth line |
| `痣` | beauty mark; `表情`, `眼角` expression marks, corner of the eye (never clothing) |
| `髮…`, `髪…`, `发…` | hair (`前发` front, `后发` back, `侧发` sides; `髮2`… other locks) |
| `肌…` | body skin; `手`, `手2` hands; `脖`, `首` neck; `指甲` nails |
| `体` | clothed one-piece body (skin and outfit in the same material): filed with the outfit, cut by geometry |
| `<name>+` | layer placed over the material `<name>` (back side of a double-sided part, printed pattern) |
| any other name | outfit; its name also gives the kind of accessory (`clothes/library.py`, `KINDS_NAME`: 武器 weapon, 镜 glasses, 面具 mask, 帽 headwear, 翅膀 wings, 尾 tail, 包 bag, 项链 jewellery, 腰带 belt…) and the render surface (`workshop/page/viewer.js`, `MATERIALS`: 针织/毛衣 knit, 皮/鞋/靴 leather, 袜 stockings, 缎/丝 satin, 镜/宝石 glass) |

Names written by Chimera itself (outputs, never to be translated): `颜_A`, `颜_A2` (stitched face), `肌_tete`, `肌_cou`,
`肌_bande`.

### What the table does not translate

- **Heads read from a GLB** (`head/head_glb.py`, `HeadGlb`): skin, eyes, eyebrows and hair are chosen there by
  `SKIN_PATTERNS`, `EYES_PATTERNS`… This is not a name-by-name translation: over all the materials of a GLB, the
  first pattern that finds something wins and rules out those of the following patterns, with shared exclusions. On
  the 105 GLBs of NTE, a name-by-name translation would change 3 choices (`face_gaoguagn`, a typo for a highlight,
  would become skin twice; an `eyehight` would become eye). And `HeadGlb` is called without knowing which game the
  GLB comes from (`hair/measure.py` and research tools not published). Its patterns mix NTE (`face_new`, `_eyes`, `meimao`,
  `jiemao`, `bantou`, `touming`, `gaoguang`), the generic names also used for Outerplane (`_face`, `_eye`,
  `hair`) and the house set (Keijo: a French name for hair).
- **Cutting an Outerplane face** (`head/outerplane.py`): a single `FACE` material carries skin, eyes, highlights,
  fold, mouth, blush; its pieces are classified by their position, not by a name. Only the role of each material
  (FACE, FACE_EYE, HAIR, BODY) goes through its table.
- **Morphs**: Outerplane's `morphs_mmd` renames them in its importer; there is no morph translation at the point
  where PMX files are read.

A game that is not in PMX needs an importer in `tools/head` or `tools/clothes` that writes PMX files in this
language: `head/outerplane.py` is the example (GLB → donor PMX, roles from `materials`, morphs renamed by
`morphs_mmd`).
