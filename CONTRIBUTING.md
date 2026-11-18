# Adding support for a game or a format

This guide is for developers. It describes the code as it is: there is no plugin system. Part of the
work is done by writing a description, without touching the code. The rest requires changing the core,
and this guide names the files. What already exists: [COMPATIBILITY.md](COMPATIBILITY.md).
The keys of a description are detailed in [tools/games/README.md](tools/games/README.md).

Never attach a game file (model, texture, extract) to a contribution.

## How the core sees a game

- The core (workshop, heads, clothes, atlas) names no game. It asks `tools/games`
  (`games.active()`, `J.chemin(source)`, `J.valeur(clé)`) for what it needs to know.
- A game is a folder `tools/games/<game>/` with a `game.json` file.
- The core reads **PMX** files (the MikuMikuDance format). It expects:
  - MMD bone and morph names (`センター`, `頭`, `まばたき`, `あ`…);
  - material names in a "common language" inherited from ZZZ (`颜` face skin, `目` eye,
    `髮` hair, `体` clothed body…).
- Only one game is **active** at a time (setting `games.active`). The models and the base head come
  from it.

## 1. What a description alone can do

A description is enough when your models are PMX files with MMD bones and morphs, and only their
material names or their folder layout differ.

1. **Create `tools/games/<game>/game.json`**, starting from `tools/games/zzz/game.json`. Keys that
   start with `_` are notes: the code never reads them.
2. **`sources`**: the default folder of each source, under `{data}/sources/<game>/…`. Each
   user can move it with `games.<game>.<source>` in their `settings.json`. The PMX source is called
   `mmd_extrait`: it is the one read by `head/pivot.py`, `head/library.py` and
   `clothes/model.py`.
3. **`modeles`**: `{character: PMX path relative to mmd_extrait}`, or `"*"` when every user brings
   their own models (any `<Name>/<Name>.pmx` folder in `mmd_extrait`). One head per human
   character, with no outfit variant.
4. **`tete_base`**: the character onto which all faces are stitched (`nom`), the distance between its
   eyes and their midpoint, and the head used for the first pass of the pivot head (`premier_passage`).
   Without it, the game cannot be active.
5. **`materiaux`**: the table that translates your game's material names into the common language.
   It is a list of `[pattern, common name]` pairs; the first pattern that matches wins. The translation
   happens when reading, in `tools/atlas/pmx.py` (`lire`). It only affects PMX files stored in this
   game's source folders. Without a table, nothing is translated.
6. **`catalogue`**: the faces and hairstyles that Chimera offers, and the `petit` level for a trial.
7. **`personnages`**: the special cases of a character, for instance the strength of an expression
   (`expressions`) or a garment that the mesh alone does not separate (`exception_vetement`).
8. **`compatibilite`**: what you declare. It needs the level, the formats, the provenance, the
   expected content of each source, a level and a detail per heading, the steps and the limits.
   Shape and rules: `tools/games/README.md`, "Declaring compatibility".
9. **A new garment type or a new kind of accessory**, placed in an existing tab:
   `layers.default.json`, no code. See `tools/workshop/README.md`, the section on tuning the layers.

### Checking it

Run these commands from the repository root. `<python>` is a Python with numpy (setting
`programs.python`).

```
python3 -I tools/games                      # does your game show up, are its folders found?
python3 -I tools/games check             # declaration complete? COMPATIBILITY.md up to date?
python3 -I tools/games write               # rewrites COMPATIBILITY.md from the descriptions
CHIMERA_GAME=<game> <python> -I tools/head/pivot.py --output <test folder>
CHIMERA_GAME=<game> <python> -I tools/workshop/catalog.py prepare --level small
cd tools/workshop && <python> tests/test_workshop.py
```

- `pivot.py` reads all your models and says which ones it cannot measure.
- `catalog.py prepare --level small` takes two faces and two hairstyles, plus a few parts.
  `catalog.py summary` then gives the parts kept and the reasons for the rejections.
- To try without touching your data, point `CHIMERA_DATA` and `CHIMERA_CACHE` at other
  folders.
- `test_jeu_factice_traduit_a_la_lecture` (in `tools/workshop/tests/test_workshop.py`) shows how to
  check a `materiaux` table on a PMX written for the test, without any game file.

## 2. What requires touching the core

Today, no description covers the following cases.

| What you want | Where it happens | What to do |
|---|---|---|
| Read a format other than PMX (GLB, FBX, VRM…) | `tools/atlas/pmx.py` is the only model reader of the core | Write an **importer** that converts your files into PMX in the common language, placed in `tools/head` or `tools/clothes`. Two examples. `tools/vroid/importer.py`: a whole .vrm (head, hair, outfit) to a PMX that the pipeline then reads with no option, with renamed bones (`os_mmd`), arms-down pose, skin cut at the bottom of the neck, renamed morphs; its test `tools/vroid/tests/test_importer.py` builds a fake file. `tools/head/outerplane.py`: GLB to a donor PMX (head only), roles from the `materiaux` table, morphs renamed by `morphes_mmd`, then `library.py extract --outerplane`. A head-only importer also needs an option in `tools/head/library.py`, modelled on `--outerplane`; a whole-model importer, like the VRoid one, does not. |
| Recognize the heads of a GLB | `tools/head/head_glb.py`: `PEAU_MOTIFS`, `YEUX_MOTIFS`, `SOURCILS_MOTIFS`, `CHEVEUX_MOTIFS` | Add your patterns to these lists. They apply to every GLB, whatever its game: the first pattern that matches wins. Check that the GLBs already recognized keep their choices. These patterns are only used by the research tools (`mesures_visage.py`, `spectre_signature.py`) and by `outerplane.py`, not by Chimera itself. |
| Other bone names, another skeleton | MMD names (`頭`, `首`, `センター`, `上半身`…) are hard-coded in many files: `head/head.py`, `head/hairstyle.py`, `head/retarget_head.py`, `custom_body/skeleton.py`, `custom_body/plug.py`, `workshop/prepare.py`, `workshop/vrm.py`, `cheveux/*`, `atlas/passe_*.py`… | The simplest way: rename the bones to MMD names in your importer. |
| Other morph (expression) names | `workshop/expressions.py` (`EXPRESSIONS`, MMD names); the face library records morphs under their own names | Rename them to MMD names in your importer, like Outerplane's `morphes_mmd`. No morph translation exists at the point where PMX files are read. |
| A material name that is not in the common language | `head/head.py` (`PEAU = "颜"`…), `head/hairstyle.py` (`CHEVEUX`), `clothes/library.py` (`SORTES_NOM`), `atlas/primitives.py` (`GROUPES_MATERIAU`) | First, try to translate it with the `materiaux` table. Otherwise, add it to the lists in these files. Example: ZZZ leaves out Norma and Verina, whose face skin is called `脸`. |
| The surface of a part in the render (knit, leather, satin, glass) | `tools/workshop/page/viewer.js`, constant `MATIERES` (regular expressions on the material name) | Add your words to these expressions. |
| A new tab in the page | `tools/workshop/page/workshop.js`, constant `ONGLETS` (face, hairstyle, top, bottom, shoes, accessories) | Add the tab there, then place types in it in `layers.default.json`. The page only knows the tabs in this list. |
| Another render (light, shadows, outline, background) | A render is a JSON preset, `tools/workshop/page/presets/<id>.json`, declared in `tools/workshop/page/presets/index.json`; its settings are read by `tools/workshop/page/viewer.js` | This is not touching the core: tune the render in the light and shadow panel of the page, export it as JSON, put the file in `page/rendus/` and add it to `index.json`. No game provides a render. |
| Another game as the active game, with no base head | `tools/workshop/common.py` (`TETE_BASE = JEU.valeur("tete_base.nom", None)`) | Possible with a built base (`tools/vroid_base`, `tools/free_base`): Chimera then offers only its face and hairstyle. The game's faces need a base head in the face catalogue. |

If you touch the core, add a test in `tools/workshop/tests/` that works without any game file,
like the fake PMX of `test_jeu_factice_traduit_a_la_lecture`.

## What your contribution must contain

- **The description**: `tools/games/<game>/game.json`, with its `compatibilite` key, plus the code
  touched, if any.
- **The declared level**, per heading (`eprouve`, `partiel` or `non`), and `COMPATIBILITY.md` rewritten
  by `python3 -I tools/games write`.
- **The models you tested it on**: their names and their number, and what
  `catalog.py summary` says (kept, rejected, reasons).
- **No game file**: no model, no texture, no extract, no image taken from a model. The repository
  keeps only descriptions and code.
