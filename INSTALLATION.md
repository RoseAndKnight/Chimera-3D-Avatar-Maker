# Installing Chimera

Chimera assembles heads, hairstyles and clothes on a base body, in a local web page.
The repository contains code only. It provides no model: no body, no head, no clothes. You bring
your own.

## Requirements

- **Linux.** It is the only system tried.
- **Python 3 with numpy.** The server and the computations need numpy only. Tried: Python 3.13 (the one
  shipped with Blender 5.2) and Python 3.14 with numpy 2.3. The Python shipped with Blender is enough:
  `<Blender folder>/5.2/python/bin/python3.13`.
- **Blender 5.2** (tried: 5.2.2). It prepares the catalogue and assembles the characters. Other versions
  have not been tried.
- **A browser with WebGL2.** Without WebGL2, the page falls back to a "simplified preview".

## First run

From the repository root, three commands. You need a VRoid Studio character exported as .vrm (VRM 0.x):
your own, or a VRoid Studio sample model (AvatarSample_D to G are CC0).

    python3 -I outils/base_vroid/fabriquer.py <character.vrm>   # bare body, face and hairstyle; no Blender
    python3 -I outils/atelier/catalogue.py preparer              # the catalogue of the page; needs Blender
    bin/atelier                                                  # http://127.0.0.1:8767, opened in the browser

The page opens on the character's bare body, its face and its hairstyle: enough to try the tool and
"Assemble" a model (PMX and VRM). Its clothes, and those of your other models, are added to the catalogue
(see below). The terms of use read from the file are copied next to the base (`PROVENANCE.md`): they apply
to every model assembled on it.

Without a .vrm file, the free base (MakeHuman, CC0) is the fallback: `python3 -I outils/base_libre/fabriquer.py`
(it downloads a few MB of MakeHuman from GitHub, once). As long as the VRoid base is not built, Chimera
uses the free base if that one is built, with its own cache (`atelier-base-libre`).

    bin/atelier 8780       # on another port

While something is missing, the server starts anyway. The page then says that Chimera does not yet have
what it needs to open, and lists what is missing. The terminal prints the same list.

To see where the catalogue stands:

    python3 -I outils/atelier/catalogue.py bilan

## Ready-made library (optional)

Instead of preparing models of your own, you can install a library that is already prepared: about 480
VRoid Hub models whose authors allow redistribution and modification (faces, hairstyles, about 2,000 clothing
parts, with names, tags and a link to each original model). It is about 8.6 GB to download and takes about 10 GB
once installed. You need neither Blender nor `catalogue.py preparer` for it; you still need Blender to assemble.

    python3 -I outils/bibliotheque/bibliotheque.py installer    # downloads, checks (sha256), installs into chemins.donnees
    bin/atelier

The address is `bibliotheque.adresse` in the settings. A folder or another URL can be given instead:
`installer <folder or URL>`. An interrupted download resumes when the same command is run again. Models of your
own that are already there are never overwritten: the command stops, and `--remplacer` sets them aside in
`<donnees>/.chimera-bibliotheque/`. You can add your own models afterwards, as below; `preparer` then
prepares only those. Each model keeps its author's terms (see `CREDITS.md` of the library). Details:
`outils/bibliotheque/README.md`.

## Settings

The code contains no machine path. The public values are in `reglages.defaut.json`: do not edit it.
Write yours in `reglages.json`, at the repository root. That file is not tracked by git.
Put only the keys you want to change:

    {
     "chemins": {"donnees": "~/my-data"},
     "programmes": {
      "blender": "~/blender-5.2.2-linux-x64/blender",
      "python": "~/blender-5.2.2-linux-x64/5.2/python/bin/python3.13"
     }
    }

- `chemins.donnees`: the folder of your data (models, libraries, cache). Default: `donnees/` in the
  repository, ignored by git.
- `programmes.blender`: Blender. Default: `blender`, looked up in the PATH.
- `programmes.python`: the Python with numpy. Default: the one running the command.
- `atelier.corps`, `atelier.cache`: Chimera's base body and cache (see below).

To see the values read:

    python3 -I outils/reglages.py tout

All the keys and environment variables: `outils/atelier/README.md`, settings section.
Restart the server after changing `reglages.json`.

## Where to put your models

The formats read, the expected folders and the description of a game are in `COMPATIBILITE.md`.

## What the repository does not provide

No model. The page needs three things to open: a **base body**, at least one **face** and at least one
**hairstyle**. All three are built together from a VRoid Studio character by
`outils/base_vroid/fabriquer.py` (see "First run" and `outils/base_vroid/README.md`), or, without a
.vrm file, from MakeHuman (CC0) by `outils/base_libre/fabriquer.py`.

- **The base body** (`atelier.corps`, default `{donnees}/base-vroid/corps.pmx`; free base:
  `{donnees}/base-libre/corps.pmx`). The character's bare skin, bones with MMD names, arms-down pose: parts
  from MMD models are worn on it through their bone names. To start from another reference body:
  `outils/corps_refait` (mesh), `outils/forme_maison` (shape and skin), `outils/corps_maison` (skeleton),
  each with its README. Another body needs a cache of its own (`atelier.cache`).
- **The face and hairstyle of the base** (`tete.pmx`, `coiffure.pmx`, next to the body). They are used as long
  as the base head is not in the face catalogue.
- **The base head of a game.** Each game names its own (`tete_base` in `outils/jeux/<jeu>/jeu.json`).
  VRoid's is the head of the built base (`base-vroid/base.json`). Once its entry is in the face
  catalogue, the other faces of the game are stitched onto it (separate cache: change `atelier.cache`).
- **Faces, hairstyles, clothes.** They come from your models, laid out as `COMPATIBILITE.md` says.

The shortest path to characters of your own is **VRoid Studio**: a character exported as .vrm
(VRM 0.x) gives Chimera its face, its hairstyle and its clothes.

    python3 -I outils/vroid/importer.py <file.vrm>... --sortie <folder>

Then, in `reglages.json`, set `jeux.actif` = `vroid` (the default) and `jeux.vroid.mmd_extrait` = that folder.
The base head is the one of the base character. Next steps and limits: `COMPATIBILITE.md`,
VRoid section.

Once your models are in place, prepare the catalogue (Blender, several minutes):

    python3 -I outils/atelier/catalogue.py preparer --niveau petit    # a trial, a few parts
    python3 -I outils/atelier/catalogue.py preparer                   # everything; run again, it resumes where it stopped

If the base body or the base head is missing, `preparer` says so and runs nothing. Then reload the page.

## Checking

    python3 -I outils/atelier/tests/test_atelier.py
    python3 -I outils/vroid/tests/test_importeur.py
    python3 -I outils/base_vroid/tests/test_base_vroid.py

(`python3` always means a Python with numpy, for instance Blender's.)
