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

    python3 -I tools/vroid_base/build.py <character.vrm>   # bare body, face and hairstyle; no Blender
    python3 -I tools/workshop/catalog.py prepare              # the catalogue of the page; needs Blender
    bin/chimera                                                  # http://127.0.0.1:8767, opened in the browser

The page opens on the character's bare body, its face and its hairstyle: enough to try the tool and
"Assemble" a model (PMX and VRM). Its clothes, and those of your other models, are added to the catalogue
(see below). The terms of use read from the file are copied next to the base (`PROVENANCE.md`): they apply
to every model assembled on it.

Without a .vrm file, the free base (MakeHuman, CC0) is the fallback: `python3 -I tools/free_base/build.py`
(it downloads a few MB of MakeHuman from GitHub, once). As long as the VRoid base is not built, Chimera
uses the free base if that one is built, with its own cache (`workshop-free-base`).

    bin/chimera 8780       # on another port

While something is missing, the server starts anyway. The page then says that Chimera does not yet have
what it needs to open, and lists what is missing. The terminal prints the same list.

To see where the catalogue stands:

    python3 -I tools/workshop/catalog.py summary

## Ready-made library (optional)

Instead of preparing models of your own, you can install a library that is already prepared: about 480
VRoid Hub models whose authors allow redistribution and modification (faces, hairstyles, about 2,000 clothing
parts, with names, tags and a link to each original model). It is about 8.6 GB to download and takes about 10 GB
once installed. You need neither Blender nor `catalog.py prepare` for it; you still need Blender to assemble.

    python3 -I tools/library/library.py install    # downloads, checks (sha256), installs into paths.data
    bin/chimera

The address is `library.url` in the settings. A folder or another URL can be given instead:
`install <folder or URL>`. An interrupted download resumes when the same command is run again. Models of your
own that are already there are never overwritten: the command stops, and `--replace` sets them aside in
`<data>/.chimera-library/`. You can add your own models afterwards, as below; `prepare` then
prepares only those. Each model keeps its author's terms (see `CREDITS.md` of the library). Details:
`tools/library/README.md`.

### Installed before format 2?

Since format 2 every data name is in English (folders, files, JSON keys; table: [FORMAT-2.md](FORMAT-2.md)). The code
reads only format 2: on a folder installed or prepared earlier (format 1) it stops and gives the command. Either
convert that folder once, in place (a few minutes; nothing is downloaded or prepared again; running it again
does no harm):

    python3 -I tools/library/library.py migrate <your data folder> --settings <your settings.json>
    python3 -I tools/library/library.py migrate <your data folder> --check      # 0 left: done

or install the library again into an empty folder (`paths.data`). A format-1 package is refused by the installation step
of this code (and the other way round): update the code (`git pull`) or the library.

## Settings

The code contains no machine path. The public values are in `settings.default.json`: do not edit it.
Write yours in `settings.json`, at the repository root. That file is not tracked by git.
Put only the keys you want to change:

    {
     "paths": {"data": "~/my-data"},
     "programs": {
      "blender": "~/blender-5.2.2-linux-x64/blender",
      "python": "~/blender-5.2.2-linux-x64/5.2/python/bin/python3.13"
     }
    }

- `paths.data`: the folder of your data (models, libraries, cache). Default: `data/` in the
  repository, ignored by git.
- `programs.blender`: Blender. Default: `blender`, looked up in the PATH.
- `programs.python`: the Python with numpy. Default: the one running the command.
- `workshop.body`, `workshop.cache`: Chimera's base body and cache (see below).

To see the values read:

    python3 -I tools/settings.py all

All the keys and environment variables: `tools/workshop/README.md`, settings section.
Restart the server after changing `settings.json`.

## Where to put your models

The formats read, the expected folders and the description of a game are in `COMPATIBILITY.md`.

## What the repository does not provide

No model. The page needs three things to open: a **base body**, at least one **face** and at least one
**hairstyle**. All three are built together from a VRoid Studio character by
`tools/vroid_base/build.py` (see "First run" and `tools/vroid_base/README.md`), or, without a
.vrm file, from MakeHuman (CC0) by `tools/free_base/build.py`.

- **The base body** (`workshop.body`, default `{data}/vroid-base/body.pmx`; free base:
  `{data}/free-base/body.pmx`). The character's bare skin, bones with MMD names, arms-down pose: parts
  from MMD models are worn on it through their bone names. To start from another reference body:
  `tools/rebuilt_body` (mesh), `tools/custom_body` (skeleton) (the shape and skin tool is not published),
  each with its README. Another body needs a cache of its own (`workshop.cache`).
- **The face and hairstyle of the base** (`head.pmx`, `hairstyle.pmx`, next to the body). They are used as long
  as the base head is not in the face catalogue.
- **The base head of a game.** Each game names its own (`base_head` in `tools/games/<game>/game.json`).
  VRoid's is the head of the built base (`vroid-base/base.json`). Once its entry is in the face
  catalogue, the other faces of the game are stitched onto it (separate cache: change `workshop.cache`).
- **Faces, hairstyles, clothes.** They come from your models, laid out as `COMPATIBILITY.md` says.

The shortest path to characters of your own is **VRoid Studio**: a character exported as .vrm
(VRM 0.x) gives Chimera its face, its hairstyle and its clothes.

    python3 -I tools/vroid/importer.py <file.vrm>... --output <folder>

Then, in `settings.json`, set `games.active` = `vroid` (the default) and `games.vroid.mmd_extracted` = that folder.
The base head is the one of the base character. Next steps and limits: `COMPATIBILITY.md`,
VRoid section.

Once your models are in place, prepare the catalogue (Blender, several minutes):

    python3 -I tools/workshop/catalog.py prepare --level small    # a trial, a few parts
    python3 -I tools/workshop/catalog.py prepare                   # everything; run again, it resumes where it stopped

If the base body or the base head is missing, `prepare` says so and runs nothing. Then reload the page.

## Checking

    python3 -I tools/workshop/tests/test_workshop.py
    python3 -I tools/vroid/tests/test_importer.py
    python3 -I tools/vroid_base/tests/test_vroid_base.py

(`python3` always means a Python with numpy, for instance Blender's.)
