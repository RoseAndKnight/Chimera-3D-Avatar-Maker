# tools/library: the ready-made library

The public repository ships no model. This tool fetches a library that is already prepared (faces, hairstyles,
clothes, with their names, tags and links to the original models) and installs it where the workshop looks for
its data, so that `bin/chimera` opens on the whole catalogue and assembles characters at once, with no
preparation in Blender.

    python3 -I tools/library/library.py install                 # from library.url (settings.default.json)
    python3 -I tools/library/library.py install <URL or folder> # another address, or a package on disk
    python3 -I tools/library/library.py install --replace     # set aside (never delete) what is in the way

Maintainer side (needs the workshop's Python, with numpy):

    python3 -I tools/library/library.py build <installation> <output> --statuses <statuses.json>
    python3 -I tools/library/library.py check <output> --statuses <statuses.json>

`<installation>` is a folder with a `settings.json` naming prepared data (`paths.data`, `workshop.cache`,
`workshop.body`); nothing is written there. The output folder is a Hugging Face *dataset* repository as is:
`index.json`, `archives/*.tar.gz`, `README.md` (the dataset card), `CREDITS.md`, `sources.json`, `upload.sh`.

## What the workshop reads (and so what the package holds)

Established from the code (`tools/workshop/server.py`, `catalog.py`, `assemble_blender.py`, `inputs.py`,
`tools/head/hairstyle.py`, `tools/head/library.py`, `tools/clothes/wear.py`, `tools/workshop/vrm.py`).
The package roots, and where the install code puts them:

| Root | Folder on the machine | Read by |
|---|---|---|
| `cache/` | `workshop.cache` | the page and the assembly |
| `base/` | folder of `workshop.body` | the assembly and the preparation |
| `face/` | `<data>/face-library` | the assembly (hairstyle posing) |
| `clothes/` | `<data>/clothes-library` | the assembly (fitting each part) |
| `mmd_extracted/` | `games.vroid.mmd_extracted` (`<data>/sources/vroid/mmd-extracted`) | the assembly (hairstyle posing) |
| `pivot/` | `<data>/pivot-head` | the hairstyle posing (pivot head) |

The page (`server.py`, `page/*.js`) reads only the cache: `catalog.json` (rebuilt at start from the records
`heads|hairstyles|parts/<id>/meta.json`), `labels.json` (names, descriptions, tags), `sources.json` (page,
author, licence of each model), `body/body.glb`, and per item `heads/<id>/{head.glb, body.bin, hidden_skin.bin,
thumbnail.png}`, `hairstyles/<id>/{hairstyle.glb, <face>.bin, thumbnail.png}`, `parts/<id>/{part.glb, hidden_skin.bin,
covers.bin, thumbnail.png}`, and `tex/<fingerprint>.webp` named inside those glb files and records.

The assembly (`assemble_blender.py`) reads, for the chosen face `T`, hairstyle `H` and parts `P…`:
- `cache/heads/T/meta.json` and `cache/heads/T/model/` (the face sewn on the base head; its textures in `cache/base/`);
- `cache/hairstyles/H/meta.json`; `hairstyle.place` reads `cache/library/H` and `cache/library/<T's library name>`
  (links to `face/<name>/`: `record.json`, `part.npz`, `hairstyle.npz`…), the source models they name
  (`mmd_extracted/<name>/<name>.pmx` and its textures) and the pivot head (`pivot/pivot-head.json`, which names a source model);
- `cache/parts/P/meta.json` → `"folder"`: `clothes/<model>/<part>/` (PMX, `record.json`) and the textures that PMX
  names (`clothes/<model>/tex/`); `covers.bin`, `hidden_skin.bin` of the part;
- the base body (`base/body.pmx`) through the settings.

The export (PMX, VRM: `assemble_blender.leave_vrm`, `vrm.py`, `inputs.export`) reads only the assembled model in
`cache/assemblies/<id>/`, made by the assembly.

Left out: the original `.vrm` files, `cache/extractions/`, `cache/imports-in-progress/`, `cache/assemblies/`, cache
textures no kept record names, garment-library parts that have no record in the cache (the next `prepare` would
fit them for hours), and the source folders of refused parts; of a refused item only its record and thumbnail stay
(the page lists refused items).

## What the install code does to the files

- **Machine paths.** The records of the owner's machine hold absolute paths. In the package they read `{cache}`,
  `{mmd_extracted}`… and the install code writes this machine's folders there.
- **Fingerprints.** A record keeps the fingerprint of its source file (`common.fingerprint_file`: path, size, date);
  a different one makes `catalog.py prepare` prepare the item again. `index.json` lists the owner's fingerprints
  with their files (`fingerprints`); the install code computes them again on the installed files and rewrites them. Adding
  your own model afterwards prepares that model only.
- **Identical copies.** The cache keeps one file under many names (a hairstyle's pairs: the same `.bin` under every
  face's name, about 225,000 names). The archives hold each content once; `index.json` (`duplicates`) lists the other
  names and the install code makes them as hard links (a copy if the disk refuses links).
- **The face library links** `cache/library/<name>` → `face/<name>` are made by the install code (`links`), as
  `catalog.prepare_library` makes them; the archives hold no link.
- **Shared dictionaries** (`labels.json`, `sources.json`, `face-library/index.json`,
  `clothes-library/check.json`) are merged: your own entries stay.
- **Someone else's files.** An item already present that a previous installation did not put there (your own
  model, another base body) stops the installation before anything is downloaded; `--replace` moves those items
  to `<data>/.chimera-library/replaced-<date>/`. The manifest of the last installation is
  `<cache>/library.json`.
- **Safety.** sha256 of each archive checked before unpacking; members: regular files and folders only, relative,
  inside a known root, no `..`; every path named by `index.json` checked the same way; files written by the
  install code itself (never `tarfile.extract`); nothing executed. Downloads go to `<data>/.chimera-library/archives/`;
  an interrupted one resumes (HTTP Range) or starts again, a finished and checked archive is not fetched again;
  the archives are removed after a successful installation (`--keep-archives` keeps them).

## Format 2 and `migrate`

The package and the data are in format 2 (English names, `FORMAT-2.md`); `index.json` says `"format": 2` and the
data folder holds `chimera-data.json`. The install code refuses a package of another format ("update the code or the
library"); the code refuses a format-1 data folder and names the converter. `library.py migrate <data>
[--settings <settings.json>] [--check]` (`migrate.py`) converts a format-1 folder in place, in phases: names
(folders and files, deepest first, `os.rename` keeps hard links and dates), records (every JSON rewritten through
a new file and `os.replace`, so a `cp -al` backup keeps the old content), models' paths, fingerprints computed again
the way the install code does, links of the face library, settings paths, then the marker. Each phase says what it
did and is skipped when done; `--check` lists what is still format 1 and exits 1 if anything is.

## Which models (`build`)

A VRoid Hub model `h<id>`: the status for original models in the statuses file (`--statuses`) **and** licence (`sources.json` of the cache) with
`allowRedistribution: true` and `modification: allowModificationRedistribution`. Another model: only if its own file
says so (VRM meta CC0, or a VRoid Hub licence URL with `redistribution=allow&modification=allow`; for the base,
`provenance.json` of the base). The rest is left out everywhere. `check` reads every archive back: sha256, safe
members, no path into a home folder, no id of a model left out (in paths, records, texts), no `cache/tex` texture that no kept record
names, every unit present.

Tests: `python3 -I tools/library/tests/test_installation.py` (standard library only).
