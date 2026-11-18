# outils/bibliotheque: the ready-made library

The public repository ships no model. This tool fetches a library that is already prepared (faces, hairstyles,
clothes, with their names, tags and links to the original models) and installs it where the workshop looks for
its data, so that `bin/atelier` opens on the whole catalogue and assembles characters at once, with no
preparation in Blender.

    python3 -I outils/bibliotheque/bibliotheque.py installer                 # from bibliotheque.adresse (reglages.defaut.json)
    python3 -I outils/bibliotheque/bibliotheque.py installer <URL or folder> # another address, or a package on disk
    python3 -I outils/bibliotheque/bibliotheque.py installer --remplacer     # set aside (never delete) what is in the way

Maintainer side (needs the workshop's Python, with numpy):

    python3 -I outils/bibliotheque/bibliotheque.py construire <installation> <output> --statuts <statuts.json>
    python3 -I outils/bibliotheque/bibliotheque.py verifier <output> --statuts <statuts.json>

`<installation>` is a folder with a `reglages.json` naming prepared data (`chemins.donnees`, `atelier.cache`,
`atelier.corps`); nothing is written there. The output folder is a Hugging Face *dataset* repository as is:
`index.json`, `archives/*.tar.gz`, `README.md` (the dataset card), `CREDITS.md`, `sources.json`, `mettre-en-ligne.sh`.

## What the workshop reads (and so what the package holds)

Established from the code (`outils/atelier/serveur.py`, `catalogue.py`, `assembler_blender.py`, `entrees.py`,
`outils/tete/coiffure.py`, `outils/tete/bibliotheque.py`, `outils/vetements/porter.py`, `outils/atelier/vrm.py`).
The package roots, and where the installer puts them:

| Root | Folder on the machine | Read by |
|---|---|---|
| `cache/` | `atelier.cache` | the page and the assembly |
| `base/` | folder of `atelier.corps` | the assembly and the preparation |
| `visage/` | `<donnees>/bibliotheque-visage` | the assembly (hairstyle posing) |
| `vetements/` | `<donnees>/bibliotheque-vetements` | the assembly (fitting each part) |
| `mmd_extrait/` | `jeux.vroid.mmd_extrait` (`<donnees>/sources/vroid/mmd-extrait`) | the assembly (hairstyle posing) |
| `pivot/` | `<donnees>/tete-pivot` | the hairstyle posing (pivot head) |

The page (`serveur.py`, `page/*.js`) reads only the cache: `catalogue.json` (rebuilt at start from the records
`tetes|coiffures|pieces/<id>/meta.json`), `etiquettes.json` (names, descriptions, tags), `sources.json` (page,
author, licence of each model), `corps/corps.glb`, and per item `tetes/<id>/{tete.glb, corps.bin, peau_cachee.bin,
vignette.png}`, `coiffures/<id>/{coiffure.glb, <face>.bin, vignette.png}`, `pieces/<id>/{piece.glb, peau_cachee.bin,
couvre.bin, vignette.png}`, and `tex/<fingerprint>.webp` named inside those glb files and records.

The assembly (`assembler_blender.py`) reads, for the chosen face `T`, hairstyle `H` and parts `P…`:
- `cache/tetes/T/meta.json` and `cache/tetes/T/modele/` (the face sewn on the base head; its textures in `cache/base/`);
- `cache/coiffures/H/meta.json`; `coiffure.poser` reads `cache/biblio/H` and `cache/biblio/<T's library name>`
  (links to `visage/<name>/`: `fiche.json`, `partie.npz`, `coiffure.npz`…), the source models they name
  (`mmd_extrait/<name>/<name>.pmx` and its textures) and the pivot head (`pivot/tete-pivot.json`, which names a source model);
- `cache/pieces/P/meta.json` → `"dossier"`: `vetements/<model>/<part>/` (PMX, `fiche.json`) and the textures that PMX
  names (`vetements/<model>/tex/`); `couvre.bin`, `peau_cachee.bin` of the part;
- the base body (`base/corps.pmx`) through the settings.

The export (PMX, VRM: `assembler_blender.sortir_vrm`, `vrm.py`, `entrees.export`) reads only the assembled model in
`cache/assemblages/<id>/`, made by the assembly.

Left out: the original `.vrm` files, `cache/extractions/`, `cache/imports-en-cours/`, `cache/assemblages/`, cache
textures no kept record names, garment-library parts that have no record in the cache (the next `preparer` would
fit them for hours), and the source folders of refused parts; of a refused item only its record and thumbnail stay
(the page lists refused items).

## What the installer does to the files

- **Machine paths.** The records of the owner's machine hold absolute paths. In the package they read `{cache}`,
  `{mmd_extrait}`… and the installer writes this machine's folders there.
- **Fingerprints.** A record keeps the fingerprint of its source file (`commun.empreinte_fichier`: path, size, date);
  a different one makes `catalogue.py preparer` prepare the item again. `index.json` lists the owner's fingerprints
  with their files (`empreintes`); the installer computes them again on the installed files and rewrites them. Adding
  your own model afterwards prepares that model only.
- **Identical copies.** The cache keeps one file under many names (a hairstyle's pairs: the same `.bin` under every
  face's name, about 225,000 names). The archives hold each content once; `index.json` (`doubles`) lists the other
  names and the installer makes them as hard links (a copy if the disk refuses links).
- **The face library links** `cache/biblio/<name>` → `visage/<name>` are made by the installer (`liens`), as
  `catalogue.preparer_biblio` makes them; the archives hold no link.
- **Shared dictionaries** (`etiquettes.json`, `sources.json`, `bibliotheque-visage/index.json`,
  `bibliotheque-vetements/controle.json`) are merged: your own entries stay.
- **Someone else's files.** An item already present that a previous installation did not put there (your own
  model, another base body) stops the installation before anything is downloaded; `--remplacer` moves those items
  to `<donnees>/.chimera-bibliotheque/remplaces-<date>/`. The manifest of the last installation is
  `<cache>/bibliotheque.json`.
- **Safety.** sha256 of each archive checked before unpacking; members: regular files and folders only, relative,
  inside a known root, no `..`; every path named by `index.json` checked the same way; files written by the
  installer itself (never `tarfile.extract`); nothing executed. Downloads go to `<donnees>/.chimera-bibliotheque/archives/`;
  an interrupted one resumes (HTTP Range) or starts again, a finished and checked archive is not fetched again;
  the archives are removed after a successful installation (`--garder-archives` keeps them).

## Which models (`construire`)

A VRoid Hub model `h<id>`: status `originale` in `statuts.json` **and** licence (`sources.json` of the cache) with
`allowRedistribution: true` and `modification: allowModificationRedistribution`. Another model: only if its own file
says so (VRM meta CC0, or a VRoid Hub licence URL with `redistribution=allow&modification=allow`; for the base,
`provenance.json` of the base). The rest is left out everywhere. `verifier` reads every archive back: sha256, safe
members, no path into a home folder, no id of a model left out (in paths, records, texts), no `cache/tex` texture that no kept record
names, every unit present.

Tests: `python3 -I outils/bibliotheque/tests/test_installation.py` (standard library only).
