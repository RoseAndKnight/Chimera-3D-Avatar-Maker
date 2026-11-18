# Character workshop

Choose a face, a hairstyle and an outfit in a page, see the character in 3D, then get
a single finished PMX.

## Running

    bin/atelier          # opens http://127.0.0.1:8767 in the browser
    bin/atelier 9000     # on another port

Installing on a new machine: `INSTALLATION.md` (repository root). The repository provides no model, not
even the base body: as long as the prepared body, a face or a hairstyle is missing, `/` serves a page that says
what to do (`serveur.accueil`, `catalogue.a_faire`), and `catalogue.py preparer` stops without a body or a base head.

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
of the body with a skeleton with readable names (`outils/corps_maison/brancher.py`), when its data exist
(`<data>/corps-maison/squelette.json`); the version on the original bones then stays next to it
(`modele/resultat-zzz.pmx`). Without these data, or with `atelier.squelette` = `zzz`, the model keeps the bones of
its base body, and the VRM comes out of it as is.

## Layers

The page and the assembly run the same computation (`combinaison.py`), starting from parts that are all fitted
on the bare skin. The skin is the common frame: each garment vertex has its line (the direction
going away from the skin) and its height above it, and moves only along its line. The
rule: along a line, the parts are met in the order of their ranks, at least 1.5 mm apart
(more for coarse meshes). It is measured by exact rays (`rayons.py`)
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

Check independent of the adjustment: `controle_couches.py` counts, on the displayed triangles, the
edges of one part that cross a face of the other and the area of the lower part that shows in front of
the upper one, on six reference outfits (or `--pieces a,b,c`). To run again after any change to
`combinaison.py`.

The hair falls over the adjusted outfit (`combinaison.coiffer`): a strand under the outer face
of a part comes out by the shortest way, except under a hat, a hood or glasses.


## Catalogue

    python3 -I outils/atelier/catalogue.py preparer [--niveau petit|complet]   # only redoes what is missing
    python3 -I outils/atelier/catalogue.py preparer --perimees                  # also redoes the parts fitted by another method (V_PORTAGE) or on another body
    python3 -I outils/atelier/catalogue.py ajouter <part folder>               # one more part
    python3 -I outils/atelier/catalogue.py bilan                               # counts and rejection reasons

(`python3` = a Python with numpy, for example Blender's: `programmes.python` in the settings, see below.)
A part in the library format (`fiche.json`, `<part>.pmx`, `tex/`) is fitted on the
common body, measured (`mesure.py`) then cached. Rejected are those whose fitting or
measurement fails, or that fail `sous_la_peau` (more than 1 % of the vertices inside the body, counted by
the fitting without a depth limit: `traversee_repos` only sees down to 10 mm), `traversee_repos`,
`traversee_repos_p90_mm` or `poses_corps_seul`, as well as shoes without an upper (mere
soles, the foot would stay bare). A check without a value is not a passed check: the record
names it (`non_mesures`), and `bilan` counts them. An accessory whose garment measurement
discards all the parts (wing, bag, rivets) is no longer rejected for that (`mesure_sans_objet`);
placed as one block, it is stuck into the body by nature and `sous_la_peau` does not apply to it.
The record keeps the fitting method (`v_portage`) and the body (`corps`): `bilan` counts the
outdated parts, `preparer --perimees` redoes them.
The other checks, irrelevant on a bare body or tied to the source, stay recorded in the record.

## Where the files are

The cache (light parts, assembled models) is in `atelier.cache` of the settings (`ATELIER_CACHE`), by default
`<data>/atelier-base-vroid`, for the base body `<data>/base-vroid/corps.pmx`: bare body, face
(`tete.pmx`) and hairstyle (`coiffure.pmx`) of a VRoid Studio character, with `base.json` (base head of the VRoid
game and its measurements), `PROVENANCE.md` (terms of use read from the file) and `controles.json`, written by
`outils/base_vroid/fabriquer.py <modele.vrm>`. As long as that base does not exist, the free MakeHuman base
(`<data>/base-libre/corps.pmx`, `outils/base_libre`) is used if it is built, with the cache
`<data>/atelier-base-libre`. Another base body is set with `atelier.corps`, with its own cache
(`atelier.cache`): a cache is only valid for the body its parts were fitted on. The assembled models are in `<cache>/assemblages/<date>/`. `ATELIER_LANCEUR` prefixes Blender launches (for example
`systemd-run --user --scope -p MemoryMax=8G --`).

## Settings

No machine path is written in the code: everything goes through `outils/reglages.py`. Three layers, each one
overriding the previous one key by key:

1. `reglages.defaut.json` (repository root, tracked): public values, not to be changed for your machine;
2. `reglages.json` (repository root, ignored by git; `ATELIER_REGLAGES` names another file): your own,
   same shape, only the keys to change;
3. environment variables.

| Key | Public default | Variable | Role |
|---|---|---|---|
| `chemins.donnees` | `donnees` (in the repository) | `ATELIER_ATLAS` | data root (`<data>`: libraries, bodies, heads) |
| `chemins.squelettes` | `{donnees}/squelettes` | | bone namings per game (`maison.json`, `biped.json`) |
| `atelier.cache` | `{donnees}/atelier-base-vroid` | `ATELIER_CACHE` | Chimera's cache (fallback: `{donnees}/atelier-base-libre`) |
| `atelier.corps` | `{donnees}/base-vroid/corps.pmx` | `ATELIER_CORPS` | base body (`outils/base_vroid`; fallback: `{donnees}/base-libre/corps.pmx`) |
| `atelier.squelette` | `maison` | `ATELIER_SQUELETTE` | `zzz` keeps the game's bones |
| `programmes.blender` | `blender` (looked up in the PATH) | `ATELIER_BLENDER` | Blender of the queue |
| `programmes.python` | the running interpreter | `ATELIER_PYTHON` | Python (numpy) of the sub-computations and of `bin/atelier` |
| `calcul.places` | one per physical core, 2.5 GB each, 3 at least | `ATELIER_PLACES` (`BLENDER_PLACES` for the queue alone) | heavy computations at once |
| `calcul.lanceur` | nothing | `ATELIER_LANCEUR` | prefix of Blender launches |
| `calcul.file_attente` | `$TMPDIR/blender-file-<uid>` | | folder of the locks of `bin/blender-file` |
| `jeux.actif` | `vroid` | `ATELIER_JEU` | game the models come from (its description: `outils/jeux/<jeu>/jeu.json`) |
| `jeux.<jeu>.<source>` | the one in the description (`sources`) | | source folders of a game for your machine (e.g. `jeux.zzz.mmd_extrait`, `jeux.outerplane.personnages`) |

What the code knows about a game (characters, base head, special cases) is in its
description, `outils/jeux/<jeu>/`; parts you made yourself are declared in `outils/jeux/maison/`.
Adding a game: `outils/jeux/README.md`.

In a value, `{donnees}` (or any dotted key between braces) is replaced; a relative path starts from the
repository root; `~` is expanded. Example of `reglages.json`:

    {"chemins": {"donnees": "~/mes-donnees"}, "programmes": {"blender": "~/blender-5.2/blender"}}

To check: `python3 -I outils/reglages.py tout` (or `chemin atelier.cache`, `programme blender`). Blender
launches go through the queue `bin/blender-file`, which reads the same settings.

### Adjusting the layers

The garment types, the accessory kinds, their ranks (what goes over what), the outfit rules
and the stacking thresholds are in `couches.defaut.json` (repository root, tracked), read by
`outils/reglages.py` right after `reglages.defaut.json`. The meaning of each key is written in the file itself
(`_aide` of each section). To change something, do not touch that file: write in your
`reglages.json`, under `"couches"`, only what changes. The merge is done key by key, down into an
entry; a list (`bas_fermes`, `occupe`…) is replaced as a whole; `null` removes an entry.

    {"couches": {
      "vetements": {"jupe": {"rang": 33},
                    "kimono": {"onglet": "haut", "emplacement": "dessus", "rang": 57, "nom": "kimono"}},
      "accessoires": {"broche": {"rang": 79, "exclusif": false}},
      "seuils": {"sortie_max_mm": 10}}}

| Section | An entry | Role |
|---|---|---|
| `vetements` | `onglet`, `emplacement`, `rang` (required), `nom`, `occupe` | `type` of `fiche.json`: tab of the page, exclusive slot (two parts of the same slot are not worn together), rank (the smallest underneath), displayed name, extra slots taken (a dress takes top, bottom and underneath) |
| `accessoires` | `rang` (required), `exclusif`, `nom`, `comme_ceinture` | `sorte` of `fiche.json`; always in the extras tab; `comme_ceinture`: goes around the body, adjusted with the garments, placed on the fabric without tightening it |
| `tenue` | `rangs_dans_la_tenue`, `bas_fermes`, `hauts`, `bas` | rank imposed during the adjustment (belt 36: over the bottom, under the jacket); bottoms that cover the underwear (it is not worn); a top whose hem comes out of a bottom is worn over it |
| `seuils` | millimetres (`_mm`), `signal_part`, `passes` | margins between layers and for accessories, floor above the skin, free flap, largest push-out, sag of coarse meshes, masking of hidden faces, flagging of a clipping |

What stays in the code of `combinaison.py`, on purpose: the search ranges (`PORTEE_RENTREE`, `PORTEE_PEAU`,
`PORTEE`), the geometric criteria (`ACCORD`, `EMPREINTE`, `CONTACT`, `FIN`) and the grids (`CELLULE`, `VOXEL`,
`VOXEL_CORPS`). These are choices of the algorithm, not render preferences: changing them means rereading
the computation and running `controle_couches.py` again. The same goes for the rule "a sock lower than the underwear goes
under it" and the flags of `couvre.bin`.

When the change takes effect:

- the settings are read at startup: **restart the server** (`bin/atelier`) after changing `reglages.json`;
- a rank, a tab, a slot, `occupe`: nothing to redo. Each part record (`meta.json`) keeps
  those of the day it was prepared, but the catalogue, the layer adjustment and the assembly read them again from
  today's settings (`commun.ranger`). Only the displayed name of an already prepared part stays the one in
  its record;
- a threshold or an outfit rule: nothing to redo, no outfit adjustment is kept on disk (the server
  keeps them in memory while it runs). Models already assembled keep the settings of the
  day they were assembled;
- an added type or kind: its parts enter the catalogue at the next `catalogue.py preparer` (which only
  does what is missing) or through `catalogue.py ajouter <folder>`. A part of a removed type stays in the cache
  and in the catalogue until it is deleted;
- the page knows only four part tabs (`haut`, `bas`, `chaussures`, `accessoires`, written in
  `page/atelier.js`): a type filed under another tab does not appear there.

To see the merged values: `python3 -I outils/reglages.py valeur couches.vetements.jupe`.

Tests: `python3 -I outils/atelier/tests/test_atelier.py`.


## Security

The server listens only on `127.0.0.1` and answers only its own page: a request that names it other
than by its local address, or that comes from another site open in the browser, is refused (403). Do not
expose it on a network: it launches Blender and writes into the cache at the page's request.
