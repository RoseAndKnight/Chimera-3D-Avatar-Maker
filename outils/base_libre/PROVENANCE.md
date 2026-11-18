# Provenance of the free base

The body (`corps.pmx`, `forme.pkl`, `maison/`), the face (`tete.pmx`, `yeux.png`) and the hairstyle
(`coiffure.pmx`, `cheveux.png`) are built by `outils/base_libre/fabriquer.py`. Nothing in them comes from a game.

- **Shape.** MakeHuman's base mesh (`makehuman/data/3dobjs/base.obj`, group `body`) and its shape
  targets (`makehuman/data/targets/...`; the list and the doses are in `silhouette.json`), taken from the
  public repository https://github.com/makehumancommunity/makehuman at the version recorded in `fabriquer.py`.
  These files are published under **CC0 1.0** (public domain): see MakeHuman's `LICENSE.md`, section C,
  downloaded next to `base.obj`.
- **Head.** The head of the same mesh, joined to the neck.
- **Eyes.** MakeHuman's simple eyes (`makehuman/data/eyes/low-poly`) and their texture
  (`eyes/materials/brown_eye.png`), same repository, same version, same CC0 licence.
- **Hairstyle.** The helper volume `helper-hair` of `base.obj` (CC0), trimmed here into a bob with a fringe, and a
  cap drawn from the skull skin. Flat colour.
- **Body mesh, skeleton, pose, weights.** Computed by the tools of this repository (`outils/corps_refait`,
  `outils/corps_maison`, `outils/base_libre`): our grid laid on MakeHuman's skin, the limb joints taken
  from the `joint-*` markers of `base.obj`, the neck straightened and thinned, the arms and legs put in
  the A-pose of MMD models.
- **Bone names.** Those of MMD models (首, 頭, 腕, 足…), so that the parts of these models can be worn on it.
- **Skin.** A flat colour (`peau.png`).

The built set can be reused and redistributed without conditions (CC0 for the MakeHuman part).
