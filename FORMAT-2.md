# Data format 2: old names → new names

Format 2 writes every data name in English. The code reads only format 2; faced with a format-1 folder it stops
and names the conversion command. Generated from `tools/dataformat.py` (the tables the code and the converter use),
plus the routes and options below. Step 1 renamed the code's folders, files, commands and settings keys;
this step renames the data and what the server and the page exchange.

Convert a format-1 data folder once (rerunnable, says what it did, keeps the files' dates and hard links):

    python3 -I tools/library/library.py migrate <data folder> [--settings <settings.json>]
    python3 -I tools/library/library.py migrate <data folder> --check     # lists what is still format 1

The data folder gets `chimera-data.json` (`{"format": 2}`); a ready-made package says `"format": 2` in its `index.json`.

## What stays as it is

- proper names: model and character names (`Sendagaya_Shino`, `Vivi`), bone and material names of the models (Japanese or Chinese);
- the bilingual labels `[english, french]` (`labels.json`, `label`);
- settings keys already renamed in step 1 (`mmd_extrait` in `game.json` stays the name of the path setting; its default folder is now `mmd-extracted`).

## Folders

| format 1 | format 2 |
|---|---|
| `atelier-base-vroid` | `workshop-vroid-base` |
| `atelier-base-libre` | `workshop-free-base` |
| `maison` | `own-rig` |
| `base-vroid` | `vroid-base` |
| `base-libre` | `free-base` |
| `bibliotheque-vetements` | `clothes-library` |
| `bibliotheque-visage` | `face-library` |
| `tete-pivot` | `pivot-head` |
| `mmd-extrait` | `mmd-extracted` |
| `.chimera-bibliotheque` | `.chimera-library` |
| `squelettes` | `skeletons` |
| `cheveux` | `hair` |
| `doubles` | `duplicates` |
| `en_cours` | `in_progress` |
| `tetes` | `heads` |
| `coiffures` | `hairstyles` |
| `pieces` | `parts` |
| `biblio` | `library` |
| `corps` | `body` |
| `assemblages` | `assemblies` |
| `imports-en-cours` | `imports-in-progress` |
| `modele` | `model` |
| `teinte` | `tint` |
| `rendus` | `renders` |
| `1-coiffure` | `1-hairstyle` |
| `2-greffe` | `2-graft` |
| `3-pieces` | `3-parts` |
| `vetements` | `clothes` |
| `accessoires` | `accessories` |

## Files

Plus these patterns: `^(\d+)_peau\.png$` → `\1_skin.png`, `^reflet-([0-9a-f]+\.png)$` → `highlight-\1`, `^atelier-suite-(.+\.json)$` → `workshop-batch-\1`

| format 1 | format 2 |
|---|---|
| `corps.pmx` | `body.pmx` |
| `corps.png` | `body.png` |
| `corps.glb` | `body.glb` |
| `corps.bin` | `body.bin` |
| `corps_pos.bin` | `body_pos.bin` |
| `tete.pmx` | `head.pmx` |
| `tete.png` | `head.png` |
| `tete.glb` | `head.glb` |
| `coiffure.pmx` | `hairstyle.pmx` |
| `coiffure.glb` | `hairstyle.glb` |
| `coiffure.json` | `hairstyle.json` |
| `coiffure.npz` | `hairstyle.npz` |
| `peau.png` | `skin.png` |
| `yeux.png` | `eyes.png` |
| `resultat.pmx` | `result.pmx` |
| `resultat.vrm` | `result.vrm` |
| `piece.glb` | `part.glb` |
| `pied.bin` | `foot.bin` |
| `vignette.png` | `thumbnail.png` |
| `planche.png` | `sheet.png` |
| `trois_quarts.png` | `three_quarters.png` |
| `dos.png` | `back.png` |
| `peau_cachee.bin` | `hidden_skin.bin` |
| `couvre.bin` | `covers.bin` |
| `recalage.npz` | `registration.npz` |
| `partie.npz` | `part.npz` |
| `formes.npz` | `shapes.npz` |
| `fiche.json` | `record.json` |
| `catalogue.json` | `catalog.json` |
| `etiquettes.json` | `labels.json` |
| `mesures.json` | `measures.json` |
| `teinte.json` | `tint.json` |
| `portage.json` | `fitting.json` |
| `controles.json` | `checks.json` |
| `controle.json` | `check.json` |
| `etat.json` | `state.json` |
| `journal.txt` | `log.txt` |
| `greffe.json` | `graft.json` |
| `decoupe.json` | `cutout.json` |
| `tete-pivot.json` | `pivot-head.json` |
| `bibliotheque.json` | `library.json` |
| `demande.json` | `request.json` |
| `ombre-visage.png` | `face-shade.png` |
| `ombre-visage.json` | `face-shade.json` |
| `cartes.bin` | `maps.bin` |
| `cartes-corps.bin` | `body-maps.bin` |
| `erreur.json` | `error.json` |
| `raccord.json` | `seam.json` |
| `poids.npz` | `weights.npz` |
| `forme.pkl` | `shape.pkl` |
| `forme-makehuman.pkl` | `shape-makehuman.pkl` |
| `squelette.json` | `skeleton.json` |
| `cheveux.png` | `hair.png` |
| `recette.json` | `recipe.json` |
| `poids_texture.npz` | `texture_weights.npz` |
| `visage_compose.png` | `composed_face.png` |
| `visage.pmx` | `face.pmx` |
| `tete-moyenne.pmx` | `average-head.pmx` |
| `piece_tete.pmx` | `head_part.pmx` |
| `peau_corps.png` | `body_skin.png` |
| `reste_donneur.png` | `donor_rest.png` |
| `piece.avant-reprise.glb` | `part.before-repair.glb` |
| `peau_cachee.avant.bin` | `hidden_skin.before.bin` |

## Garment and accessory types (words, ids and file names: `haut-2` → `top-2`, `<perso>__haut` → `<perso>__top`, `haut+garnitures.pmx` → `top+trims.pmx`)

| format 1 | format 2 |
|---|---|
| `culotte` | `underwear` |
| `chaussettes` | `socks` |
| `haut` | `top` |
| `manches` | `sleeves` |
| `short` | `shorts` |
| `pantalon` | `trousers` |
| `jupe` | `skirt` |
| `robe` | `dress` |
| `combinaison` | `jumpsuit` |
| `chaussures` | `shoes` |
| `bottes` | `boots` |
| `gants` | `gloves` |
| `col` | `collar` |
| `gilet` | `waistcoat` |
| `veste` | `jacket` |
| `manteau` | `coat` |
| `capuche` | `hood` |
| `ceinture` | `belt` |
| `harnais` | `harness` |
| `jarretiere` | `garter` |
| `chevillere` | `anklet` |
| `brassard` | `armband` |
| `bague` | `ring` |
| `cravate` | `tie` |
| `echarpe` | `scarf` |
| `collier` | `necklace` |
| `bijou` | `jewel` |
| `sac` | `bag` |
| `lunettes` | `glasses` |
| `masque` | `mask` |
| `couvre_chef` | `headwear` |
| `casque_audio` | `headphones` |
| `queue` | `tail` |
| `ailes` | `wings` |
| `oreilles` | `ears` |
| `autre` | `other` |
| `cheveux` | `hair` |
| `ornement` | `ornament` |
| `sangle` | `strap` |
| `boucle_bouton` | `fastener` |
| `ecusson` | `crest` |
| `poche` | `pocket` |
| `peluche` | `plush` |
| `objet` | `object` |
| `vetement` | `garment` |
| `accessoire` | `accessory` |

## Tabs

| format 1 | format 2 |
|---|---|
| `haut` | `top` |
| `bas` | `bottom` |
| `chaussures` | `shoes` |
| `accessoires` | `accessories` |

## Slots

| format 1 | format 2 |
|---|---|
| `dessous` | `under` |
| `chaussettes` | `socks` |
| `haut` | `top` |
| `manches` | `sleeves` |
| `bas` | `bottom` |
| `jupe` | `skirt` |
| `robe` | `dress` |
| `chaussures` | `shoes` |
| `mains` | `hands` |
| `col` | `collar` |
| `gilet` | `waistcoat` |
| `dessus` | `outer` |
| `capuche` | `hood` |

## Limbs (in keys such as `右bras` → `右arm`)

| format 1 | format 2 |
|---|---|
| `avant-bras` | `forearm` |
| `bras` | `arm` |
| `cuisse` | `thigh` |
| `jambe` | `leg` |

## Check codes (also inside `reason` and `message` texts)

| format 1 | format 2 |
|---|---|
| `ample_bord_libre` | `loose_free_edge` |
| `ample_ecart_p50_mm` | `loose_gap_p50_mm` |
| `ample_normales_coupees_sur_coutures` | `loose_normals_split_on_seams` |
| `budget_part_modele` | `model_share_budget` |
| `contour_sommet_p1` | `outline_vertex_p1` |
| `contour_taille` | `waist_outline` |
| `double_face` | `double_sided` |
| `influences_max_4` | `max_4_influences` |
| `moulant_bord_soude` | `tight_welded_edge` |
| `moulant_plis_p90_mm` | `tight_folds_p90_mm` |
| `moulant_rien_dessous` | `tight_nothing_under` |
| `moulant_sans_os_propres` | `tight_without_own_bones` |
| `peau_cachee` | `hidden_skin` |
| `phys_heurte_jambes` | `phys_hits_legs` |
| `phys_ignore_son_groupe` | `phys_ignores_own_group` |
| `phys_rebond_nul` | `phys_zero_bounce` |
| `phys_ressorts_nuls` | `phys_zero_springs` |
| `phys_torsion_bloquee` | `phys_twist_locked` |
| `pied_dans_chaussure` | `foot_in_shoe` |
| `poses_corps_seul` | `body_only_poses` |
| `sans_sdef` | `no_sdef` |
| `sous_la_peau` | `under_skin` |
| `texels_sur_peau` | `texels_on_skin` |
| `traversee_repos` | `rest_crossing` |
| `traversee_repos_p90_mm` | `rest_crossing_p90_mm` |
| `cheveux_hors_peau` | `hair_off_skin` |
| `couture` | `seam` |
| `oeil_dans_la_peau` | `eye_in_skin` |
| `qualite` | `quality` |
| `retournements_morphes` | `morph_flips` |
| `crane_sous_cheveux` | `skull_under_hair` |

## Word values (steps, modes, kinds…)

| format 1 | format 2 |
|---|---|
| `attente` | `waiting` |
| `coiffure` | `hairstyle` |
| `greffe` | `graft` |
| `pieces` | `parts` |
| `couleur` | `color` |
| `controles` | `checks` |
| `squelette` | `skeleton` |
| `rendu` | `render` |
| `fin` | `done` |
| `erreur` | `error` |
| `annulee` | `cancelled` |
| `piece` | `part` |
| `tete` | `head` |
| `fabriquer` | `build` |
| `catalogue` | `catalog` |
| `visages` | `faces` |
| `coiffures` | `hairstyles` |
| `vetements` | `clothes` |
| `visage` | `face` |
| `cible` | `target` |
| `tous` | `all` |
| `reperes` | `landmarks` |
| `bandes` | `bands` |
| `haut_ample` | `loose_top` |

## JSON keys

| format 1 | format 2 |
|---|---|
| `cadre` | `frame` |
| `axe` | `axis` |
| `autres_echecs` | `other_failures` |
| `erreur` | `error` |
| `entree` | `input` |
| `etat` | `state` |
| `jeu_actif` | `active_game` |
| `sortie` | `output` |
| `squelette_sortie` | `output_skeleton` |
| `squelettes` | `skeletons` |
| `cheveux` | `hair` |
| `doubles` | `duplicates` |
| `en_cours` | `in_progress` |
| `ordre` | `order` |
| `decalages` | `offsets` |
| `masques` | `masks` |
| `retirees` | `removed` |
| `accessoire_reste` | `accessory_left` |
| `accessoires` | `accessories` |
| `accessoires_par_sorte` | `accessories_by_sort` |
| `alertes` | `warnings` |
| `allongement` | `elongation` |
| `anneau` | `loop` |
| `annulation` | `cancellation` |
| `annulee` | `cancelled` |
| `applique` | `applied` |
| `arete_A` | `edge_A` |
| `aretes_bord` | `border_edges` |
| `aretes_bord_attendues` | `expected_border_edges` |
| `aretes_non_varietes` | `non_manifold_edges` |
| `aretes_orientation_contraire` | `flipped_edges` |
| `aretes_ouvertes_ajoutees` | `open_edges_added` |
| `atomes` | `atoms` |
| `auteur` | `author` |
| `avance` | `progress` |
| `avant_bras` | `forearm` |
| `bande` | `band` |
| `bande_iod` | `band_iod` |
| `base_passage_1` | `base_pass_1` |
| `bassin` | `pelvis` |
| `biblio` | `library` |
| `bilan` | `assessment` |
| `bloc` | `block` |
| `bloquant` | `blocking` |
| `bord_patch` | `patch_border` |
| `bord_trou` | `hole_border` |
| `bouche` | `mouth` |
| `bouche_bas` | `mouth_bottom` |
| `bouche_d` | `mouth_r` |
| `bouche_g` | `mouth_l` |
| `bouche_haut` | `mouth_top` |
| `boucles` | `loops` |
| `boucles_bord` | `border_loops` |
| `bras` | `arm` |
| `buste` | `bust` |
| `calques` | `texture_layers` |
| `calques_sans_jumelle` | `texture_layers_without_twin` |
| `catalogue` | `catalog` |
| `centre_yeux` | `eyes_center` |
| `chausse` | `on_foot` |
| `chemin` | `path` |
| `cheville` | `ankle` |
| `cible` | `target` |
| `classe` | `class` |
| `clignement` | `blink` |
| `coiffure` | `hairstyle` |
| `coiffures` | `hairstyles` |
| `controle` | `check` |
| `controles` | `checks` |
| `copie` | `copy` |
| `corps` | `body` |
| `corps_cible_mm` | `target_body_mm` |
| `corps_source_connu` | `source_body_known` |
| `corps_source_mm` | `source_body_mm` |
| `cou` | `neck` |
| `couche` | `layer` |
| `couches` | `layers` |
| `couches_internes` | `inner_layers` |
| `couches_internes_admises` | `allowed_inner_layers` |
| `couleur_peau_lab` | `skin_color_lab` |
| `cousu_a_la_peau_mm` | `sewn_to_skin_mm` |
| `couture` | `seam` |
| `couture_bande_retournes` | `seam_band_flipped` |
| `couture_bande_triangles` | `seam_band_triangles` |
| `couture_dE_apres` | `seam_dE_after` |
| `couture_dE_avant` | `seam_dE_before` |
| `couture_delta` | `seam_delta` |
| `couture_ecart_max` | `seam_gap_max` |
| `couture_ecart_median` | `seam_gap_median` |
| `couture_encoches_retirees` | `seam_notches_removed` |
| `couvercles` | `lids` |
| `couverture` | `coverage` |
| `couverture_cheveux` | `hair_coverage` |
| `crane` | `skull` |
| `cuisse` | `thigh` |
| `de` | `from` |
| `dE_apres` | `dE_after` |
| `dE_avant` | `dE_before` |
| `debut` | `start` |
| `dedans_oreilles` | `inside_ears` |
| `dehors` | `outside` |
| `dehors_avant` | `outside_before` |
| `deplacement_couture_max` | `seam_shift_max` |
| `deplacement_max` | `shift_max` |
| `deplacement_mm` | `shift_mm` |
| `dessous` | `under` |
| `dessus` | `over` |
| `dessus_du_pied` | `instep` |
| `deux_vetements` | `two_garments` |
| `diedre_max_deg` | `dihedral_max_deg` |
| `diedre_moy_deg` | `dihedral_mean_deg` |
| `diedre_q99_deg` | `dihedral_q99_deg` |
| `distance_a_la_moyenne_iod` | `distance_to_mean_iod` |
| `distance_moyenne_iod` | `mean_distance_iod` |
| `dist_moy` | `dist_mean` |
| `dossier` | `folder` |
| `droite` | `right` |
| `duree` | `duration` |
| `duree_s` | `duration_s` |
| `ecart_aux_autres_iod` | `gap_to_others_iod` |
| `ecart_cou_mm` | `neck_gap_mm` |
| `ecartes` | `discarded` |
| `ecart_greffe_mm` | `graft_gap_mm` |
| `ecart_relatif_max` | `relative_gap_max` |
| `ecart_relatif_moyen` | `relative_gap_mean` |
| `ecart_yeux` | `eye_gap` |
| `echecs` | `failures` |
| `echelle` | `scale` |
| `echelle_globale` | `global_scale` |
| `echelle_mediane` | `median_scale` |
| `echelle_source` | `source_scale` |
| `elargi` | `widened` |
| `emplacement` | `slot` |
| `empreinte` | `fingerprint` |
| `empreintes` | `fingerprints` |
| `entrees` | `entries` |
| `essais` | `attempts` |
| `etape` | `step` |
| `etapes` | `steps` |
| `etendue_mm` | `extent_mm` |
| `etiquette` | `label` |
| `extrait` | `extracted` |
| `faces_bande` | `band_faces` |
| `faces_cachees` | `hidden_faces` |
| `faces_calque` | `layer_faces` |
| `faces_coiffure` | `hairstyle_faces` |
| `faces_corps` | `body_faces` |
| `faces_en_2` | `faces_in_2` |
| `faces_en_4` | `faces_in_4` |
| `faces_en_double` | `duplicate_faces` |
| `faces_hors_corps` | `faces_off_body` |
| `faces_marquees` | `marked_faces` |
| `faces_oreilles` | `ear_faces` |
| `faces_peau` | `skin_faces` |
| `faces_peau_retirees` | `skin_faces_removed` |
| `faces_perdues` | `lost_faces` |
| `faces_pieces` | `part_faces` |
| `faces_tenue` | `outfit_faces` |
| `faces_tete` | `head_faces` |
| `faces_zone` | `zone_faces` |
| `fiabilite` | `reliability` |
| `fichier` | `file` |
| `fini` | `done` |
| `finis` | `finished` |
| `forme_moyenne` | `mean_shape` |
| `front` | `forehead` |
| `front_haut` | `forehead_top` |
| `gaine` | `sheath` |
| `garnie` | `trimmed` |
| `garnitures` | `trims` |
| `gauche` | `left` |
| `genre` | `kind` |
| `harmonie_couture` | `seam_harmony` |
| `hauteur` | `height` |
| `hauteurs_du_tronc` | `torso_heights` |
| `incrustations_admises` | `allowed_inlays` |
| `jambe` | `leg` |
| `jeu` | `game` |
| `largeur` | `width` |
| `lever` | `lift` |
| `libre` | `free` |
| `licence` | `license` |
| `lissage` | `smoothing` |
| `lot` | `batch` |
| `lots` | `batches` |
| `lueur` | `glow` |
| `main` | `hand` |
| `marge_iod` | `margin_iod` |
| `marge_mm` | `margin_mm` |
| `materiaux` | `materials` |
| `materiaux_recolores` | `recolored_materials` |
| `menton` | `chin` |
| `mesure_sans_objet` | `measure_not_applicable` |
| `mesures` | `measures` |
| `mesures_echecs` | `measure_failures` |
| `meta_vrm` | `vrm_meta` |
| `modele` | `model` |
| `modeles` | `models` |
| `morphes` | `morphs` |
| `morphes_absents` | `missing_morphs` |
| `morphes_de_A_ajoutes` | `morphs_from_A_added` |
| `morphes_harmonises` | `harmonized_morphs` |
| `morphes_resultat` | `result_morphs` |
| `nez` | `nose` |
| `nom` | `name` |
| `nombres` | `counts` |
| `noms` | `names` |
| `non_fournies` | `not_provided` |
| `non_mesures` | `not_measured` |
| `n_paires` | `n_pairs` |
| `n_sommets` | `n_vertices` |
| `occupe` | `occupies` |
| `oeil_dans_la_peau` | `eye_in_skin` |
| `oeil_d_bas` | `eye_r_bottom` |
| `oeil_d_ext` | `eye_r_outer` |
| `oeil_d_haut` | `eye_r_top` |
| `oeil_d_int` | `eye_r_inner` |
| `oeil_g_bas` | `eye_l_bottom` |
| `oeil_g_ext` | `eye_l_outer` |
| `oeil_g_haut` | `eye_l_top` |
| `oeil_g_int` | `eye_l_inner` |
| `ombre` | `shade` |
| `onglet` | `tab` |
| `oreille_d` | `ear_r` |
| `oreille_g` | `ear_l` |
| `oreilles` | `ears` |
| `oreilles_liees_faces` | `ears_linked_faces` |
| `orteils` | `toes` |
| `os` | `bones` |
| `os_ajoutes` | `added_bones` |
| `os_attache` | `attach_bones` |
| `os_coiffure` | `hairstyle_bones` |
| `os_humanoides` | `humanoid_bones` |
| `os_propres` | `own_bones` |
| `os_recales` | `registered_bones` |
| `outil` | `tool` |
| `paires` | `pairs` |
| `parametres` | `parameters` |
| `par_membre` | `per_limb` |
| `par_personnage_tete_pivot` | `per_person_pivot_head` |
| `part` | `fraction` |
| `part_aretes_sup15` | `fraction_edges_over15` |
| `part_dedans` | `fraction_inside` |
| `parties` | `regions` |
| `part_peau` | `fraction_skin` |
| `part_transportee` | `fraction_transported` |
| `part_valence_diff6` | `fraction_valence_not6` |
| `pas` | `stage` |
| `passe` | `passed` |
| `peau` | `skin` |
| `peau_A_lab` | `skin_A_lab` |
| `peau_B_lab` | `skin_B_lab` |
| `peau_cachee` | `hidden_skin` |
| `peau_d_origine` | `original_skin` |
| `peau_fermee` | `closed_skin` |
| `peau_opaque` | `opaque_skin` |
| `peau_source_connue` | `source_skin_known` |
| `perso` | `character` |
| `personnage` | `person` |
| `personnages` | `people` |
| `persos` | `characters` |
| `physique` | `physics` |
| `piece` | `part_name` |
| `piece_courante` | `current_part` |
| `pieces` | `parts` |
| `pieces_total` | `parts_total` |
| `pied` | `foot` |
| `pied_en_bloc` | `foot_as_block` |
| `pied_refuse` | `foot_refused` |
| `pire_mm` | `worst_mm` |
| `pixels_touches` | `touched_pixels` |
| `planche` | `sheet` |
| `planche_note` | `sheet_note` |
| `poids_corriges_par_la_source` | `weights_fixed_by_source` |
| `poids_repris_du_corps` | `weights_from_body` |
| `pointure` | `shoe_size` |
| `portage` | `fitting` |
| `portes_visage` | `worn_on_face` |
| `pose_bras_deg` | `arm_pose_deg` |
| `pose_rigide` | `rigid_pose` |
| `pourquoi` | `why` |
| `premier_doublon` | `first_duplicate` |
| `premier_passage` | `first_pass` |
| `preparees` | `prepared` |
| `prof_max_mm` | `depth_max_mm` |
| `prof_p50_mm` | `depth_p50_mm` |
| `proportions_visage_A` | `face_proportions_A` |
| `propre` | `own` |
| `qualite` | `quality` |
| `quoi` | `what` |
| `raccord` | `join` |
| `raccords` | `joins` |
| `raccord_visage` | `face_join` |
| `raffinage_B` | `refinement_B` |
| `raison` | `reason` |
| `rang` | `rank` |
| `rapport` | `report` |
| `rapport_a` | `ratio_to` |
| `rayon_cou_iod` | `neck_radius_iod` |
| `recalage_distance_iod` | `registration_distance_iod` |
| `recentre` | `recentred` |
| `recouvre` | `covers` |
| `refaire` | `redo` |
| `refus` | `refusals` |
| `refusees` | `refused` |
| `regard` | `gaze` |
| `releve` | `reading` |
| `reperes` | `landmarks` |
| `reste_dedans` | `stays_inside` |
| `resultat` | `result` |
| `resultat_a_iod2` | `result_a_iod2` |
| `resultat_cligne` | `result_blink` |
| `resultat_repos` | `result_rest` |
| `resultat_repos_iod2` | `result_rest_iod2` |
| `resume` | `summary` |
| `retenue` | `kept` |
| `retenues` | `kept_count` |
| `retenus` | `kept_points` |
| `retournements_morphes` | `morph_flips` |
| `retournes_couture` | `seam_flipped` |
| `retournes_resultat` | `result_flipped` |
| `retournes_sources` | `sources_flipped` |
| `rigides` | `rigid_bodies` |
| `rigides_coiffure` | `hairstyle_rigid_bodies` |
| `rigides_physiques` | `physics_rigid_bodies` |
| `rigides_sommets` | `vertex_rigid_bodies` |
| `semelle` | `sole` |
| `seuil` | `threshold` |
| `seuil_q99` | `threshold_q99` |
| `sommets` | `vertices` |
| `sommets_ajoutes` | `added_vertices` |
| `sommets_bouges` | `moved_vertices` |
| `sommets_cou` | `neck_vertices` |
| `sommets_couture` | `seam_vertices` |
| `sommets_gardes` | `kept_vertices` |
| `sommets_libres` | `free_vertices` |
| `sommets_oeil` | `eye_vertices` |
| `sommets_peau` | `skin_vertices` |
| `sommets_symetrises` | `mirrored_vertices` |
| `sorte` | `sort` |
| `sorte_pourquoi` | `sort_reason` |
| `sortis` | `left_out` |
| `source_cligne` | `source_blink` |
| `source_repos` | `source_rest` |
| `source_repos_iod2` | `source_rest_iod2` |
| `sourcils` | `eyebrows` |
| `sous_la_peau` | `under_skin` |
| `squelette` | `skeleton` |
| `suivi_icp` | `icp_track` |
| `sur` | `on` |
| `taille_cm` | `height_cm` |
| `teinte` | `tint` |
| `tete` | `head` |
| `tete_base` | `base_head` |
| `tete_pivot` | `pivot_head` |
| `tetes` | `heads` |
| `texels_loin` | `far_texels` |
| `titre` | `title` |
| `topologie` | `topology` |
| `torse` | `torso` |
| `traverse` | `crosses` |
| `trou_faces` | `hole_faces` |
| `type_fr` | `type_name` |
| `type_mesure` | `measure_type` |
| `unites` | `units` |
| `valeur` | `value` |
| `ventre` | `belly` |
| `vetements` | `clothes` |
| `vetements_par_type` | `clothes_by_type` |
| `vignette` | `thumbnail` |
| `visage` | `face` |
| `visages` | `face_count` |
| `visage_w>0.5` | `face_w>0.5` |
| `v_peau` | `v_skin` |
| `v_portage` | `v_fitting` |
| `taille` | `size` |
| `taille_extraite` | `extracted_size` |
| `taille_installee` | `installed_size` |
| `partages` | `shared` |
| `liens` | `links` |
| `lien` | `link` |
| `vers` | `to` |
| `tetes_bin` | `heads_bin` |

## Server ↔ page

Every JSON body and answer uses the keys and words above. Routes:

| format 1 | format 2 |
|---|---|
| `/api/catalogue` | `/api/catalog` |
| `/api/modeles` | `/api/models` |
| `/api/travaux/<id>` | `/api/jobs/<id>` |
| `/api/assembler` | `/api/assemble` |
| `/api/assemblages` | `/api/assemblies` |
| `/api/extraire` | `/api/extract` |
| `/api/extraction/annuler` | `/api/extraction/cancel` |
| `/api/ouvrir` | `/api/open` |
| `/api/importer?nom=` | `/api/import?name=` |
| `/api/vignette/(tetes|coiffures|pieces)/<id>` | `/api/thumbnail/(heads|hairstyles|parts)/<id>` |
| `/data/tetes/…, /data/coiffures/…, /data/pieces/…` | `/data/heads/…, /data/hairstyles/…, /data/parts/…` |
| `—` | `/api/dataformat (new: the tables, for the page)` |

## Options of the internal scripts

| format 1 | format 2 |
|---|---|
| `--biblio` | `--library` |
| `--cadre` | `--frame` |
| `--coiffure` | `--hairstyle` |
| `--coiffures` | `--hairstyles` |
| `--corps` | `--body` |
| `--couture-mm` | `--seam-mm` |
| `--delai` | `--timeout` |
| `--ecart-cou-max` | `--max-neck-gap` |
| `--ecart-min` | `--min-gap` |
| `--echantillons` | `--samples` |
| `--forme` | `--shape` |
| `--garder-crane` | `--keep-skull` |
| `--graine` | `--seed` |
| `--mats-cheveux` | `--hair-mats` |
| `--modele` | `--model` |
| `--nom` | `--name` |
| `--noter` | `--note` |
| `--peau` | `--skin` |
| `--peau-tete` | `--head-skin` |
| `--piece` | `--part` |
| `--pieces` | `--parts` |
| `--planche` | `--sheet` |
| `--prendre` | `--take` |
| `--raccord` | `--seam` |
| `--recalage` | `--realign` |
| `--recette` | `--recipe` |
| `--ressorts` | `--springs` |
| `--retirer` | `--remove` |
| `--saines` | `--healthy` |
| `--sans-garde` | `--no-guard` |
| `--sans-teinte` | `--no-tint` |
| `--seuls` | `--only` |
| `--sortie` | `--output` |
| `--suivre-os` | `--follow-bones` |
| `--tableau` | `--table` |
| `--taille` | `--size` |
| `--teinte-base` | `--base-tint` |
| `--tete` | `--head` |
| `--textures-pour` | `--textures-for` |
| `--tour` | `--turn` |
| `--tous` | `--all` |
| `--tout` | `--all` |
| `--valeurs` | `--values` |
| `--visage` | `--face` |
| `--vues` | `--views` |
| `--yeux` | `--eyes` |

## Ready-made package

| format 1 | format 2 |
|---|---|
| `{mmd_extrait}` in paths | `{mmd_extracted}` |
| roots `visage`, `vetements`, `mmd_extrait` | `face`, `clothes`, `mmd_extracted` |
| archives `tetes-*`, `coiffures-*`, `pieces-*` | `heads-*`, `hairstyles-*`, `parts-*` |
| `"*tetes"` in duplicates | `"*heads"` |
| `.chimera-bibliotheque/` | `.chimera-library/` |
| `index.json` keys | the JSON keys above (`format`: 2) |
