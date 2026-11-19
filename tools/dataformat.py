"""Chimera's data format: the English names of format 2 and the table from format 1 (French names).

Format 2 (19/11/2026): every folder, file, JSON key and word value of the data is English, and the code
uses the same names. This module only serves the conversion of a format-1 folder (tools/library/migrate.py:
python3 -I tools/library/library.py migrate <data folder>) and the refusal of a format-1 folder by the tools.
"""
import json
import os
import re

FORMAT = 2
REVISION = 1                          # format 2, revision 1: names inside the files (see REVISED below)
MARKER = "chimera-data.json"          # in the data folder: {"format": 2}
MIGRATE = "python3 -I tools/library/library.py migrate"

# ---------------------------------------------------------------- folders
DIRS = {
    # top of the data folder
    "atelier-base-vroid": "workshop-vroid-base",
    "atelier-base-libre": "workshop-free-base",
    "maison": "own-rig",
    "base-vroid": "vroid-base",
    "base-libre": "free-base",
    "bibliotheque-vetements": "clothes-library",
    "bibliotheque-visage": "face-library",
    "tete-pivot": "pivot-head",
    "mmd-extrait": "mmd-extracted",
    ".chimera-bibliotheque": ".chimera-library",
    "squelettes": "skeletons", "cheveux": "hair", "doubles": "duplicates", "en_cours": "in_progress",
    # workshop cache
    "tetes": "heads",
    "coiffures": "hairstyles",
    "pieces": "parts",
    "biblio": "library",
    "corps": "body",
    "assemblages": "assemblies",
    "imports-en-cours": "imports-in-progress",
    # inside
    "modele": "model",
    "teinte": "tint",
    "rendus": "renders",
    "1-coiffure": "1-hairstyle",
    "2-greffe": "2-graft",
    "3-pieces": "3-parts",
    "vetements": "clothes",
    "accessoires": "accessories",
}

# ---------------------------------------------------------------- files
FILES = {
    "corps.pmx": "body.pmx", "corps.png": "body.png", "corps.glb": "body.glb", "corps.bin": "body.bin",
    "corps_pos.bin": "body_pos.bin",
    "tete.pmx": "head.pmx", "tete.png": "head.png", "tete.glb": "head.glb",
    "coiffure.pmx": "hairstyle.pmx", "coiffure.glb": "hairstyle.glb", "coiffure.json": "hairstyle.json",
    "coiffure.npz": "hairstyle.npz",
    "peau.png": "skin.png", "yeux.png": "eyes.png",
    "resultat.pmx": "result.pmx", "resultat.vrm": "result.vrm",
    "piece.glb": "part.glb", "pied.bin": "foot.bin",
    "vignette.png": "thumbnail.png", "planche.png": "sheet.png", "trois_quarts.png": "three_quarters.png",
    "dos.png": "back.png",
    "peau_cachee.bin": "hidden_skin.bin", "couvre.bin": "covers.bin",
    "recalage.npz": "registration.npz", "partie.npz": "part.npz", "formes.npz": "shapes.npz",
    "fiche.json": "record.json", "catalogue.json": "catalog.json", "etiquettes.json": "labels.json",
    "mesures.json": "measures.json", "teinte.json": "tint.json", "portage.json": "fitting.json",
    "controles.json": "checks.json", "controle.json": "check.json", "etat.json": "state.json",
    "journal.txt": "log.txt", "greffe.json": "graft.json", "decoupe.json": "cutout.json",
    "tete-pivot.json": "pivot-head.json", "bibliotheque.json": "library.json", "demande.json": "request.json",
    "ombre-visage.png": "face-shade.png", "ombre-visage.json": "face-shade.json", "cartes.bin": "maps.bin",
    "cartes-corps.bin": "body-maps.bin", "erreur.json": "error.json", "raccord.json": "seam.json",
    "poids.npz": "weights.npz", "forme.pkl": "shape.pkl", "forme-makehuman.pkl": "shape-makehuman.pkl", "squelette.json": "skeleton.json", "cheveux.png": "hair.png",
    "recette.json": "recipe.json", "poids_texture.npz": "texture_weights.npz", "visage_compose.png": "composed_face.png",
    "visage.pmx": "face.pmx", "tete-moyenne.pmx": "average-head.pmx", "piece_tete.pmx": "head_part.pmx",
    "peau_corps.png": "body_skin.png", "reste_donneur.png": "donor_rest.png",
    "piece.avant-reprise.glb": "part.before-repair.glb", "peau_cachee.avant.bin": "hidden_skin.before.bin",
}
# names made of a word: 1_peau.png (tint layers), reflet-<hash>.png (highlights), atelier-suite-<n>.json (batches)
PATTERNS = [
    (re.compile(r"^(\d+)_peau\.png$"), r"\1_skin.png"),
    (re.compile(r"^reflet-([0-9a-f]+\.png)$"), r"highlight-\1"),
    (re.compile(r"^atelier-suite-(.+\.json)$"), r"workshop-batch-\1"),
    (re.compile(r"^resultat-(.+\.pmx)$"), r"result-\1"),          # assemblies: the PMX on the source bones
]

# ---------------------------------------------------------------- vocabulary (word values, type folders)
TYPES = {
    # garments
    "culotte": "underwear", "chaussettes": "socks", "haut": "top", "manches": "sleeves", "short": "shorts",
    "pantalon": "trousers", "jupe": "skirt", "robe": "dress", "combinaison": "jumpsuit", "chaussures": "shoes",
    "bottes": "boots", "gants": "gloves", "col": "collar", "gilet": "waistcoat", "veste": "jacket",
    "manteau": "coat", "capuche": "hood",
    # accessories
    "ceinture": "belt", "harnais": "harness", "jarretiere": "garter", "chevillere": "anklet", "brassard": "armband",
    "bracelet": "bracelet", "bague": "ring", "cravate": "tie", "echarpe": "scarf", "collier": "necklace",
    "bijou": "jewel", "sac": "bag", "lunettes": "glasses", "masque": "mask", "couvre_chef": "headwear",
    "casque_audio": "headphones", "queue": "tail", "ailes": "wings", "oreilles": "ears",
    # other pieces of a cut-out model
    "autre": "other", "cheveux": "hair", "ornement": "ornament", "sangle": "strap", "boucle_bouton": "fastener",
    "ecusson": "crest", "poche": "pocket", "peluche": "plush", "objet": "object", "badge": "badge",
    "vetement": "garment", "accessoire": "accessory",
}
TABS = {"haut": "top", "bas": "bottom", "chaussures": "shoes", "accessoires": "accessories"}
SLOTS = {"dessous": "under", "chaussettes": "socks", "haut": "top", "manches": "sleeves", "bas": "bottom",
         "jupe": "skirt", "robe": "dress", "chaussures": "shoes", "mains": "hands", "col": "collar",
         "gilet": "waistcoat", "dessus": "outer", "capuche": "hood"}
TRIMS = ("+garnitures", "+trims")      # <type>+garnitures.pmx: the piece with its trims
# words of other values (steps of an assembly or an import, kinds, what is being prepared)
WORDS = {
    "attente": "waiting", "coiffure": "hairstyle", "greffe": "graft", "pieces": "parts", "couleur": "color",
    "controles": "checks", "squelette": "skeleton", "rendu": "render", "fin": "done", "erreur": "error",
    "annulee": "cancelled", "piece": "part", "tete": "head", "base": "base", "fabriquer": "build",
    "catalogue": "catalog", "visages": "faces", "coiffures": "hairstyles", "vetements": "clothes",
    "visage": "face", "cible": "target", "tous": "all", "fractions": "fractions", "reperes": "landmarks",
    "bandes": "bands", "haut_ample": "loose_top",
}
# codes of checks and failures (controles[].nom, echecs, autres_echecs, mesures_echecs, non_mesures)
CHECKS = {
    "ample_bord_libre": "loose_free_edge", "ample_ecart_p50_mm": "loose_gap_p50_mm",
    "ample_normales_coupees_sur_coutures": "loose_normals_split_on_seams",
    "budget_part_modele": "model_share_budget", "contour_sommet_p1": "outline_vertex_p1",
    "contour_taille": "outline_size", "double_face": "double_sided", "influences_max_4": "max_4_influences",
    "moulant_bord_soude": "tight_welded_edge", "moulant_plis_p90_mm": "tight_folds_p90_mm",
    "moulant_rien_dessous": "tight_nothing_under", "moulant_sans_os_propres": "tight_without_own_bones",
    "peau_cachee": "hidden_skin", "phys_heurte_jambes": "phys_hits_legs",
    "phys_ignore_son_groupe": "phys_ignores_own_group", "phys_rebond_nul": "phys_zero_bounce",
    "phys_ressorts_nuls": "phys_zero_springs", "phys_torsion_bloquee": "phys_twist_locked",
    "pied_dans_chaussure": "foot_in_shoe", "poses_corps_seul": "body_only_poses", "sans_sdef": "no_sdef",
    "sous_la_peau": "under_skin", "texels_sur_peau": "texels_on_skin", "traversee_repos": "rest_crossing",
    "traversee_repos_p90_mm": "rest_crossing_p90_mm", "cheveux_hors_peau": "hair_off_skin",
    "couture": "seam", "oeil_dans_la_peau": "eye_in_skin", "qualite": "quality",
    "retournements_morphes": "morph_flips", "crane_sous_cheveux": "skull_under_hair",
}

# ---------------------------------------------------------------- JSON keys
KEYS = {
    "cadre": "frame", "axe": "axis", "autres_echecs": "other_failures", "erreur": "error", "entree": "input",
    "etat": "state", "jeu_actif": "active_game", "sortie": "output", "squelette_sortie": "output_skeleton",
    "squelettes": "skeletons", "cheveux": "hair", "doubles": "duplicates", "en_cours": "in_progress", "ordre": "order", "decalages": "offsets", "masques": "masks", "retirees": "removed",
    "accessoire_reste": "accessory_left", "accessoires": "accessories", "accessoires_par_sorte": "accessories_by_sort",
    "alertes": "warnings", "allongement": "elongation", "anneau": "loop", "annulation": "cancellation",
    "annulee": "cancelled", "applique": "applied", "arete_A": "edge_A", "aretes_bord": "border_edges",
    "aretes_bord_attendues": "expected_border_edges", "aretes_non_varietes": "non_manifold_edges",
    "aretes_orientation_contraire": "flipped_edges", "aretes_ouvertes_ajoutees": "open_edges_added",
    "atomes": "atoms", "auteur": "author", "avance": "progress", "avant_bras": "forearm", "bande": "band",
    "bande_iod": "band_iod", "base_passage_1": "base_pass_1", "bassin": "pelvis", "biblio": "library",
    "bilan": "assessment", "bloc": "block", "bloquant": "blocking", "bord_patch": "patch_border",
    "bord_trou": "hole_border", "bouche": "mouth", "bouche_bas": "mouth_bottom", "bouche_d": "mouth_r",
    "bouche_g": "mouth_l", "bouche_haut": "mouth_top", "boucles": "loops", "boucles_bord": "border_loops",
    "bras": "arm", "buste": "bust", "calques": "texture_layers", "calques_sans_jumelle": "texture_layers_without_twin",
    "catalogue": "catalog", "centre_yeux": "eyes_center", "chausse": "on_foot", "chemin": "path",
    "cheville": "ankle", "cible": "target", "classe": "class", "clignement": "blink", "coiffure": "hairstyle",
    "coiffures": "hairstyles", "controle": "check", "controles": "checks", "copie": "copy", "corps": "body",
    "corps_cible_mm": "target_body_mm", "corps_source_connu": "source_body_known", "corps_source_mm": "source_body_mm",
    "cou": "neck", "couche": "layer", "couches": "layers", "couches_internes": "inner_layers",
    "couches_internes_admises": "allowed_inner_layers", "couleur_peau_lab": "skin_color_lab",
    "cousu_a_la_peau_mm": "sewn_to_skin_mm", "couture": "seam", "couture_bande_retournes": "seam_band_flipped",
    "couture_bande_triangles": "seam_band_triangles", "couture_dE_apres": "seam_dE_after",
    "couture_dE_avant": "seam_dE_before", "couture_delta": "seam_delta", "couture_ecart_max": "seam_gap_max",
    "couture_ecart_median": "seam_gap_median", "couture_encoches_retirees": "seam_notches_removed",
    "couvercles": "lids", "couverture": "coverage", "couverture_cheveux": "hair_coverage", "crane": "skull",
    "cuisse": "thigh", "de": "from", "dE_apres": "dE_after", "dE_avant": "dE_before", "debut": "start",
    "dedans_oreilles": "inside_ears", "dehors": "outside", "dehors_avant": "outside_before",
    "deplacement_couture_max": "seam_shift_max", "deplacement_max": "shift_max", "deplacement_mm": "shift_mm",
    "dessous": "under", "dessus": "over", "dessus_du_pied": "instep", "deux_vetements": "two_garments",
    "diedre_max_deg": "dihedral_max_deg", "diedre_moy_deg": "dihedral_mean_deg", "diedre_q99_deg": "dihedral_q99_deg",
    "distance_a_la_moyenne_iod": "distance_to_mean_iod", "distance_moyenne_iod": "mean_distance_iod",
    "dist_moy": "dist_mean", "dossier": "folder", "droite": "right", "duree": "duration", "duree_s": "duration_s",
    "ecart_aux_autres_iod": "gap_to_others_iod", "ecart_cou_mm": "neck_gap_mm", "ecartes": "discarded",
    "ecart_greffe_mm": "graft_gap_mm", "ecart_relatif_max": "relative_gap_max",
    "ecart_relatif_moyen": "relative_gap_mean", "ecart_yeux": "eye_gap", "echecs": "failures", "echelle": "scale",
    "echelle_globale": "global_scale", "echelle_mediane": "median_scale", "echelle_source": "source_scale",
    "elargi": "widened", "emplacement": "slot", "empreinte": "fingerprint", "empreintes": "fingerprints",
    "entrees": "entries", "essais": "attempts", "etape": "step", "etapes": "steps", "etendue_mm": "extent_mm",
    "etiquette": "label", "extrait": "extracted", "faces_bande": "band_faces", "faces_cachees": "hidden_faces",
    "faces_calque": "layer_faces", "faces_coiffure": "hairstyle_faces", "faces_corps": "body_faces",
    "faces_en_2": "faces_in_2", "faces_en_4": "faces_in_4", "faces_en_double": "duplicate_faces",
    "faces_hors_corps": "faces_off_body", "faces_marquees": "marked_faces", "faces_oreilles": "ear_faces",
    "faces_peau": "skin_faces", "faces_peau_retirees": "skin_faces_removed", "faces_perdues": "lost_faces",
    "faces_pieces": "part_faces", "faces_tenue": "outfit_faces", "faces_tete": "head_faces",
    "faces_zone": "zone_faces", "fiabilite": "reliability", "fichier": "file", "fini": "done",
    "finis": "finished", "forme_moyenne": "mean_shape", "front": "forehead", "front_haut": "forehead_top",
    "gaine": "sheath", "garnie": "trimmed", "garnitures": "trims", "gauche": "left", "genre": "kind",
    "harmonie_couture": "seam_harmony", "hauteur": "height", "hauteurs_du_tronc": "torso_heights",
    "incrustations_admises": "allowed_inlays", "jambe": "leg", "jeu": "game", "largeur": "width",
    "lever": "lift", "libre": "free", "licence": "license", "lissage": "smoothing", "lot": "batch",
    "lots": "batches", "lueur": "glow", "main": "hand", "marge_iod": "margin_iod", "marge_mm": "margin_mm",
    "materiaux": "materials", "materiaux_recolores": "recolored_materials", "menton": "chin",
    "mesure_sans_objet": "measure_not_applicable", "mesures": "measures", "mesures_echecs": "measure_failures",
    "meta_vrm": "vrm_meta", "modele": "model", "modeles": "models", "morphes": "morphs",
    "morphes_absents": "missing_morphs", "morphes_de_A_ajoutes": "morphs_from_A_added",
    "morphes_harmonises": "harmonized_morphs", "morphes_resultat": "result_morphs", "nez": "nose",
    "nom": "name", "nombres": "counts", "noms": "names", "non_fournies": "not_provided",
    "non_mesures": "not_measured", "n_paires": "n_pairs", "n_sommets": "n_vertices", "occupe": "occupies",
    "oeil_dans_la_peau": "eye_in_skin", "oeil_d_bas": "eye_r_bottom", "oeil_d_ext": "eye_r_outer",
    "oeil_d_haut": "eye_r_top", "oeil_d_int": "eye_r_inner", "oeil_g_bas": "eye_l_bottom",
    "oeil_g_ext": "eye_l_outer", "oeil_g_haut": "eye_l_top", "oeil_g_int": "eye_l_inner", "ombre": "shade",
    "onglet": "tab", "oreille_d": "ear_r", "oreille_g": "ear_l", "oreilles": "ears",
    "oreilles_liees_faces": "ears_linked_faces", "orteils": "toes", "os": "bones", "os_ajoutes": "added_bones",
    "os_attache": "attach_bones", "os_coiffure": "hairstyle_bones", "os_humanoides": "humanoid_bones",
    "os_propres": "own_bones", "os_recales": "registered_bones", "outil": "tool", "paires": "pairs",
    "parametres": "parameters", "par_membre": "per_limb", "par_personnage_tete_pivot": "per_person_pivot_head",
    "part": "fraction", "part_aretes_sup15": "fraction_edges_over15", "part_dedans": "fraction_inside",
    "parties": "regions", "part_peau": "fraction_skin", "part_transportee": "fraction_transported",
    "part_valence_diff6": "fraction_valence_not6", "pas": "stage", "passe": "passed",
    "peau": "skin", "peau_A_lab": "skin_A_lab", "peau_B_lab": "skin_B_lab", "peau_cachee": "hidden_skin",
    "peau_d_origine": "original_skin", "peau_fermee": "closed_skin", "peau_opaque": "opaque_skin",
    "peau_source_connue": "source_skin_known", "perso": "character", "personnage": "person",
    "personnages": "people", "persos": "characters", "physique": "physics", "piece": "part_name",
    "piece_courante": "current_part", "pieces": "parts", "pieces_total": "parts_total", "pied": "foot",
    "pied_en_bloc": "foot_as_block", "pied_refuse": "foot_refused", "pire_mm": "worst_mm",
    "pixels_touches": "touched_pixels", "planche": "sheet", "planche_note": "sheet_note",
    "poids_corriges_par_la_source": "weights_fixed_by_source", "poids_repris_du_corps": "weights_from_body",
    "pointure": "shoe_size", "portage": "fitting", "portes_visage": "worn_on_face", "pose_bras_deg": "arm_pose_deg",
    "pose_rigide": "rigid_pose", "pourquoi": "why", "premier_doublon": "first_duplicate",
    "premier_passage": "first_pass", "preparees": "prepared", "prof_max_mm": "depth_max_mm",
    "prof_p50_mm": "depth_p50_mm", "proportions_visage_A": "face_proportions_A", "propre": "own",
    "qualite": "quality", "quoi": "what", "raccord": "join", "raccords": "joins", "raccord_visage": "face_join",
    "raffinage_B": "refinement_B", "raison": "reason", "rang": "rank", "rapport": "report",
    "rapport_a": "ratio_to", "rayon_cou_iod": "neck_radius_iod", "recalage_distance_iod": "registration_distance_iod",
    "recentre": "recentred", "recouvre": "covers", "refaire": "redo", "refus": "refusals", "refusees": "refused",
    "regard": "gaze", "releve": "reading", "reperes": "landmarks", "reste_dedans": "stays_inside",
    "resultat": "result", "resultat_a_iod2": "result_a_iod2", "resultat_cligne": "result_blink",
    "resultat_repos": "result_rest", "resultat_repos_iod2": "result_rest_iod2", "resume": "summary",
    "retenue": "kept", "retenues": "kept_count", "retenus": "kept_points",
    "retournements_morphes": "morph_flips", "retournes_couture": "seam_flipped",
    "retournes_resultat": "result_flipped", "retournes_sources": "sources_flipped", "rigides": "rigid_bodies",
    "rigides_coiffure": "hairstyle_rigid_bodies", "rigides_physiques": "physics_rigid_bodies",
    "rigides_sommets": "vertex_rigid_bodies", "semelle": "sole", "seuil": "threshold",
    "seuil_q99": "threshold_q99", "sommets": "vertices", "sommets_ajoutes": "added_vertices",
    "sommets_bouges": "moved_vertices", "sommets_cou": "neck_vertices", "sommets_couture": "seam_vertices",
    "sommets_gardes": "kept_vertices", "sommets_libres": "free_vertices", "sommets_oeil": "eye_vertices",
    "sommets_peau": "skin_vertices", "sommets_symetrises": "mirrored_vertices", "sorte": "sort",
    "sorte_pourquoi": "sort_reason", "sortis": "left_out", "source_cligne": "source_blink",
    "source_repos": "source_rest", "source_repos_iod2": "source_rest_iod2", "sourcils": "eyebrows",
    "sous_la_peau": "under_skin", "squelette": "skeleton", "suivi_icp": "icp_track", "sur": "on",
    "taille_cm": "height_cm", "teinte": "tint", "tete": "head", "tete_base": "base_head",
    "tete_pivot": "pivot_head", "tetes": "heads", "texels_loin": "far_texels", "titre": "title",
    "topologie": "topology", "torse": "torso", "traverse": "crosses", "trou_faces": "hole_faces",
    "type_fr": "type_name", "type_mesure": "measure_type", "unites": "units", "valeur": "value",
    "ventre": "belly", "vetements": "clothes", "vetements_par_type": "clothes_by_type", "vignette": "thumbnail",
    "visage": "face", "visages": "face_count", "visage_w>0.5": "face_w>0.5", "v_peau": "v_skin",
    "v_portage": "v_fitting",
    # library package (index.json)
    "taille": "size", "taille_extraite": "extracted_size", "taille_installee": "installed_size",
    "partages": "shared", "liens": "links", "lien": "link", "vers": "to", "tetes_bin": "heads_bin",
}
LIMBS = {"avant-bras": "forearm", "bras": "arm", "cuisse": "thigh", "jambe": "leg"}   # 右bras, 左cuisse...
# subtree left whole: what the VRM file says about a model
OPAQUE = ("meta_vrm",)
# dicts whose keys are garment types (cutout.json/parts, index.json): vocabulary, not keys
TYPE_KEYED = ("pieces", "vetements_par_type", "accessoires_par_sorte", "decalages", "masques", "retirees")
# keys whose values are codes of checks, or words of WORDS
CHECK_LISTS = ("echecs", "autres_echecs", "mesures_echecs", "non_mesures")
WORD_VALUES = ("etape", "pas", "quoi", "mode", "proportions", "valeur", "hauteurs_du_tronc", "sorte_pourquoi",
               "type_mesure", "etapes")


KEYS_BACK = {v: k for k, v in KEYS.items()}
# keys of the games' descriptions (tools/games/<game>/game.json) that a setting games.<game>.* may override
DESCRIPTION_KEYS = {"atelier": "workshop",
    "compatibilite": "compatibility",
    "contenu": "content",
    "depuis_base": "since_base",
    "dossiers_avec": "folders_with",
    "ecart_yeux_typique": "typical_eye_gap",
    "exception_vetement": "exception_garment",
    "exclure_os": "exclude_os",
    "exclure_pieces": "exclude_parts",
    "exclus": "excluded",
    "fabriquees": "built",
    "fichiers_de": "files_of",
    "garder_pieces": "keep_parts",
    "limites": "limits",
    "mmd_extrait": "mmd_extracted",
    "morphes_mmd": "morphs_mmd",
    "niveau": "level",
    "panneaux": "panels",
    "petit": "small",
    "prise_en_charge": "support",
    "provenance": "origin",
    "tenues_couches": "outfits_layers",
    "uv_boites": "uv_boxes",
    "uv_clarte_min": "uv_clarity_min",
    "uv_exclure": "uv_exclude",
    "verifier": "check"}
# a slot is a word of SLOTS or, for an exclusive accessory or a dress taking a skirt's place, a type
SLOT_WORDS = {**TYPES, **SLOTS}
_KEY_TO = dict(KEYS)


# ---------------------------------------------------------------- names
def type_name(word):
    """haut -> top, haut-2 -> top-2, haut+garnitures -> top+trims; anything else unchanged."""
    de, to = TRIMS
    m = re.fullmatch(r"(.+?)(-\d+)?(%s)?" % re.escape(de), word, re.S)
    if not m:
        return word
    base, n, trims = m.group(1), m.group(2) or "", m.group(3)
    if base not in TYPES:
        return word
    return TYPES[base] + n + (to if trims else "")


def part_id(ident):
    """Base_Female__jupe-2 -> Base_Female__skirt-2 (character__type of the workshop's parts)."""
    if "__" in ident:
        character, _, t = ident.rpartition("__")
        return character + "__" + type_name(t)
    return ident


def dir_name(name):
    if name in DIRS:
        return DIRS[name]
    if "__" in name:
        return part_id(name)
    return type_name(name)


def file_name(name):
    if name in FILES:
        return FILES[name]
    for pattern, to in PATTERNS:
        if pattern.match(name):
            return pattern.sub(to, name)
    stem, ext = os.path.splitext(name)
    if ext in (".pmx", ".json"):
        return type_name(stem) + ext
    return name


def path(p):
    """A path (absolute, relative, a /data/ URL, with / or \\) with the format-2 names. Nothing inside a
    model's sources (sources/<game>/<source>/<model>/...) is renamed but the source folder."""
    if not isinstance(p, str) or not p:
        return p
    sep = "\\" if ("\\" in p and "/" not in p) else "/"
    parts = p.split(sep)
    out = []
    in_sources = False
    for i, c in enumerate(parts):
        last = i == len(parts) - 1
        if c in ("", ".", ".."):
            out.append(c)
            continue
        if in_sources:
            out.append(DIRS.get(c, c) if c == "mmd-extrait" else c)
            continue
        if c == "sources":
            in_sources = True
        if last and "." in c and not c.startswith("."):
            out.append(file_name(c))
        else:
            out.append(dir_name(c))
    return sep.join(out)


# ---------------------------------------------------------------- records (migrate)
def _key(k, table, limbs):
    """A key of a format-1 record -> its format-2 name (右bras -> 右arm, passe_2 -> pass_2)."""
    if not isinstance(k, str):
        return k
    if k in table:
        return table[k]
    if k[:1] in ("右", "左") and k[1:] in limbs:
        return k[0] + limbs[k[1:]]
    m = re.fullmatch(r"passe_(\d+)", k)
    if m:
        return "pass_" + m.group(1)
    return k


# codes of checks quoted inside a sentence ("raison", "message"): only the codes with an underscore
_CODES = re.compile(r"\b(" + "|".join(sorted((k for k in CHECKS if "_" in k), key=len, reverse=True)) + r")\b")


def codes_in_text(text):
    return _CODES.sub(lambda m: CHECKS[m.group(1)], text)


# ---------------------------------------------------------------- format of a data folder
FORMAT1_SIGNS = ("bibliotheque-vetements", "bibliotheque-visage", "base-vroid", "base-libre", "atelier-base-vroid",
                 "atelier-base-libre", "tete-pivot", ".chimera-bibliotheque",
                 # a workshop cache or a base body kept outside the data folder (workshop.cache, workshop.body)
                 "tetes", "coiffures", "catalogue.json", "assemblages", "corps.pmx", "forme.pkl")


def format_of(data):
    """2 if the folder says so, 1 if it holds format-1 names, None if it is empty or unknown."""
    try:
        with open(os.path.join(data, MARKER), encoding="utf-8") as f:
            return int(json.load(f).get("format", 0)) or None
    except (FileNotFoundError, ValueError, AttributeError):
        pass
    if any(os.path.exists(os.path.join(data, n)) for n in FORMAT1_SIGNS):
        return 1
    return None


# ---------------------------------------------------------------- revision 1 (19/11/2026)
# Names the first conversion did not open: the arrays inside .npz files, the house skeleton (skeleton.json,
# weights.npz, poses.npz, shape.pkl) and one check code. A format-2 folder of revision 0 gets them once.
REVISED_CHECKS = {"waist_outline": "outline_size"}
REGIONS = {"yeux": "eyes", "sourcils": "eyebrows", "nez": "nose", "bouche": "mouth", "base": "base"}
NPZ = {
    "hairstyle.npz": {"coiffure": "hairstyle", "oreilles": "ears", "os": "bones"},
    "part.npz": {"echelle": "scale", "fiable": "reliable", "noms": "names",
                 **{"pieces_" + k: "parts_" + v for k, v in REGIONS.items()}},
    "texture_weights.npz": {"visage": "face", **{"poids_" + k: "weight_" + v for k, v in REGIONS.items()}},
    "registration.npz": {"trou": "hole", "p_cou": "p_neck"},
    "shapes.npz": {"formes": "shapes", "noms": "names", "poids": "weights"},
    "weights.npz": {"poids": "weights", "os": "bones"},
    "poses.npz": {"repos": "rest"},
}
NPZ_PREFIXES = {"poses.npz": {"maison_": "own_"}}
# house skeleton: bone names (racine, bras_torsion1_G...), keys and words
HOUSE_MULTI = {"avant_bras": "forearm", "petit_doigt": "little", "haut_poitrine": "upper_chest"}
HOUSE_WORDS = {"racine": "root", "bassin": "pelvis", "colonne": "spine", "poitrine": "chest", "cou": "neck",
               "tete": "head", "epaule": "shoulder", "bras": "arm", "main": "hand", "pouce": "thumb", "majeur": "middle",
               "annulaire": "ring", "cuisse": "thigh", "jambe": "leg", "pied": "foot", "orteils": "toes", "orteil": "toe",
               "torsion": "twist"}
SKELETON_KEYS = {"unite": "unit", "repere": "frame", "torsion": "twist", "mesure": "measure", "sorte": "sort",
                 "axe": "axis", "part": "fraction"}
TWIST_SORTS = {"retient": "holds", "suit": "follows"}
SKELETON_TEXTS = {"repere": "y up, front -z, character's left +x",
                  "torsion": "holds: the bone undoes (1 - fraction) of the roll of its source about the axis; "
                             "follows: the bone takes the fraction of the roll of its source about the axis"}
# shape.pkl of a rebuilt body: keys, map names (tronc, doigt_pouce, bras_D...), zones, construction notes
SHAPE_KEYS = {"P_avant": "P_before", "vois": "neighbors", "miroir": "mirror", "cartes": "maps", "libre": "free",
              "tete": "head"}
SHAPE_HEAD_KEYS = {"pont": "bridge", "anneau": "loop"}
SHAPE_MAP_WORDS = {"tronc": "trunk", "cou": "neck", "bras": "arm", "bande": "band", "jambe": "leg", "doigt": "digit",
                   "pouce": "thumb", "orteil": "toe"}
SHAPE_ZONES = {"aucune": "none", "corps": "body", "main": "hand", "orteil": "toe"}
SHAPE_INFO = {"emmanchure_cm": "armhole_cm", "y_aisselle": "y_armpit", "y_base_cou": "y_base_neck",
              "bras_perimetres_cm": "arm_perimeters_cm", "jambe_s0": "leg_s0", "doigts_debut_cm": "digits_start_cm",
              "orteils_z": "toes_z", "pouce": "thumb"}


def house_bone(name):
    """racine -> root, avant_bras_torsion2_G -> forearm_twist2_L, petit_doigt3_D -> little3_R."""
    if not isinstance(name, str):
        return name
    base, side = (name[:-2], name[-1]) if re.search(r"_[GD]$", name) else (name, "")
    for a, b in HOUSE_MULTI.items():
        base = re.sub(r"(?<![a-z])%s(?![a-z])" % a, b, base)
    base = re.sub(r"[a-z]+", lambda m: HOUSE_WORDS.get(m.group(0), m.group(0)), base)
    return base + ("_" + {"G": "L", "D": "R"}[side] if side else "")


def map_name(name):
    """tronc -> trunk, doigt_doigt0 -> digit_finger0, bras_D -> arm_R."""
    if not isinstance(name, str):
        return name
    base, side = (name[:-2], "_R") if name.endswith("_D") else (name, "")
    parts = base.split("_")
    if len(parts) > 1 and parts[0] == "doigt" and re.fullmatch(r"doigt\d*", parts[1]):
        parts[1] = "finger" + parts[1][5:]
    return "_".join(re.sub(r"[a-z]+", lambda m: SHAPE_MAP_WORDS.get(m.group(0), m.group(0)), x) for x in parts) + side


def revision_of(data):
    try:
        with open(os.path.join(data, MARKER), encoding="utf-8") as f:
            return int(json.load(f).get("revision", 0))
    except (FileNotFoundError, ValueError, AttributeError):
        return 0


class Format1(SystemExit):
    pass


def refused_format1(data, cache=None):
    """Stops with the conversion command if the data folder (or the cache) is still in format 1."""
    if data and format_of(data) == 1:
        raise Format1("%s holds format-1 data (French names). Convert it once: %s %s\n"
                      "(or install the library again into an empty folder)." % (data, MIGRATE, data))
    for d in (data, cache):
        if d and os.path.isdir(d) and format_of(d) == FORMAT and revision_of(d) < REVISION:
            raise Format1("%s is in format 2 but before revision %d (names inside the .npz files and the house "
                          "skeleton). Convert it once more: %s %s" % (d, REVISION, MIGRATE, d))
    if cache and os.path.isdir(cache) and format_of(cache) == 1:
        raise Format1("%s is a format-1 workshop cache (French names). Convert it once: %s %s\n"
                      "(and its data folder the same way)." % (cache, MIGRATE, cache))
