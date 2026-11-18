// Chimera 3D Avatar Maker — the page. Three views: Create (choose, look, tune the render, assemble),
// Library (import models, extract their parts), Characters (download what was assembled).
import * as V from "/page/visionneuse.js";
const { creerVisionneuse, lireGLB } = V;

const $=s=>document.querySelector(s);
const JSON_H={"Content-Type":"application/json"};
// Names come from model files: they are shown as text, never as HTML.
const ech=t=>String(t??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const garde=(cle,defaut)=>{ try{ return localStorage.getItem(cle)??defaut; }catch(e){ return defaut; } };
const noter=(cle,v)=>{ try{ localStorage.setItem(cle,v); }catch(e){} };

// Interface texts: [English, French]. The server speaks English; its closed labels are translated below (SERVEUR).
const TXT={
  vue_creer:["Create","Créer"], vue_bibliotheque:["Library","Bibliothèque"], vue_personnages:["Characters","Personnages"],
  visage:["Face","Visage"], coiffure:["Hair","Coiffure"], haut:["Top","Haut"], bas:["Bottom","Bas"], chaussures:["Shoes","Chaussures"], accessoires:["Extras","Accessoires"],
  chercher:["Search…","Chercher…"], couleur:["Colour","Couleur"], couleur_origine:["Original colour","Couleur d'origine"], tout:["all","tout"],
  cadre_corps:["Body","Corps"], cadre_visage:["Face","Visage"], tourner:["Turn","Tourner"], repos:["Neutral","Repos"],
  chargement:["Loading…","Chargement…"],
  simplifie:["Simplified preview: this browser has no WebGL2, so the light and shadow settings are off.","Aperçu simplifié : ce navigateur n'a pas WebGL2, les réglages de lumière et d'ombres sont éteints."],
  volet_personnage:["Character","Personnage"], volet_rendu:["Light & shadows","Lumière et ombres"],
  nom_personnage:["Character name","Nom du personnage"], sans_nom:["My character","Mon personnage"],
  assembler:["Assemble","Assembler"], assemblage:["Assembling…","Assemblage…"],
  aide_assembler:["Builds one finished model from these parts, then offers it for download.","Fabrique un seul modèle fini à partir de ces pièces, puis le propose au téléchargement."],
  inacheve:["unfinished","inachevé"],
  refusees:["Rejected parts","Pièces refusées"], retirer:["Remove","Retirer"],
  origines:["Original models and licences","Modèles d'origine et licences"], par:["by","par"], page_origine:["Original model page","Page du modèle d'origine"],
  lic_releve:["Terms as read on {d}; the page of the model has today's.","Conditions lues le {d} ; la page du modèle donne celles d'aujourd'hui."],
  lic_credit_oui:["credit required","crédit obligatoire"], lic_credit_non:["credit not required","crédit non requis"],
  lic_commercial:["commercial use allowed","usage commercial permis"], lic_perso:["personal use only","usage personnel seulement"],
  lic_modif:["modification and redistribution allowed","modification et redistribution permises"], lic_modif_non:["modification restricted","modification restreinte"],
  lic_autre:["Other terms","Autres conditions"], lic_vrm:["VRM licence","Licence VRM"],
  garde_coiffure:["Does not sit cleanly on this face","Ne se pose pas proprement sur ce visage"],
  rayon_vide:["Nothing here yet.","Rien ici pour l'instant."], ouvrir_bibliotheque:["Open the library","Ouvrir la bibliothèque"],
  attente:["Waiting…","En attente…"], echec:["Failed: ","Échec : "], serveur_muet:["The local server stopped answering: ","Le serveur local ne répond plus : "],
  controles_ok:["Checks passed","Contrôles passés"], controles_ko:["Checks to review","Contrôles à revoir"],
  images_absentes:["Preview images not rendered: ","Images non rendues : "], vrm_absent:["No VRM: ","VRM non livré : "],
  telecharger:["Download","Télécharger"], ouvrir_dossier:["Open folder","Ouvrir le dossier"],
  prereglage:["Preset","Préréglage"], personnalise:["Custom","Personnalisé"], nom_rendu:["Render name","Nom du rendu"],
  rendu_importer:["Import…","Importer…"], rendu_exporter:["Export","Exporter"], rendu_defaut:["Reset","Réinitialiser"],
  rendu_importe:["Render imported: ","Rendu importé : "], rendu_refuse:["This file is not a render: ","Ce fichier n'est pas un rendu : "],
  rendu_exporte:["Render saved as ","Rendu enregistré sous "],
  importer_titre:["Import models","Importer des modèles"],
  importer_aide:["Bring in existing characters. Nothing leaves this computer: files are copied into the local data folder.","Faites entrer des personnages existants. Rien ne quitte cet ordinateur : les fichiers sont copiés dans le dossier de données local."],
  depot_titre:["Drop model files here, or click to choose","Déposez des fichiers de modèles ici, ou cliquez pour choisir"],
  formats_entree:["Accepted formats","Formats acceptés"], squelettes:["Recognised skeletons","Squelettes reconnus"],
  eprouve:["tested","éprouvé"], partiel:["partial","partiel"],
  extraire_titre:["Extract the parts","Extraire les pièces"],
  extraire_aide:["Cuts each model into a face, a hairstyle and clothes, fits them on the common body and adds them to the catalogue. This runs Blender and can take several minutes per model.","Découpe chaque modèle en visage, coiffure et vêtements, les porte sur le corps commun et les ajoute au catalogue. Blender travaille : comptez plusieurs minutes par modèle."],
  visages:["Faces","Visages"], coiffures:["Hairstyles","Coiffures"], vetements:["Clothes","Vêtements"], extraire:["Extract","Extraire"],
  aucun_modele:["No model imported yet.","Aucun modèle importé pour l'instant."], choisir_modeles:["Tick at least one model.","Cochez au moins un modèle."],
  pieces:["parts","pièces"], refus:["rejected","refusés"], envoi:["Importing ","Import de "], importe:["Imported: ","Importé : "], importes:["{n} models imported","{n} modèles importés"], ecartes:["{n} file(s) set aside (weapons, already imported)","{n} fichier(s) écarté(s) (armes, déjà importés)"],
  extraction_finie:["Extraction finished.","Extraction terminée."],
  extraction_en_cours:["Extracting…","Extraction en cours…"], annuler:["Cancel","Annuler"], annulation:["Cancelling…","Annulation…"],
  extraction_serveur:["Closing this tab does not stop the extraction: it runs on the local server. Come back to this page to follow it.","Fermer l'onglet n'arrête pas l'extraction : elle tourne sur le serveur local. Revenez sur cette page pour la suivre."],
  extraction_annulee:["Extraction cancelled. What was finished stays in the catalogue; extracting again resumes from there.","Extraction annulée. Ce qui était fini reste au catalogue ; extraire de nouveau reprend à partir de là."],
  modele_k:["Model {k} of {n}: {nom} — {e}","Modèle {k} sur {n} : {nom} — {e}"], modeles_k:["Models {k}–{m} of {n}: {e}","Modèles {k} à {m} sur {n} : {e}"],
  pas_pivot:["pivot head","tête-pivot"], pas_visages:["faces","visages"], pas_coiffures:["hairstyles","coiffures"], pas_vetements:["cutting the clothes","découpe des vêtements"],
  pas_catalogue:["fitting the faces","pose des visages"], pas_coiffure:["hairstyle","coiffure"], pas_pieces:["clothes","vêtements"], indisponible:["Not available on this server.","Indisponible sur ce serveur."],
  personnages_titre:["Assembled characters","Personnages assemblés"],
  personnages_aide:["Every character you assembled, ready to download.","Chaque personnage assemblé, prêt à télécharger."],
  aucun_personnage:["No character yet: assemble one in Create.","Aucun personnage pour l'instant : assemblez-en un dans Créer."],
  manque:["The catalogue is empty. Import models and extract their parts to start.","Le catalogue est vide. Importez des modèles et extrayez leurs pièces pour commencer."],
  nombres:["{v} faces · {c} hairstyles · {p} parts","{v} visages · {c} coiffures · {p} pièces"],
  coiffure_mal_posee:["This hairstyle does not sit cleanly on this face ({e}).","Cette coiffure ne se pose pas proprement sur ce visage ({e})."],
  traverse:["{a} clips through {b}: {p} % of the points, up to {mm} mm.","{a} traverse {b} : {p} % des points, jusqu'à {mm} mm."],
  ressort:["{a} pokes out of {b}: {p} % of the points, up to {mm} mm.","{a} ressort de {b} : {p} % des points, jusqu'à {mm} mm."],
  couches_echec:["Layers not checked for this combination: the computation failed.","Couches non vérifiées pour cette combinaison : le calcul a échoué."],
};
let LANGUE=garde("chimera.langue",(navigator.language||"en").toLowerCase().startsWith("fr")?"fr":"en");
const L=()=>LANGUE==="fr"?1:0;
const t=(k,v)=>{ let s=(TXT[k]||[k,k])[L()]; for(const n in v||{}) s=s.replace(`{${n}}`,v[n]); return s; };

// Closed labels that come from the server or from data, as [English, French]. The server sends English (older caches
// may hold the French one): either is shown in the page's language. Free diagnostic messages (rejection reasons,
// errors, logs) stay in English in both languages.
const TYPES={   // piece types and accessory kinds (couches.defaut.json), by identifier
  culotte:["underwear","culotte"], chaussettes:["socks","chaussettes"], haut:["top","haut"], manches:["sleeves","manches"],
  short:["shorts","short"], pantalon:["trousers","pantalon"], jupe:["skirt","jupe"], robe:["dress","robe"], combinaison:["jumpsuit","combinaison"],
  chaussures:["shoes","chaussures"], bottes:["boots","bottes"], gants:["gloves","gants"], col:["collar","col"], gilet:["waistcoat","gilet"],
  veste:["jacket","veste"], manteau:["coat","manteau"], capuche:["hood","capuche"],
  ceinture:["belt","ceinture"], harnais:["harness","harnais"], jarretiere:["garter","jarretière"], chevillere:["anklet","chevillère"],
  brassard:["armband","brassard"], bracelet:["bracelet","bracelet"], bague:["ring","bague"], cravate:["tie","cravate"], echarpe:["scarf","écharpe"],
  collier:["necklace","collier"], bijou:["jewel","bijou"], sac:["bag","sac"], lunettes:["glasses","lunettes"], masque:["mask","masque"],
  couvre_chef:["headwear","couvre-chef"], casque_audio:["headphones","casque audio"], queue:["tail","queue"], ailes:["wings","ailes"], oreilles:["ears","oreilles"],
};
const EXPRESSIONS={Sourire:["Smile","Sourire"],Rire:["Laugh","Rire"],Parle:["Talking","Parle"],Surprise:["Surprise","Surprise"],
  "Colère":["Anger","Colère"],Triste:["Sad","Triste"],"Yeux fermés":["Eyes closed","Yeux fermés"],Yandere:["Yandere","Yandere"]};   // names are keys (expressions.py)
const ECHECS={crane_couvert:["scalp poorly covered","crâne mal couvert"],crane_sous_cheveux:["scalp showing","crâne visible"],cheveux_hors_peau:["hair inside the skin","cheveux dans la peau"]};
const PREREGLAGES={maison:["Studio","Studio"],jour:["Day","Jour"],couchant:["Sunset","Couchant"],nuit:["Night","Nuit"],simple:["Plain","Simple"]};
const SERVEUR=[
  ["face","visage"],["hairstyle","coiffure"],["part","pièce"],["Free head","Tête libre"],["Free bob","Carré libre"],["Base face","Visage de base"],["Base hairstyle","Coiffure de base"],
  // extraction states (entrees.Extraction)
  ["Waiting…","En attente…"],["Cancelling…","Annulation…"],["Extraction finished.","Extraction terminée."],["The extraction failed.","L'extraction a échoué."],
  ["Pivot head: the head closest to the game's average…","Tête-pivot : la tête la plus proche de la moyenne du jeu…"],
  ["Extraction cancelled. What was finished stays in the catalogue; extracting again resumes from there.","Extraction annulée. Ce qui était fini reste au catalogue ; extraire de nouveau reprend à partir de là."],
  // /api/formats (entrees.formats); a long note is matched sentence by sentence
  ["PMX (MikuMikuDance), in a .zip with its textures (one or more models per archive)","PMX (MikuMikuDance), dans un .zip avec ses textures (un ou plusieurs modèles par archive)"],
  ["Names must follow the common language of ZZZ's official MMD models (face skin \"颜\", hair \"髮\", skin \"肌\", MMD bone \"頭\"): a PMX without them is rejected at import.",
   "Les noms doivent suivre la langue commune des modèles MMD officiels de ZZZ (peau du visage « 颜 », cheveux « 髮 », peau « 肌 », os MMD « 頭 ») : un PMX qui ne les a pas est refusé à l'import."],
  ["An archive may hold several models: each PMX that follows the common language is imported, the others (a weapon, a prop) are set aside.",
   "Une archive peut contenir plusieurs modèles : chaque PMX qui suit la langue commune est importé, les autres (une arme, un décor) sont écartés."],
  ["The assembled model and its textures, as a .zip.","Le modèle assemblé et ses textures, en .zip."],
  ["No physics (no rigid bodies for the hair, no springs).","Sans physique (ni corps rigides de cheveux, ni ressorts)."],
  ["Humanoid bones, 13 expressions, MToon materials, in T-pose.","Os humanoïdes, 13 expressions, matériaux MToon, en T-pose."],
  ["A VRM is a binary glTF file: renamed to .glb, a glTF viewer opens it (without the VRM extensions).","Un VRM est un fichier glTF binaire : renommé en .glb, un lecteur glTF l'ouvre (sans les extensions VRM)."],
  ["Missing if writing it failed.","Absent si son écriture a échoué."],
  ["Humanoid bones renamed to MMD bones at import, breast bones merged into their parent, springs not kept.","Os humanoïdes renommés en os MMD à l'import, os de poitrine fondus dans leur parent, ressorts non repris."],
  ["Bones of the MMD standard (センター, 上半身, 頭…) and materials of the common language, no conversion.","Os du standard MMD (センター, 上半身, 頭…) et matériaux de la langue commune, sans conversion."],
  ["A VRM that is not a VRoid character (no VRoid Studio face material, \"..._Face_00_SKIN\") is refused with that reason: the face, hair and outfit are found through VRoid's material names.","Un VRM qui n'est pas un personnage VRoid (aucun matériau de visage VRoid Studio, « ..._Face_00_SKIN ») est refusé avec cette raison : le visage, les cheveux et la tenue se trouvent par les noms de matériaux de VRoid."],
  ["Tried on three VRoid Studio sample models (15/11/2026) and on eight characters through the workshop page (18/11/2026); physics and spring bones are never read.",
   "Essayé sur trois modèles d'exemple de VRoid Studio (15/11/2026) et sur huit personnages par la page de l'atelier (18/11/2026) ; la physique et les ressorts ne sont jamais lus."],
  ["No physics: hair and garment flaps do not move on their own.","Aucune physique : cheveux et pans de vêtements ne bougent pas seuls."],
];
// sentences built around names: [pattern on the English text, [English, French]] (the captured names are translated too)
const MODELES=[
  [/^(.+) worn over (.+)$/,[(a,b)=>`${a} worn over ${b}`,(a,b)=>`${a} porté par-dessus ${b}`]],
  [/^(.+) removed: the (.+) covers it$/,[(a,b)=>`${a} removed: the ${b} covers it`,(a,b)=>`${a} retirée : le ${b} la couvre`]],
  [/^(.+): VRM humanoid$/,[a=>`${a}: VRM humanoid`,a=>`${a} : humanoïde VRM`]],
  [/^(.+): standard MMD bones$/,[a=>`${a}: standard MMD bones`,a=>`${a} : os MMD standard`]],
  [/^part ([a-z_]+)(-\d+)?$/,[(a,b)=>`part ${genre(a)}${b}`,(a,b)=>`pièce ${genre(a)}${b}`]],   // a rejected part of an import, by its folder (<kind>[-n])
  [/^part (\S+)$/,[a=>`part ${a}`,a=>`pièce ${a}`]],
  [/^(face|hairstyle) (.+)$/,[(a,b)=>`${a} ${b}`,(a,b)=>`${a} ${b}`]],   // a rejected catalogue item of a model
  // render files read by the viewer (visionneuse.normaliserRendu)
  [/^This file is not a render: unreadable JSON \((.*)\)\.$/,[a=>`unreadable JSON (${a}).`,a=>`JSON illisible (${a}).`]],   // after the "not a render" prefix of rendu_refuse
  [/^Not a render: a JSON object is expected\.$/,[()=>"Not a render: a JSON object is expected.",()=>"Ce n'est pas un rendu : un objet JSON est attendu."]],
  [/^Not a render: "format" must be "chimera-rendu" \(read: (.*)\)\.$/,[a=>`Not a render: "format" must be "chimera-rendu" (read: ${a}).`,a=>`Ce n'est pas un rendu : "format" doit valoir "chimera-rendu" (lu : ${a}).`]],
  [/^Unreadable render: "version" must be an integer \(1\)\.$/,[()=>`Unreadable render: "version" must be an integer (1).`,()=>`Rendu illisible : "version" doit être un entier (1).`]],
  [/^Render version (.+): this viewer only reads version 1\.$/,[a=>`Render version ${a}: this viewer only reads version 1.`,a=>`Rendu de version ${a} : cette visionneuse ne lit que la version 1.`]],
  [/^(.+) · ([^·]+)$/,[(a,b)=>`${a} · ${b}`,(a,b)=>`${a} · ${b}`]],   // a part: "<character> · <type>"
];
// kinds of part the workshop does not offer on their own (trims, objects), named in refusal reasons and folders
const GENRES={sangle:["strap","sangle"],boucle_bouton:["buckle or button","boucle ou bouton"],ornement:["ornament","ornement"],poche:["pocket","poche"],
  ecusson:["patch","écusson"],badge:["badge","badge"],chaine:["chain","chaîne"],corde:["cord","corde"],fermeture:["fastener","fermeture"],boucle:["buckle","boucle"],
  arme:["weapon","arme"],peluche:["plush toy","peluche"],objet:["object","objet"],cheveux:["hair","cheveux"],"no kind":["no kind","sans genre"]};
const genre=k=>(GENRES[k]||TYPES[k]||[k,k])[L()];
// blocking checks of a part (preparer.BLOQUANTS, assembler_blender.BLOQUANTS)
const CONTROLES={sous_la_peau:["inside the skin","sous la peau"],traversee_repos:["clipping at rest","traversée au repos"],
  traversee_repos_p90_mm:["clipping at rest, 90th percentile (mm)","traversée au repos, 90e centile (mm)"],poses_corps_seul:["test poses on the body alone","épreuves sur le corps seul"]};
const controle=k=>CONTROLES[k]?CONTROLES[k][L()]:k;
const echecs=s=>s.split(", ").map(c=>ECHECS[c]?ECHECS[c][L()]:c).join(", ");
// refusal reasons and errors written by the server: [pattern on the English text, French, English if it must change too]
const RAISONS=[
  [/^the workshop does not offer this kind of part on its own \((.+)\)$/,a=>`l'atelier ne propose pas ce genre de pièce seule (${genre(a)})`,a=>`the workshop does not offer this kind of part on its own (${genre(a)})`],
  [/^sole without an upper: (\d+) % of the top of the foot covered \(minimum (\d+) %\)$/,(a,b)=>`semelle sans dessus : ${a} % du dessus du pied couvert (minimum ${b} %)`],
  [/^fails on its own head : (.+)$/,a=>`échoue sur sa propre tête : ${echecs(a)}`,a=>`fails on its own head: ${echecs(a)}`],
  [/^original head missing$/,()=>"tête d'origine absente"],
  [/^sits cleanly on only (\d+) heads out of (\d+)$/,(a,b)=>`ne se pose proprement que sur ${a} têtes sur ${b}`],
  [/^shape damaged by fitting \(review score (.+), limit (.+)\)$/,(a,b)=>`forme abîmée par le portage (note ${a} à la revue, limite ${b})`],
  [/^part with no extent \((.+) mm\)$/,a=>`pièce sans étendue (${a} mm)`],
  [/^fitting failed: (.*)$/s,a=>`port impossible : ${a}`],
  [/^measurement failed: (.*)$/s,a=>`mesure impossible : ${a}`],
  [/^preparation failed: (.*)$/s,a=>`préparation impossible : ${a}`],
  [/^cutting failed: (.*)$/s,a=>`découpe en échec : ${a}`],
  [/^unknown$/,()=>"inconnue"],
  [/^to prepare again: zero-weight bone mishandled by the graft \(fixed, run preparer again\)$/,()=>"à repréparer : os de poids nul mal repris par la greffe (corrigé, relancer preparer)"],
  [/^(\w+) = (.+) \(threshold (.+)\)$/,(a,b,c)=>`${controle(a)} = ${b} (seuil ${c})`,(a,b,c)=>`${controle(a)} = ${b} (threshold ${c})`],
  // assembly (assembler_blender.py, serveur.py)
  [/^(.+): (\w+) = (.+) \(threshold (.+)\)$/s,(n,a,b,c)=>`${srv(n)} : ${controle(a)} = ${b} (seuil ${c})`,(n,a,b,c)=>`${srv(n)}: ${controle(a)} = ${b} (threshold ${c})`],
  [/^(.+): measurement impossible \((.*)\)$/s,(n,a)=>`${srv(n)} : mesure impossible (${a})`],
  [/^(\d+) vertices, (\d+) materials, (\d+) bones; (\d+) parts measured, (\d+) layer pairs, (\d+) clippings$/,(a,b,c,d,e,f)=>`${a} sommets, ${b} matériaux, ${c} os ; ${d} pièces mesurées, ${e} paires de couches, ${f} traversées`],
  [/^Blender stopped \(code (-?\d+)\), see journal\.txt$/,a=>`Blender s'est arrêté (code ${a}), voir journal.txt`],
  [/^could not start Blender: (.*)$/s,a=>`impossible de lancer Blender : ${a}`],
  [/^assembly interrupted, see journal\.txt$/,()=>"assemblage interrompu, voir journal.txt"],
  [/^assembly interrupted \(the server stopped during the job\)$/,()=>"assemblage interrompu (le serveur s'est arrêté pendant le travail)"],
  [/^Unknown job\.$/,()=>"Travail inconnu."],
  // import and extraction (entrees.py, serveur.py)
  [/^The active game "(.+)" has no mmd_extrait folder\.$/,a=>`Le jeu actif « ${a} » n'a pas de dossier mmd_extrait.`],
  [/^Unreadable file name: rename the file with Latin letters or digits\.$/,()=>"Nom de fichier illisible : renommer le fichier avec des lettres latines ou des chiffres."],
  [/^Empty file\.$/,()=>"Fichier vide."],
  [/^File too large \((\d+) MB at most\)\.$/,a=>`Fichier trop gros (${a} Mo au plus).`],
  [/^Upload interrupted\.$/,()=>"Envoi interrompu."],
  [/^This file is not a VRM \(nor a binary glTF\)\.$/,()=>"Ce fichier n'est pas un VRM (ni un glTF binaire)."],
  [/^This file is not a readable VRM \(glTF 2\.0 expected\)\.$/,()=>"Ce fichier n'est pas un VRM lisible (glTF 2.0 attendu)."],
  [/^This file is not a readable VRM: damaged header\.$/,()=>"Ce fichier n'est pas un VRM lisible : en-tête abîmé."],
  [/^This file declares a VRM version other than 0\.x and 1\.x: it is not read\.$/,()=>"Ce fichier déclare une version de VRM autre que 0.x et 1.x : il n'est pas lu."],
  [/^This file is a glTF without the VRM extension: only a VRM from VRoid \(Studio or Hub\) is read\.$/,()=>"Ce fichier est un glTF sans extension VRM : seul un VRM de VRoid (Studio ou Hub) est lu."],
  [/^This VRM is not read: the file is not a VRoid character: no material has a VRoid Studio face name \("\.\.\._Face_00_SKIN"\), and the face, hair and outfit are found through those names\.$/s,
   ()=>"Ce VRM n'est pas lu : ce n'est pas un personnage VRoid, aucun matériau ne porte le nom de visage de VRoid Studio (« …_Face_00_SKIN »), et c'est par ces noms qu'on trouve le visage, la coiffure et la tenue."],
  [/^This VRM is not read: (.*)\.$/s,a=>`Ce VRM n'est pas lu : ${a}.`],
  [/^Archive rejected: a file escapes its folder \((.*)\)\.$/s,a=>`Archive refusée : un fichier sort de son dossier (${a}).`],
  [/^This file is not a readable \.zip archive\.$/,()=>"Ce fichier n'est pas une archive .zip lisible."],
  [/^Archive rejected: more than (\d+) files\.$/,a=>`Archive refusée : plus de ${a} fichiers.`],
  [/^Encrypted archive: unpack it yourself and make a \.zip again without a password\.$/,()=>"Archive chiffrée : la décompresser soi-même et refaire un .zip sans mot de passe."],
  [/^Archive rejected: it contains a link \((.*)\)\.$/s,a=>`Archive refusée : elle contient un lien (${a}).`],
  [/^Archive rejected: abnormal compression \((.*)\)\.$/s,a=>`Archive refusée : compression anormale (${a}).`],
  [/^Archive rejected: more than (\d+) GB once unpacked\.$/,a=>`Archive refusée : plus de ${a} Go une fois décompressée.`],
  [/^The archive contains no \.pmx file\.$/,()=>"L'archive ne contient aucun fichier .pmx."],
  [/^Archive rejected: a file is larger than it declares\.$/,()=>"Archive refusée : un fichier est plus gros que ce qu'elle déclare."],
  [/^This \.pmx file is not readable: (.*)$/s,a=>`Ce fichier .pmx n'est pas lisible : ${a}`],
  [/^This PMX does not follow the workshop's common language: (.*)\. See outils\/jeux\/README\.md\.$/s,a=>`Ce PMX ne suit pas la langue commune de l'atelier : ${a}. Voir outils/jeux/README.md.`],
  [/^Unsupported format \((.*)\): a \.vrm \(VRoid Studio or VRoid Hub\) or a \.zip containing a \.pmx\.$/,a=>`Format non pris en charge (${a==="no extension"?"sans extension":a}) : un .vrm (VRoid Studio ou VRoid Hub) ou un .zip contenant un .pmx.`],
  [/^Unsupported format: a \.vrm \(VRoid Studio or VRoid Hub\) or a \.zip containing a \.pmx\.$/,()=>"Format non pris en charge : un .vrm (VRoid Studio ou VRoid Hub) ou un .zip contenant un .pmx."],
  [/^A model "(.+)" is already imported: rename the file to make another one\.$/,a=>`Un modèle « ${a} » est déjà importé : renommer le fichier pour en faire un autre.`],
  [/^A model "(.+)" is already imported\.$/,a=>`Un modèle « ${a} » est déjà importé.`],
  [/^The VRM conversion failed: (.*)$/s,a=>`La conversion du VRM a échoué : ${a}`],
  [/^None of the (\d+) \.pmx files of the archive is a readable character\. The first one: (.*)$/s,(a,b)=>`Aucun des ${a} fichiers .pmx de l'archive n'est un personnage lisible. Le premier : ${srv(b)}`],
  [/^Every model of this archive is already imported\.$/,()=>"Tous les modèles de cette archive sont déjà importés."],
  [/^Unreadable request\.$/,()=>"Demande illisible."],
  [/^Choose at least one model \("modeles"\) or all of them \("tous": true\)\.$/,()=>"Choisir au moins un modèle (« modeles ») ou tous (« tous »: true)."],
  [/^Model unknown to the active game: (.*)\.$/s,a=>`Modèle inconnu du jeu actif : ${a}.`],
  [/^No model in the active game: import one first\.$/,()=>"Aucun modèle dans le jeu actif : en importer un d'abord."],
  [/^"quoi": a list among (.*)\.$/,a=>`« quoi » : une liste parmi ${a}.`],
  [/^An extraction is already running \((.+)\): wait for it to finish\.$/,a=>`Une extraction est déjà en cours (${a}) : attendre qu'elle finisse.`],
  [/^No extraction is running\.$/,()=>"Aucune extraction n'est en cours."],
  [/^The pivot head could not be computed \(see the log\)\.$/,()=>"La tête-pivot n'a pas pu être calculée (voir le journal)."],
  [/^The catalogue cannot be prepared: (.*)$/s,a=>`Le catalogue ne peut pas être préparé : ${a}`],
  [/^Preparing the catalogue stopped with an error \(see the log\)\.$/,()=>"La préparation du catalogue s'est arrêtée en erreur (voir le journal)."],
  [/^Unknown assembly or format\.$/,()=>"Assemblage ou format inconnu."],
  [/^This assembly has no VRM\.$/,()=>"Cet assemblage n'a pas de VRM."],
  [/^This assembly has no PMX\.$/,()=>"Cet assemblage n'a pas de PMX."],
  [/^Content-Type application\/octet-stream expected\.$/,()=>"Content-Type application/octet-stream attendu."],
  [/^Unreadable file size\.$/,()=>"Taille du fichier illisible."],
];
const PAIRES=new Map();
for(const x of [...SERVEUR,...Object.values(TYPES)]) for(const s of x) if(!PAIRES.has(s)) PAIRES.set(s,x);
// A label from the server or the data, in the page's language (unknown text is shown as it is).
function srv(s){
  s=String(s??""); const x=PAIRES.get(s); if(x) return x[L()];
  for(const [re,fr,en] of RAISONS){ const m=s.match(re); if(m) return L()?fr(...m.slice(1)):en?en(...m.slice(1)):s; }
  for(const [re,f] of MODELES){ const m=s.match(re); if(m) return f[L()](...m.slice(1).map(srv)); }
  const morceaux=s.split("; ");   // several reasons in a row
  if(morceaux.length>1){ const tr=morceaux.map(srv); if(tr.every((p,i)=>p!==morceaux[i])) return tr.join(L()?" ; ":"; "); }
  const phrases=s.split(/(?<=\.)\s+(?=\S)/);
  if(phrases.length>1&&phrases.every(p=>PAIRES.has(p))) return phrases.map(srv).join(" ");
  return s;
}
const nomType=p=>{ const x=TYPES[p.type]||PAIRES.get(p.type_fr); return x?x[L()]:(p.type_fr||p.type||""); };
// displayed name of a catalogue item: its label's name when it has one, otherwise its code name
// (a part is "<character> · <type>", in the page's language)
const nomCode=it=>it.perso?`${it.perso} · ${nomType(it)}`:srv(it.nom);
const nomElement=it=>it.etiquette?.nom?.[L()]||nomCode(it);
// the model an item comes from (cat.sources: page, name, author, licence terms as read at download), if known
const sourceDe=it=>(cat.sources||{})[it.perso||it.id]||null;
const lienSur=u=>/^https:\/\//.test(String(u||""))?String(u):null;
function licenceCourte(s){
  const l=s.licence||{}, m=l.nom?[l.nom]:[];
  if(l.creditNotation) m.push(t(l.creditNotation==="required"?"lic_credit_oui":"lic_credit_non"));
  if(l.commercialUsage) m.push(t(l.commercialUsage==="personalNonProfit"?"lic_perso":"lic_commercial"));
  if(l.modification) m.push(t(l.modification==="allowModificationRedistribution"?"lic_modif":"lic_modif_non"));
  return m.join(" · ");
}
// tooltip of a tile: name, description, tags, then the code name (the model it comes from)
const bulleElement=it=>{ const e=it.etiquette, s=sourceDe(it), o=s?`${s.nom} ${t("par")} ${s.auteur} (${s.site||""})`:null;
  if(!e) return [nomCode(it),o].filter(Boolean).join("\n");
  return [nomElement(it),e.description?.[L()],(e.tags?.[L()]||[]).map(x=>"#"+x).join(" "),nomCode(it),o].filter(Boolean).join("\n"); };
// what the search box reads: names, description and tags, in both languages, and the code name
const texteRecherche=it=>it._rech??=[nomCode(it),it.id,...(it.etiquette?[...(it.etiquette.nom||[]),...(it.etiquette.description||[]),...(it.etiquette.tags||[]).flat()]:[])].join(" ").toLowerCase();
const nomExpression=n=>EXPRESSIONS[n]?EXPRESSIONS[n][L()]:n;
function traduire(){
  document.documentElement.lang=LANGUE;
  for(const e of document.querySelectorAll("[data-i]")) e.textContent=t(e.dataset.i);
  $("#recherche").placeholder=t("chercher"); $("#langue").textContent=LANGUE==="fr"?"en":"fr"; $("#nom").placeholder=t("sans_nom");
  boutonExtraire(xEnCours);
}

const DESSINS={
  visage:`<circle cx="12" cy="12" r="8.5"/><path d="M9 10.5v.5M15 10.5v.5M9 15c1.8 1.6 4.2 1.6 6 0"/>`,
  coiffure:`<path d="M4.5 18v-5C4.5 7.5 7.5 4 12 4s7.5 3.500 7.500 9v5M8 10c2.500 1.200 5.500 1.200 8-.500"/>`,
  haut:`<path d="M8.500 4 4 6.500l1.500 4L8 9.500V20h8V9.500l2.500 1 1.500-4L15.500 4c-.800 1.300-2 2-3.500 2s-2.700-.700-3.500-2z"/>`,
  bas:`<path d="M7 4h10l1.500 16H14L12 9l-2 11H5.500z"/>`,
  chaussures:`<path d="M3 17v-3.500c2 0 3.500-1 4-4l3.500 1c.500 2 2.500 3.500 6 4 2.500.400 4.500 1 4.500 2.500z"/>`,
  accessoires:`<path d="m12 3.500 2.600 5.300 5.900.800-4.300 4.100 1 5.800L12 16.800 6.800 19.500l1-5.800-4.300-4.100 5.900-.800z"/>`,
};
const ONGLETS=["visage","coiffure","haut","bas","chaussures","accessoires"];
const COULEURS=[null,"#1e1a1c","#4a2f24","#7a4a2c","#b98a52","#e7cf9a","#e9e4dc","#8b8f9b","#a23a34","#d9648c","#7a4ba8","#3c6fb4","#3f9c97","#5f9449"];
const vue=creerVisionneuse($("#vue"));
const etat={onglet:"visage",tete:null,coiffure:null,pieces:new Set(),couleur:null,filtre:null,recherche:"",signal:[],ref:0.5};
let cat={tetes:[],coiffures:[],pieces:[],refus:[]}, parId={}, genCombi=0, genTete=0, genCoif=0, coifBase=null, coifDecal=null;   // one counter per kind: a hairstyle does not cancel the neck fitting of a face
let corpsBase=null, corpsDecal=null;   // body positions under the face graft, offset of the feet posed in a heeled shoe
let peauPieces=[], peauTete=[];   // body skin faces hidden by the parts, by the face graft
const masquerCorps=()=>vue.masque("corps",[...peauPieces,...peauTete]);
const hex=c=>({r:parseInt(c.slice(1,3),16)/255,g:parseInt(c.slice(3,5),16)/255,b:parseInt(c.slice(5,7),16)/255});
const json=async(url,opts)=>{ const r=await fetch(url,opts); let v=null; try{ v=await r.json(); }catch(e){} if(!r.ok) throw new Error((v&&v.erreur&&srv(v.erreur))||(r.status===404?t("indisponible"):`${r.status}`)); return v; };

async function lireCatalogue(){
  cat=await json("/api/catalogue"); parId={};
  for(const k of ["tetes","coiffures","pieces","refus"]) cat[k]=cat[k]||[];
  for(const p of cat.pieces) parId[p.id]=p;
  $("#nombres").textContent=t("nombres",{v:cat.tetes.length,c:cat.coiffures.length,p:cat.pieces.length});
  $("#nb-refus").textContent=cat.refus.length;
  $("#liste-refus").innerHTML=cat.refus.map(r=>`<li><b>${ech(srv(r.nom))}</b> (${ech(srv(r.genre))})<br>${ech(srv(r.raison))}</li>`).join("");
}
const vide=()=>!cat.tetes.length||!cat.coiffures.length;

function montrer(v){
  if(!["creer","bibliotheque","personnages"].includes(v)) v="creer";
  document.body.dataset.vue=v;
  for(const n of ["creer","bibliotheque","personnages"]) $("#"+n).hidden=n!==v;
  for(const b of $("#vues").children) b.classList.toggle("actif",b.dataset.v===v);
  if(v==="creer"){ vue.sale=true; window.dispatchEvent(new Event("resize")); }
  if(v==="bibliotheque") ouvrirBibliotheque();
  if(v==="personnages") ouvrirPersonnages();
  if(location.hash.slice(1)!==v) history.replaceState(null,"","#"+v);
}

async function demarrer(){
  traduire();
  $("#langue").onclick=()=>{ LANGUE=LANGUE==="fr"?"en":"fr"; noter("chimera.langue",LANGUE); traduire(); peindreOnglets(); peindre(); construireStudio(); montrer(document.body.dataset.vue); $("#nombres").textContent=t("nombres",{v:cat.tetes.length,c:cat.coiffures.length,p:cat.pieces.length}); };
  $("#vues").onclick=e=>{ const v=e.target.dataset.v; if(v) montrer(v); };
  $("#volets").onclick=e=>{ const p=e.target.dataset.p; if(p) volet(p); };
  $("#onglets").onclick=e=>{const o=e.target.closest("[data-o]"); if(o){etat.onglet=o.dataset.o;etat.filtre=null;peindre();}};
  $("#recherche").oninput=e=>{etat.recherche=e.target.value.toLowerCase();peindreGrille();};
  $("#pastilles").innerHTML=COULEURS.map((c,i)=>`<button data-c="${i}" title="${c?c:t("couleur_origine")}" style="background:${c||"#fff"}">${c?"":"×"}</button>`).join("");
  $("#pastilles").onclick=e=>{const i=e.target.dataset.c; if(i!==undefined){etat.couleur=COULEURS[+i];majPastilles();appliquerTeinte();}};
  $("#commandes").onclick=e=>{ const v=e.target.dataset.vue; if(!v)return; if(v==="tour"){vue.tourne=!vue.tourne;$("#tourner").classList.toggle("actif",vue.tourne);} else vue.cadrer(v);};
  $("#expression").onchange=e=>vue.exprimer&&vue.exprimer(e.target.value||null);
  vue.surTourne=b=>$("#tourner").classList.toggle("actif",b);
  vue.surTexture=(u,m)=>{ if(etat.imgCoiffure===u){etat.ref=m; appliquerTeinte();} };
  $("#assembler").onclick=assembler;
  $("#nom").value=garde("chimera.nom","");
  $("#nom").oninput=e=>noter("chimera.nom",e.target.value);
  $("#simplifie").hidden=!vue.logiciel;
  volet("personnage"); brancherBibliotheque();
  await demarrerStudio();
  await lireCatalogue(); peindreOnglets();
  document.addEventListener("visibilitychange",()=>{ if(!document.hidden) suivreExtraction(); });
  if(vide()){ peindre(); montrer("bibliotheque"); suivreExtraction(); return; }
  montrer(location.hash.slice(1));
  suivreExtraction();
  await monterScene();
}
let scene=false;
async function monterScene(){
  scene=true;
  await vue.placer("corps","/data/corps/corps.glb");
  etat.tete=cat.tetes[0].id; etat.coiffure=cat.coiffures[0].id;
  await majTete(); await majCoiffure(); await majPieces(); peindre();
  genererVignettes();
}
// The catalogue changed on the server (an extraction made a model usable): new lists, same choices, same scene.
async function majCatalogue(){
  await lireCatalogue(); peindreOnglets(); $("#manque").hidden=!vide();
  if(!scene){ if(!vide()) await monterScene(); else peindre(); return; }
  peindre(); genererVignettes();
}
function volet(p){
  for(const b of $("#volets").children) b.classList.toggle("actif",b.dataset.p===p);
  $("#volet-personnage").hidden=p!=="personnage"; $("#volet-rendu").hidden=p!=="rendu";
}
function peindreOnglets(){
  const n={visage:cat.tetes.length,coiffure:cat.coiffures.length}; for(const p of cat.pieces) n[p.onglet]=(n[p.onglet]||0)+1;
  $("#onglets").innerHTML=ONGLETS.map(k=>`<button data-o="${k}"><svg viewBox="0 0 24 24" aria-hidden="true">${DESSINS[k]}</svg>${t(k)}<b>${n[k]||0}</b></button>`).join("");
}
function peindre(){
  for(const b of $("#onglets").children) b.classList.toggle("actif",b.dataset.o===etat.onglet);
  $("#titre-onglet").textContent=t(etat.onglet);
  $("#couleurs").hidden=etat.onglet!=="coiffure";
  majPastilles(); peindreFiltres(); peindreGrille(); peindreFiche();
}
function elements(){
  if(etat.onglet==="visage") return cat.tetes.map(t=>({...t,genre:"tete"}));
  if(etat.onglet==="coiffure") return cat.coiffures.map(c=>({...c,genre:"coiffure",garde:c.paires[etat.tete]&&!c.paires[etat.tete].passe}));
  return cat.pieces.filter(p=>p.onglet===etat.onglet).map(p=>({...p,genre:"piece"}));
}
function peindreFiltres(){
  const ex={}; for(const e of elements()) if(e.type_fr&&!(e.type_fr in ex)) ex[e.type_fr]=e;
  const types=Object.keys(ex);
  $("#filtres").innerHTML=types.length>1?[`<button data-t="" class="${etat.filtre?"":"actif"}">${t("tout")}</button>`,...types.map(t=>`<button data-t="${ech(t)}" class="${etat.filtre===t?"actif":""}">${ech(nomType(ex[t]))}</button>`)].join(""):"";
  $("#filtres").onclick=e=>{const t=e.target.dataset.t; if(t!==undefined){etat.filtre=t||null;peindre();}};
}
function peindreGrille(){
  let l=elements();
  if(etat.filtre) l=l.filter(e=>e.type_fr===etat.filtre);
  if(etat.recherche) { const mots=etat.recherche.split(/\s+/).filter(Boolean); l=l.filter(e=>{const x=texteRecherche(e); return mots.every(m=>x.includes(m));}); }
  const g=$("#grille"); g.textContent="";
  if(!l.length){ g.innerHTML=`<div class="rien">${t("rayon_vide")}<br><button data-biblio>${t("ouvrir_bibliotheque")}</button></div>`; g.querySelector("[data-biblio]").onclick=()=>montrer("bibliotheque"); return; }
  for(const it of l){
    const b=document.createElement("button"); b.className="tuile"+(estChoisi(it)?" choisie":""); b.title=it.garde?t("garde_coiffure"):bulleElement(it); b.dataset.id=it.id;
    const u=vignette(it);
    b.innerHTML=(u?`<img loading="lazy" src="${ech(u)}" alt="">`:`<div class="vide">…</div>`)+`<span>${ech(nomElement(it))}</span>`+(it.garde?`<b class="mise-en-garde">!</b>`:"")+(sourceDe(it)?`<i class="origine" title="${t("page_origine")}">↗</i>`:"");
    b.onclick=e=>{ if(e.target.classList.contains("origine")){ e.stopPropagation(); window.open(sourceDe(it).url,"_blank","noopener"); return; } choisir(it); }; g.append(b);
  }
}
const vignettes={};
const vignette=it=>it.vignette||vignettes[it.genre+it.id]||null;
const estChoisi=it=>it.genre==="tete"?etat.tete===it.id:it.genre==="coiffure"?etat.coiffure===it.id:etat.pieces.has(it.id);

async function choisir(it){
  if(it.genre==="tete"){etat.tete=it.id;peindre();await majTete();await majCoiffure();}
  else if(it.genre==="coiffure"){etat.coiffure=it.id;peindre();await majCoiffure();}
  else {basculer(it);peindre();await majPieces();}
  peindre();
}
function basculer(p,force){
  if(etat.pieces.has(p.id)&&!force){etat.pieces.delete(p.id);return;}
  for(const id of [...etat.pieces]){const q=parId[id];
    if((p.emplacement&&q.emplacement===p.emplacement)||(q.occupe||[]).includes(p.emplacement)||(p.occupe||[]).includes(q.emplacement)) etat.pieces.delete(id);
  }
  etat.pieces.add(p.id);
}
async function avecCharge(f){ const c=$("#charge"); c.hidden=false; try{await f();}catch(e){console.error(e);} finally{c.hidden=true;} }

async function majTete(){
  const g=++genTete, te=cat.tetes.find(x=>x.id===etat.tete);
  await avecCharge(async()=>{
    await vue.placer("tete",`/data/tetes/${encodeURIComponent(te.id)}/tete.glb`);
    if(g!==genTete) return;
    const ex=$("#expression"); if(ex&&vue.exprimer){ ex.innerHTML=`<option value="">${t("repos")}</option>`+vue.exprimer(null).map(n=>`<option value="${ech(n)}">${ech(nomExpression(n))}</option>`).join(""); ex.hidden=ex.options.length<2; }
    const dos=`/data/tetes/${encodeURIComponent(te.id)}/`;
    const [m,som,cache]=await Promise.all([vue.placer("corps","/data/corps/corps.glb",{peau:te.peau}),
      fetch(dos+"corps.bin").then(r=>r.ok?r.arrayBuffer():null), fetch(dos+"peau_cachee.bin").then(r=>r.ok?r.arrayBuffer():null)]);
    if(g!==genTete) return;
    // body skin as the graft leaves it under this face (preparer.raccord_peau)
    const n=m.d.pos.length, s=som&&som.byteLength===8*n?new Float32Array(som):null;
    corpsBase=s?s.subarray(0,n):m.d.pos;
    vue.sommets("corps",corpsBase,s?s.subarray(n):m.d.nor); poserCorps();
    peauTete=cache?Array.from(new Uint32Array(cache)):[]; masquerCorps();
    await vue.cartes("corps",dos+"cartes-corps.bin");
  });
}
async function majCoiffure(){
  const g=++genCoif;
  await avecCharge(async()=>{
    const id=encodeURIComponent(etat.coiffure);
    const [m,pos]=await Promise.all([vue.placer("coiffure",`/data/coiffures/${id}/coiffure.glb`,{cheveux:true}),
      fetch(`/data/coiffures/${id}/${encodeURIComponent(etat.tete)}.bin`).then(r=>r.ok?r.arrayBuffer():null)]);
    if(g!==genCoif) return;
    coifBase=pos?new Float32Array(pos):m.d.pos; coifDecal=null; poserCheveux();
    etat.imgCoiffure=m.prims[0]&&m.prims[0].image;
    const tex=etat.imgCoiffure&&vue.texture(etat.imgCoiffure); if(tex&&tex.moyenne) etat.ref=tex.moyenne;
    appliquerTeinte();
  });
  peindreFiche();
  if(etat.pieces.size) await majPieces();   // the hair falls on the outfit: to redo for this hairstyle
}
// body with its feet posed in a heeled shoe (offset returned by /api/combinaison, combinaison.poser_tenue)
function poserCorps(){
  if(!corpsBase) return;
  let p=corpsBase;
  if(corpsDecal&&corpsDecal.length===p.length){ p=new Float32Array(p.length); for(let i=0;i<p.length;i++) p[i]=corpsBase[i]+corpsDecal[i]; }
  vue.positions("corps",p);
}
// hair fitted on the head, moved over the outfit (offset returned by /api/combinaison)
function poserCheveux(){
  if(!coifBase) return;
  let p=coifBase;
  if(coifDecal&&coifDecal.length===p.length){ p=new Float32Array(p.length); for(let i=0;i<p.length;i++) p[i]=coifBase[i]+coifDecal[i]; }
  vue.positions("coiffure",p);
}
function appliquerTeinte(){
  const m=vue.modeles.get("coiffure"); if(m){m.teinte=etat.couleur?{...hex(etat.couleur),ref:etat.ref}:null; vue.sale=true;}
}
function majPastilles(){ for(const b of $("#pastilles").children) b.classList.toggle("actif",(COULEURS[+b.dataset.c]===etat.couleur)); }

async function majPieces(){
  const g=++genCombi, ids=[...etat.pieces], peau=new Map();
  document.body.dataset.couches="calcul";   // read by tests/partie.py
  const unionPeau=()=>{const f=[]; for(const a of peau.values()) for(const x of a) f.push(x); return f;};
  await avecCharge(async()=>{
    for(const c of [...vue.modeles.keys()]) if(c.startsWith("piece:")&&!etat.pieces.has(c.slice(6))) vue.retirer(c);
    await Promise.all(ids.map(async id=>{
      await vue.placer("piece:"+id,`/data/pieces/${encodeURIComponent(id)}/piece.glb`);
      vue.masque("piece:"+id,[]);
      const r=await fetch(`/data/pieces/${encodeURIComponent(id)}/peau_cachee.bin`);
      if(r.ok) peau.set(id,new Uint32Array(await r.arrayBuffer()));
    }));
    if(g!==genCombi) return;
    peauPieces=unionPeau(); masquerCorps();
  });
  etat.signal=[]; etat.notes=[]; etat.combiImpossible=false; peindreFiche();
  if(!ids.length){ coifDecal=null; poserCheveux(); corpsDecal=null; poserCorps(); if(g===genCombi) document.body.dataset.couches="pret"; return; }
  const rep=await fetch("/api/combinaison",{method:"POST",headers:JSON_H,body:JSON.stringify({pieces:ids,coiffure:etat.coiffure,tete:etat.tete})});
  if(g!==genCombi) return;
  if(!rep.ok){ etat.combiImpossible=true; peindreFiche(); document.body.dataset.couches="pret"; return; }
  const v=await rep.json();
  for(const id of ids){
    if(v.retirees[id]){ vue.masque("piece:"+id,true); peau.delete(id); continue; }
    await rebase(id,v.decalages[id]||null);
    vue.masque("piece:"+id,v.masques[id]?faces(v.masques[id]):[]);
  }
  coifDecal=v.coiffure?new Float32Array(Uint8Array.from(atob(v.coiffure),c=>c.charCodeAt(0)).buffer):null; poserCheveux();
  corpsDecal=v.corps?new Float32Array(Uint8Array.from(atob(v.corps),c=>c.charCodeAt(0)).buffer):null; poserCorps();
  peauPieces=unionPeau(); masquerCorps();
  etat.signal=v.signal; etat.notes=v.notes||[]; peindreFiche();
  if(g===genCombi) document.body.dataset.couches="pret";
}
function faces(b64){ const o=atob(b64),f=[]; for(let i=0;i<o.length;i++) if(o.charCodeAt(i)) f.push(i); return f; }
async function rebase(id,b64){
  const url=`/data/pieces/${encodeURIComponent(id)}/piece.glb`, d=await lireGLB(url);
  if(!vue.modeles.has("piece:"+id)) return;
  if(!b64){ vue.positions("piece:"+id,d.pos); return; }
  const o=new Float32Array(Uint8Array.from(atob(b64),c=>c.charCodeAt(0)).buffer), p=new Float32Array(d.pos.length);
  for(let i=0;i<p.length;i++) p[i]=d.pos[i]+o[i];
  vue.positions("piece:"+id,p);
}

const lienOrigine=it=>{ const s=it&&sourceDe(it); return s?`<a class="origine" href="${ech(s.url)}" target="_blank" rel="noopener noreferrer" title="${ech(t("page_origine")+" : "+s.nom+" "+t("par")+" "+s.auteur)}">↗</a>`:""; };
function peindreFiche(){
  const te=cat.tetes.find(x=>x.id===etat.tete), c=cat.coiffures.find(x=>x.id===etat.coiffure);
  const ids=[...etat.pieces].sort((a,b)=>parId[a].rang-parId[b].rang);
  $("#selection").innerHTML=`<li><b>${t("visage")}</b><span>${te?ech(nomElement(te)):"—"}</span>${lienOrigine(te)}</li><li><b>${t("coiffure")}</b><span>${c?ech(nomElement(c)):"—"}</span>${lienOrigine(c)}</li>`+
    ids.map(i=>`<li><b>${ech(nomType(parId[i]))}</b><span title="${ech(bulleElement(parId[i]))}">${ech(parId[i].etiquette?nomElement(parId[i]):parId[i].perso)}</span>${lienOrigine(parId[i])}<button data-r="${ech(i)}" title="${t("retirer")}">×</button></li>`).join("");
  const vus=new Map();   // the original models of the current choice, each once
  const fournit=new Map();   // model (page and name: two models may share a page) -> the slots of the choice it supplies
  for(const [r,it] of [[t("visage"),te],[t("coiffure"),c],...ids.map(i=>[nomType(parId[i]),parId[i]])]){ const s=it&&sourceDe(it);
    if(s){ const k=s.url+"|"+s.nom; s._cle=k; vus.set(k,s); fournit.set(k,[...(fournit.get(k)||[]),`${r.charAt(0).toUpperCase()+r.slice(1)} : ${nomElement(it)}`]); } }
  $("#origines").hidden=!vus.size;
  $("#origines-liste").innerHTML=[...vus.values()].map(s=>{ const l=s.licence||{}, o=lienSur(l.otherLicenseUrl), v=lienSur(l.licenseUrl);
    return `<li><a href="${ech(s.url)}" target="_blank" rel="noopener noreferrer">${ech(s.nom)} ↗</a> <small>${t("par")} ${ech(s.auteur)} · ${ech(s.site||"")}</small>`+
      `<ul class="fournit">${fournit.get(s._cle).map(x=>`<li>${ech(x)}</li>`).join("")}</ul>`+
      `<div class="${l.creditNotation==="required"?"credit":""}">${ech(licenceCourte(s))}</div>`+
      (v?`<div><a href="${ech(v)}" target="_blank" rel="noopener noreferrer">${ech(l.nom||t("lic_vrm"))} ↗</a></div>`:"")+
      (l.otherLicenseUrl||l.thirdPartyLicenses?`<div>${t("lic_autre")} : ${o?`<a href="${ech(o)}" target="_blank" rel="noopener noreferrer">${ech(o)}</a>`:ech(l.otherLicenseUrl||"")} ${ech(l.thirdPartyLicenses||"")}</div>`:"")+
      `<div class="releve">${ech(t("lic_releve",{d:s.releve||"?"}))}</div></li>`; }).join("");
  $("#selection").onclick=e=>{const r=e.target.dataset.r; if(r){etat.pieces.delete(r);peindre();majPieces();}};
  let al="";
  if(c&&c.paires[etat.tete]&&!c.paires[etat.tete].passe) al+=`<div class="alerte">${ech(t("coiffure_mal_posee",{e:c.paires[etat.tete].echecs.map(nomEchec).join(", ")}))}</div>`;
  const nomPiece=(id,nom)=>parId[id]?nomType(parId[id]):srv(nom);
  for(const s of etat.signal) al+=`<div class="alerte">${ech(t("traverse",{a:nomPiece(s.dessous,s.nom_dessous),b:nomPiece(s.dessus,s.nom_dessus),p:(100*s.part).toFixed(1),mm:s.pire_mm}))}</div>`;
  for(const n of etat.notes||[]) al+=`<div class="note">${ech(srv(n))}</div>`;
  if(etat.combiImpossible) al+=`<div class="alerte">${t("couches_echec")}</div>`;
  $("#alertes").innerHTML=al;
}
const nomEchec=e=>ECHECS[e]?ECHECS[e][L()]:e;

// ---------- Assemble, then download ----------
async function assembler(){
  const sel={tete:etat.tete,coiffure:etat.coiffure,pieces:[...etat.pieces],teinte:etat.couleur?{...hex(etat.couleur),ref:etat.ref}:null,nom:$("#nom").value.trim()||t("sans_nom")};
  const b=$("#assembler"); b.disabled=true; b.textContent=t("assemblage");
  $("#travail").hidden=false; $("#resultat").innerHTML=""; $("#etape").textContent=t("attente"); $("#barre div").style.width="3%";
  try{ const {id}=await json("/api/assembler",{method:"POST",headers:JSON_H,body:JSON.stringify(sel)}); etat.travail=id; suivre(id); }
  catch(e){ $("#etape").textContent=t("echec")+e.message; finAss(); }
}
const ETAPES={attente:["Blender…","Blender…",3],coiffure:["Fitting the hair…","Pose de la coiffure…",12],greffe:["Grafting the head…","Greffe de la tête…",24],pieces:["Outfit parts…","Pièces de la tenue…",50],couleur:["Hair colour…","Couleur des cheveux…",62],controles:["Checks…","Contrôles…",74],squelette:["Skeleton…","Squelette…",79],vrm:["VRM export…","Export VRM…",83],rendu:["Rendering the views…","Rendu des vues…",90],fin:["Done","Terminé",100]};
// download links of an assembled character: one per delivered format
const sorties=(id,formats)=>`<div class="sorties">${(formats||[]).map(f=>`<a download href="/api/assemblages/${encodeURIComponent(id)}/${encodeURIComponent(f)}">${t("telecharger")} <b>${ech(f.toUpperCase())}</b></a>`).join("")}<button data-ouvrir="${ech(id)}">${t("ouvrir_dossier")}</button></div>`;
document.addEventListener("click",e=>{ const id=e.target.dataset&&e.target.dataset.ouvrir; if(id) fetch("/api/ouvrir",{method:"POST",headers:JSON_H,body:JSON.stringify({id})}); });
async function suivre(id){
  let e;
  try{ e=await json("/api/travaux/"+id); }
  catch(x){ $("#etape").textContent=t("serveur_muet")+x.message; finAss(); return; }
  if(e.etape==="erreur"){ $("#etape").textContent=t("echec")+srv(e.erreur); finAss(); return; }
  const [en,fr,p]=ETAPES[e.etape]||[e.etape,e.etape,10];
  $("#etape").textContent=(LANGUE==="fr"?fr:en)+(e.etape==="pieces"&&e.pieces_total?` ${e.piece_courante||0}/${e.pieces_total}`:"");
  $("#barre div").style.width=p+"%";
  if(!e.fini){ setTimeout(()=>suivre(id),1500); return; }
  let h=`<div>${e.ok?t("controles_ok"):t("controles_ko")} · ${ech(srv(e.resume))}</div>`;
  if(e.echecs&&e.echecs.length) h+=`<div class="alerte">${e.echecs.map(x=>ech(srv(x))).join("<br>")}</div>`;
  for(const c of e.traverse||[]) h+=`<div class="alerte">${ech(t("ressort",{a:srv(c.dessous),b:srv(c.dessus),p:(100*c.sortis).toFixed(1),mm:c.pire_mm.toFixed(1)}))}</div>`;
  h+=e.planche===false?`<div class="alerte">${t("images_absentes")}${ech(e.planche_note)}</div>`:`<a href="/data/assemblages/${id}/planche.png" target="_blank"><img src="/data/assemblages/${id}/planche.png" alt=""></a>`;
  h+=sorties(id,e.formats||(e.vrm?["pmx","vrm"]:["pmx"]));
  if(!e.vrm&&e.vrm_note) h+=`<div class="alerte">${t("vrm_absent")}${ech(srv(e.vrm_note))}</div>`;
  $("#resultat").innerHTML=h;
  finAss();
}
function finAss(){ const b=$("#assembler"); b.disabled=false; b.textContent=t("assembler"); }

// ---------- Render studio: each setting of the viewer (REGLAGES_RENDU) has its control; a render is a JSON file ----------
const REGLAGES=V.REGLAGES_RENDU||[];
const studioPret=()=>!vue.logiciel&&REGLAGES.length&&typeof vue.regler==="function";
const lire=(o,cle)=>cle.split(".").reduce((a,k)=>a==null?a:a[k],o);
const gam=(x,lin)=>lin?(x<=0.0031308?12.92*x:1.055*Math.pow(x,1/2.4)-0.055):x, dgam=(x,lin)=>lin?(x<=0.04045?x/12.92:Math.pow((x+0.055)/1.055,2.4)):x;
const versHex=(c,lin)=>"#"+[0,1,2].map(i=>Math.round(255*Math.min(1,Math.max(0,gam(c[i]||0,lin)))).toString(16).padStart(2,"0")).join("");
const deHex=(h,lin)=>[1,3,5].map(i=>+dgam(parseInt(h.slice(i,i+2),16)/255,lin).toFixed(4));
let prereglages=[], nomRendu="";
async function demarrerStudio(){
  try{ prereglages=await json("/page/rendus/index.json"); }catch(e){ prereglages=[]; }
  construireStudio();
  if(!studioPret()) return;
  const q=new URLSearchParams(location.search).get("rendu"), sauve=garde("chimera.rendu","");
  try{
    if(q&&prereglages.some(p=>p.id===q)) await choisirPrereglage(q);
    else if(sauve){ const o=vue.chargerRendu(JSON.parse(sauve)); nomRendu=o.nom||""; peindreStudio(); }
  }catch(e){ console.warn("rendu",e); }
}
// The studio in French: the settings carry their English name in visionneuse.js (REGLAGES_RENDU), the French is here, by key.
const FR_GROUPES={"Light":"Lumière","Shadows":"Ombres","Skin":"Peau","Materials":"Matières","Outline":"Trait","Background and floor":"Fond et sol","Finish":"Fin d'image"};
const FR_RENDU={
  "lumiere.angle":["Angle","Côté d'où vient la lumière, en degrés, compté depuis la caméra (0 : de face ; 90 : de côté ; 180 : à contre-jour). Elle tourne avec la vue."],
  "lumiere.hauteur":["Hauteur","Hauteur de la lumière au-dessus de l'horizon, en degrés (négatif : par en dessous)."],
  "lumiere.couleur":["Couleur","Couleur de la lumière principale, sur les parties éclairées."],
  "lumiere.force":["Force","Force de la lumière principale : écart entre le clair et l'ombre."],
  "lumiere.contour":["Couleur du contour","Couleur de la lumière de contour, venue de derrière, qui détache la silhouette du fond."],
  "lumiere.forceContour":["Lumière de contour","Force de la lumière de contour (0 : aucune)."],
  "lumiere.ciel":["Ciel dans l'ombre","Lumière du ciel renvoyée dans l'ombre, sur les surfaces tournées vers le haut."],
  "lumiere.terre":["Sol dans l'ombre","Lumière du sol renvoyée dans l'ombre, sur les surfaces tournées vers le bas."],
  "ombre.seuil":["Seuil","Où passe la limite de l'ombre sur les vêtements et les cheveux : plus haut, plus d'ombre."],
  "ombre.douceur":["Douceur","Largeur du passage de la lumière à l'ombre (petite : limite nette)."],
  "ombre.teinte":["Teinte des vêtements","Couleur par laquelle la couleur d'un vêtement est multipliée dans l'ombre."],
  "ombre.teinteCheveux":["Teinte des cheveux","Couleur par laquelle la couleur des cheveux est multipliée dans l'ombre."],
  "ombre.saturation":["Saturation","Couleur plus vive (au-dessus de 1) ou plus grise dans l'ombre des vêtements et des cheveux."],
  "ombre.lisere":["Liseré","Bande plus saturée le long de la limite d'ombre, comme la lumière qui traverse la peau."],
  "ombre.creux":["Creux","Ombre des creux (aisselles, plis, sous les mèches), tirée des cartes d'ouverture."],
  "ombre.portees":["Ombres portées","Ce qui est devant la lumière fait de l'ombre (cheveux sur le visage, bras sur le corps, personnage sur le sol)."],
  "ombre.forcePortees":["Force des ombres portées","Noirceur des ombres portées (1 : aussi sombres que l'ombre propre)."],
  "ombre.contact":["Ombre de contact","Ce qui est posé tout près d'une surface l'assombrit : col, ceinture, mèche sur l'épaule (0 : aucune)."],
  "peau.seuil":["Seuil","Où passe la limite de l'ombre sur la peau et le visage."],
  "peau.douceur":["Douceur","Largeur du passage de la lumière à l'ombre sur la peau."],
  "peau.teinte":["Teinte de l'ombre","Couleur par laquelle la peau est multipliée dans l'ombre (les yeux et la bouche en suivent une version plus froide)."],
  "peau.transmission":["Lumière qui traverse","À contre-jour, la lumière traverse la peau fine (oreilles, doigts) et les mèches fines : lueur chaude."],
  "peau.rose":["Rose du visage","Rose des joues, du nez et des lèvres, et petit reflet de la lèvre."],
  "peau.ombreYeux":["Ombre sur les yeux","Bande sombre sur les yeux que portent certaines expressions (0 : jamais)."],
  "matiere.vetements":["Matières des vêtements","Reflets selon la matière lue dans le nom du vêtement (tissu, tricot, cuir, collant, soie, verre, métal). Éteint : vêtements mats."],
  "matiere.cheveux":["Reflets des cheveux","Bande de reflet en travers des mèches, racines plus sombres et pointes plus claires (0 : cheveux mats)."],
  "trait.actif":["Trait","Trait au bord des formes, de la couleur de la surface en plus sombre."],
  "trait.epaisseur":["Épaisseur","Épaisseur du trait (il s'affine de loin)."],
  "trait.opacite":["Opacité","Opacité du trait."],
  "trait.couleur":["Couleur","Couleur fixe du trait, mêlée à la couleur tirée de la surface selon « Part de la couleur »."],
  "trait.partCouleur":["Part de la couleur","0 : le trait prend la couleur de la surface, assombrie ; 1 : la couleur fixe partout."],
  "fond.centre":["Fond au centre","Couleur du fond derrière le personnage."],
  "fond.bord":["Fond au bord","Couleur du fond vers les bords de l'image."],
  "sol.actif":["Sol","Un rond de lumière sous les pieds, qui reçoit l'ombre du personnage."],
  "sol.lumiere":["Sol éclairé","Couleur du sol au centre du rond, en pleine lumière."],
  "sol.ombre":["Sol dans l'ombre","Couleur de l'ombre du personnage sur le sol."],
  "image.exposition":["Exposition","Clarté de toute l'image (sous 1, les clairs restent sous le blanc et le halo fait le reste)."],
  "image.saturation":["Saturation","Couleur de toute l'image : 1 inchangée, 0 noir et blanc."],
  "image.halo":["Halo","Lueur autour des parties claires (0 : aucune)."],
  "image.flou":["Profondeur de champ","Flou de ce qui est loin du sujet, surtout en gros plan (0 : tout net)."],
  "image.vignette":["Vignette","Assombrissement des coins de l'image."],
  "image.grain":["Grain","Grain léger sur toute l'image."],
};
const nomReglage=r=>LANGUE==="fr"&&FR_RENDU[r.cle]?FR_RENDU[r.cle][0]:r.nom, aideReglage=r=>LANGUE==="fr"&&FR_RENDU[r.cle]?FR_RENDU[r.cle][1]:r.aide;
const nomGroupe=g=>LANGUE==="fr"&&FR_GROUPES[g]||g, nomPrereglage=p=>PREREGLAGES[p.id]?PREREGLAGES[p.id][L()]:p.nom;
function construireStudio(){
  const z=$("#reglages"), avant=$("#prereglage").value; z.textContent="";
  $("#prereglage").innerHTML=`<option value="">${t("personnalise")}</option>`+prereglages.map(p=>`<option value="${ech(p.id)}">${ech(nomPrereglage(p))}</option>`).join("");
  $("#prereglage").value=avant;
  if(!studioPret()){ $("#rendu-note").textContent=t("simplifie"); for(const b of ["#rendu-importer","#rendu-exporter","#rendu-defaut","#prereglage"]) $(b).disabled=true; return; }
  const groupes=new Map(); for(const r of REGLAGES){ if(!groupes.has(r.groupe)) groupes.set(r.groupe,[]); groupes.get(r.groupe).push(r); }
  let premier=true;
  for(const [g,l] of groupes){
    const d=document.createElement("details"); d.open=premier; premier=false;
    d.innerHTML=`<summary>${ech(nomGroupe(g))}</summary><div class="corps">${l.map(r=>{ const id="r-"+r.cle.replace(/\W/g,"-"), a=r.aide?` title="${ech(aideReglage(r))}"`:"";
      if(r.type==="couleur") return `<div class="reglage couleur"${a}><label for="${id}">${ech(nomReglage(r))}</label><input type="color" id="${id}" data-cle="${ech(r.cle)}"></div>`;
      if(r.type==="bascule") return `<div class="reglage bascule"${a}><label for="${id}">${ech(nomReglage(r))}</label><input type="checkbox" id="${id}" data-cle="${ech(r.cle)}"></div>`;
      if(r.type==="choix") return `<div class="reglage choix"${a}><label for="${id}">${ech(nomReglage(r))}</label><select id="${id}" data-cle="${ech(r.cle)}">${(r.options||[]).map(o=>`<option>${ech(o)}</option>`).join("")}</select></div>`;
      return `<div class="reglage"${a}><label for="${id}">${ech(nomReglage(r))}</label><output></output><input type="range" id="${id}" data-cle="${ech(r.cle)}" min="${r.min}" max="${r.max}" step="${r.pas||0.01}"></div>`; }).join("")}</div>`;
    z.append(d);
  }
  z.oninput=e=>{ const cle=e.target.dataset.cle, r=REGLAGES.find(x=>x.cle===cle); if(!r) return;
    const v=r.type==="couleur"?deHex(e.target.value,r.espace==="lineaire"):r.type==="bascule"?e.target.checked:r.type==="choix"?e.target.value:+e.target.value;
    vue.regler(cle,v); if(r.type==="nombre"||!r.type) e.target.previousElementSibling.textContent=(+e.target.value).toFixed(2);
    $("#prereglage").value=""; noter("chimera.prereglage",""); memoriserRendu(); };
  $("#prereglage").onchange=e=>{ if(e.target.value) choisirPrereglage(e.target.value).catch(x=>dire(t("echec")+x.message)); };
  $("#rendu-importer").onclick=()=>$("#rendu-fichier").click();
  $("#rendu-fichier").onchange=async e=>{ const f=e.target.files[0]; e.target.value=""; if(!f) return;
    try{ const o=vue.chargerRendu(JSON.parse(await f.text())); nomRendu=o.nom||f.name.replace(/\.json$/i,""); $("#prereglage").value=""; noter("chimera.prereglage",""); peindreStudio(); memoriserRendu(); dire(t("rendu_importe")+nomRendu); }
    catch(x){ dire(t("rendu_refuse")+srv(x.message)); } };
  $("#rendu-exporter").onclick=()=>{ const nom=nomRendu||t("personnalise"), fichier=(nom.normalize("NFD").replace(/[^\w -]/g,"").trim().replace(/\s+/g,"-").toLowerCase()||"rendu")+".chimera-rendu.json";
    const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([JSON.stringify(vue.exporterRendu(nom),null,1)],{type:"application/json"})); a.download=fichier; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),2000); dire(t("rendu_exporte")+fichier); };
  $("#rendu-defaut").onclick=()=>{ vue.chargerRendu(V.RENDU_DEFAUT); nomRendu=V.RENDU_DEFAUT.nom||""; const p=prereglages.find(x=>x.nom===nomRendu); $("#prereglage").value=p?p.id:""; noter("chimera.prereglage",p?p.id:""); peindreStudio(); memoriserRendu(); };
  const d0=prereglages.find(x=>x.nom===(V.RENDU_DEFAUT||{}).nom); $("#prereglage").value=garde("chimera.rendu","")?garde("chimera.prereglage",""):(d0?d0.id:"");
  peindreStudio();
}
const dire=m=>{ $("#rendu-note").textContent=m; };
const memoriserRendu=()=>noter("chimera.rendu",JSON.stringify(vue.exporterRendu(nomRendu)));
async function choisirPrereglage(id){
  const o=vue.chargerRendu(await json(`/page/rendus/${encodeURIComponent(id)}.json`)); nomRendu=o.nom||id;
  $("#prereglage").value=id; noter("chimera.prereglage",id); peindreStudio(); memoriserRendu(); dire("");
}
function peindreStudio(){
  if(!studioPret()) return; const o=vue.rendu;
  for(const e of $("#reglages").querySelectorAll("[data-cle]")){ const r=REGLAGES.find(x=>x.cle===e.dataset.cle), v=lire(o,r.cle); if(v===undefined) continue;
    if(r.type==="couleur") e.value=versHex(v,r.espace==="lineaire"); else if(r.type==="bascule") e.checked=!!v; else { e.value=v; if(r.type!=="choix") e.previousElementSibling.textContent=(+v).toFixed(2); } }
}

// ---------- Library: import models, extract their parts ----------
let formats=null, xEnCours=false, xSuivi=null, xId=null, xVersion=null;
const nb=x=>Array.isArray(x)?x.length:(x||0);
function brancherBibliotheque(){
  const d=$("#depot"), f=$("#fichier-modele");
  f.onchange=()=>{ importer([...f.files]); f.value=""; };
  d.ondragover=e=>{ e.preventDefault(); d.classList.add("dessus"); }; d.ondragleave=()=>d.classList.remove("dessus");
  d.ondrop=e=>{ e.preventDefault(); d.classList.remove("dessus"); importer([...e.dataTransfer.files]); };
  $("#extraire").onclick=extraire; $("#annuler").onclick=annuler;
}
async function ouvrirBibliotheque(){
  $("#manque").hidden=!vide(); $("#manque").textContent=t("manque");
  try{
    formats=formats||await json("/api/formats");
    const li=x=>`<li>${x.etat?`<span class="etat ${x.etat==="eprouve"?"":"partiel"}">${t(x.etat==="eprouve"?"eprouve":"partiel")}</span>`:""}<b>${ech(srv(x.nom))}</b>${x.extensions?` <code>${ech(x.extensions.join(" "))}</code>`:""}${x.note?`<small>${ech(srv(x.note))}</small>`:""}</li>`;
    $("#formats-entree").innerHTML=formats.entree.map(li).join(""); $("#liste-squelettes").innerHTML=formats.squelettes.map(li).join("");
    const ext=[...new Set(formats.entree.flatMap(x=>x.extensions||[]))]; $("#fichier-modele").accept=ext.join(","); $("#depot-formats").textContent=ext.join("  ·  ");
    const ja=formats.jeu_actif||{}; if(ja.import===false){ $("#fichier-modele").disabled=true; $("#depot").style.opacity=.5; $("#import-etat").innerHTML=`<div class="alerte">${ech(srv(ja.raison))}</div>`; }
  }catch(e){ $("#formats-entree").innerHTML=`<li>${ech(e.message)}</li>`; }
  await peindreModeles();
  suivreExtraction();
}
async function peindreModeles(){
  let l=[]; try{ l=await json("/api/modeles"); }catch(e){ $("#modeles").innerHTML=`<div class="rien">${ech(e.message)}</div>`; return; }
  if(l.modeles) l=l.modeles;
  $("#modeles").innerHTML=l.length?l.map(m=>{ const x=m.extrait||{}, r=(m.refus||[]).length;
    return `<label title="${ech((m.refus||[]).map(y=>`${srv(y.nom)}${LANGUE==="fr"?" :":":"} ${srv(y.raison)}`).join("\n"))}"><input type="checkbox" value="${ech(m.id)}" ${x.visage&&x.coiffure&&x.pieces?"":"checked"}><b>${ech(m.nom||m.id)}</b><span class="puces">`+
      `<span class="puce ${x.visage?"oui":""}">${t("visage")}</span><span class="puce ${x.coiffure?"oui":""}">${t("coiffure")}</span><span class="puce ${x.pieces?"oui":""}">${x.pieces||0} ${t("pieces")}</span>${r?`<span class="puce non">${r} ${t("refus")}</span>`:""}</span></label>`; }).join(""):`<div class="rien">${t("aucun_modele")}</div>`;
}
async function importer(fichiers){
  const z=$("#import-etat");
  for(const f of fichiers){
    z.innerHTML=`<pre>${ech(t("envoi")+f.name)}…</pre>`;
    try{
      let r=await json("/api/importer?nom="+encodeURIComponent(f.name),{method:"PUT",headers:{"Content-Type":"application/octet-stream"},body:f});
      while(r&&r.id&&!r.modele){ const e=await json("/api/travaux/"+r.id); if(e.etape==="erreur") throw new Error(srv(e.erreur)); if(e.fini){ r=e; break; } z.innerHTML=`<pre>${ech(t("envoi")+f.name)} — ${ech(srv(e.message||e.etape||""))}</pre>`; await new Promise(s=>setTimeout(s,1200)); }
      const nm=(r.modeles||[]).length, ne=(r.ecartes||[]).length; z.innerHTML=`<pre>${ech((nm>1?t("importes",{n:nm}):t("importe")+((r.modele&&(r.modele.nom||r.modele.id))||f.name))+(ne&&nm>1?" · "+t("ecartes",{n:ne}):""))}</pre>`;
    }catch(e){ z.innerHTML=`<div class="alerte">${ech(f.name)} : ${ech(e.message)}</div>`; }
    await peindreModeles();
  }
}
// The extraction runs on the server: its state comes from GET /api/extraction (on load, on opening this view, when
// the tab comes back), never from this page, so a reload or another tab finds the bar where it is.
function boutonExtraire(enCours){
  xEnCours=enCours; const b=$("#extraire"), a=$("#annuler");
  b.disabled=enCours; b.classList.toggle("en-cours",enCours); b.textContent=t(enCours?"extraction_en_cours":"extraire");
  a.hidden=!enCours; if(!enCours){ a.disabled=false; a.textContent=t("annuler"); } else if(!a.disabled) a.textContent=t("annuler");
  $("#extraction-serveur").hidden=!enCours;
}
function messageExtraction(e){
  if(e.annulation&&!e.fini) return t("annulation");
  const pas=k=>TXT["pas_"+k]?t("pas_"+k):(k||"");
  if(e.modele&&e.modele.nom) return t("modele_k",{k:e.modele.rang,n:e.modele.total,nom:e.modele.nom,e:pas(e.modele.pas)});
  if(e.lot&&e.pas){ const l=e.lot; return l.de===l.a?t("modele_k",{k:l.de,n:l.total,nom:(l.noms||[""])[0],e:pas(e.pas)}):t("modeles_k",{k:l.de,m:l.a,n:l.total,e:pas(e.pas)}); }
  if(e.pas==="pivot") return pas("pivot");
  return e.message?srv(e.message):t("attente");
}
function peindreExtraction(e){
  $("#extraction").hidden=false;
  const et=$("#extraction-etape"), jo=$("#extraction-journal");
  if(e.journal){ jo.textContent=Array.isArray(e.journal)?e.journal.join("\n"):e.journal; jo.scrollTop=jo.scrollHeight; }
  if(!e.fini){ boutonExtraire(true); et.textContent=messageExtraction(e); $("#extraction-barre div").style.width=Math.max(2,Math.round(100*(e.part||0)))+"%";
    if(e.annulation){ $("#annuler").disabled=true; $("#annuler").textContent=t("annulation"); }
    return; }
  boutonExtraire(false);
  if(e.annulee){ et.textContent=t("extraction_annulee"); return; }
  if(e.etape==="erreur"||e.ok===false){ et.innerHTML=`<div class="alerte">${ech(t("echec")+srv(e.erreur))}</div>`;return; }
  et.textContent=t("extraction_finie")+(e.bilan?` ${nb(e.bilan.visages)} ${t("visages").toLowerCase()}, ${nb(e.bilan.coiffures)} ${t("coiffures").toLowerCase()}, ${nb(e.bilan.pieces)} ${t("pieces")}.`:"");
  $("#extraction-barre div").style.width="100%";
}
async function suivreExtraction(){
  if(xSuivi) return xSuivi;
  xSuivi=(async()=>{
    for(;;){
      let e;
      try{ e=await json("/api/extraction"); }catch(err){ if(xEnCours) $("#extraction-etape").innerHTML=`<div class="alerte">${ech(t("serveur_muet")+err.message)}</div>`; await new Promise(s=>setTimeout(s,3000)); if(!xEnCours) return; continue; }
      if(!e.id){   // nothing running: the one this page followed has just finished
        if(xId){ const f=await json("/api/travaux/"+xId).catch(()=>null); xId=null; if(f) peindreExtraction(f); await majCatalogue(); await peindreModeles(); }
        boutonExtraire(false); return;
      }
      if(xId!==e.id){ xId=e.id; xVersion=null; }
      peindreExtraction(e);
      if(xVersion!==null&&e.version!==xVersion){ await majCatalogue(); await peindreModeles(); }   // a model became usable
      xVersion=e.version;
      await new Promise(s=>setTimeout(s,1500));
    }
  })().catch(e=>console.error(e)).finally(()=>{ xSuivi=null; });
  return xSuivi;
}
async function extraire(){
  if(xEnCours) return;
  const modeles=[...$("#modeles").querySelectorAll("input:checked")].map(i=>i.value), quoi=[...$("#quoi").querySelectorAll("input:checked")].map(i=>i.value);
  const z=$("#extraction"), et=$("#extraction-etape"), jo=$("#extraction-journal"); z.hidden=false; jo.textContent="";
  if(!modeles.length||!quoi.length){ et.textContent=t("choisir_modeles"); return; }
  boutonExtraire(true); et.textContent=t("attente"); $("#extraction-barre div").style.width="2%";
  try{
    const {id}=await json("/api/extraire",{method:"POST",headers:JSON_H,body:JSON.stringify({modeles,quoi})});
    xId=id; xVersion=null;
  }catch(e){ boutonExtraire(false); et.innerHTML=`<div class="alerte">${ech(t("echec")+e.message)}</div>`; }
  suivreExtraction();   // also when the server refused because one is already running: follow that one
}
async function annuler(){
  const a=$("#annuler"); a.disabled=true; a.textContent=t("annulation");
  try{ await json("/api/extraction/annuler",{method:"POST",headers:JSON_H,body:"{}"}); }
  catch(e){ a.disabled=false; a.textContent=t("annuler"); $("#extraction-etape").innerHTML=`<div class="alerte">${ech(t("echec")+e.message)}</div>`; }
  suivreExtraction();
}

// ---------- Assembled characters ----------
async function ouvrirPersonnages(){
  const g=$("#galerie");
  try{
    formats=formats||await json("/api/formats");
    $("#formats-sortie").innerHTML=formats.sortie.map(x=>`<li><b>${ech(srv(x.nom))}</b>${x.note?`<small>${ech(srv(x.note))}</small>`:""}</li>`).join("");
    let l=await json("/api/assemblages"); if(l.assemblages) l=l.assemblages;
    g.innerHTML=l.length?l.map(a=>`<article class="fiche-perso">${a.planche?`<a href="/data/assemblages/${encodeURIComponent(a.id)}/planche.png" target="_blank"><img loading="lazy" src="/data/assemblages/${encodeURIComponent(a.id)}/planche.png" alt=""></a>`:`<div class="vide">—</div>`}
      <div class="texte"><b>${ech(a.nom||a.id)}</b><small>${ech((a.date||a.id).replace("T"," "))}</small>${a.fini===false||a.erreur?`<span class="puce non">${ech(a.erreur?srv(a.erreur):t("inacheve"))}</span>`:""}${sorties(a.id,a.formats)}</div></article>`).join(""):`<div class="rien">${t("aucun_personnage")}</div>`;
  }catch(e){ g.innerHTML=`<div class="rien">${ech(e.message)}</div>`; }
}

// Missing thumbnails: drawn once, the server keeps them.
// One pass at a time (the catalogue can change during an extraction); a call during a pass asks for another one.
let vignettesPasse=null, vignettesEncore=false, visVignette=null;
function genererVignettes(){
  if(vignettesPasse){ vignettesEncore=true; return vignettesPasse; }
  vignettesPasse=(async()=>{ do{ vignettesEncore=false; await passeVignettes(); }while(vignettesEncore); })()
    .catch(e=>console.warn("vignettes",e)).finally(()=>{ vignettesPasse=null; });
  return vignettesPasse;
}
async function passeVignettes(){
  const liste=[...cat.tetes.map(x=>["tete",x,`/data/tetes/${encodeURIComponent(x.id)}/tete.glb`,"tetes"]),
    ...cat.coiffures.map(c=>["coiffure",c,`/data/coiffures/${encodeURIComponent(c.id)}/coiffure.glb`,"coiffures"]),
    ...cat.pieces.map(p=>["piece",p,`/data/pieces/${encodeURIComponent(p.id)}/piece.glb`,"pieces"])].filter(x=>!x[1].vignette&&!vignettes[x[0]+x[1].id]);
  if(!liste.length) return;
  const cv=$("#atelier-vignette");
  if(!visVignette){ cv.hidden=false; visVignette=creerVisionneuse(cv,{conserver:true}); visVignette.dpr=1; cv.hidden=true;
    if(visVignette.chargerRendu) try{ visVignette.chargerRendu(await json("/page/rendus/simple.json")); }catch(e){} }   // thumbnails: the plainest render, no backdrop
  const v=visVignette;
  for(const [g,it,url,dos] of liste){
    try{
      v.modeles.forEach((_,k)=>v.retirer(k));
      const m=await v.placer("x",url), d=m.d, lo=[1e9,1e9,1e9], hi=[-1e9,-1e9,-1e9];
      for(let i=0;i<d.pos.length;i+=3) for(let j=0;j<3;j++){lo[j]=Math.min(lo[j],d.pos[i+j]);hi[j]=Math.max(hi[j],d.pos[i+j]);}
      const c=lo.map((a,j)=>(a+hi[j])/2), r=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2])*0.5;
      v.cam={az:0.3,el:0.05,dist:r/Math.tan(0.275)*1.15,cible:c};
      // give the textures time to arrive
      m.prims.forEach(p=>v.texture(p.image));
      await new Promise(res=>{let n=0;const f=()=>{n++; if(m.prims.every(p=>{const x=v.texture(p.image);return !x||x.pret;})||n>40) res(); else setTimeout(f,100);}; f();});
      v.dessiner();
      const blob=await new Promise(res=>cv.toBlob(res,"image/png"));
      const url2=URL.createObjectURL(blob); vignettes[g+it.id]=url2; it.vignette=url2;
      fetch(`/api/vignette/${dos}/${encodeURIComponent(it.id)}`,{method:"PUT",body:blob});
      if((etat.onglet==="visage"&&g==="tete")||(etat.onglet==="coiffure"&&g==="coiffure")||(g==="piece"&&it.onglet===etat.onglet)){
        const b=[...$("#grille").children].find(x=>x.dataset.id===it.id);
        if(b&&b.querySelector(".vide")) b.querySelector(".vide").outerHTML=`<img src="${url2}" alt="">`;
      }
    }catch(e){console.warn("vignette",it.id,e);}
  }
}
// For the regard.mjs tool: drive the page without a mouse (choices, camera, photos).
window.atelier={vue,etat,cat:()=>cat,majTete,majCoiffure,majPieces,montrer};
demarrer().catch(e=>{ console.error(e); document.body.insertAdjacentHTML("beforeend",`<p class="bandeau-note" style="position:fixed;left:20px;right:20px;bottom:20px">${ech(e.message)}</p>`); });
