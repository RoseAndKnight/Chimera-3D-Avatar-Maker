// Chimera 3D Avatar Maker — the page. Three views: Create (choose, look, tune the render, assemble),
// Library (import models, extract their parts), Characters (download what was assembled).
import * as V from "/page/viewer.js";
const { createViewer, readGLB } = V;

const $=s=>document.querySelector(s);
const JSON_H={"Content-Type":"application/json"};
// Names come from model files: they are shown as text, never as HTML.
const ech=t=>String(t??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const recall=(key,fallback)=>{ try{ return localStorage.getItem(key)??fallback; }catch(e){ return fallback; } };
const store=(key,v)=>{ try{ localStorage.setItem(key,v); }catch(e){} };
// Keys written by earlier versions under French names: moved once to their English names.
const V1_NAMES={keys:[["chimera.langue","chimera.language"],["chimera.nom","chimera.name"],["chimera.rendu","chimera.render"],["chimera.prereglage","chimera.preset"]],renderParam:"rendu"};   // fr-ok: names of earlier versions (localStorage keys, ?rendu= links), read once to move them
for(const [o,n] of V1_NAMES.keys){ try{ const v=localStorage.getItem(o); if(v!==null&&localStorage.getItem(n)===null) localStorage.setItem(n,v); localStorage.removeItem(o); }catch(e){} }

// Interface texts: [English, French]. The server speaks English; its closed labels are translated below (SERVER).
// french-labels: begin (the French texts of the page, next to their English ones)
const TXT={
  view_create:["Create","Créer"], view_library:["Library","Bibliothèque"], view_characters:["Characters","Personnages"],
  face:["Face","Visage"], hairstyle:["Hair","Coiffure"], top:["Top","Haut"], bottom:["Bottom","Bas"], shoes:["Shoes","Chaussures"], accessories:["Extras","Accessoires"],
  search:["Search…","Chercher…"], color:["Colour","Couleur"], color_origin:["Original colour","Couleur d'origine"], all:["all","tout"],
  frame_body:["Body","Corps"], frame_face:["Face","Visage"], spin:["Turn","Tourner"], rest:["Neutral","Repos"],
  loading:["Loading…","Chargement…"],
  simplified:["Simplified preview: this browser has no WebGL2, so the light and shadow settings are off.","Aperçu simplifié : ce navigateur n'a pas WebGL2, les réglages de lumière et d'ombres sont éteints."],
  panel_character:["Character","Personnage"], panel_render:["Light & shadows","Lumière et ombres"],
  character_name:["Character name","Nom du personnage"], without_name:["My character","Mon personnage"],
  assemble:["Assemble","Assembler"], assembly:["Assembling…","Assemblage…"],
  help_assemble:["Builds one finished model from these parts, then offers it for download.","Fabrique un seul modèle fini à partir de ces pièces, puis le propose au téléchargement."],
  unfinished:["unfinished","inachevé"],
  refused:["Rejected parts","Pièces refusées"], remove:["Remove","Retirer"],
  origins:["Original models and licences","Modèles d'origine et licences"], by:["by","par"], page_origin:["Original model page","Page du modèle d'origine"],
  license_read:["Terms as read on {d}; the page of the model has today's.","Conditions lues le {d} ; la page du modèle donne celles d'aujourd'hui."],
  license_credit_yes:["credit required","crédit obligatoire"], license_credit_no:["credit not required","crédit non requis"],
  license_commercial:["commercial use allowed","usage commercial permis"], license_personal:["personal use only","usage personnel seulement"],
  license_edit:["modification and redistribution allowed","modification et redistribution permises"], license_edit_no:["modification restricted","modification restreinte"],
  license_other:["Other terms","Autres conditions"], license_vrm:["VRM licence","Licence VRM"],
  hairstyle_warning:["Does not sit cleanly on this face","Ne se pose pas proprement sur ce visage"],
  shelf_empty:["Nothing here yet.","Rien ici pour l'instant."], open_library:["Open the library","Ouvrir la bibliothèque"],
  waiting:["Waiting…","En attente…"], failure:["Failed: ","Échec : "], server_silent:["The local server stopped answering: ","Le serveur local ne répond plus : "],
  checks_ok:["Checks passed","Contrôles passés"], checks_ko:["Checks to review","Contrôles à revoir"],
  images_absent:["Preview images not rendered: ","Images non rendues : "], vrm_absent:["No VRM: ","VRM non livré : "],
  download:["Download","Télécharger"], open_folder:["Open folder","Ouvrir le dossier"],
  preset:["Preset","Préréglage"], custom:["Custom","Personnalisé"], render_name:["Render name","Nom du rendu"],
  render_import:["Import…","Importer…"], render_export:["Export","Exporter"], render_default:["Reset","Réinitialiser"],
  render_imported:["Render imported: ","Rendu importé : "], render_refused:["This file is not a render: ","Ce fichier n'est pas un rendu : "],
  render_exported:["Render saved as ","Rendu enregistré sous "],
  import_title:["Import models","Importer des modèles"],
  import_help:["Bring in existing characters. Nothing leaves this computer: files are copied into the local data folder.","Faites entrer des personnages existants. Rien ne quitte cet ordinateur : les fichiers sont copiés dans le dossier de données local."],
  drop_title:["Drop model files here, or click to choose","Déposez des fichiers de modèles ici, ou cliquez pour choisir"],
  formats_input:["Accepted formats","Formats acceptés"], skeletons:["Recognised skeletons","Squelettes reconnus"],
  tested:["tested","éprouvé"], partial:["partial","partiel"],
  extract_title:["Extract the parts","Extraire les pièces"],
  extract_help:["Cuts each model into a face, a hairstyle and clothes, fits them on the common body and adds them to the catalogue. This runs Blender and can take several minutes per model.","Découpe chaque modèle en visage, coiffure et vêtements, les porte sur le corps commun et les ajoute au catalogue. Blender travaille : comptez plusieurs minutes par modèle."],
  face_count:["Faces","Visages"], hairstyles:["Hairstyles","Coiffures"], clothes:["Clothes","Vêtements"], extract:["Extract","Extraire"],
  no_model:["No model imported yet.","Aucun modèle importé pour l'instant."], choose_models:["Tick at least one model.","Cochez au moins un modèle."],
  parts:["parts","pièces"], refusals:["rejected","refusés"], sending:["Importing ","Import de "], imported:["Imported: ","Importé : "], imported_n:["{n} models imported","{n} modèles importés"], discarded:["{n} file(s) set aside (weapons, already imported)","{n} fichier(s) écarté(s) (armes, déjà importés)"],
  extraction_done:["Extraction finished.","Extraction terminée."],
  extraction_running:["Extracting…","Extraction en cours…"], cancel:["Cancel","Annuler"], cancellation:["Cancelling…","Annulation…"],
  extraction_server:["Closing this tab does not stop the extraction: it runs on the local server. Come back to this page to follow it.","Fermer l'onglet n'arrête pas l'extraction : elle tourne sur le serveur local. Revenez sur cette page pour la suivre."],
  extraction_cancelled:["Extraction cancelled. What was finished stays in the catalogue; extracting again resumes from there.","Extraction annulée. Ce qui était fini reste au catalogue ; extraire de nouveau reprend à partir de là."],
  model_k:["Model {k} of {n}: {name} — {e}","Modèle {k} sur {n} : {name} — {e}"], models_k:["Models {k}–{m} of {n}: {e}","Modèles {k} à {m} sur {n} : {e}"],
  step_pivot:["pivot head","tête-pivot"], step_faces:["faces","visages"], step_hairstyles:["hairstyles","coiffures"], step_clothes:["cutting the clothes","découpe des vêtements"],
  step_catalog:["fitting the faces","pose des visages"], step_hairstyle:["hairstyle","coiffure"], step_parts:["clothes","vêtements"], unavailable:["Not available on this server.","Indisponible sur ce serveur."],
  characters_title:["Assembled characters","Personnages assemblés"],
  characters_help:["Every character you assembled, ready to download.","Chaque personnage assemblé, prêt à télécharger."],
  no_character:["No character yet: assemble one in Create.","Aucun personnage pour l'instant : assemblez-en un dans Créer."],
  missing:["The catalogue is empty. Import models and extract their parts to start.","Le catalogue est vide. Importez des modèles et extrayez leurs pièces pour commencer."],
  counts:["{v} faces · {c} hairstyles · {p} parts","{v} visages · {c} coiffures · {p} pièces"],
  hairstyle_bad_placed:["This hairstyle does not sit cleanly on this face ({e}).","Cette coiffure ne se pose pas proprement sur ce visage ({e})."],
  crosses:["{a} clips through {b}: {p} % of the points, up to {mm} mm.","{a} traverse {b} : {p} % des points, jusqu'à {mm} mm."],
  spring:["{a} pokes out of {b}: {p} % of the points, up to {mm} mm.","{a} ressort de {b} : {p} % des points, jusqu'à {mm} mm."],
  layers_failure:["Layers not checked for this combination: the computation failed.","Couches non vérifiées pour cette combinaison : le calcul a échoué."],
};
let LANGUAGE=recall("chimera.language",(navigator.language||"en").toLowerCase().startsWith("fr")?"fr":"en");
const L=()=>LANGUAGE==="fr"?1:0;
const t=(k,v)=>{ let s=(TXT[k]||[k,k])[L()]; for(const n in v||{}) s=s.replace(`{${n}}`,v[n]); return s; };

// Closed labels that come from the server or from data, as [English, French]. The server sends English (older caches
// may hold the French one): either is shown in the page's language. Free diagnostic messages (rejection reasons,
// errors, logs) stay in English in both languages.
const TYPES={   // piece types and accessory kinds (layers.default.json), by identifier
  underwear:["underwear","culotte"], socks:["socks","chaussettes"], top:["top","haut"], sleeves:["sleeves","manches"],
  shorts:["shorts","short"], trousers:["trousers","pantalon"], skirt:["skirt","jupe"], dress:["dress","robe"], jumpsuit:["jumpsuit","combinaison"],
  shoes:["shoes","chaussures"], boots:["boots","bottes"], gloves:["gloves","gants"], collar:["collar","col"], waistcoat:["waistcoat","gilet"],
  jacket:["jacket","veste"], coat:["coat","manteau"], hood:["hood","capuche"],
  belt:["belt","ceinture"], harness:["harness","harnais"], garter:["garter","jarretière"], anklet:["anklet","chevillère"],
  armband:["armband","brassard"], bracelet:["bracelet","bracelet"], ring:["ring","bague"], tie:["tie","cravate"], scarf:["scarf","écharpe"],
  necklace:["necklace","collier"], jewel:["jewel","bijou"], bag:["bag","sac"], glasses:["glasses","lunettes"], mask:["mask","masque"],
  headwear:["headwear","couvre-chef"], headphones:["headphones","casque audio"], tail:["tail","queue"], wings:["wings","ailes"], ears:["ears","oreilles"],
};
const EXPRESSIONS={Smile:["Smile","Sourire"],Laugh:["Laugh","Rire"],Talking:["Talking","Parle"],Surprise:["Surprise","Surprise"],
  Anger:["Anger","Colère"],Sad:["Sad","Triste"],"Eyes closed":["Eyes closed","Yeux fermés"],Yandere:["Yandere","Yandere"]};   // names are keys (expressions.py)
const FAILURES={skull_covered:["scalp poorly covered","crâne mal couvert"],skull_under_hair:["scalp showing","crâne visible"],hair_off_skin:["hair inside the skin","cheveux dans la peau"]};
const PRESETS={home:["Studio","Studio"],day:["Day","Jour"],sunset:["Sunset","Couchant"],night:["Night","Nuit"],simple:["Plain","Simple"]};
const SERVER=[
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
const MODELS=[
  [/^(.+) worn over (.+)$/,[(a,b)=>`${a} worn over ${b}`,(a,b)=>`${a} porté par-dessus ${b}`]],
  [/^(.+) removed: the (.+) covers it$/,[(a,b)=>`${a} removed: the ${b} covers it`,(a,b)=>`${a} retirée : le ${b} la couvre`]],
  [/^(.+): VRM humanoid$/,[a=>`${a}: VRM humanoid`,a=>`${a} : humanoïde VRM`]],
  [/^(.+): standard MMD bones$/,[a=>`${a}: standard MMD bones`,a=>`${a} : os MMD standard`]],
  [/^part ([a-z_]+)(-\d+)?$/,[(a,b)=>`part ${kind(a)}${b}`,(a,b)=>`pièce ${kind(a)}${b}`]],   // a rejected part of an import, by its folder (<kind>[-n])
  [/^part (\S+)$/,[a=>`part ${a}`,a=>`pièce ${a}`]],
  [/^(face|hairstyle) (.+)$/,[(a,b)=>`${a} ${b}`,(a,b)=>`${a} ${b}`]],   // a rejected catalogue item of a model
  // render files read by the viewer (viewer.normalizeRender)
  [/^This file is not a render: unreadable JSON \((.*)\)\.$/,[a=>`unreadable JSON (${a}).`,a=>`JSON illisible (${a}).`]],   // after the "not a render" prefix of rendu_refuse
  [/^Not a render: a JSON object is expected\.$/,[()=>"Not a render: a JSON object is expected.",()=>"Ce n'est pas un rendu : un objet JSON est attendu."]],
  [/^Not a render: "format" must be "chimera-render" \(read: (.*)\)\.$/,[a=>`Not a render: "format" must be "chimera-render" (read: ${a}).`,a=>`Ce n'est pas un rendu : "format" doit valoir "chimera-render" (lu : ${a}).`]],
  [/^Unreadable render: "version" must be an integer\.$/,[()=>`Unreadable render: "version" must be an integer.`,()=>`Rendu illisible : "version" doit être un entier.`]],
  [/^Render version (.+): this viewer only reads version (\d+)\.$/,[(a,b)=>`Render version ${a}: this viewer only reads version ${b}.`,(a,b)=>`Rendu de version ${a} : cette visionneuse ne lit que la version ${b}.`]],
  [/^(.+) · ([^·]+)$/,[(a,b)=>`${a} · ${b}`,(a,b)=>`${a} · ${b}`]],   // a part: "<character> · <type>"
];
// kinds of part the workshop does not offer on their own (trims, objects), named in refusal reasons and folders
const KINDS={strap:["strap","sangle"],buckle_button:["buckle or button","boucle ou bouton"],ornament:["ornament","ornement"],pocket:["pocket","poche"],
  patch:["patch","écusson"],badge:["badge","badge"],chain:["chain","chaîne"],cord:["cord","corde"],fastener:["fastener","fermeture"],buckle:["buckle","boucle"],
  weapon:["weapon","arme"],plush:["plush toy","peluche"],object:["object","objet"],hair:["hair","cheveux"],"no kind":["no kind","sans genre"]};
const kind=k=>(KINDS[k]||TYPES[k]||[k,k])[L()];
// blocking checks of a part (preparer.BLOQUANTS, assembler_blender.BLOQUANTS)
const CHECKS={under_skin:["inside the skin","sous la peau"],rest_crossing:["clipping at rest","traversée au repos"],
  rest_crossing_p90_mm:["clipping at rest, 90th percentile (mm)","traversée au repos, 90e centile (mm)"],body_only_poses:["test poses on the body alone","épreuves sur le corps seul"]};
const check=k=>CHECKS[k]?CHECKS[k][L()]:k;
const failures=s=>s.split(", ").map(c=>FAILURES[c]?FAILURES[c][L()]:c).join(", ");
// refusal reasons and errors written by the server: [pattern on the English text, French, English if it must change too]
const REASONS=[
  [/^the workshop does not offer this kind of part on its own \((.+)\)$/,a=>`l'atelier ne propose pas ce genre de pièce seule (${kind(a)})`,a=>`the workshop does not offer this kind of part on its own (${kind(a)})`],
  [/^sole without an upper: (\d+) % of the top of the foot covered \(minimum (\d+) %\)$/,(a,b)=>`semelle sans dessus : ${a} % du dessus du pied couvert (minimum ${b} %)`],
  [/^fails on its own head : (.+)$/,a=>`échoue sur sa propre tête : ${failures(a)}`,a=>`fails on its own head: ${failures(a)}`],
  [/^original head missing$/,()=>"tête d'origine absente"],
  [/^sits cleanly on only (\d+) heads out of (\d+)$/,(a,b)=>`ne se pose proprement que sur ${a} têtes sur ${b}`],
  [/^shape damaged by fitting \(review score (.+), limit (.+)\)$/,(a,b)=>`forme abîmée par le portage (note ${a} à la revue, limite ${b})`],
  [/^part with no extent \((.+) mm\)$/,a=>`pièce sans étendue (${a} mm)`],
  [/^fitting failed: (.*)$/s,a=>`port impossible : ${a}`],
  [/^measurement failed: (.*)$/s,a=>`mesure impossible : ${a}`],
  [/^preparation failed: (.*)$/s,a=>`préparation impossible : ${a}`],
  [/^cutting failed: (.*)$/s,a=>`découpe en échec : ${a}`],
  [/^unknown$/,()=>"inconnue"],
  [/^to prepare again: zero-weight bone mishandled by the graft \(fixed, run prepare again\)$/,()=>"à repréparer : os de poids nul mal repris par la greffe (corrigé, relancer prepare)"],
  [/^(\w+) = (.+) \(threshold (.+)\)$/,(a,b,c)=>`${check(a)} = ${b} (seuil ${c})`,(a,b,c)=>`${check(a)} = ${b} (threshold ${c})`],
  // assembly (assemble_blender.py, server.py)
  [/^(.+): (\w+) = (.+) \(threshold (.+)\)$/s,(n,a,b,c)=>`${srv(n)} : ${check(a)} = ${b} (seuil ${c})`,(n,a,b,c)=>`${srv(n)}: ${check(a)} = ${b} (threshold ${c})`],
  [/^(.+): measurement impossible \((.*)\)$/s,(n,a)=>`${srv(n)} : mesure impossible (${a})`],
  [/^(\d+) vertices, (\d+) materials, (\d+) bones; (\d+) parts measured, (\d+) layer pairs, (\d+) clippings$/,(a,b,c,d,e,f)=>`${a} sommets, ${b} matériaux, ${c} os ; ${d} pièces mesurées, ${e} paires de couches, ${f} traversées`],
  [/^Blender stopped \(code (-?\d+)\), see journal\.txt$/,a=>`Blender s'est arrêté (code ${a}), voir journal.txt`],
  [/^could not start Blender: (.*)$/s,a=>`impossible de lancer Blender : ${a}`],
  [/^assembly interrupted, see log\.txt$/,()=>"assemblage interrompu, voir log.txt"],
  [/^assembly interrupted \(the server stopped during the job\)$/,()=>"assemblage interrompu (le serveur s'est arrêté pendant le travail)"],
  [/^Unknown job\.$/,()=>"Travail inconnu."],
  // import and extraction (inputs.py, server.py)
  [/^The active game "(.+)" has no mmd_extracted folder\.$/,a=>`Le jeu actif « ${a} » n'a pas de dossier mmd_extracted.`],
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
  [/^This PMX does not follow the workshop's common language: (.*)\. See outils\/jeux\/README\.md\.$/s,a=>`Ce PMX ne suit pas la langue commune de l'atelier : ${a}. Voir tools/games/README.md.`],
  [/^Unsupported format \((.*)\): a \.vrm \(VRoid Studio or VRoid Hub\) or a \.zip containing a \.pmx\.$/,a=>`Format non pris en charge (${a==="no extension"?"sans extension":a}) : un .vrm (VRoid Studio ou VRoid Hub) ou un .zip contenant un .pmx.`],
  [/^Unsupported format: a \.vrm \(VRoid Studio or VRoid Hub\) or a \.zip containing a \.pmx\.$/,()=>"Format non pris en charge : un .vrm (VRoid Studio ou VRoid Hub) ou un .zip contenant un .pmx."],
  [/^A model "(.+)" is already imported: rename the file to make another one\.$/,a=>`Un modèle « ${a} » est déjà importé : renommer le fichier pour en faire un autre.`],
  [/^A model "(.+)" is already imported\.$/,a=>`Un modèle « ${a} » est déjà importé.`],
  [/^The VRM conversion failed: (.*)$/s,a=>`La conversion du VRM a échoué : ${a}`],
  [/^None of the (\d+) \.pmx files of the archive is a readable character\. The first one: (.*)$/s,(a,b)=>`Aucun des ${a} fichiers .pmx de l'archive n'est un personnage lisible. Le premier : ${srv(b)}`],
  [/^Every model of this archive is already imported\.$/,()=>"Tous les modèles de cette archive sont déjà importés."],
  [/^Unreadable request\.$/,()=>"Demande illisible."],
  [/^Choose at least one model \("models"\) or all of them \("all": true\)\.$/,()=>"Choisir au moins un modèle (« models ») ou tous (« all »: true)."],
  [/^Model unknown to the active game: (.*)\.$/s,a=>`Modèle inconnu du jeu actif : ${a}.`],
  [/^No model in the active game: import one first\.$/,()=>"Aucun modèle dans le jeu actif : en importer un d'abord."],
  [/^"what": a list among (.*)\.$/,a=>`« what » : une liste parmi ${a}.`],
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
// the assembly steps: [English, French, progress %]
const STEPS={waiting:["Blender…","Blender…",3],hairstyle:["Fitting the hair…","Pose de la coiffure…",12],graft:["Grafting the head…","Greffe de la tête…",24],parts:["Outfit parts…","Pièces de la tenue…",50],color:["Hair colour…","Couleur des cheveux…",62],checks:["Checks…","Contrôles…",74],skeleton:["Skeleton…","Skeleton…",79],vrm:["VRM export…","Export VRM…",83],render:["Rendering the views…","Rendu des vues…",90],done:["Done","Terminé",100]};
// The studio in French: the settings carry their English name in viewer.js (RENDER_SETTINGS), the French is here, by key.
const FR_GROUPS={"Light":"Lumière","Shadows":"Ombres","Skin":"Peau","Materials":"Matières","Outline":"Trait","Background and floor":"Fond et sol","Finish":"Fin d'image"};
const FR_RENDER={
  "light.angle":["Angle","Côté d'où vient la lumière, en degrés, compté depuis la caméra (0 : de face ; 90 : de côté ; 180 : à contre-jour). Elle tourne avec la vue."],
  "light.height":["Hauteur","Hauteur de la lumière au-dessus de l'horizon, en degrés (négatif : par en dessous)."],
  "light.color":["Couleur","Couleur de la lumière principale, sur les parties éclairées."],
  "light.force":["Force","Force de la lumière principale : écart entre le clair et l'ombre."],
  "light.rim":["Couleur du contour","Couleur de la lumière de contour, venue de derrière, qui détache la silhouette du fond."],
  "light.forceRim":["Lumière de contour","Force de la lumière de contour (0 : aucune)."],
  "light.sky":["Ciel dans l'ombre","Lumière du ciel renvoyée dans l'ombre, sur les surfaces tournées vers le haut."],
  "light.ground":["Sol dans l'ombre","Lumière du sol renvoyée dans l'ombre, sur les surfaces tournées vers le bas."],
  "shadow.threshold":["Seuil","Où passe la limite de l'ombre sur les vêtements et les cheveux : plus haut, plus d'ombre."],
  "shadow.softness":["Douceur","Largeur du passage de la lumière à l'ombre (petite : limite nette)."],
  "shadow.tint":["Teinte des vêtements","Couleur par laquelle la couleur d'un vêtement est multipliée dans l'ombre."],
  "shadow.tintHair":["Teinte des cheveux","Couleur par laquelle la couleur des cheveux est multipliée dans l'ombre."],
  "shadow.saturation":["Saturation","Couleur plus vive (au-dessus de 1) ou plus grise dans l'ombre des vêtements et des cheveux."],
  "shadow.edgeBand":["Liseré","Bande plus saturée le long de la limite d'ombre, comme la lumière qui traverse la peau."],
  "shadow.crevices":["Creux","Ombre des creux (aisselles, plis, sous les mèches), tirée des cartes d'ouverture."],
  "shadow.casts":["Ombres portées","Ce qui est devant la lumière fait de l'ombre (cheveux sur le visage, bras sur le corps, personnage sur le sol)."],
  "shadow.castStrength":["Force des ombres portées","Noirceur des ombres portées (1 : aussi sombres que l'ombre propre)."],
  "shadow.contact":["Ombre de contact","Ce qui est posé tout près d'une surface l'assombrit : col, ceinture, mèche sur l'épaule (0 : aucune)."],
  "skin.threshold":["Seuil","Où passe la limite de l'ombre sur la peau et le visage."],
  "skin.softness":["Douceur","Largeur du passage de la lumière à l'ombre sur la peau."],
  "skin.tint":["Teinte de l'ombre","Couleur par laquelle la peau est multipliée dans l'ombre (les yeux et la bouche en suivent une version plus froide)."],
  "skin.transmission":["Lumière qui traverse","À contre-jour, la lumière traverse la peau fine (oreilles, doigts) et les mèches fines : lueur chaude."],
  "skin.blush":["Rose du visage","Rose des joues, du nez et des lèvres, et petit reflet de la lèvre."],
  "skin.shadowEyes":["Ombre sur les yeux","Bande sombre sur les yeux que portent certaines expressions (0 : jamais)."],
  "material.clothes":["Matières des vêtements","Reflets selon la matière lue dans le nom du vêtement (tissu, tricot, cuir, collant, soie, verre, métal). Éteint : vêtements mats."],
  "material.hair":["Reflets des cheveux","Bande de reflet en travers des mèches, racines plus sombres et pointes plus claires (0 : cheveux mats)."],
  "outline.active":["Trait","Trait au bord des formes, de la couleur de la surface en plus sombre."],
  "outline.thickness":["Épaisseur","Épaisseur du trait (il s'affine de loin)."],
  "outline.opacity":["Opacité","Opacité du trait."],
  "outline.color":["Couleur","Couleur fixe du trait, mêlée à la couleur tirée de la surface selon « Part de la couleur »."],
  "outline.partColor":["Part de la couleur","0 : le trait prend la couleur de la surface, assombrie ; 1 : la couleur fixe partout."],
  "background.center":["Fond au centre","Couleur du fond derrière le personnage."],
  "background.border":["Fond au bord","Couleur du fond vers les bords de l'image."],
  "floor.active":["Sol","Un rond de lumière sous les pieds, qui reçoit l'ombre du personnage."],
  "floor.light":["Sol éclairé","Couleur du sol au centre du rond, en pleine lumière."],
  "floor.shadow":["Sol dans l'ombre","Couleur de l'ombre du personnage sur le sol."],
  "image.exposure":["Exposition","Clarté de toute l'image (sous 1, les clairs restent sous le blanc et le halo fait le reste)."],
  "image.saturation":["Saturation","Couleur de toute l'image : 1 inchangée, 0 noir et blanc."],
  "image.halo":["Halo","Lueur autour des parties claires (0 : aucune)."],
  "image.blur":["Profondeur de champ","Flou de ce qui est loin du sujet, surtout en gros plan (0 : tout net)."],
  "image.thumbnail":["Vignette","Assombrissement des coins de l'image."],
  "image.grain":["Grain","Grain léger sur toute l'image."],
};
// french-labels: end
const PAIRS=new Map();
for(const x of [...SERVER,...Object.values(TYPES)]) for(const s of x) if(!PAIRS.has(s)) PAIRS.set(s,x);
// A label from the server or the data, in the page's language (unknown text is shown as it is).
function srv(s){
  s=String(s??""); const x=PAIRS.get(s); if(x) return x[L()];
  for(const [re,fr,en] of REASONS){ const m=s.match(re); if(m) return L()?fr(...m.slice(1)):en?en(...m.slice(1)):s; }
  for(const [re,f] of MODELS){ const m=s.match(re); if(m) return f[L()](...m.slice(1).map(srv)); }
  const chunks=s.split("; ");   // several reasons in a row
  if(chunks.length>1){ const tr=chunks.map(srv); if(tr.every((p,i)=>p!==chunks[i])) return tr.join(L()?" ; ":"; "); }
  const phrases=s.split(/(?<=\.)\s+(?=\S)/);
  if(phrases.length>1&&phrases.every(p=>PAIRS.has(p))) return phrases.map(srv).join(" ");
  return s;
}
const nameType=p=>{ const x=TYPES[p.type]||PAIRS.get(p.type_name); return x?x[L()]:(p.type_name||p.type||""); };
// displayed name of a catalogue item: its label's name when it has one, otherwise its code name
// (a part is "<character> · <type>", in the page's language)
const nameCode=it=>it.character?`${it.character} · ${nameType(it)}`:srv(it.name);
const nameElement=it=>it.label?.name?.[L()]||nameCode(it);
// the model an item comes from (cat.sources: page, name, author, licence terms as read at download), if known
const sourceOf=it=>(cat.sources||{})[it.character||it.id]||null;
const safeLink=u=>/^https:\/\//.test(String(u||""))?String(u):null;
function licenseShort(s){
  const l=s.license||{}, m=l.name?[l.name]:[];
  if(l.creditNotation) m.push(t(l.creditNotation==="required"?"license_credit_yes":"license_credit_no"));
  if(l.commercialUsage) m.push(t(l.commercialUsage==="personalNonProfit"?"license_personal":"license_commercial"));
  if(l.modification) m.push(t(l.modification==="allowModificationRedistribution"?"license_edit":"license_edit_no"));
  return m.join(" · ");
}
// tooltip of a tile: name, description, tags, then the code name (the model it comes from)
const bubbleElement=it=>{ const e=it.label, s=sourceOf(it), o=s?`${s.name} ${t("by")} ${s.author} (${s.site||""})`:null;
  if(!e) return [nameCode(it),o].filter(Boolean).join("\n");
  return [nameElement(it),e.description?.[L()],(e.tags?.[L()]||[]).map(x=>"#"+x).join(" "),nameCode(it),o].filter(Boolean).join("\n"); };
// what the search box reads: names, description and tags, in both languages, and the code name
const textSearch=it=>it._search??=[nameCode(it),it.id,...(it.label?[...(it.label.name||[]),...(it.label.description||[]),...(it.label.tags||[]).flat()]:[])].join(" ").toLowerCase();
const nameExpression=n=>EXPRESSIONS[n]?EXPRESSIONS[n][L()]:n;
function translate(){
  document.documentElement.lang=LANGUAGE;
  for(const e of document.querySelectorAll("[data-i]")) e.textContent=t(e.dataset.i);
  $("#search").placeholder=t("search"); $("#language").textContent=LANGUAGE==="fr"?"en":"fr"; $("#name").placeholder=t("without_name");
  buttonExtract(xRunning);
}

const DRAWINGS={
  face:`<circle cx="12" cy="12" r="8.5"/><path d="M9 10.5v.5M15 10.5v.5M9 15c1.8 1.6 4.2 1.6 6 0"/>`,
  hairstyle:`<path d="M4.5 18v-5C4.5 7.5 7.5 4 12 4s7.5 3.500 7.500 9v5M8 10c2.500 1.200 5.500 1.200 8-.500"/>`,
  top:`<path d="M8.500 4 4 6.500l1.500 4L8 9.500V20h8V9.500l2.500 1 1.500-4L15.500 4c-.800 1.300-2 2-3.500 2s-2.700-.700-3.500-2z"/>`,
  bottom:`<path d="M7 4h10l1.500 16H14L12 9l-2 11H5.500z"/>`,
  shoes:`<path d="M3 17v-3.500c2 0 3.500-1 4-4l3.500 1c.500 2 2.500 3.500 6 4 2.500.400 4.500 1 4.500 2.500z"/>`,
  accessories:`<path d="m12 3.500 2.600 5.300 5.900.800-4.300 4.100 1 5.800L12 16.800 6.800 19.500l1-5.800-4.300-4.100 5.900-.800z"/>`,
};
const TABS=["face","hairstyle","top","bottom","shoes","accessories"];
const COLORS=[null,"#1e1a1c","#4a2f24","#7a4a2c","#b98a52","#e7cf9a","#e9e4dc","#8b8f9b","#a23a34","#d9648c","#7a4ba8","#3c6fb4","#3f9c97","#5f9449"];
const view=createViewer($("#view"));
const state={tab:"face",head:null,hairstyle:null,parts:new Set(),color:null,filter:null,search:"",signal:[],ref:0.5};
let cat={heads:[],hairstyles:[],parts:[],refusals:[]}, byId={}, genCombo=0, genHead=0, genHair=0, hairBase=null, hairOffset=null;   // one counter per kind: a hairstyle does not cancel the neck fitting of a face
let bodyBase=null, bodyOffset=null;   // body positions under the face graft, offset of the feet posed in a heeled shoe
let skinParts=[], skinHead=[];   // body skin faces hidden by the parts, by the face graft
const hideBody=()=>view.mask("body",[...skinParts,...skinHead]);
const hex=c=>({r:parseInt(c.slice(1,3),16)/255,g:parseInt(c.slice(3,5),16)/255,b:parseInt(c.slice(5,7),16)/255});
const json=async(url,opts)=>{ const r=await fetch(url,opts); let v=null; try{ v=await r.json(); }catch(e){} if(!r.ok) throw new Error((v&&v.error&&srv(v.error))||(r.status===404?t("unavailable"):`${r.status}`)); return v; };

async function readCatalog(){
  cat=await json("/api/catalog"); byId={};
  for(const k of ["heads","hairstyles","parts","refusals"]) cat[k]=cat[k]||[];
  for(const p of cat.parts) byId[p.id]=p;
  $("#counts").textContent=t("counts",{v:cat.heads.length,c:cat.hairstyles.length,p:cat.parts.length});
  $("#nb-refusals").textContent=cat.refusals.length;
  $("#list-refusals").innerHTML=cat.refusals.map(r=>`<li><b>${ech(srv(r.name))}</b> (${ech(srv(r.kind))})<br>${ech(srv(r.reason))}</li>`).join("");
}
const empty=()=>!cat.heads.length||!cat.hairstyles.length;

function show(v){
  if(!["create","library","characters"].includes(v)) v="create";
  document.body.dataset.view=v;
  for(const n of ["create","library","characters"]) $("#"+n).hidden=n!==v;
  for(const b of $("#views").children) b.classList.toggle("active",b.dataset.v===v);
  if(v==="create"){ view.sale=true; window.dispatchEvent(new Event("resize")); }
  if(v==="library") openLibrary();
  if(v==="characters") openCharacters();
  if(location.hash.slice(1)!==v) history.replaceState(null,"","#"+v);
}

async function init(){
  translate();
  $("#language").onclick=()=>{ LANGUAGE=LANGUAGE==="fr"?"en":"fr"; store("chimera.language",LANGUAGE); translate(); paintTabs(); paint(); buildStudio(); show(document.body.dataset.view); $("#counts").textContent=t("counts",{v:cat.heads.length,c:cat.hairstyles.length,p:cat.parts.length}); };
  $("#views").onclick=e=>{ const v=e.target.dataset.v; if(v) show(v); };
  $("#panes").onclick=e=>{ const p=e.target.dataset.p; if(p) pane(p); };
  $("#tabs").onclick=e=>{const o=e.target.closest("[data-o]"); if(o){state.tab=o.dataset.o;state.filter=null;paint();}};
  $("#search").oninput=e=>{state.search=e.target.value.toLowerCase();paintGrid();};
  $("#dots").innerHTML=COLORS.map((c,i)=>`<button data-c="${i}" title="${c?c:t("color_origin")}" style="background:${c||"#fff"}">${c?"":"×"}</button>`).join("");
  $("#dots").onclick=e=>{const i=e.target.dataset.c; if(i!==undefined){state.color=COLORS[+i];updateDots();applyTint();}};
  $("#controls").onclick=e=>{ const v=e.target.dataset.view; if(!v)return; if(v==="turn"){view.spinning=!view.spinning;$("#spin").classList.toggle("active",view.spinning);} else view.fit(v);};
  $("#expression").onchange=e=>view.express&&view.express(e.target.value||null);
  view.onSpinning=b=>$("#spin").classList.toggle("active",b);
  view.onTexture=(u,m)=>{ if(state.imgHairstyle===u){state.ref=m; applyTint();} };
  $("#assemble").onclick=assemble;
  $("#name").value=recall("chimera.name","");
  $("#name").oninput=e=>store("chimera.name",e.target.value);
  $("#simplified").hidden=!view.software;
  pane("character"); connectLibrary();
  await initStudio();
  await readCatalog(); paintTabs();
  document.addEventListener("visibilitychange",()=>{ if(!document.hidden) followExtraction(); });
  if(empty()){ paint(); show("library"); followExtraction(); return; }
  show(location.hash.slice(1));
  followExtraction();
  await buildScene();
}
let scene=false;
async function buildScene(){
  scene=true;
  await view.placer("body","/data/body/body.glb");
  state.head=cat.heads[0].id; state.hairstyle=cat.hairstyles[0].id;
  await updateHead(); await updateHairstyle(); await updateParts(); paint();
  generateThumbnails();
}
// The catalogue changed on the server (an extraction made a model usable): new lists, same choices, same scene.
async function updateCatalog(){
  await readCatalog(); paintTabs(); $("#missing").hidden=!empty();
  if(!scene){ if(!empty()) await buildScene(); else paint(); return; }
  paint(); generateThumbnails();
}
function pane(p){
  for(const b of $("#panes").children) b.classList.toggle("active",b.dataset.p===p);
  $("#pane-character").hidden=p!=="character"; $("#pane-render").hidden=p!=="render";
}
function paintTabs(){
  const n={face:cat.heads.length,hairstyle:cat.hairstyles.length}; for(const p of cat.parts) n[p.tab]=(n[p.tab]||0)+1;
  $("#tabs").innerHTML=TABS.map(k=>`<button data-o="${k}"><svg viewBox="0 0 24 24" aria-hidden="true">${DRAWINGS[k]}</svg>${t(k)}<b>${n[k]||0}</b></button>`).join("");
}
function paint(){
  for(const b of $("#tabs").children) b.classList.toggle("active",b.dataset.o===state.tab);
  $("#title-tab").textContent=t(state.tab);
  $("#colors").hidden=state.tab!=="hairstyle";
  updateDots(); paintFilters(); paintGrid(); paintDetails();
}
function elements(){
  if(state.tab==="face") return cat.heads.map(t=>({...t,kind:"head"}));
  if(state.tab==="hairstyle") return cat.hairstyles.map(c=>({...c,kind:"hairstyle",warn:c.pairs[state.head]&&!c.pairs[state.head].passed}));
  return cat.parts.filter(p=>p.tab===state.tab).map(p=>({...p,kind:"part"}));
}
function paintFilters(){
  const ex={}; for(const e of elements()) if(e.type_name&&!(e.type_name in ex)) ex[e.type_name]=e;
  const types=Object.keys(ex);
  $("#filters").innerHTML=types.length>1?[`<button data-t="" class="${state.filter?"":"active"}">${t("all")}</button>`,...types.map(t=>`<button data-t="${ech(t)}" class="${state.filter===t?"active":""}">${ech(nameType(ex[t]))}</button>`)].join(""):"";
  $("#filters").onclick=e=>{const t=e.target.dataset.t; if(t!==undefined){state.filter=t||null;paint();}};
}
function paintGrid(){
  let l=elements();
  if(state.filter) l=l.filter(e=>e.type_name===state.filter);
  if(state.search) { const words=state.search.split(/\s+/).filter(Boolean); l=l.filter(e=>{const x=textSearch(e); return words.every(m=>x.includes(m));}); }
  const g=$("#grid"); g.textContent="";
  if(!l.length){ g.innerHTML=`<div class="nothing">${t("shelf_empty")}<br><button data-library>${t("open_library")}</button></div>`; g.querySelector("[data-library]").onclick=()=>show("library"); return; }
  for(const it of l){
    const b=document.createElement("button"); b.className="tile"+(isChosen(it)?" chosen":""); b.title=it.warn?t("hairstyle_warning"):bubbleElement(it); b.dataset.id=it.id;
    const u=thumbnail(it);
    b.innerHTML=(u?`<img loading="lazy" src="${ech(u)}" alt="">`:`<div class="empty">…</div>`)+`<span>${ech(nameElement(it))}</span>`+(it.warn?`<b class="warning">!</b>`:"")+(sourceOf(it)?`<i class="origin" title="${t("page_origin")}">↗</i>`:"");
    b.onclick=e=>{ if(e.target.classList.contains("origin")){ e.stopPropagation(); window.open(sourceOf(it).url,"_blank","noopener"); return; } choose(it); }; g.append(b);
  }
}
const thumbnails={};
const thumbnail=it=>it.thumbnail||thumbnails[it.kind+it.id]||null;
const isChosen=it=>it.kind==="head"?state.head===it.id:it.kind==="hairstyle"?state.hairstyle===it.id:state.parts.has(it.id);

async function choose(it){
  if(it.kind==="head"){state.head=it.id;paint();await updateHead();await updateHairstyle();}
  else if(it.kind==="hairstyle"){state.hairstyle=it.id;paint();await updateHairstyle();}
  else {flip(it);paint();await updateParts();}
  paint();
}
function flip(p,force){
  if(state.parts.has(p.id)&&!force){state.parts.delete(p.id);return;}
  for(const id of [...state.parts]){const q=byId[id];
    if((p.slot&&q.slot===p.slot)||(q.occupies||[]).includes(p.slot)||(p.occupies||[]).includes(q.slot)) state.parts.delete(id);
  }
  state.parts.add(p.id);
}
async function withLoad(f){ const c=$("#loader"); c.hidden=false; try{await f();}catch(e){console.error(e);} finally{c.hidden=true;} }

async function updateHead(){
  const g=++genHead, te=cat.heads.find(x=>x.id===state.head);
  await withLoad(async()=>{
    await view.placer("head",`/data/heads/${encodeURIComponent(te.id)}/head.glb`);
    if(g!==genHead) return;
    const ex=$("#expression"); if(ex&&view.express){ ex.innerHTML=`<option value="">${t("rest")}</option>`+view.express(null).map(n=>`<option value="${ech(n)}">${ech(nameExpression(n))}</option>`).join(""); ex.hidden=ex.options.length<2; }
    const back=`/data/heads/${encodeURIComponent(te.id)}/`;
    const [m,som,cache]=await Promise.all([view.placer("body","/data/body/body.glb",{skin:te.skin}),
      fetch(back+"body.bin").then(r=>r.ok?r.arrayBuffer():null), fetch(back+"hidden_skin.bin").then(r=>r.ok?r.arrayBuffer():null)]);
    if(g!==genHead) return;
    // body skin as the graft leaves it under this face (prepare.join_skin)
    const n=m.d.pos.length, s=som&&som.byteLength===8*n?new Float32Array(som):null;
    bodyBase=s?s.subarray(0,n):m.d.pos;
    view.vertices("body",bodyBase,s?s.subarray(n):m.d.nor); placeBody();
    skinHead=cache?Array.from(new Uint32Array(cache)):[]; hideBody();
    await view.maps("body",back+"body-maps.bin");
  });
}
async function updateHairstyle(){
  const g=++genHair;
  await withLoad(async()=>{
    const id=encodeURIComponent(state.hairstyle);
    const [m,pos]=await Promise.all([view.placer("hairstyle",`/data/hairstyles/${id}/hairstyle.glb`,{hair:true}),
      fetch(`/data/hairstyles/${id}/${encodeURIComponent(state.head)}.bin`).then(r=>r.ok?r.arrayBuffer():null)]);
    if(g!==genHair) return;
    hairBase=pos?new Float32Array(pos):m.d.pos; hairOffset=null; placeHair();
    state.imgHairstyle=m.prims[0]&&m.prims[0].image;
    const tex=state.imgHairstyle&&view.texture(state.imgHairstyle); if(tex&&tex.average) state.ref=tex.average;
    applyTint();
  });
  paintDetails();
  if(state.parts.size) await updateParts();   // the hair falls on the outfit: to redo for this hairstyle
}
// body with its feet posed in a heeled shoe (offset returned by /api/combination, outfit.place_outfit)
function placeBody(){
  if(!bodyBase) return;
  let p=bodyBase;
  if(bodyOffset&&bodyOffset.length===p.length){ p=new Float32Array(p.length); for(let i=0;i<p.length;i++) p[i]=bodyBase[i]+bodyOffset[i]; }
  view.positions("body",p);
}
// hair fitted on the head, moved over the outfit (offset returned by /api/combination)
function placeHair(){
  if(!hairBase) return;
  let p=hairBase;
  if(hairOffset&&hairOffset.length===p.length){ p=new Float32Array(p.length); for(let i=0;i<p.length;i++) p[i]=hairBase[i]+hairOffset[i]; }
  view.positions("hairstyle",p);
}
function applyTint(){
  const m=view.models.get("hairstyle"); if(m){m.tint=state.color?{...hex(state.color),ref:state.ref}:null; view.sale=true;}
}
function updateDots(){ for(const b of $("#dots").children) b.classList.toggle("active",(COLORS[+b.dataset.c]===state.color)); }

async function updateParts(){
  const g=++genCombo, ids=[...state.parts], skin=new Map();
  document.body.dataset.layers="computing";   // read by tests/session.py
  const unionSkin=()=>{const f=[]; for(const a of skin.values()) for(const x of a) f.push(x); return f;};
  await withLoad(async()=>{
    for(const c of [...view.models.keys()]) if(c.startsWith("part:")&&!state.parts.has(c.slice(6))) view.remove(c);
    await Promise.all(ids.map(async id=>{
      await view.placer("part:"+id,`/data/parts/${encodeURIComponent(id)}/part.glb`);
      view.mask("part:"+id,[]);
      const r=await fetch(`/data/parts/${encodeURIComponent(id)}/hidden_skin.bin`);
      if(r.ok) skin.set(id,new Uint32Array(await r.arrayBuffer()));
    }));
    if(g!==genCombo) return;
    skinParts=unionSkin(); hideBody();
  });
  state.signal=[]; state.notes=[]; state.comboImpossible=false; paintDetails();
  if(!ids.length){ hairOffset=null; placeHair(); bodyOffset=null; placeBody(); if(g===genCombo) document.body.dataset.layers="ready"; return; }
  const rep=await fetch("/api/combination",{method:"POST",headers:JSON_H,body:JSON.stringify({parts:ids,hairstyle:state.hairstyle,head:state.head})});
  if(g!==genCombo) return;
  if(!rep.ok){ state.comboImpossible=true; paintDetails(); document.body.dataset.layers="ready"; return; }
  const v=await rep.json();
  for(const id of ids){
    if(v.removed[id]){ view.mask("part:"+id,true); skin.delete(id); continue; }
    await rebase(id,v.offsets[id]||null);
    view.mask("part:"+id,v.masks[id]?faces(v.masks[id]):[]);
  }
  hairOffset=v.hairstyle?new Float32Array(Uint8Array.from(atob(v.hairstyle),c=>c.charCodeAt(0)).buffer):null; placeHair();
  bodyOffset=v.body?new Float32Array(Uint8Array.from(atob(v.body),c=>c.charCodeAt(0)).buffer):null; placeBody();
  skinParts=unionSkin(); hideBody();
  state.signal=v.signal; state.notes=v.notes||[]; paintDetails();
  if(g===genCombo) document.body.dataset.layers="ready";
}
function faces(b64){ const o=atob(b64),f=[]; for(let i=0;i<o.length;i++) if(o.charCodeAt(i)) f.push(i); return f; }
async function rebase(id,b64){
  const url=`/data/parts/${encodeURIComponent(id)}/part.glb`, d=await readGLB(url);
  if(!view.models.has("part:"+id)) return;
  if(!b64){ view.positions("part:"+id,d.pos); return; }
  const o=new Float32Array(Uint8Array.from(atob(b64),c=>c.charCodeAt(0)).buffer), p=new Float32Array(d.pos.length);
  for(let i=0;i<p.length;i++) p[i]=d.pos[i]+o[i];
  view.positions("part:"+id,p);
}

const originLink=it=>{ const s=it&&sourceOf(it); return s?`<a class="origin" href="${ech(s.url)}" target="_blank" rel="noopener noreferrer" title="${ech(t("page_origin")+" : "+s.name+" "+t("by")+" "+s.author)}">↗</a>`:""; };
function paintDetails(){
  const te=cat.heads.find(x=>x.id===state.head), c=cat.hairstyles.find(x=>x.id===state.hairstyle);
  const ids=[...state.parts].sort((a,b)=>byId[a].rank-byId[b].rank);
  $("#selection").innerHTML=`<li><b>${t("face")}</b><span>${te?ech(nameElement(te)):"—"}</span>${originLink(te)}</li><li><b>${t("hairstyle")}</b><span>${c?ech(nameElement(c)):"—"}</span>${originLink(c)}</li>`+
    ids.map(i=>`<li><b>${ech(nameType(byId[i]))}</b><span title="${ech(bubbleElement(byId[i]))}">${ech(byId[i].label?nameElement(byId[i]):byId[i].character)}</span>${originLink(byId[i])}<button data-r="${ech(i)}" title="${t("remove")}">×</button></li>`).join("");
  const seen=new Map();   // the original models of the current choice, each once
  const provides=new Map();   // model (page and name: two models may share a page) -> the slots of the choice it supplies
  for(const [r,it] of [[t("face"),te],[t("hairstyle"),c],...ids.map(i=>[nameType(byId[i]),byId[i]])]){ const s=it&&sourceOf(it);
    if(s){ const k=s.url+"|"+s.name; s._key=k; seen.set(k,s); provides.set(k,[...(provides.get(k)||[]),`${r.charAt(0).toUpperCase()+r.slice(1)} : ${nameElement(it)}`]); } }
  $("#origins").hidden=!seen.size;
  $("#origins-list").innerHTML=[...seen.values()].map(s=>{ const l=s.license||{}, o=safeLink(l.otherLicenseUrl), v=safeLink(l.licenseUrl);
    return `<li><a href="${ech(s.url)}" target="_blank" rel="noopener noreferrer">${ech(s.name)} ↗</a> <small>${t("by")} ${ech(s.author)} · ${ech(s.site||"")}</small>`+
      `<ul class="provides">${provides.get(s._key).map(x=>`<li>${ech(x)}</li>`).join("")}</ul>`+
      `<div class="${l.creditNotation==="required"?"credit":""}">${ech(licenseShort(s))}</div>`+
      (v?`<div><a href="${ech(v)}" target="_blank" rel="noopener noreferrer">${ech(l.name||t("license_vrm"))} ↗</a></div>`:"")+
      (l.otherLicenseUrl||l.thirdPartyLicenses?`<div>${t("license_other")} : ${o?`<a href="${ech(o)}" target="_blank" rel="noopener noreferrer">${ech(o)}</a>`:ech(l.otherLicenseUrl||"")} ${ech(l.thirdPartyLicenses||"")}</div>`:"")+
      `<div class="read">${ech(t("license_read",{d:s.read||"?"}))}</div></li>`; }).join("");
  $("#selection").onclick=e=>{const r=e.target.dataset.r; if(r){state.parts.delete(r);paint();updateParts();}};
  let al="";
  if(c&&c.pairs[state.head]&&!c.pairs[state.head].passed) al+=`<div class="alert">${ech(t("hairstyle_bad_placed",{e:c.pairs[state.head].failures.map(nameFailure).join(", ")}))}</div>`;
  const partName=(id,name)=>byId[id]?nameType(byId[id]):srv(name);
  for(const s of state.signal) al+=`<div class="alert">${ech(t("crosses",{a:partName(s.under,s.name_under),b:partName(s.over,s.name_over),p:(100*s.fraction).toFixed(1),mm:s.worst_mm}))}</div>`;
  for(const n of state.notes||[]) al+=`<div class="note">${ech(srv(n))}</div>`;
  if(state.comboImpossible) al+=`<div class="alert">${t("layers_failure")}</div>`;
  $("#warnings").innerHTML=al;
}
const nameFailure=e=>FAILURES[e]?FAILURES[e][L()]:e;

// ---------- Assemble, then download ----------
async function assemble(){
  const sel={head:state.head,hairstyle:state.hairstyle,parts:[...state.parts],tint:state.color?{...hex(state.color),ref:state.ref}:null,name:$("#name").value.trim()||t("without_name")};
  const b=$("#assemble"); b.disabled=true; b.textContent=t("assembly");
  $("#work").hidden=false; $("#result").innerHTML=""; $("#step").textContent=t("waiting"); $("#bar div").style.width="3%";
  try{ const {id}=await json("/api/assemble",{method:"POST",headers:JSON_H,body:JSON.stringify(sel)}); state.work=id; follow(id); }
  catch(e){ $("#step").textContent=t("failure")+e.message; endAss(); }
}
// download links of an assembled character: one per delivered format
const outputs=(id,formats)=>`<div class="outputs">${(formats||[]).map(f=>`<a download href="/api/assemblies/${encodeURIComponent(id)}/${encodeURIComponent(f)}">${t("download")} <b>${ech(f.toUpperCase())}</b></a>`).join("")}<button data-open="${ech(id)}">${t("open_folder")}</button></div>`;
document.addEventListener("click",e=>{ const id=e.target.dataset&&e.target.dataset.open; if(id) fetch("/api/open",{method:"POST",headers:JSON_H,body:JSON.stringify({id})}); });
async function follow(id){
  let e;
  try{ e=await json("/api/jobs/"+id); }
  catch(x){ $("#step").textContent=t("server_silent")+x.message; endAss(); return; }
  if(e.step==="error"){ $("#step").textContent=t("failure")+srv(e.error); endAss(); return; }
  const [en,fr,p]=STEPS[e.step]||[e.step,e.step,10];
  $("#step").textContent=(LANGUAGE==="fr"?fr:en)+(e.step==="parts"&&e.parts_total?` ${e.current_part||0}/${e.parts_total}`:"");
  $("#bar div").style.width=p+"%";
  if(!e.done){ setTimeout(()=>follow(id),1500); return; }
  let h=`<div>${e.ok?t("checks_ok"):t("checks_ko")} · ${ech(srv(e.summary))}</div>`;
  if(e.failures&&e.failures.length) h+=`<div class="alert">${e.failures.map(x=>ech(srv(x))).join("<br>")}</div>`;
  for(const c of e.crosses||[]) h+=`<div class="alert">${ech(t("spring",{a:srv(c.under),b:srv(c.over),p:(100*c.left_out).toFixed(1),mm:c.worst_mm.toFixed(1)}))}</div>`;
  h+=e.sheet===false?`<div class="alert">${t("images_absent")}${ech(e.sheet_note)}</div>`:`<a href="/data/assemblies/${id}/sheet.png" target="_blank"><img src="/data/assemblies/${id}/sheet.png" alt=""></a>`;
  h+=outputs(id,e.formats||(e.vrm?["pmx","vrm"]:["pmx"]));
  if(!e.vrm&&e.vrm_note) h+=`<div class="alert">${t("vrm_absent")}${ech(srv(e.vrm_note))}</div>`;
  $("#result").innerHTML=h;
  endAss();
}
function endAss(){ const b=$("#assemble"); b.disabled=false; b.textContent=t("assemble"); }

// ---------- Render studio: each setting of the viewer (RENDER_SETTINGS) has its control; a render is a JSON file ----------
const SETTINGS=V.RENDER_SETTINGS||[];
const studioReady=()=>!view.software&&SETTINGS.length&&typeof view.adjust==="function";
const read=(o,key)=>key.split(".").reduce((a,k)=>a==null?a:a[k],o);
const gam=(x,lin)=>lin?(x<=0.0031308?12.92*x:1.055*Math.pow(x,1/2.4)-0.055):x, dgam=(x,lin)=>lin?(x<=0.04045?x/12.92:Math.pow((x+0.055)/1.055,2.4)):x;
const toHex=(c,lin)=>"#"+[0,1,2].map(i=>Math.round(255*Math.min(1,Math.max(0,gam(c[i]||0,lin)))).toString(16).padStart(2,"0")).join("");
const fromHex=(h,lin)=>[1,3,5].map(i=>+dgam(parseInt(h.slice(i,i+2),16)/255,lin).toFixed(4));
let presets=[], renderName="";
async function initStudio(){
  try{ presets=await json("/page/presets/index.json"); }catch(e){ presets=[]; }
  buildStudio();
  if(!studioReady()) return;
  const q=new URLSearchParams(location.search).get("render")??new URLSearchParams(location.search).get(V1_NAMES.renderParam), saved=recall("chimera.render","");
  try{
    if(q&&presets.some(p=>p.id===q)) await choosePreset(q);
    else if(saved){ const o=view.loadRender(JSON.parse(saved)); renderName=o.name||""; paintStudio(); }
  }catch(e){ console.warn("render",e); }
}
const nameSetting=r=>LANGUAGE==="fr"&&FR_RENDER[r.key]?FR_RENDER[r.key][0]:r.name, helpSetting=r=>LANGUAGE==="fr"&&FR_RENDER[r.key]?FR_RENDER[r.key][1]:r.help;
const nameGroup=g=>LANGUAGE==="fr"&&FR_GROUPS[g]||g, namePreset=p=>PRESETS[p.id]?PRESETS[p.id][L()]:p.name;
function buildStudio(){
  const z=$("#settings"), before=$("#preset").value; z.textContent="";
  $("#preset").innerHTML=`<option value="">${t("custom")}</option>`+presets.map(p=>`<option value="${ech(p.id)}">${ech(namePreset(p))}</option>`).join("");
  $("#preset").value=before;
  if(!studioReady()){ $("#render-note").textContent=t("simplified"); for(const b of ["#render-import","#render-export","#render-default","#preset"]) $(b).disabled=true; return; }
  const groups=new Map(); for(const r of SETTINGS){ if(!groups.has(r.group)) groups.set(r.group,[]); groups.get(r.group).push(r); }
  let first=true;
  for(const [g,l] of groups){
    const d=document.createElement("details"); d.open=first; first=false;
    d.innerHTML=`<summary>${ech(nameGroup(g))}</summary><div class="body">${l.map(r=>{ const id="r-"+r.key.replace(/\W/g,"-"), a=r.help?` title="${ech(helpSetting(r))}"`:"";
      if(r.type==="color") return `<div class="setting color"${a}><label for="${id}">${ech(nameSetting(r))}</label><input type="color" id="${id}" data-key="${ech(r.key)}"></div>`;
      if(r.type==="toggle") return `<div class="setting toggle"${a}><label for="${id}">${ech(nameSetting(r))}</label><input type="checkbox" id="${id}" data-key="${ech(r.key)}"></div>`;
      if(r.type==="choice") return `<div class="setting choice"${a}><label for="${id}">${ech(nameSetting(r))}</label><select id="${id}" data-key="${ech(r.key)}">${(r.options||[]).map(o=>`<option>${ech(o)}</option>`).join("")}</select></div>`;
      return `<div class="setting"${a}><label for="${id}">${ech(nameSetting(r))}</label><output></output><input type="range" id="${id}" data-key="${ech(r.key)}" min="${r.min}" max="${r.max}" step="${r.step||0.01}"></div>`; }).join("")}</div>`;
    z.append(d);
  }
  z.oninput=e=>{ const key=e.target.dataset.key, r=SETTINGS.find(x=>x.key===key); if(!r) return;
    const v=r.type==="color"?fromHex(e.target.value,r.space==="linear"):r.type==="toggle"?e.target.checked:r.type==="choice"?e.target.value:+e.target.value;
    view.adjust(key,v); if(r.type==="number"||!r.type) e.target.previousElementSibling.textContent=(+e.target.value).toFixed(2);
    $("#preset").value=""; store("chimera.preset",""); rememberRender(); };
  $("#preset").onchange=e=>{ if(e.target.value) choosePreset(e.target.value).catch(x=>say(t("failure")+x.message)); };
  $("#render-import").onclick=()=>$("#render-file").click();
  $("#render-file").onchange=async e=>{ const f=e.target.files[0]; e.target.value=""; if(!f) return;
    try{ const o=view.loadRender(JSON.parse(await f.text())); renderName=o.name||f.name.replace(/\.json$/i,""); $("#preset").value=""; store("chimera.preset",""); paintStudio(); rememberRender(); say(t("render_imported")+renderName); }
    catch(x){ say(t("render_refused")+srv(x.message)); } };
  $("#render-export").onclick=()=>{ const name=renderName||t("custom"), file=(name.normalize("NFD").replace(/[^\w -]/g,"").trim().replace(/\s+/g,"-").toLowerCase()||"render")+".chimera-render.json";
    const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([JSON.stringify(view.exportRender(name),null,1)],{type:"application/json"})); a.download=file; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),2000); say(t("render_exported")+file); };
  $("#render-default").onclick=()=>{ view.loadRender(V.RENDER_DEFAULT); renderName=V.RENDER_DEFAULT.name||""; const p=presets.find(x=>x.name===renderName); $("#preset").value=p?p.id:""; store("chimera.preset",p?p.id:""); paintStudio(); rememberRender(); };
  const d0=presets.find(x=>x.name===(V.RENDER_DEFAULT||{}).name); $("#preset").value=recall("chimera.render","")?recall("chimera.preset",""):(d0?d0.id:"");
  paintStudio();
}
const say=m=>{ $("#render-note").textContent=m; };
const rememberRender=()=>store("chimera.render",JSON.stringify(view.exportRender(renderName)));
async function choosePreset(id){
  const o=view.loadRender(await json(`/page/presets/${encodeURIComponent(id)}.json`)); renderName=o.name||id;
  $("#preset").value=id; store("chimera.preset",id); paintStudio(); rememberRender(); say("");
}
function paintStudio(){
  if(!studioReady()) return; const o=view.render;
  for(const e of $("#settings").querySelectorAll("[data-key]")){ const r=SETTINGS.find(x=>x.key===e.dataset.key), v=read(o,r.key); if(v===undefined) continue;
    if(r.type==="color") e.value=toHex(v,r.space==="linear"); else if(r.type==="toggle") e.checked=!!v; else { e.value=v; if(r.type!=="choice") e.previousElementSibling.textContent=(+v).toFixed(2); } }
}

// ---------- Library: import models, extract their parts ----------
let formats=null, xRunning=false, xProgress=null, xId=null, xVersion=null;
const nb=x=>Array.isArray(x)?x.length:(x||0);
function connectLibrary(){
  const d=$("#drop"), f=$("#file-model");
  f.onchange=()=>{ importFiles([...f.files]); f.value=""; };
  d.ondragover=e=>{ e.preventDefault(); d.classList.add("over"); }; d.ondragleave=()=>d.classList.remove("over");
  d.ondrop=e=>{ e.preventDefault(); d.classList.remove("over"); importFiles([...e.dataTransfer.files]); };
  $("#extract").onclick=extract; $("#cancel").onclick=cancel;
}
async function openLibrary(){
  $("#missing").hidden=!empty(); $("#missing").textContent=t("missing");
  try{
    formats=formats||await json("/api/formats");
    const li=x=>`<li>${x.state?`<span class="state ${x.state==="tested"?"":"partial"}">${t(x.state==="tested"?"tested":"partial")}</span>`:""}<b>${ech(srv(x.name))}</b>${x.extensions?` <code>${ech(x.extensions.join(" "))}</code>`:""}${x.note?`<small>${ech(srv(x.note))}</small>`:""}</li>`;
    $("#formats-input").innerHTML=formats.input.map(li).join(""); $("#list-skeletons").innerHTML=formats.skeletons.map(li).join("");
    const ext=[...new Set(formats.input.flatMap(x=>x.extensions||[]))]; $("#file-model").accept=ext.join(","); $("#drop-formats").textContent=ext.join("  ·  ");
    const ja=formats.active_game||{}; if(ja.import===false){ $("#file-model").disabled=true; $("#drop").style.opacity=.5; $("#import-state").innerHTML=`<div class="alert">${ech(srv(ja.reason))}</div>`; }
  }catch(e){ $("#formats-input").innerHTML=`<li>${ech(e.message)}</li>`; }
  await paintModels();
  followExtraction();
}
async function paintModels(){
  let l=[]; try{ l=await json("/api/models"); }catch(e){ $("#models").innerHTML=`<div class="nothing">${ech(e.message)}</div>`; return; }
  if(l.models) l=l.models;
  $("#models").innerHTML=l.length?l.map(m=>{ const x=m.extracted||{}, r=(m.refusals||[]).length;
    return `<label title="${ech((m.refusals||[]).map(y=>`${srv(y.name)}${LANGUAGE==="fr"?" :":":"} ${srv(y.reason)}`).join("\n"))}"><input type="checkbox" value="${ech(m.id)}" ${x.face&&x.hairstyle&&x.parts?"":"checked"}><b>${ech(m.name||m.id)}</b><span class="chips">`+
      `<span class="chip ${x.face?"yes":""}">${t("face")}</span><span class="chip ${x.hairstyle?"yes":""}">${t("hairstyle")}</span><span class="chip ${x.parts?"yes":""}">${x.parts||0} ${t("parts")}</span>${r?`<span class="chip no">${r} ${t("refusals")}</span>`:""}</span></label>`; }).join(""):`<div class="nothing">${t("no_model")}</div>`;
}
async function importFiles(files){
  const z=$("#import-state");
  for(const f of files){
    z.innerHTML=`<pre>${ech(t("sending")+f.name)}…</pre>`;
    try{
      let r=await json("/api/import?name="+encodeURIComponent(f.name),{method:"PUT",headers:{"Content-Type":"application/octet-stream"},body:f});
      while(r&&r.id&&!r.model){ const e=await json("/api/jobs/"+r.id); if(e.step==="error") throw new Error(srv(e.error)); if(e.done){ r=e; break; } z.innerHTML=`<pre>${ech(t("sending")+f.name)} — ${ech(srv(e.message||e.step||""))}</pre>`; await new Promise(s=>setTimeout(s,1200)); }
      const nm=(r.models||[]).length, nd=(r.discarded||[]).length; z.innerHTML=`<pre>${ech((nm>1?t("imported_n",{n:nm}):t("imported")+((r.model&&(r.model.name||r.model.id))||f.name))+(nd&&nm>1?" · "+t("discarded",{n:nd}):""))}</pre>`;
    }catch(e){ z.innerHTML=`<div class="alert">${ech(f.name)} : ${ech(e.message)}</div>`; }
    await paintModels();
  }
}
// The extraction runs on the server: its state comes from GET /api/extraction (on load, on opening this view, when
// the tab comes back), never from this page, so a reload or another tab finds the bar where it is.
function buttonExtract(running){
  xRunning=running; const b=$("#extract"), a=$("#cancel");
  b.disabled=running; b.classList.toggle("running",running); b.textContent=t(running?"extraction_running":"extract");
  a.hidden=!running; if(!running){ a.disabled=false; a.textContent=t("cancel"); } else if(!a.disabled) a.textContent=t("cancel");
  $("#extraction-server").hidden=!running;
}
function messageExtraction(e){
  if(e.cancellation&&!e.done) return t("cancellation");
  const stage=k=>TXT["step_"+k]?t("step_"+k):(k||"");
  if(e.model&&e.model.name) return t("model_k",{k:e.model.rank,n:e.model.total,name:e.model.name,e:stage(e.model.stage)});
  if(e.batch&&e.stage){ const l=e.batch; return l.from===l.to?t("model_k",{k:l.from,n:l.total,name:(l.names||[""])[0],e:stage(e.stage)}):t("models_k",{k:l.from,m:l.to,n:l.total,e:stage(e.stage)}); }
  if(e.stage==="pivot") return stage("pivot");
  return e.message?srv(e.message):t("waiting");
}
function paintExtraction(e){
  $("#extraction").hidden=false;
  const stepEl=$("#extraction-step"), jo=$("#extraction-log");
  if(e.log){ jo.textContent=Array.isArray(e.log)?e.log.join("\n"):e.log; jo.scrollTop=jo.scrollHeight; }
  if(!e.done){ buttonExtract(true); stepEl.textContent=messageExtraction(e); $("#extraction-bar div").style.width=Math.max(2,Math.round(100*(e.fraction||0)))+"%";
    if(e.cancellation){ $("#cancel").disabled=true; $("#cancel").textContent=t("cancellation"); }
    return; }
  buttonExtract(false);
  if(e.cancelled){ stepEl.textContent=t("extraction_cancelled"); return; }
  if(e.step==="error"||e.ok===false){ stepEl.innerHTML=`<div class="alert">${ech(t("failure")+srv(e.error))}</div>`;return; }
  stepEl.textContent=t("extraction_done")+(e.assessment?` ${nb(e.assessment.face_count)} ${t("face_count").toLowerCase()}, ${nb(e.assessment.hairstyles)} ${t("hairstyles").toLowerCase()}, ${nb(e.assessment.parts)} ${t("parts")}.`:"");
  $("#extraction-bar div").style.width="100%";
}
async function followExtraction(){
  if(xProgress) return xProgress;
  xProgress=(async()=>{
    for(;;){
      let e;
      try{ e=await json("/api/extraction"); }catch(err){ if(xRunning) $("#extraction-step").innerHTML=`<div class="alert">${ech(t("server_silent")+err.message)}</div>`; await new Promise(s=>setTimeout(s,3000)); if(!xRunning) return; continue; }
      if(!e.id){   // nothing running: the one this page followed has just finished
        if(xId){ const f=await json("/api/jobs/"+xId).catch(()=>null); xId=null; if(f) paintExtraction(f); await updateCatalog(); await paintModels(); }
        buttonExtract(false); return;
      }
      if(xId!==e.id){ xId=e.id; xVersion=null; }
      paintExtraction(e);
      if(xVersion!==null&&e.version!==xVersion){ await updateCatalog(); await paintModels(); }   // a model became usable
      xVersion=e.version;
      await new Promise(s=>setTimeout(s,1500));
    }
  })().catch(e=>console.error(e)).finally(()=>{ xProgress=null; });
  return xProgress;
}
async function extract(){
  if(xRunning) return;
  const models=[...$("#models").querySelectorAll("input:checked")].map(i=>i.value), what=[...$("#what").querySelectorAll("input:checked")].map(i=>i.value);
  const z=$("#extraction"), stepEl=$("#extraction-step"), jo=$("#extraction-log"); z.hidden=false; jo.textContent="";
  if(!models.length||!what.length){ stepEl.textContent=t("choose_models"); return; }
  buttonExtract(true); stepEl.textContent=t("waiting"); $("#extraction-bar div").style.width="2%";
  try{
    const {id}=await json("/api/extract",{method:"POST",headers:JSON_H,body:JSON.stringify({models,what})});
    xId=id; xVersion=null;
  }catch(e){ buttonExtract(false); stepEl.innerHTML=`<div class="alert">${ech(t("failure")+e.message)}</div>`; }
  followExtraction();   // also when the server refused because one is already running: follow that one
}
async function cancel(){
  const a=$("#cancel"); a.disabled=true; a.textContent=t("cancellation");
  try{ await json("/api/extraction/cancel",{method:"POST",headers:JSON_H,body:"{}"}); }
  catch(e){ a.disabled=false; a.textContent=t("cancel"); $("#extraction-step").innerHTML=`<div class="alert">${ech(t("failure")+e.message)}</div>`; }
  followExtraction();
}

// ---------- Assembled characters ----------
async function openCharacters(){
  const g=$("#gallery");
  try{
    formats=formats||await json("/api/formats");
    $("#formats-output").innerHTML=formats.output.map(x=>`<li><b>${ech(srv(x.name))}</b>${x.note?`<small>${ech(srv(x.note))}</small>`:""}</li>`).join("");
    let l=await json("/api/assemblies"); if(l.assemblies) l=l.assemblies;
    g.innerHTML=l.length?l.map(a=>`<article class="details-character">${a.sheet?`<a href="/data/assemblies/${encodeURIComponent(a.id)}/sheet.png" target="_blank"><img loading="lazy" src="/data/assemblies/${encodeURIComponent(a.id)}/sheet.png" alt=""></a>`:`<div class="empty">—</div>`}
      <div class="text"><b>${ech(a.name||a.id)}</b><small>${ech((a.date||a.id).replace("T"," "))}</small>${a.done===false||a.error?`<span class="chip no">${ech(a.error?srv(a.error):t("unfinished"))}</span>`:""}${outputs(a.id,a.formats)}</div></article>`).join(""):`<div class="nothing">${t("no_character")}</div>`;
  }catch(e){ g.innerHTML=`<div class="nothing">${ech(e.message)}</div>`; }
}

// Missing thumbnails: drawn once, the server keeps them.
// One pass at a time (the catalogue can change during an extraction); a call during a pass asks for another one.
let thumbnailsPass=null, thumbnailsAgain=false, visThumbnail=null;
function generateThumbnails(){
  if(thumbnailsPass){ thumbnailsAgain=true; return thumbnailsPass; }
  thumbnailsPass=(async()=>{ do{ thumbnailsAgain=false; await thumbnailPass(); }while(thumbnailsAgain); })()
    .catch(e=>console.warn("thumbnails",e)).finally(()=>{ thumbnailsPass=null; });
  return thumbnailsPass;
}
async function thumbnailPass(){
  const list=[...cat.heads.map(x=>["head",x,`/data/heads/${encodeURIComponent(x.id)}/head.glb`,"heads"]),
    ...cat.hairstyles.map(c=>["hairstyle",c,`/data/hairstyles/${encodeURIComponent(c.id)}/hairstyle.glb`,"hairstyles"]),
    ...cat.parts.map(p=>["part",p,`/data/parts/${encodeURIComponent(p.id)}/part.glb`,"parts"])].filter(x=>!x[1].thumbnail&&!thumbnails[x[0]+x[1].id]);
  if(!list.length) return;
  const cv=$("#workshop-thumbnail");
  if(!visThumbnail){ cv.hidden=false; visThumbnail=createViewer(cv,{keep:true}); visThumbnail.dpr=1; cv.hidden=true;
    if(visThumbnail.loadRender) try{ visThumbnail.loadRender(await json("/page/presets/simple.json")); }catch(e){} }   // thumbnails: the plainest render, no backdrop
  const v=visThumbnail;
  for(const [g,it,url,back] of list){
    try{
      v.models.forEach((_,k)=>v.remove(k));
      const m=await v.placer("x",url), d=m.d, lo=[1e9,1e9,1e9], hi=[-1e9,-1e9,-1e9];
      for(let i=0;i<d.pos.length;i+=3) for(let j=0;j<3;j++){lo[j]=Math.min(lo[j],d.pos[i+j]);hi[j]=Math.max(hi[j],d.pos[i+j]);}
      const c=lo.map((a,j)=>(a+hi[j])/2), r=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2])*0.5;
      v.cam={az:0.3,el:0.05,dist:r/Math.tan(0.275)*1.15,aim:c};
      // give the textures time to arrive
      m.prims.forEach(p=>v.texture(p.image));
      await new Promise(res=>{let n=0;const f=()=>{n++; if(m.prims.every(p=>{const x=v.texture(p.image);return !x||x.ready;})||n>40) res(); else setTimeout(f,100);}; f();});
      v.draw();
      const blob=await new Promise(res=>cv.toBlob(res,"image/png"));
      const url2=URL.createObjectURL(blob); thumbnails[g+it.id]=url2; it.thumbnail=url2;
      fetch(`/api/thumbnail/${back}/${encodeURIComponent(it.id)}`,{method:"PUT",body:blob});
      if((state.tab==="face"&&g==="head")||(state.tab==="hairstyle"&&g==="hairstyle")||(g==="part"&&it.tab===state.tab)){
        const b=[...$("#grid").children].find(x=>x.dataset.id===it.id);
        if(b&&b.querySelector(".empty")) b.querySelector(".empty").outerHTML=`<img src="${url2}" alt="">`;
      }
    }catch(e){console.warn("thumbnail",it.id,e);}
  }
}
// For the page_photos.mjs tool: drive the page without a mouse (choices, camera, photos).
window.workshop={view,state,cat:()=>cat,updateHead,updateHairstyle,updateParts,show};
init().catch(e=>{ console.error(e); document.body.insertAdjacentHTML("beforeend",`<p class="banner-note" style="position:fixed;left:20px;right:20px;bottom:20px">${ech(e.message)}</p>`); });
