// Photos of the real workshop page, large and headless: you see what the user sees.
// node tools/workshop/page_photos.mjs [--tete A[,B]] [--coiffure X|aucune] [--tenue] [--pieces a,b] [--sortie folder] [--taille 1400]
//   [--vues visage,cou-dessous] [--couleurs] [--rendu home,day,simple] [--rendu-fichier my-render.json]
//   [--lumiere 60] [--trait 0] [--regler ombre.seuil=0.5,trait.epaisseur=2] [--expression Sourire]
// Without --tete, the head the page already shows. Per head: the neck from the front, three-quarter, profile, below, nape, and the whole face.
// --couleurs paints each skin material in one colour (颜 red, 颜_A green, 肌_cou and 肌_bande blue, 肌_tete purple, body grey):
// you see which surface a defect belongs to.
// --rendu: one photo per preset named (page/presets/<id>.json, file name suffixed); --rendu-fichier: an exported render.
// --lumiere: angle of the light from the camera, in degrees; --trait 0 removes the outline, 1.5 thickens it;
// --regler: any render setting (keys of REGLAGES_RENDU), applied after the preset.
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, l) =>
  x.startsWith("--") ? [...a, [x.slice(2), l[i + 1] && !l[i + 1].startsWith("--") ? l[i + 1] : true]] : a, []));
const tetes = args.tete ? String(args.tete).split(",") : [null];
const sortie = args.sortie || "regard";
const taille = +(args.taille || 1400);
const url = args.url || "http://localhost:8768/";

// az: 0 = front; el: height of the gaze; dist in metres; cible: height aimed at on the body
const VUES = {
  "visage":        { az: 0.0,  el: 0.03,  dist: 0.55, y: 1.47 },
  "visage-trois-q":{ az: 0.7,  el: 0.03,  dist: 0.55, y: 1.47 },
  "visage-autre":  { az: -0.9, el: 0.03,  dist: 0.55, y: 1.47 },
  "yeux":          { az: 0.2,  el: 0.0,   dist: 0.22, y: 1.555 },
  "joue":          { az: 1.1,  el: -0.1,  dist: 0.2,  y: 1.53 },
  "bouche":        { az: 0.25, el: 0.0,   dist: 0.16, y: 1.505, z: 0.08 },
  "bouche-face":   { az: 0.0,  el: 0.0,   dist: 0.16, y: 1.505, z: 0.08 },
  "oreille":       { az: 2.2,  el: 0.0,   dist: 0.2,  y: 1.55, x: 0.07 },
  "tete":          { az: 0.0,  el: 0.1,   dist: 0.9,  y: 1.58 },
  "tete-trois-q":  { az: 0.8,  el: 0.25,  dist: 0.9,  y: 1.58 },
  "tete-dos":      { az: 2.6,  el: 0.25,  dist: 0.9,  y: 1.58 },
  "cou-face":      { az: 0.0,  el: 0.0,   dist: 0.32, y: 1.46 },
  "cou-trois-q":   { az: 0.7,  el: 0.0,   dist: 0.32, y: 1.46 },
  "cou-profil":    { az: 1.57, el: 0.0,   dist: 0.32, y: 1.46 },
  "cou-dessous":   { az: 0.3,  el: -0.6,  dist: 0.32, y: 1.47 },
  "cou-nuque":     { az: 3.14, el: 0.1,   dist: 0.32, y: 1.46 },
  "buste-profil":  { az: 1.57, el: 0.0,   dist: 0.50, y: 1.42 },
  "buste-trois-q": { az: 0.7,  el: 0.0,   dist: 0.50, y: 1.42 },
  "buste-dos":     { az: 2.4,  el: 0.0,   dist: 0.50, y: 1.42 },
  "buste-face":    { az: 0.0,  el: 0.0,   dist: 1.3,  y: 1.25 },
  "collier":       { az: 0.0,  el: 0.0,   dist: 0.28, y: 1.33 },
  "ruban":         { az: 0.0,  el: 0.0,   dist: 0.2,  y: 1.255, x: 0.2, z: 0.12 },
  "ruban-cote":    { az: 1.3,  el: 0.0,   dist: 0.2,  y: 1.255, x: 0.2, z: 0.12 },
  "ruban-dessus":  { az: 0.4,  el: 0.7,   dist: 0.2,  y: 1.255, x: 0.2, z: 0.12 },
  "corps":         { az: 0.5,  el: 0.05,  dist: 3.2,  y: 0.85 },
  "dos":           { az: 3.14, el: 0.05,  dist: 3.2,  y: 0.85 },
  "fesses-dos":    { az: 3.14, el: 0.0,   dist: 0.75, y: 0.86 },
  "fesses-trois-q":{ az: 2.4,  el: 0.0,   dist: 0.75, y: 0.86 },
  "fesses-profil": { az: 1.57, el: 0.0,   dist: 0.75, y: 0.86 },
};

fs.mkdirSync(sortie, { recursive: true });
// Playwright's Chromium if it is installed, otherwise the system Chrome (CHROME=… to pick another one)
const chrome = process.env.CHROME || (fs.existsSync("/usr/bin/google-chrome") ? "/usr/bin/google-chrome" : undefined);
const nav = await chromium.launch({ executablePath: fs.existsSync(chromium.executablePath()) ? undefined : chrome, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await nav.newPage({ viewport: { width: taille, height: taille } });
page.on("pageerror", e => console.error("page:", e.message));
page.setDefaultTimeout(180000);   // loaded machine: a photo can wait more than 30 s
await page.goto(url);
// wait for the page to finish starting up (face, hairstyle, outfit in place), otherwise our calls cross its own
await page.waitForFunction(() => window.atelier && window.atelier.cat() && window.atelier.vue.modeles.has("tete") && document.body.dataset.couches === "pret", null, { timeout: 120000 });
// the view fills the whole window
await page.addStyleTag({ content: "#bandeau,#choix,#fiche,#commandes,#charge,#simplifie{display:none!important} body{display:block!important;margin:0} #scene{position:fixed;inset:0} #vue{width:100vw!important;height:100vh!important}" });

// command-line settings: --lumiere, --trait, --regler key=value,…
const nombre = x => (x === "true" ? true : x === "false" ? false : isNaN(+x) ? x : +x);
const reglages = [];
if (args.lumiere != null) reglages.push(["lumiere.angle", +args.lumiere]);
if (args.trait != null) { reglages.push(["trait.actif", +args.trait > 0]); if (+args.trait > 0) reglages.push(["trait.epaisseur", +args.trait]); }
if (args.regler) for (const x of String(args.regler).split(",")) { const i = x.indexOf("="); reglages.push([x.slice(0, i), nombre(x.slice(i + 1))]); }

const vues = args.vues ? Object.fromEntries(String(args.vues).split(",").map(n => [n, VUES[n]])) : VUES;
const COULEURS = { "颜": [0.95, 0.35, 0.35, 1], "颜_A": [0.35, 0.85, 0.35, 1], "肌_cou": [0.35, 0.5, 0.95, 1], "肌_bande": [0.35, 0.5, 0.95, 1], "肌_tete": [0.8, 0.4, 0.9, 1] };

for (let tete of tetes) {
  tete = await page.evaluate(async ({ tete, coiffure, tenue, pieces }) => {
    const A = window.atelier, e = A.etat;
    if (tete) e.tete = tete;
    if (coiffure === "aucune") e.coiffure = null; else if (coiffure) e.coiffure = coiffure;
    if (pieces) { e.pieces.clear(); for (const id of pieces) e.pieces.add(id); } else if (!tenue) e.pieces.clear();
    await A.majTete();
    if (e.coiffure) await A.majCoiffure(); else A.vue.retirer && A.vue.modeles.has("coiffure") && A.vue.retirer("coiffure");
    await A.majPieces();
    return e.tete;
  }, { tete, coiffure: args.coiffure, tenue: !!args.tenue, pieces: args.pieces ? String(args.pieces).split(',') : null });
  // check: the drawn body must be the one the graft leaves under this face (corps.bin), otherwise the
  // common body's head pokes out under the face and the photo lies
  const ecart = await page.evaluate(async (tete) => {
    const w = window.atelier.vue, g = w.g, m = w.modeles.get("corps");
    if (!g || !g.getBufferSubData) return null;
    const r = await fetch(`/data/tetes/${encodeURIComponent(tete)}/corps.bin`);
    if (!r.ok) return null;
    const a = new Float32Array(m.d.pos.length), b = new Float32Array(await r.arrayBuffer());
    g.bindBuffer(g.ARRAY_BUFFER, m.bpos); g.getBufferSubData(g.ARRAY_BUFFER, 0, a);
    let x = 0; for (let i = 0; i < a.length; i++) x = Math.max(x, Math.abs(a[i] - b[i]));
    return x;
  }, tete);
  if (ecart !== null && ecart > 1e-6) { console.error(`${tete}: drawn body ≠ corps.bin (gap ${ecart.toFixed(4)} m), photo rejected`); process.exitCode = 1; continue; }
  // textures
  await page.waitForFunction(() => [...window.atelier.vue.tex.values()].every(t => t.pret), null, { timeout: 60000 }).catch(() => {});
  if (args.couleurs) await page.evaluate(({ C }) => {
    const w = window.atelier.vue;
    for (const [cle, m] of w.modeles) for (const p of m.prims) {
      if (cle === "coiffure") continue;
      if (cle === "tete") m.peau = null;
      if (cle === "corps") { m.peau = null; p.image = null; p.diffus = [0.75, 0.75, 0.75, 1]; }
      else if (cle === "tete") { p.image = null; p.diffus = C[p.nom] || [0.95, 0.85, 0.3, 1]; }
    }
    w.sale = true;
  }, { C: COULEURS });
  if (args.expression) await page.evaluate(e => window.atelier.vue.exprimer(e), args.expression);
  for (const [nom, v] of Object.entries(vues)) {
    await page.evaluate(v => { const w = window.atelier.vue; w.vise = null; w.tourne = false;
      w.cam = { az: v.az, el: v.el, dist: v.dist, cible: [v.x || 0, v.y, v.z || 0] }; w.sale = true; }, v);
    // textures are only requested on the first draw: draw, then wait for them (loaded machine, they are slow)
    await page.evaluate(() => window.atelier.vue.dessiner());
    await page.waitForFunction(() => { const w = window.atelier.vue; if ([...w.tex.values()].every(t => t.pret)) { w.sale = true; return true; } return false; }, null, { timeout: 120000 }).catch(() => {});
    for (const rendu of args.rendu ? String(args.rendu).split(",") : [null]) {
      // the render: preset (or exported file), then the command-line settings
      await page.evaluate(async ({ rendu, doc, reglages }) => { const w = window.atelier.vue;
        if (rendu) w.chargerRendu(await fetch(`/page/presets/${encodeURIComponent(rendu)}.json`).then(r => { if (!r.ok) throw new Error("unknown preset: " + rendu); return r.json(); }));
        else if (doc) w.chargerRendu(doc);
        for (const [cle, val] of reglages) w.regler(cle, val);
        w.sale = true; }, { rendu, doc: args["rendu-fichier"] ? fs.readFileSync(args["rendu-fichier"], "utf8") : null, reglages });
      await page.waitForTimeout(400);
      const f = path.join(sortie, `${tete}-${nom}${rendu ? "-" + rendu : ""}.png`);
      await page.locator("#vue").screenshot({ path: f });
      console.log(f);
    }
  }
}
await nav.close();
