// Photos of the real workshop page, large and headless: you see what the user sees.
// node tools/workshop/page_photos.mjs [--head A[,B]] [--hairstyle X|none] [--outfit] [--parts a,b] [--output folder] [--size 1400]
//   [--views face,neck-under] [--colors] [--render home,day,simple] [--render-file my-render.json]
//   [--light 60] [--outline 0] [--adjust shadow.threshold=0.5,outline.thickness=2] [--expression Smile]
// Without --head, the head the page already shows. Per head: the neck from the front, three-quarter, profile, below, nape, and the whole face.
// --colors paints each skin material in one colour (颜 red, 颜_A green, 肌_cou and 肌_bande blue, 肌_tete purple, body grey):
// you see which surface a defect belongs to.
// --render: one photo per preset named (page/presets/<id>.json, file name suffixed); --render-file: an exported render.
// --light: angle of the light from the camera, in degrees; --outline 0 removes the outline, 1.5 thickens it;
// --adjust: any render setting (keys of RENDER_SETTINGS), applied after the preset.
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, l) =>
  x.startsWith("--") ? [...a, [x.slice(2), l[i + 1] && !l[i + 1].startsWith("--") ? l[i + 1] : true]] : a, []));
const heads = args.head ? String(args.head).split(",") : [null];
const output = args.output || "gaze";
const size = +(args.size || 1400);
const url = args.url || "http://localhost:8768/";

// az: 0 = front; el: height of the gaze; dist in metres; y: height aimed at on the body
const VIEWS = {
  "face":        { az: 0.0,  el: 0.03,  dist: 0.55, y: 1.47 },
  "face-three-q":{ az: 0.7,  el: 0.03,  dist: 0.55, y: 1.47 },
  "face-other":  { az: -0.9, el: 0.03,  dist: 0.55, y: 1.47 },
  "eyes":          { az: 0.2,  el: 0.0,   dist: 0.22, y: 1.555 },
  "cheek":         { az: 1.1,  el: -0.1,  dist: 0.2,  y: 1.53 },
  "mouth":        { az: 0.25, el: 0.0,   dist: 0.16, y: 1.505, z: 0.08 },
  "mouth-face":   { az: 0.0,  el: 0.0,   dist: 0.16, y: 1.505, z: 0.08 },
  "ear":       { az: 2.2,  el: 0.0,   dist: 0.2,  y: 1.55, x: 0.07 },
  "head":          { az: 0.0,  el: 0.1,   dist: 0.9,  y: 1.58 },
  "head-three-q":  { az: 0.8,  el: 0.25,  dist: 0.9,  y: 1.58 },
  "head-back":      { az: 2.6,  el: 0.25,  dist: 0.9,  y: 1.58 },
  "neck-face":      { az: 0.0,  el: 0.0,   dist: 0.32, y: 1.46 },
  "neck-three-q":   { az: 0.7,  el: 0.0,   dist: 0.32, y: 1.46 },
  "neck-profile":    { az: 1.57, el: 0.0,   dist: 0.32, y: 1.46 },
  "neck-under":   { az: 0.3,  el: -0.6,  dist: 0.32, y: 1.47 },
  "neck-nape":     { az: 3.14, el: 0.1,   dist: 0.32, y: 1.46 },
  "bust-profile":  { az: 1.57, el: 0.0,   dist: 0.50, y: 1.42 },
  "bust-three-q": { az: 0.7,  el: 0.0,   dist: 0.50, y: 1.42 },
  "bust-back":     { az: 2.4,  el: 0.0,   dist: 0.50, y: 1.42 },
  "bust-face":    { az: 0.0,  el: 0.0,   dist: 1.3,  y: 1.25 },
  "necklace":       { az: 0.0,  el: 0.0,   dist: 0.28, y: 1.33 },
  "ribbon":         { az: 0.0,  el: 0.0,   dist: 0.2,  y: 1.255, x: 0.2, z: 0.12 },
  "ribbon-side":    { az: 1.3,  el: 0.0,   dist: 0.2,  y: 1.255, x: 0.2, z: 0.12 },
  "ribbon-over":  { az: 0.4,  el: 0.7,   dist: 0.2,  y: 1.255, x: 0.2, z: 0.12 },
  "body":         { az: 0.5,  el: 0.05,  dist: 3.2,  y: 0.85 },
  "back":           { az: 3.14, el: 0.05,  dist: 3.2,  y: 0.85 },
  "buttocks-back":    { az: 3.14, el: 0.0,   dist: 0.75, y: 0.86 },
  "buttocks-three-q":{ az: 2.4,  el: 0.0,   dist: 0.75, y: 0.86 },
  "buttocks-profile": { az: 1.57, el: 0.0,   dist: 0.75, y: 0.86 },
};

fs.mkdirSync(output, { recursive: true });
// Playwright's Chromium if it is installed, otherwise the system Chrome (CHROME=… to pick another one)
const chrome = process.env.CHROME || (fs.existsSync("/usr/bin/google-chrome") ? "/usr/bin/google-chrome" : undefined);
const nav = await chromium.launch({ executablePath: fs.existsSync(chromium.executablePath()) ? undefined : chrome, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await nav.newPage({ viewport: { width: size, height: size } });
page.on("pageerror", e => console.error("page:", e.message));
page.setDefaultTimeout(180000);   // loaded machine: a photo can wait more than 30 s
await page.goto(url);
// wait for the page to finish starting up (face, hairstyle, outfit in place), otherwise our calls cross its own
await page.waitForFunction(() => window.workshop && window.workshop.cat() && window.workshop.view.models.has("head") && document.body.dataset.layers === "ready", null, { timeout: 120000 });
// the view fills the whole window
await page.addStyleTag({ content: "#topbar,#choice,#details,#controls,#loader,#simplified{display:none!important} body{display:block!important;margin:0} #scene{position:fixed;inset:0} #view{width:100vw!important;height:100vh!important}" });

// command-line settings: --light, --trait, --adjust key=value,…
const number = x => (x === "true" ? true : x === "false" ? false : isNaN(+x) ? x : +x);
const settings = [];
if (args.light != null) settings.push(["light.angle", +args.light]);
if (args.outline != null) { settings.push(["outline.active", +args.outline > 0]); if (+args.outline > 0) settings.push(["outline.thickness", +args.outline]); }
if (args.adjust) for (const x of String(args.adjust).split(",")) { const i = x.indexOf("="); settings.push([x.slice(0, i), number(x.slice(i + 1))]); }

const views = args.views ? Object.fromEntries(String(args.views).split(",").map(n => [n, VIEWS[n]])) : VIEWS;
const COLORS = { "颜": [0.95, 0.35, 0.35, 1], "颜_A": [0.35, 0.85, 0.35, 1], "肌_cou": [0.35, 0.5, 0.95, 1], "肌_bande": [0.35, 0.5, 0.95, 1], "肌_tete": [0.8, 0.4, 0.9, 1] };

for (let head of heads) {
  head = await page.evaluate(async ({ head, hairstyle, outfit, parts }) => {
    const A = window.workshop, e = A.state;
    if (head) e.head = head;
    if (hairstyle === "none") e.hairstyle = null; else if (hairstyle) e.hairstyle = hairstyle;
    if (parts) { e.parts.clear(); for (const id of parts) e.parts.add(id); } else if (!outfit) e.parts.clear();
    await A.updateHead();
    if (e.hairstyle) await A.updateHairstyle(); else A.view.remove && A.view.models.has("hairstyle") && A.view.remove("hairstyle");
    await A.updateParts();
    return e.head;
  }, { head, hairstyle: args.hairstyle, outfit: !!args.outfit, parts: args.parts ? String(args.parts).split(',') : null });
  // check: the drawn body must be the one the graft leaves under this face (corps.bin), otherwise the
  // common body's head pokes out under the face and the photo lies
  const gap = await page.evaluate(async (head) => {
    const w = window.workshop.view, g = w.g, m = w.models.get("body");
    if (!g || !g.getBufferSubData) return null;
    const r = await fetch(`/data/heads/${encodeURIComponent(head)}/body.bin`);
    if (!r.ok) return null;
    const a = new Float32Array(m.d.pos.length), b = new Float32Array(await r.arrayBuffer());
    g.bindBuffer(g.ARRAY_BUFFER, m.bpos); g.getBufferSubData(g.ARRAY_BUFFER, 0, a);
    let x = 0; for (let i = 0; i < a.length; i++) x = Math.max(x, Math.abs(a[i] - b[i]));
    return x;
  }, head);
  if (gap !== null && gap > 1e-6) { console.error(`${head}: drawn body ≠ body.bin (gap ${gap.toFixed(4)} m), photo rejected`); process.exitCode = 1; continue; }
  // textures
  await page.waitForFunction(() => [...window.workshop.view.tex.values()].every(t => t.ready), null, { timeout: 60000 }).catch(() => {});
  if (args.colors) await page.evaluate(({ C }) => {
    const w = window.workshop.view;
    for (const [key, m] of w.models) for (const p of m.prims) {
      if (key === "hairstyle") continue;
      if (key === "head") m.skin = null;
      if (key === "body") { m.skin = null; p.image = null; p.diffuse = [0.75, 0.75, 0.75, 1]; }
      else if (key === "head") { p.image = null; p.diffuse = C[p.name] || [0.95, 0.85, 0.3, 1]; }
    }
    w.sale = true;
  }, { C: COLORS });
  if (args.expression) await page.evaluate(e => window.workshop.view.express(e), args.expression);
  for (const [name, v] of Object.entries(views)) {
    await page.evaluate(v => { const w = window.workshop.view; w.target = null; w.spinning = false;
      w.cam = { az: v.az, el: v.el, dist: v.dist, aim: [v.x || 0, v.y, v.z || 0] }; w.sale = true; }, v);
    // textures are only requested on the first draw: draw, then wait for them (loaded machine, they are slow)
    await page.evaluate(() => window.workshop.view.draw());
    await page.waitForFunction(() => { const w = window.workshop.view; if ([...w.tex.values()].every(t => t.ready)) { w.sale = true; return true; } return false; }, null, { timeout: 120000 }).catch(() => {});
    for (const render of args.render ? String(args.render).split(",") : [null]) {
      // the render: preset (or exported file), then the command-line settings
      await page.evaluate(async ({ render, doc, settings }) => { const w = window.workshop.view;
        if (render) w.loadRender(await fetch(`/page/presets/${encodeURIComponent(render)}.json`).then(r => { if (!r.ok) throw new Error("unknown preset: " + render); return r.json(); }));
        else if (doc) w.loadRender(doc);
        for (const [key, val] of settings) w.adjust(key, val);
        w.sale = true; }, { render, doc: args["render-file"] ? fs.readFileSync(args["render-file"], "utf8") : null, settings });
      await page.waitForTimeout(400);
      const f = path.join(output, `${head}-${name}${render ? "-" + render : ""}.png`);
      await page.locator("#view").screenshot({ path: f });
      console.log(f);
    }
  }
}
await nav.close();
