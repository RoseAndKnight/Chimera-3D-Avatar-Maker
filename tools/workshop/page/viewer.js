// Dependency-free WebGL2 viewer: GLB models from the cache, a single toon render tunable like a studio, hair tint
// (the computation of assemble_blender.recolor). The render is a document (RENDER_DEFAULT, RENDER_SETTINGS): direction and
// colour of the light, two-tone shadows (the form, what stands in front, the crevices), shadow tint per material, outline,
// background and floor, finish (contact shadows, bloom, depth of field, grain). It is exported and imported as JSON
// ("chimera-render" format); the presets are in page/presets/. Parts only have their base texture: occlusion,
// gloss and face shadow come from the computed maps (maps.py), see the README.
const bottom=a=>{ let h=1e9; for(let i=1;i<a.length;i+=3) if(a[i]<h) h=a[i]; return h; };
const height=a=>{ let h=-1e9; for(let i=1;i<a.length;i+=3) if(a[i]>h) h=a[i]; return h; };

// ---- The render: a tunable document ----
// Each setting: dotted key in the document, displayed group and name, type, bounds, default (= the "Studio" render, the evening one).
// Colours: [r,g,b] from 0 to 1; `space` says whether the value is linear (it multiplies the light) or sRGB (it is displayed as is).
export const RENDER_FORMAT="chimera-render", RENDER_VERSION=2;
// Version 1 named its settings in French: read, then rewritten with these names.
const RENDER_V1={format:"chimera-rendu",version:1,name:"nom",keys:{"lumiere.angle":"light.angle","lumiere.hauteur":"light.height","lumiere.couleur":"light.color","lumiere.force":"light.force","lumiere.contour":"light.rim","lumiere.forceContour":"light.forceRim","lumiere.ciel":"light.sky","lumiere.terre":"light.ground","ombre.seuil":"shadow.threshold","ombre.douceur":"shadow.softness","ombre.teinte":"shadow.tint","ombre.teinteCheveux":"shadow.tintHair","ombre.saturation":"shadow.saturation","ombre.lisere":"shadow.edgeBand","ombre.creux":"shadow.crevices","ombre.portees":"shadow.casts","ombre.forcePortees":"shadow.castStrength","ombre.contact":"shadow.contact","peau.seuil":"skin.threshold","peau.douceur":"skin.softness","peau.teinte":"skin.tint","peau.transmission":"skin.transmission","peau.rose":"skin.blush","peau.ombreYeux":"skin.shadowEyes","matiere.vetements":"material.clothes","matiere.cheveux":"material.hair","trait.actif":"outline.active","trait.epaisseur":"outline.thickness","trait.opacite":"outline.opacity","trait.couleur":"outline.color","trait.partCouleur":"outline.partColor","fond.centre":"background.center","fond.bord":"background.border","sol.actif":"floor.active","sol.lumiere":"floor.light","sol.ombre":"floor.shadow","image.exposition":"image.exposure","image.saturation":"image.saturation","image.halo":"image.halo","image.flou":"image.blur","image.vignette":"image.thumbnail","image.grain":"image.grain"}};   // fr-ok: the French setting names of version 1, read to convert them
const DEG=180/Math.PI;
export const RENDER_SETTINGS=Object.freeze([
  // Light
  {key:"light.angle", group:"Light", name:"Angle", type:"number", min:-180, max:180, step:1, unit:"°", default:0.5*DEG, help:"Side the light comes from, in degrees from the camera (0: front; 90: side; 180: backlit). It turns with the view."},
  {key:"light.height", group:"Light", name:"Height", type:"number", min:-30, max:89, step:1, unit:"°", default:Math.atan2(0.6,0.8)*DEG, help:"Height of the light above the horizon, in degrees (negative: from below)."},
  {key:"light.color", group:"Light", name:"Colour", type:"color", space:"linear", default:[1,0.97/1.03,0.90/1.03], help:"Colour of the main light, on lit areas."},
  {key:"light.force", group:"Light", name:"Strength", type:"number", min:0, max:2, step:0.01, default:1.03, help:"Strength of the main light: gap between lit and shaded."},
  {key:"light.rim", group:"Light", name:"Rim colour", type:"color", space:"linear", default:[1.0,0.93,0.88], help:"Colour of the rim light, from behind, that separates the silhouette from the background."},
  {key:"light.forceRim", group:"Light", name:"Rim light", type:"number", min:0, max:3, step:0.05, default:1, help:"Strength of the rim light (0: none)."},
  {key:"light.sky", group:"Light", name:"Sky in the shade", type:"color", space:"linear", default:[0.02,0.03,0.06], help:"Sky light bounced into the shade, on surfaces facing up."},
  {key:"light.ground", group:"Light", name:"Ground in the shade", type:"color", space:"linear", default:[0.06,0.03,0.02], help:"Ground light bounced into the shade, on surfaces facing down."},
  // Shadows
  {key:"shadow.threshold", group:"Shadows", name:"Threshold", type:"number", min:0, max:1, step:0.01, default:0.46, help:"Where the shadow edge falls on clothes and hair: higher means more shadow."},
  {key:"shadow.softness", group:"Shadows", name:"Softness", type:"number", min:0.001, max:0.5, step:0.005, default:0.025, help:"Width of the passage from light to shadow (small: hard edge)."},
  {key:"shadow.tint", group:"Shadows", name:"Clothes tint", type:"color", space:"linear", default:[0.32,0.27,0.46], help:"Colour by which a garment's colour is multiplied in the shade."},
  {key:"shadow.tintHair", group:"Shadows", name:"Hair tint", type:"color", space:"linear", default:[0.36,0.27,0.44], help:"Colour by which the hair colour is multiplied in the shade."},
  {key:"shadow.saturation", group:"Shadows", name:"Saturation", type:"number", min:0, max:2, step:0.01, default:1.15, help:"More vivid (above 1) or greyer colour in the shade of clothes and hair."},
  {key:"shadow.edgeBand", group:"Shadows", name:"Edge band", type:"number", min:0, max:3, step:0.05, default:1, help:"More saturated band along the shadow edge, like light passing through skin."},
  {key:"shadow.crevices", group:"Shadows", name:"Crevices", type:"number", min:0, max:2, step:0.05, default:1, help:"Shade in crevices (armpits, folds, under strands), from the occlusion maps."},
  {key:"shadow.casts", group:"Shadows", name:"Cast shadows", type:"toggle", default:true, help:"What stands before the light casts a shadow (hair on the face, arm on the body, character on the floor)."},
  {key:"shadow.castStrength", group:"Shadows", name:"Cast shadow strength", type:"number", min:0, max:1, step:0.01, default:1, help:"Darkness of cast shadows (1: as dark as the form shadow)."},
  {key:"shadow.contact", group:"Shadows", name:"Contact shadow", type:"number", min:0, max:3, step:0.05, default:1, help:"What lies close to a surface darkens it: collar, belt, strand on the shoulder (0: none)."},
  // Skin
  {key:"skin.threshold", group:"Skin", name:"Threshold", type:"number", min:0, max:1, step:0.01, default:0.46, help:"Where the shadow edge falls on skin and face."},
  {key:"skin.softness", group:"Skin", name:"Softness", type:"number", min:0.001, max:0.5, step:0.005, default:0.09, help:"Width of the passage from light to shadow on skin."},
  {key:"skin.tint", group:"Skin", name:"Shade tint", type:"color", space:"linear", default:[0.60,0.36,0.47], help:"Colour the skin is multiplied by in the shade (eyes and mouth follow a cooler version)."},
  {key:"skin.transmission", group:"Skin", name:"Light through skin", type:"number", min:0, max:3, step:0.05, default:1, help:"Backlit, light passes through thin skin (ears, fingers) and thin strands: warm glow."},
  {key:"skin.blush", group:"Skin", name:"Face blush", type:"number", min:0, max:2, step:0.05, default:1, help:"Pink of the cheeks, nose and lips, and the small lip highlight."},
  {key:"skin.shadowEyes", group:"Skin", name:"Shade over the eyes", type:"number", min:0, max:1, step:0.05, default:1, help:"Dark band over the eyes carried by some expressions (0: never)."},
  // Materials
  {key:"material.clothes", group:"Materials", name:"Clothes materials", type:"toggle", default:true, help:"Highlights by material read from the garment name (cloth, knit, leather, tights, silk, glass, metal). Off: matte clothes."},
  {key:"material.hair", group:"Materials", name:"Hair highlights", type:"number", min:0, max:3, step:0.05, default:1, help:"Highlight band across the strands, darker roots and lighter tips (0: matte hair)."},
  // Outline
  {key:"outline.active", group:"Outline", name:"Outline", type:"toggle", default:false, help:"Line at the edge of shapes, in a darker version of the surface colour."},
  {key:"outline.thickness", group:"Outline", name:"Thickness", type:"number", min:0.25, max:3, step:0.05, default:1, help:"Line thickness (it thins with distance)."},
  {key:"outline.opacity", group:"Outline", name:"Opacity", type:"number", min:0, max:1, step:0.01, default:1, help:"Line opacity."},
  {key:"outline.color", group:"Outline", name:"Colour", type:"color", space:"srgb", default:[0.16,0.09,0.13], help:"Fixed line colour, mixed with the surface colour according to \"Colour share\"."},
  {key:"outline.partColor", group:"Outline", name:"Colour share", type:"number", min:0, max:1, step:0.01, default:0, help:"0: the line takes the darkened surface colour; 1: the fixed colour everywhere."},
  // Background and floor
  {key:"background.center", group:"Background and floor", name:"Background center", type:"color", space:"srgb", default:[0.27,0.19,0.27], help:"Background colour behind the character."},
  {key:"background.border", group:"Background and floor", name:"Background edge", type:"color", space:"srgb", default:[0.075,0.05,0.09], help:"Background colour towards the edges of the image."},
  {key:"floor.active", group:"Background and floor", name:"Floor", type:"toggle", default:true, help:"A disc of light under the feet, which receives the character's shadow."},
  {key:"floor.light", group:"Background and floor", name:"Lit floor", type:"color", space:"srgb", default:[0.36,0.27,0.33], help:"Floor colour at the center of the disc, in full light."},
  {key:"floor.shadow", group:"Background and floor", name:"Floor in the shade", type:"color", space:"srgb", default:[0.11,0.075,0.14], help:"Colour of the character's shadow on the floor."},
  // Finish
  {key:"image.exposure", group:"Finish", name:"Exposure", type:"number", min:0.2, max:1.6, step:0.01, default:0.77, help:"Brightness of the whole image (below 1, highlights stay under white and the bloom does the rest)."},
  {key:"image.saturation", group:"Finish", name:"Saturation", type:"number", min:0, max:2, step:0.01, default:1.08, help:"Colour of the whole image: 1 unchanged, 0 black and white."},
  {key:"image.halo", group:"Finish", name:"Bloom", type:"number", min:0, max:3, step:0.05, default:1, help:"Glow around bright areas (0: none)."},
  {key:"image.blur", group:"Finish", name:"Depth of field", type:"number", min:0, max:3, step:0.05, default:1, help:"Blur of what is far from the subject, mostly in close-ups (0: all sharp)."},
  {key:"image.thumbnail", group:"Finish", name:"Thumbnail", type:"number", min:0, max:1, step:0.01, default:0.20, help:"Darkening of the image corners."},
  {key:"image.grain", group:"Finish", name:"Grain", type:"number", min:0, max:0.2, step:0.005, default:0.045, help:"Light grain over the whole image."},
].map(Object.freeze));
const BY_KEY=new Map(RENDER_SETTINGS.map(r=>[r.key,r]));
const read=(o,key)=>key.split(".").reduce((x,k)=>x!=null&&typeof x==="object"?x[k]:undefined,o);
const place=(o,key,v)=>{ const k=key.split("."), d=k.pop(); let x=o; for(const a of k) x=(x[a]&&typeof x[a]==="object")?x[a]:(x[a]={}); x[d]=v; };
const copy=o=>JSON.parse(JSON.stringify(o));
const freeze=o=>{ for(const v of Object.values(o)) if(v&&typeof v==="object") freeze(v); return Object.freeze(o); };
const lin1=x=>Math.pow(x,2.2), enc1=x=>Math.pow(Math.max(0,Math.min(1,x)),1/2.2);
// A value for this setting, clamped; undefined if it is not one.
function value(r,v){
  if(r.type==="number"){ const x=typeof v==="string"&&v.trim()!==""?Number(v):v; return typeof x==="number"&&Number.isFinite(x)?Math.min(r.max,Math.max(r.min,x)):undefined; }
  if(r.type==="toggle") return typeof v==="boolean"?v:(v===0||v===1?!!v:undefined);
  if(r.type==="choice") return r.options.includes(v)?v:undefined;
  if(r.type==="color"){
    if(typeof v==="string"&&/^#?[0-9a-f]{6}$/i.test(v.trim())){ const h=v.trim().replace("#",""); v=[0,2,4].map(i=>parseInt(h.slice(i,i+2),16)/255); if(r.space==="linear") v=v.map(lin1); }
    return Array.isArray(v)&&v.length===3&&v.every(x=>typeof x==="number"&&Number.isFinite(x))?v.map(x=>Math.min(1,Math.max(0,x))):undefined;
  }
}
export const RENDER_DEFAULT=(()=>{ const o={format:RENDER_FORMAT,version:RENDER_VERSION,name:"Studio"}; for(const r of RENDER_SETTINGS) place(o,r.key,copy(r.default)); return freeze(o); })();
// Document read (object or JSON text) -> complete render: format and version checked, unknown keys ignored, values clamped,
// missing or unreadable ones replaced by the default.
export function normalizeRender(o){
  if(typeof o==="string"){ try{ o=JSON.parse(o); }catch(e){ throw new Error("This file is not a render: unreadable JSON ("+e.message+")."); } }
  if(!o||typeof o!=="object"||Array.isArray(o)) throw new Error("Not a render: a JSON object is expected.");
  if(o.format===RENDER_V1.format&&o.version===RENDER_V1.version){
    const n={format:RENDER_FORMAT,version:RENDER_VERSION,name:o[RENDER_V1.name]};
    for(const [k1,k2] of Object.entries(RENDER_V1.keys)){ const x=read(o,k1); if(x!==undefined) place(n,k2,x); }
    o=n;
  }
  if(o.format!==RENDER_FORMAT) throw new Error(`Not a render: "format" must be "${RENDER_FORMAT}"${o.format!==undefined?` (read: ${JSON.stringify(o.format)})`:""}.`);
  if(!Number.isInteger(o.version)||o.version<1) throw new Error("Unreadable render: \"version\" must be an integer.");
  if(o.version>RENDER_VERSION) throw new Error(`Render version ${o.version}: this viewer only reads version ${RENDER_VERSION}.`);
  const n={format:RENDER_FORMAT,version:RENDER_VERSION,name:typeof o.name==="string"&&o.name.trim()?o.name.trim().slice(0,80):"Untitled"};
  for(const r of RENDER_SETTINGS){ const v=value(r,read(o,r.key)); place(n,r.key,v===undefined?copy(r.default):v); }
  return n;
}

const VS = `#version 300 es
layout(location=0) in vec3 p; layout(location=1) in vec3 n; layout(location=2) in vec2 t; layout(location=3) in vec4 k; layout(location=4) in vec3 p0;
uniform mat4 VP; out vec3 N; out vec2 UV; out vec3 P; out vec4 K; out vec3 P0;
void main(){ N=n; UV=t; P=p; K=k; P0=p0; gl_Position=VP*vec4(p,1.0); }`;
const FS = `#version 300 es
precision highp float; precision highp sampler2D; precision lowp sampler2DShadow; in vec3 N; in vec2 UV; in vec3 P; in vec4 K; in vec3 P0; out vec4 c;   // P0: the point's rest position (head: without the expression)
uniform sampler2D T; uniform bool aTex; uniform vec4 diffuse; uniform vec3 L; uniform vec4 tint; uniform vec3 ref; uniform int sort; uniform vec3 eye;
// part maps (maps.py): K = occlusion, gloss, face lit from +x, from −x; height of the top of the hair
uniform bool aMaps; uniform float top;
// depth maps seen from the light (everything; everything but the head), garment material (0 cloth, 1 knit, 2 leather, 3 tights, 4 silk, 5 glass)
uniform mat4 LM; uniform mat4 LMt; uniform sampler2DShadow Sh; uniform sampler2DShadow ShadowNoHead; uniform float perMeter; uniform int material;
// drawn face map (maps.py, per character): image wrapped around the head, red = light from +x, green = from −x;
// its frame in angle and height, and the vertical axis (z) around which the angle is measured
uniform bool aDrawing; uniform sampler2D Drawing; uniform vec4 frame; uniform float axis;
vec3 drawing(vec3 p){ vec2 q=(vec2(atan(p.x,p.z-axis),p.y)-frame.xy)/(frame.zw-frame.xy); return texture(Drawing,vec2(q.x,1.0-q.y)).rgb; }
// layered eyes: part_ 1 = white (白目), 2 = iris (目), 0 = the rest; iris[0] at +x, iris[1] at −x: center x, y, half-width, half-height
uniform int part_; uniform vec4 iris[2]; uniform vec4 mouth;   // part_: 1 eye white, 2 iris, 3 teeth, 4 inside of the mouth; mouth: center x, y, z and half-width of the lip line
uniform sampler2D OmZ;   // the light's depth map, read as is: thickness of matter crossed
uniform vec4 gaze;   // head expression: iris glow, shade over the eyes (already multiplied by its setting), eye height, iris half-height
// The render settings (RENDER_SETTINGS): light (colour × strength), rim, sky and ground bounced into the shade;
// shadow tints (clothes, hair, skin); marche: threshold and softness for clothes, then for skin; casting: strength of
// cast shadows (0: off, the maps are not read)
uniform vec3 key, rimC, sky, ground, omV, omC, omP; uniform vec4 marche;
uniform float rimF, satO, edgeBand, crevices, casting, transm, blush, reflection, expo, satF; uniform bool materials;
// sort: 0 garment, 1 skin, 2 hair, 3 eyes and eyebrows, 4 face
vec3 lin(vec3 x){ return pow(x,vec3(2.2)); }
vec3 enc(vec3 x){ return pow(clamp(x,0.0,1.0),vec3(1.0/2.2)); }
// Share of light reaching the point (1: nothing in front), softened over a radius given in metres, but never with reads
// more than one map texel apart: each read is a bilinear comparison whose ramp is one texel wide, so only then do the 16
// add up to one continuous edge. Wider apart (the map is framed on the view, 16 texels between reads in a face close-up),
// the edge splits into flat steps, stretched into bands that follow the face's relief under a grazing light.
float castBlur(sampler2DShadow S, mat4 M, vec3 n, float radius){
  vec3 s=(M*vec4(P+n*0.010,1.0)).xyz*0.5+0.5;
  if(s.x<0.0||s.x>1.0||s.y<0.0||s.y>1.0||s.z>1.0) return 1.0;
  float e=min(radius*perMeter/1.5,1.0/float(textureSize(S,0).x)), t=0.0;
  for(int i=-1;i<=2;i++) for(int j=-1;j<=2;j++) t+=texture(S,vec3(s.xy+(vec2(float(i),float(j))-0.5)*e,s.z-0.0012));
  return t/16.0;
}
// The same, tighter, for the body and clothes: the expected depth follows the plane of the surface from one kernel
// point to the next, which allows a 2 mm offset on skin (5 on cloth, which has folds) instead of 16; a cord or a pendant lying on the skin then casts
// its shadow there, whole and without bands (with the wide offset, it only appeared halfway, in rim lines).
float castFine(sampler2DShadow S, mat4 M, vec3 n, float radius){
  // surface almost along the light's axis: move further off it, otherwise it streaks with its own shadow. The face itself counts,
  // not only the smoothed normal: at a sharp fold (under the chest) a triangle can lie along the light while its normal faces it,
  // and it then showed as one dark triangle on lit skin
  vec3 ng=normalize(cross(dFdx(P),dFdy(P))); if(dot(ng,n)<0.0) ng=-ng;
  float grazing=1.0-clamp(min(dot(n,L),dot(ng,L)),0.0,1.0);
  vec3 s=(M*vec4(P+n*((sort==1?0.0015:0.005)+0.008*grazing*grazing),1.0)).xyz*0.5+0.5;
  if(s.x<0.0||s.x>1.0||s.y<0.0||s.y>1.0||s.z>1.0) return 1.0;
  vec3 dx=dFdx(s), dy=dFdy(s); float det=dx.x*dy.y-dx.y*dy.x;
  vec2 slope=abs(det)>1e-14?vec2(dy.y*dx.z-dx.y*dy.z,dx.x*dy.z-dy.x*dx.z)/det:vec2(0.0);
  float e=min(radius*perMeter/1.5,2.0/float(textureSize(S,0).x)), t=0.0;   // never more than two map texels between two reads: beyond that, a thin cord gives four shadows side by side
  for(int i=-1;i<=2;i++) for(int j=-1;j<=2;j++){ vec2 d=(vec2(float(i),float(j))-0.5)*e;
    t+=texture(S,vec3(s.xy+d,s.z+clamp(dot(d,slope),-0.0016,0.0016)-0.0001)); }
  return t/16.0;
}
// Applied cast shadow: k, its share of darkness at full strength (hair is never fully in shadow).
float castShadow(sampler2DShadow S, mat4 M, vec3 n, float radius, float k){ return casting>0.0?mix(1.0,castBlur(S,M,n,radius),k*casting):1.0; }
vec3 vif(vec3 x, float k){ return max(mix(vec3(dot(x,vec3(0.3,0.59,0.11))),x,k),0.0); }
// Shade over the eyes (expression that carries it): from the top of the head down to below the eyes, a dark, cool band;
// the cheek and mouth stay in the light. The iris keeps some of its colour there.
vec3 shadowEyes(vec3 o, bool isIris){
  if(gaze.y<=0.0||gaze.w<=0.0) return o;
  float f=smoothstep(gaze.z-gaze.w*2.4,gaze.z-gaze.w*0.9,P.y)*(1.0-smoothstep(0.10,0.14,abs(P.x)));
  return o*mix(vec3(1.0),vec3(0.36,0.29,0.47),f*gaze.y*(isIris?1.0-gaze.x:1.0));   // a glowing iris does not go out in the shade
}
// Two-tone light, but the shadow comes from three sources: the form (normal), what stands in front (depth map)
// and the crevices (occlusion); its colour depends on the material, like the highlights.
vec3 studio(vec3 rgb, vec3 n, vec3 v){
  vec3 al=lin(rgb);
  bool skin=sort==1||sort==4;
  float nv=clamp(dot(n,v),0.0,1.0), nl=dot(n,L)*0.5+0.5;
  float opening=aMaps?smoothstep(0.2,0.8,K.x):1.0;
  float sV=marche.x, dV=marche.y, sP=marche.z, dP=marche.w;
  if(sort==3){
    // eyes, lashes, eyebrows, mouth: painted colour, under the same shadow as the face (hair, and drawn map)
    float day=castShadow(ShadowNoHead,LMt,n,0.007,0.88);
    if(aDrawing){
      vec2 lh=normalize(L.xz+vec2(0.0,1e-5)); vec3 d=drawing(P0);
      day*=smoothstep(-0.035,0.035,(lh.x>=0.0?d.r:d.g)+0.06-(0.5-0.5*lh.y));   // the eye, set back, goes into shadow a little after the skin around it
    }
    vec3 o=al*expo*mix(omP*vec3(1.0,0.50/0.36,0.68/0.47),vec3(1.0),day);   // the skin's shadow, cooler
    if(part_>=3){
      // open mouth: the lips shade what they cover, and the back fades to dark red towards the throat
      float background=smoothstep(0.003,0.030,mouth.z-P.z);
      if(part_==3) o=al*expo*mix(vec3(0.74,0.62,0.70),vec3(1.0,0.97,0.95),day*(1.0-0.75*background));
      else {
        o=vif(al,1.2)*expo*mix(vec3(1.0,0.90,0.88),vec3(0.36,0.17,0.25),background)*mix(vec3(0.70,0.55,0.66),vec3(1.0),day);
        o+=vec3(1.0,0.85,0.85)*smoothstep(0.90,0.97,dot(n,normalize(L+v)))*0.10*day*(1.0-background);   // the tongue, moist: a small glint
      }
      return enc(o);
    }
    if(part_==0||iris[0].w<=0.0) return enc(o);
    // the eye, layer by layer: the eyelid shades the top of the white and of the iris; the iris keeps its light in
    // the shade (it is what one looks at), lightens in a crescent towards the bottom; painted highlights stay white
    vec4 b=P0.x>=0.0?iris[0]:iris[1]; float t=(P0.y-b.y)/b.w, paupiere=smoothstep(0.0,1.1,t);
    o*=mix(vec3(1.0),vec3(0.55,0.50,0.68),paupiere*0.55);
    if(part_==2){
      vec3 c=vif(al,1.35);
      o=mix(o,c*1.05*mix(1.0,0.6,paupiere),0.7);
      o+=c*0.20*smoothstep(0.1,-0.9,t);
      o=mix(o,vec3(1.0),smoothstep(0.80,0.92,min(rgb.r,min(rgb.g,rgb.b))));
      if(gaze.x>0.0){
        // glowing gaze: the iris lights up by itself, with no shadow or eyelid, saturated, brighter towards the center;
        // its painted highlights stay white, and the bloom picks it up (its strongest colour is enough, see FS_SCREEN)
        vec3 lit=vif(al,1.6); lit=lit*1.5+lit*lit*0.8;
        lit*=mix(1.15,0.9,smoothstep(0.0,1.0,abs(t)));
        lit=mix(lit,vec3(1.0),smoothstep(0.80,0.92,min(rgb.r,min(rgb.g,rgb.b))));
        o=mix(o,lit,gaze.x);
      }
    }
    return enc(o);
  }
  float lum, borderV=0.0;   // borderV: share of ordinary skin lighting, at the edge of the face
  if(sort==4){
    // face: drawn boundary (map), never its own triangles; the hair and the rest cast their shadow on it
    if(aMaps){
      vec2 lh=normalize(L.xz+vec2(0.0,1e-5)); float s=lh.x>=0.0?K.z:K.w, net=0.12;
      if(aDrawing){
        // drawn map: sharp boundary; a transition one and a half screen pixels wide, whatever the slope of the map
        vec3 d=drawing(P0);
        s=lh.x>=0.0?d.r:d.g; net=clamp(fwidth(s)*1.5,0.003,0.035); borderV=1.0-smoothstep(0.0,1.0,d.b);
      }
      lum=smoothstep(-net,net,s-(0.5-0.5*lh.y));
    }
    else lum=smoothstep(sP-dP-0.02,sP+dP,dot(normalize(vec3(n.x*0.35,0.0,max(n.z,0.0)+0.6)),L)*0.5+0.5);   // without a map: a normal flattened towards the front
    lum*=castShadow(ShadowNoHead,LMt,n,0.007,0.88);
    // at the edge of the face, the lighting of the head skin that continues it: no line along the cheek
    if(borderV>0.0) lum=mix(lum,smoothstep(sP-dP,sP+dP,nl+(opening-1.0)*0.25*crevices)*castShadow(Sh,LM,n,0.004,1.0),borderV);
  } else if(sort==2){
    // the hair receives the shadow of the head and of the other strands, softened and never whole
    lum=smoothstep(sV-0.03-2.0*dV,sV-0.03+2.0*dV,nl+(opening-1.0)*0.22*crevices)*castShadow(Sh,LM,n,0.008,0.70);
  } else {
    float border=skin?dP:(materials&&material==1?dV*2.8:dV), s=skin?sP:sV;   // knit: a softer edge
    lum=smoothstep(s-border,s+border,nl+(opening-1.0)*0.25*crevices)*(casting>0.0?mix(1.0,castFine(Sh,LM,n,skin?0.004:0.0025),casting):1.0);
  }
  // shadow colour: skin, hair, cloth, each by its own tint
  vec3 om=skin?al*omP:(sort==2?vif(al,1.0+(satO-1.0)*5.0/3.0)*omC:vif(al,satO)*omV);
  vec3 o=mix(om,al,lum);
  // boundary: a more saturated edge band, like light passing through skin (subtler elsewhere)
  float fr=lum*(1.0-lum)*4.0;
  o*=mix(vec3(1.0),skin?vec3(1.06,0.74,0.66):vec3(1.0,0.90,0.94),fr*(skin?0.55:0.3)*edgeBand);
  if(sort==4&&P0.z>mouth.z-0.035&&blush>0.0){
    // blush on the cheeks and the tip of the nose, placed from the eyes and mouth (at the rest position: it follows the skin)
    vec4 b=P0.x>=0.0?iris[0]:iris[1];
    if(b.w>0.0){
      float joue=1.0-smoothstep(0.0,1.0,length((P0.xy-vec2(b.x*1.10,b.y-b.w*2.7))/vec2(b.w*2.3,b.w*1.1)));
      float nose=mouth.w>0.0?1.0-smoothstep(0.0,1.0,length((P0.xy-vec2(0.0,mix(b.y,mouth.y,0.60)))/vec2(b.w*0.5,b.w*0.42))):0.0;
      o*=mix(vec3(1.0),vec3(1.05,0.78,0.78),joue*joue*0.50*blush);
      o*=mix(vec3(1.0),vec3(1.10,0.90,0.88),nose*nose*0.45*blush);                      // the nose turns pink without darkening (darkening made a grey patch on tanned skin)
      // on dark skin, tinting does not show: the pink is added
      o+=vec3(0.20,0.03,0.05)*(joue*joue*0.50+nose*nose*0.28)*blush*(1.0-smoothstep(0.12,0.45,dot(al,vec3(0.3,0.59,0.11))))*mix(0.4,1.0,lum);
    }
    // lips: a stronger pink below the line, thinner above, and a small sharp highlight on the lower one
    if(mouth.w>0.0){
      vec2 q=(P0.xy-mouth.xy)/mouth.w;
      float bottom=1.0-smoothstep(0.45,1.0,length((q-vec2(0.0,-0.30))/vec2(0.66,0.34)));
      float over=1.0-smoothstep(0.45,1.0,length((q-vec2(0.0,0.12))/vec2(0.80,0.16)));
      o*=mix(vec3(1.0),vec3(1.04,0.74,0.77),(bottom*0.55+over*0.30)*blush);
      o+=(al*0.8+0.1)*(1.0-smoothstep(0.75,1.0,length((q-vec2(0.16*sign(L.x),-0.27))/vec2(0.17,0.055))))*0.28*lum*blush;   // highlight scaled to the skin: discreet on dark skin
    }
  }
  if(sort==1&&casting>0.0&&transm>0.0){
    // thin skin against the light (ears, fingers, nostrils): the light passing through comes out warm red
    vec3 s=(LM*vec4(P,1.0)).xyz*0.5+0.5;
    float ep=(s.z-texture(OmZ,s.xy).r)*5.0;   // metres of matter between the light and this point
    o+=al*vec3(0.95,0.26,0.16)*0.55*transm*(1.0-lum)*exp(-max(ep,0.0)/0.006)*smoothstep(0.0,0.0015,ep);
  }
  // crevices: one tone darker in the shade, almost nothing in the light
  o*=mix(1.0,mix(mix(0.78,0.94,lum),1.0,opening),(sort==4?borderV:1.0)*crevices);
  // sky from above, ground from below, in the shade only
  o+=al*(1.0-lum)*(sky*max(n.y,0.0)+ground*max(-n.y,0.0));
  vec3 h=normalize(L+v);
  if(sort==2){
    if(reflection>0.0){
      // hair: highlight band across the strands; without tangents, the strand runs down from the top of the skull
      vec3 d=P-vec3(0.0,top+0.02,-0.01); vec3 tg=normalize(d-n*dot(d,n));
      float g=dot(rgb,vec3(0.3,0.59,0.11));
      tg=normalize(tg+n*((g-0.5)*0.35-0.12));                                  // the texture shifts the band from one strand to the next
      float th=dot(tg,h), b=sqrt(max(1.0-th*th,0.0));
      float band=smoothstep(0.985,0.997,b)*smoothstep(0.15,0.45,nl)*smoothstep(0.25,0.6,opening);
      o+=band*(vif(al,0.7)+vec3(0.10,0.09,0.08)*(0.35+g))*0.18*reflection;         // on dark hair the highlight keeps their colour (white, it looked like plastic)
      o+=smoothstep(0.90,0.99,b)*smoothstep(0.15,0.45,nl)*vif(al,1.2)*0.05*lum*reflection;   // around the band, a wide, soft highlight
      o*=mix(vec3(1.0),mix(vec3(0.93,0.90,0.96),vec3(1.03,1.01,0.98),clamp((top-P.y)/0.45,0.0,1.0)),min(reflection,1.0));   // roots slightly darker and cooler, light tips
    }
    if(casting>0.0&&transm>0.0){
      // thin strands against the light: the light passes through them and lights them in a warm tone
      vec3 s=(LM*vec4(P,1.0)).xyz*0.5+0.5; float ep=(s.z-texture(OmZ,s.xy).r)*5.0;
      o+=vif(al,1.4)*vec3(1.0,0.80,0.60)*0.28*transm*(1.0-lum)*exp(-max(ep,0.0)/0.012)*smoothstep(0.0,0.002,ep);
    }
  } else if(sort==0&&materials){
    if(material==2){ float s=pow(max(dot(n,h),0.0),36.0); o+=(smoothstep(0.30,0.42,s)*0.10+s*0.10)*lum*mix(al,vec3(1.0),0.6); }   // leather: wide highlight with a sharp edge
    else if(material==1) o+=al*pow(1.0-nv,2.5)*0.16*(0.4+0.6*lum);                                                               // knit: light fuzz on the edges
    else if(material==3){ o+=al*pow(1.0-nv,1.5)*0.32*(0.5+0.5*lum); o+=vec3(0.90,0.90,1.0)*pow(max(dot(n,h),0.0),20.0)*0.10*lum; }   // tights: light veil on the edges, stretched highlight
    else if(material==4){ float s=pow(max(dot(n,h),0.0),10.0); o+=mix(al,vec3(1.0),0.35)*(s*0.28+smoothstep(0.55,0.75,s)*0.12)*lum; }    // silk, ribbon: wide, soft highlight, sharper core
    else if(material==5){ float s=pow(max(dot(n,h),0.0),90.0); o=o*(0.75+0.5*pow(1.0-nv,2.0))+al*pow(nv,3.0)*0.25+vec3(1.0)*smoothstep(0.35,0.5,s)*0.6*lum; }   // glass, stone: brighter base in the center, sharp glint
    else {
      // cloth: matte; a wide light area gives the folds their volume, and a dark cloth takes a cool veil on the edges
      float sombre=1.0-smoothstep(0.03,0.25,dot(al,vec3(0.3,0.59,0.11)));
      o+=al*pow(1.0-nv,4.0)*0.08*lum+al*0.10*lum*smoothstep(0.55,0.95,dot(n,h));
      o+=mix(al,vec3(0.55,0.58,0.75),0.5)*pow(1.0-nv,3.0)*0.14*sombre*(0.3+0.7*lum);
    }
  }
  if(K.y>0.0&&materials){
    // metal: the colour hardly matters any more, the highlight bands make the material
    float m=dot(reflect(-v,n),normalize(L+vec3(0.0,0.5,0.0)))*0.5+0.5;
    float bands=smoothstep(0.62,0.68,m)+0.45*smoothstep(0.22,0.28,m)*(1.0-smoothstep(0.40,0.46,m));
    o=mix(o,al*(0.30+1.7*bands)+vec3(0.8,0.78,0.72)*smoothstep(0.90,0.94,m),K.y);
  }
  // rim light: thin, on the side opposite the main light, to separate the silhouette without a black line
  if(rimF>0.0){
    vec3 lc=normalize(vec3(-L.x,0.35,-L.z)-v*0.4);
    float rim=smoothstep(0.52,0.72,pow(1.0-nv,2.0)*(0.35+0.65*max(dot(n,lc),0.0)))*(sort==4?0.35:1.0);
    o+=rim*mix(al,rimC,0.45)*(skin?0.30:0.24)*rimF;
  }
  // finish: light colour on the lit side, exposure, saturation
  o=vif(o*mix(vec3(1.0),key,lum)*expo,satF);
  return enc(o);
}
void main(){
  vec4 a = aTex ? texture(T,UV) : vec4(1.0);
  if(a.a<0.4) discard;
  vec3 rgb=a.rgb*diffuse.rgb;
  if(tint.a>0.0){
    float v=max(rgb.r,max(rgb.g,rgb.b));
    float vc=max(tint.r,max(tint.g,tint.b));
    float sc=vc>0.0?1.0-min(tint.r,min(tint.g,tint.b))/vc:0.0;
    float v2=min(v*vc/ref.x,1.0);
    vec3 pure=tint.rgb/max(vc,1e-5);
    rgb=v2*pure*sc+vec3(v2)*(1.0-sc);
  }
  vec3 n=normalize(N);
  // hair is made of sheets seen from both sides: the back face keeps the strand's normal, otherwise it goes dark and makes a hole
  if(!gl_FrontFacing&&sort!=2) n=-n;
  vec3 v=normalize(eye-P);
  c=vec4(shadowEyes(studio(rgb,n,v),part_==2),sort==3?0.0:(sort==4?0.5:(sort==1?0.9:1.0)));   // the alpha says how much contact shadow the surface receives; 0.9 marks the body skin (see the finish, mode 0)
}`;

// Depth map seen from the light: same vertices, the texture is only used for cutting out.
const VS_SHADOW = `#version 300 es
layout(location=0) in vec3 p; layout(location=2) in vec2 t; uniform mat4 LM; out vec2 UV;
void main(){ UV=t; gl_Position=LM*vec4(p,1.0); }`;
const FS_SHADOW = `#version 300 es
precision highp float; in vec2 UV; uniform sampler2D T; uniform bool aTex;
void main(){ if(aTex&&texture(T,UV).a<0.4) discard; }`;
const SHADOW_SIZE=2048, FLOOR_RADIUS=1.15;
// Full screen: background (0), then bloom of the bright areas (1 threshold, 2 blur, 3 add).
const VS_SCREEN = `#version 300 es
out vec2 UV; void main(){ vec2 q=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2)); UV=q; gl_Position=vec4(q*2.0-1.0,0.0,1.0); }`;
const FS_SCREEN = `#version 300 es
precision highp float; in vec2 UV; out vec4 c; uniform sampler2D S; uniform int mode; uniform vec2 stage; uniform float aspect; uniform vec3 f0, f1; uniform float halo;
void main(){
  if(mode==0){
    float d=length((UV-vec2(0.5,0.58))*vec2(aspect,1.0));
    vec3 f=mix(f0,f1,smoothstep(0.0,0.8,d));
    c=vec4(f+(fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453)-0.5)/255.0,1.0); return;   // grain: no banding in the gradient
  }
  if(mode==1){
    vec3 x=(texture(S,UV+stage)+texture(S,UV-stage)+texture(S,UV+vec2(stage.x,-stage.y))+texture(S,UV+vec2(-stage.x,stage.y))).rgb*0.25;
    c=vec4(x*smoothstep(0.60,0.92,max(dot(x,vec3(0.3,0.59,0.11)),max(x.r,max(x.g,x.b))*0.8)),1.0); return;   // an almost pure vivid colour (glowing iris) also passes
  }
  if(mode==2){
    c=vec4(texture(S,UV).rgb*0.227+(texture(S,UV+stage*1.385).rgb+texture(S,UV-stage*1.385).rgb)*0.316+(texture(S,UV+stage*3.231).rgb+texture(S,UV-stage*3.231).rgb)*0.070,1.0); return;
  }
  c=vec4(texture(S,UV).rgb*vec3(0.27,0.21,0.23)*halo,0.0);
}`;
// Finish. The scene is drawn into a buffer whose colour and depth are read back:
// 0 contact shadows (what lies close to a surface darkens it: collar, belt, strand on the shoulder),
// 1 the image under these shadows, 2 depth of field (what is far from the subject is blurred, mostly in close-ups),
// 3 softened edges and grain, by multiplication over the finished image; 4 and 5 the outline.
const NEAR=0.02, FAR=20, FIELD=0.55;   // the view's perspective
const VS_END = `#version 300 es
precision highp sampler2D;   // depth is read through Z: at the default low precision a graphics card returns it in 0.6 mm steps
uniform sampler2D Z; out vec2 UV; flat out float sujet;
float metres(float d){ return 2.0*${NEAR}*${FAR}.0/(${FAR}.0+${NEAR}-(d*2.0-1.0)*(${FAR}.0-${NEAR})); }
void main(){ vec2 q=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2)); UV=q; gl_Position=vec4(q*2.0-1.0,0.0,1.0);
  // the subject: mean distance of what fills the center of the image
  float s=0.0,n=0.0; for(int i=-2;i<=2;i++) for(int j=-2;j<=2;j++){ float d=textureLod(Z,vec2(0.5)+vec2(float(i),float(j))*0.045,0.0).r; if(d<1.0){ s+=metres(d); n+=1.0; } }
  sujet=n>0.0?s/n:0.0; }`;
const FS_END = `#version 300 es
precision highp float; precision highp sampler2D; in vec2 UV; flat in float sujet; out vec4 c;   // highp sampler: see VS_FIN (contact shadows drew rings on smooth cloth)
uniform sampler2D C; uniform sampler2D Z; uniform sampler2D A; uniform sampler2D G; uniform int mode; uniform vec2 stage, dir; uniform float aspect; uniform float outline;
uniform float contact, blur, thumbnail, grain, outlinePart; uniform vec3 outlineC;
const float T=${Math.tan(FIELD/2).toFixed(5)}, RADIUS=0.022, BLUR=0.012;   // RADIUS: reach of a contact shadow, in metres; BLUR: diameter of the largest blur, in image heights
float metres(float d){ return 2.0*${NEAR}*${FAR}.0/(${FAR}.0+${NEAR}-(d*2.0-1.0)*(${FAR}.0-${NEAR})); }
vec3 place(vec2 uv){ float z=metres(texture(Z,uv).r); return vec3((uv*2.0-1.0)*vec2(aspect*T,T)*z,-z); }
float alea(vec2 p){ return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453); }
void main(){
  if(mode==0){
    if(texture(Z,UV).r>=1.0||contact<=0.0){ c=vec4(0.0); return; }
    vec3 p=place(UV);
    // body skin takes no contact shadow from body skin: its own relief (edge of the chest, cleavage) is not something lying on it,
    // and it drew thin dark lines there; cast shadows and crevices already shade those folds
    float a0=texture(C,UV).a; bool skin=a0>0.8&&a0<0.95;
    // surface normal, from the shortest difference on each side (no false slope at the edge of a shape)
    vec3 xa=place(UV+vec2(stage.x,0.0))-p, xb=p-place(UV-vec2(stage.x,0.0)), ya=place(UV+vec2(0.0,stage.y))-p, yb=p-place(UV-vec2(0.0,stage.y));
    vec3 n=normalize(cross(abs(xa.z)<abs(xb.z)?xa:xb,abs(ya.z)<abs(yb.z)?ya:yb));
    vec2 r=RADIUS/(2.0*T*-p.z)*vec2(1.0/aspect,1.0);
    float turn=float((int(gl_FragCoord.x)&3)+((int(gl_FragCoord.y)&3)<<2))*0.3927, t=0.0;   // sixteen starting angles, smoothed afterwards over four points
    for(int i=0;i<12;i++){
      float a=turn+float(i)*2.39996, k=sqrt((float(i)+0.5)/12.0);
      vec2 q=UV+vec2(cos(a),sin(a))*k*r; vec3 v=place(q)-p; float l=length(v);
      if(skin){ float aq=texture(C,q).a; if(aq>0.8&&aq<0.95) continue; }
      t+=max(dot(n,v)/max(l,1e-5)-0.18,0.0)*(1.0-smoothstep(RADIUS*0.8,RADIUS*1.6,l));   // in front of the surface, and within reach: beyond that, it is another shape, not a contact
    }
    c=vec4(clamp(t/12.0*2.6,0.0,1.0)); return;
  }
  if(mode==1){
    vec4 x=texture(C,UV); float d=texture(Z,UV).r, o=0.0;
    for(int i=-2;i<2;i++) for(int j=-2;j<2;j++) o+=texture(A,UV+(vec2(float(i),float(j))+0.5)*stage).r;
    o=smoothstep(0.06,0.62,o/16.0)*(x.a>0.8?1.0:x.a);
    // blur at this point: zero around the subject, it grows with the distance from it, all the more as the subject is close;
    // the sharp zone reaches further behind than in front: a strand in the centre of the image must not blur the eyes
    float z=metres(d), e=(z-sujet)/z, f=sujet>0.0?clamp((e>0.0?e-0.22:-e-0.10)*0.010*blur/(2.0*T*sujet)/BLUR,0.0,1.0):0.0;
    vec3 img=x.rgb*mix(vec3(1.0),vec3(0.60,0.43,0.57),o*0.85*contact);
    // outline: drawn separately (modes 4 and 5), smoothed, then laid here; it spills slightly over both sides of the edge
    if(outline>0.0&&(d>=1.0||x.a>0.25)){ vec4 l=texture(G,UV); img=mix(img,l.rgb/max(l.a,0.004),smoothstep(0.10,0.21,l.a)*0.92*min(outline,1.0)); }
    c=vec4(img,f); return;
  }
  if(mode==4){
    // outline, first stage: the points at the edge of a shape (what is next to them is clearly further away than the slope of
    // the surface would lead one to expect), with the outline colour at that spot: the surface's, darker and
    // more vivid, never black (or the fixed colour, according to its share). Depth has only one value per point: this edge is stair-stepped.
    float d=texture(Z,UV).r; vec4 x=texture(C,UV);
    if(d>=1.0||x.a<=0.25){ c=vec4(0.0); return; }
    float z=metres(d), threshold=0.004+0.012*z, m=0.0, force=0.0; vec2 outside=vec2(0.0);
    for(int i=0;i<8;i++){
      float a=float(i)*0.7854; vec2 q=vec2(cos(a),sin(a))*stage*(i%2==0?1.0:1.4142);
      float saut=(metres(texture(Z,UV+q).r)-z)-max(z-metres(texture(Z,UV-q).r),0.0), k=smoothstep(threshold*0.8,threshold*1.3,saut);
      if(k<=0.0) continue;
      // a real edge continues beyond; two surfaces that brush against each other make only one point
      k*=smoothstep(threshold*0.4,threshold,(metres(texture(Z,UV+q*3.0).r)-z)-max(z-metres(texture(Z,UV-q*3.0).r),0.0));
      m=max(m,k); force=max(force,k*smoothstep(threshold,0.03+threshold*4.0,saut)); outside+=vec2(cos(a),sin(a))*k;
    }
    if(m<=0.0){ c=vec4(0.0); return; }
    float support=0.5+0.5*dot(normalize(outside+vec2(1e-5)),normalize(vec2(-0.45,-1.0)));   // under the shape, opposite the light
    m*=mix(0.72,1.0,force)*mix(0.84,1.0,support);                                         // full at the edge of a large shape, thin between two strands
    float g=dot(x.rgb,vec3(0.3,0.59,0.11)); vec3 vifs=max(mix(vec3(g),x.rgb,1.55),0.0);
    // on a dark material (black dress), darkening no longer shows: the outline goes almost to black there, tinted
    vec3 collar=vifs*vec3(1.0,0.58,0.76)*mix(0.60,0.42,smoothstep(0.12,0.6,g))*mix(0.22,1.0,smoothstep(0.03,0.30,g));
    c=vec4(mix(collar,outlineC,outlinePart)*m,m); return;
  }
  if(mode==5){
    // outline, second stage: this stair-stepped edge is spread (two passes, horizontal then vertical); read back with a soft
    // threshold, it gives a smooth-edged outline whose width follows the strength of the edge
    vec4 t=vec4(0.0); float n=0.0;
    for(int i=-4;i<=4;i++){ float w=exp(-0.28*float(i*i)); t+=texture(C,UV+dir*float(i)*0.75)*w; n+=w; }
    c=t/n; return;
  }
  if(mode==2){
    vec4 x=texture(C,UV); float r=x.a*BLUR*0.5;   // radius, in image heights
    if(r*float(textureSize(C,0).y)<0.6){ c=vec4(x.rgb,1.0); return; }
    vec3 s=x.rgb; float n=1.0, turn=alea(gl_FragCoord.xy)*6.2832;
    for(int i=0;i<20;i++){
      float a=turn+float(i)*2.39996, k=sqrt((float(i)+0.5)/20.0)*r;
      vec4 y=texture(C,UV+vec2(cos(a)/aspect,sin(a))*k);
      float w=smoothstep(k*0.6,k*1.1,y.a*BLUR*0.5+stage.y);   // a sharp point does not bleed onto the blur next to it
      s+=y.rgb*w; n+=w;
    }
    c=vec4(s/n,1.0); return;
  }
  float border=1.0-thumbnail*smoothstep(0.38,0.95,length((UV-0.5)*vec2(min(aspect,1.6),1.0)));
  c=vec4(vec3(0.5*border*(1.0+(alea(gl_FragCoord.xy)-0.5)*grain)),0.5);
}`;
// Floor: a disc of light under the character, which receives its shadow (third depth map).
const VS_FLOOR = `#version 300 es
uniform mat4 VP; uniform float floor; out vec3 P;
void main(){ vec2 q=vec2(float(gl_VertexID&1),float(gl_VertexID>>1))*2.0-1.0; P=vec3(q.x*3.0,floor,q.y*3.0); gl_Position=VP*vec4(P,1.0); }`;
const FS_FLOOR = `#version 300 es
precision highp float; precision lowp sampler2DShadow; in vec3 P; out vec4 c; uniform mat4 LM; uniform sampler2DShadow Sh0; uniform float perMeter; uniform vec3 s0, s1; uniform float casting;
void main(){
  float t=1.0;
  if(casting>0.0){
    vec3 s=(LM*vec4(P,1.0)).xyz*0.5+0.5; float e=min(0.006*perMeter,1.0/float(textureSize(Sh0,0).x)); t=0.0;   // reads never more than one map texel apart (see castBlur)
    for(int i=-2;i<=2;i++) for(int j=-2;j<=2;j++) t+=texture(Sh0,vec3(s.xy+vec2(float(i),float(j))*e,s.z-0.002));
    t=(s.x<0.0||s.x>1.0||s.y<0.0||s.y>1.0||s.z>1.0)?1.0:mix(1.0,t/25.0,casting);
  }
  float a=1.0-smoothstep(0.4,2.2,length(P.xz));
  c=vec4(mix(s0,s1,t)*a,a);
}`;

function mat4(){ return new Float32Array(16); }
function ortho(r,zn,zf){ const m=mat4(); m[0]=1/r;m[5]=1/r;m[10]=-2/(zf-zn);m[14]=-(zf+zn)/(zf-zn);m[15]=1;return m; }
function perspective(fov,asp,zn,zf){ const f=1/Math.tan(fov/2),m=mat4(); m[0]=f/asp;m[5]=f;m[10]=(zf+zn)/(zn-zf);m[11]=-1;m[14]=2*zf*zn/(zn-zf);return m; }
function gaze(e,c,u){ const z=norm(sub(e,c)),x=norm(cross(u,z)),y=cross(z,x),m=mat4();
  m[0]=x[0];m[4]=x[1];m[8]=x[2];m[1]=y[0];m[5]=y[1];m[9]=y[2];m[2]=z[0];m[6]=z[1];m[10]=z[2];
  m[12]=-pt(x,e);m[13]=-pt(y,e);m[14]=-pt(z,e);m[15]=1;return m; }
function mult(a,b){ const r=mat4(); for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+i]*b[j*4+k];r[j*4+i]=s;} return r; }
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], pt=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2], cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const n=Math.hypot(...a)||1;return[a[0]/n,a[1]/n,a[2]/n];};

const cacheGLB=new Map();
export function readGLB(url){
  if(!cacheGLB.has(url)) cacheGLB.set(url,fetch(url).then(r=>{ if(!r.ok) throw new Error(url); return r.arrayBuffer(); }).then(analyze));
  return cacheGLB.get(url);
}
function analyze(b){
  const dv=new DataView(b); let off=12,doc=null,bin=null;
  while(off<b.byteLength){ const n=dv.getUint32(off,true),t=dv.getUint32(off+4,true);
    if(t===0x4E4F534A) doc=JSON.parse(new TextDecoder().decode(new Uint8Array(b,off+8,n))); else bin=b.slice(off+8,off+8+n);
    off+=8+n; }
  const tab=(i,T,c)=>{const a=doc.accessors[i],v=doc.bufferViews[a.bufferView]; return new T(bin,v.byteOffset||0,a.count*c);};
  const pr=doc.meshes[0].primitives;
  const images=(doc.images||[]).map(x=>x.uri);
  const textures=doc.textures||[];
  const d={
    pos:tab(pr[0].attributes.POSITION,Float32Array,3), nor:tab(pr[0].attributes.NORMAL,Float32Array,3), uv:tab(pr[0].attributes.TEXCOORD_0,Float32Array,2),
    prims:pr.map(p=>{ const m=doc.materials[p.material], e=m.extras||{}, bt=m.pbrMetallicRoughness.baseColorTexture, face=/^(颜|肌|首|体)/.test(m.name);
      return {indices:tab(p.indices,Uint32Array,1), image:bt?images[textures[bt.index].source]:null, diffuse:m.pbrMetallicRoughness.baseColorFactor, name:m.name, material:MATERIALS.findIndex(x=>x.test(m.name))+1, skin:face||e.role==="skin", face:/^颜/.test(m.name)};}),
  };
  return d;
}
// Hair smoothing: each triangle becomes four, the new point of each edge placed on the curve drawn by
// its neighbours (butterfly: eight points around the edge). The original vertices do not move, so the
// hairstyle does not shrink and stays outside the skull; it is the silhouettes and stair-stepped shadow edges (braids,
// strands with large polygons) that get rounded. Vertices duplicated at texture seams count as one.
function smoothing(d){
  const n=d.pos.length/3, P=d.pos, welded=new Int32Array(n), vu=new Map();
  for(let i=0;i<n;i++){ const k=Math.round(P[3*i]*2e4)+","+Math.round(P[3*i+1]*2e4)+","+Math.round(P[3*i+2]*2e4); let w=vu.get(k); if(w===undefined){ w=i; vu.set(k,w); } welded[i]=w; }
  // welded edges: the opposite vertices (two at most; beyond that the edge is not smoothed)
  const key=(a,b)=>a<b?a*n+b:b*n+a, face=new Map();
  for(const p of d.prims) for(let f=0;f<p.indices.length;f+=3) for(let e=0;e<3;e++){
    const a=welded[p.indices[f+e]], b=welded[p.indices[f+(e+1)%3]], c=welded[p.indices[f+(e+2)%3]], k=key(a,b); let x=face.get(k); if(!x) face.set(k,x=[]); x.push(c); }
  const wing=(a,c,except)=>{ const x=face.get(key(a,c)); if(!x||x.length!==2) return -1; const y=x[0]===except?x[1]:x[0]; return y===except?-1:y; };
  // new vertices: one per edge of each texture piece (two duplicated vertices give two points at the same place)
  const edgeLine=new Map(), tips=[], template=[];
  const point=(i,j)=>{ const k=key(i,j); let x=edgeLine.get(k); if(x!==undefined) return x;
    x=n+tips.length/2; edgeLine.set(k,x); tips.push(i,j);
    const a=welded[i], b=welded[j], o=face.get(key(a,b));
    if(o&&o.length===2&&o[0]!==o[1]){ const c=o[0], e=o[1], w=[wing(a,c,b),wing(c,b,a),wing(b,e,a),wing(e,a,b)];
      template.push(c,e,...(w.includes(-1)?[-1,-1,-1,-1]:w)); }
    else template.push(-1,-1,-1,-1,-1,-1);
    return x; };
  const prims=d.prims.map(p=>{ const s=p.indices, o=new Uint32Array(s.length*4);
    for(let f=0,k=0;f<s.length;f+=3){ const a=s[f],b=s[f+1],c=s[f+2], ab=point(a,b), bc=point(b,c), ca=point(c,a);
      o[k++]=a;o[k++]=ab;o[k++]=ca; o[k++]=ab;o[k++]=b;o[k++]=bc; o[k++]=ca;o[k++]=bc;o[k++]=c; o[k++]=ab;o[k++]=bc;o[k++]=ca; }
    return {...p,indices:o}; });
  const nd=tips.length/2, N=n+nd;
  // positions: the original ones, then the point of each edge
  const positions=src=>{ const o=new Float32Array(N*3); o.set(src.subarray(0,n*3));
    for(let e=0;e<nd;e++){ const a=3*tips[2*e], b=3*tips[2*e+1], g=template, q=6*e;
      for(let x=0;x<3;x++){ let v=(src[a+x]+src[b+x])*0.5;
        if(g[q]>=0&&g[q+2]>=0) v+=(src[3*g[q]+x]+src[3*g[q+1]+x])*0.125-(src[3*g[q+2]+x]+src[3*g[q+3]+x]+src[3*g[q+4]+x]+src[3*g[q+5]+x])*0.0625;
        o[3*(n+e)+x]=v; } }
    return o; };
  // the rest (normals, UVs, maps): the mean of the two ends
  const average=(src,c,T,unit)=>{ const o=new T(N*c); o.set(src.subarray(0,n*c));
    for(let e=0;e<nd;e++){ const a=c*tips[2*e], b=c*tips[2*e+1]; let l=0;
      for(let x=0;x<c;x++){ const v=(src[a+x]+src[b+x])*0.5; o[c*(n+e)+x]=v; l+=v*v; }
      if(unit){ l=Math.sqrt(l)||1; for(let x=0;x<c;x++) o[c*(n+e)+x]/=l; } }
    return o; };
  return {n, N, positions, average, d:{pos:positions(P), nor:average(d.nor,3,Float32Array,true), uv:average(d.uv,2,Float32Array,false), prims}};
}
// Garment material read from the name of its material: 1 knit and wool, 2 leather, 3 tights, 4 silk and ribbon,
// 5 glass and stone, otherwise cloth.
const MATERIALS=[/针织|毛衣|毛领|围巾|tricot|pull|knit|laine/i,/腰带|皮|鞋|靴|手套|拳套|包|cuir|ceinture|botte|leather/i,/袜|collant|nylon/i,/缎|丝|领带|蝴蝶|ruban|satin|n(œ|oe)ud|soie/i,/镜|宝石|珠宝|verre|pierre|gem/i];

class ViewerBase {
  constructor(canvas,{keep=false}={}){
    this.c=canvas; this.models=new Map(); this.tex=new Map(); this.r=copy(RENDER_DEFAULT);
    this.cam={az:0.35,el:0.08,dist:3.5,aim:[0,0.8,0]}; this.target=null; this.spinning=false; this.sale=true;
    if(!keep){ this.mouse(); new ResizeObserver(()=>this.sale=true).observe(canvas); requestAnimationFrame(()=>this.buckle()); }
  }
  // ---- the current render ----
  get render(){ return copy(this.r); }
  // Changes a setting (dotted key of RENDER_SETTINGS); the value is clamped. Returns the value kept.
  adjust(key,val){
    const r=BY_KEY.get(key); if(!r) throw new Error(`Unknown render setting: ${key}`);
    const v=value(r,val); if(v===undefined) throw new Error(`Unreadable value for ${key} (${r.type}): ${JSON.stringify(val)}`);
    place(this.r,key,v); this.sale=true; return copy(v);
  }
  // Replaces the whole render with a document (object or JSON text); returns the normalised render. Throws a readable Error if it is not one.
  loadRender(o){ const n=normalizeRender(o); this.r=n; this.sale=true; return copy(n); }
  exportRender(name){ const o=copy(this.r); o.format=RENDER_FORMAT; o.version=RENDER_VERSION; if(typeof name==="string"&&name.trim()) o.name=name.trim().slice(0,80);
    return {format:o.format,version:o.version,name:o.name,...Object.fromEntries(Object.entries(o).filter(([k])=>!["format","version","name"].includes(k)))}; }
  mouse(){
    const c=this.c; let x=0,y=0,on=false;
    c.addEventListener("pointerdown",e=>{on=true;x=e.clientX;y=e.clientY;c.setPointerCapture(e.pointerId);this.spinning=false;this.onSpinning&&this.onSpinning(false);this.target=null;});
    c.addEventListener("pointermove",e=>{ if(!on) return; this.cam.az-=(e.clientX-x)*0.008; this.cam.el=Math.max(-1.2,Math.min(1.3,this.cam.el+(e.clientY-y)*0.006)); x=e.clientX;y=e.clientY;this.sale=true;});
    c.addEventListener("pointerup",()=>on=false);
    c.addEventListener("wheel",e=>{e.preventDefault();this.cam.dist=Math.max(0.35,Math.min(6,this.cam.dist*Math.exp(e.deltaY*0.001)));this.sale=true;},{passive:false});
  }
  fit(q){ this.target=q==="face"?{dist:0.55,aim:[0,1.42,0],el:0.03}:{dist:3.5,aim:[0,0.8,0],el:0.08}; this.sale=true; }
  buckle(){
    if(this.target){ const k=0.16,v=this.target,m=this.cam; m.dist+=(v.dist-m.dist)*k; m.el+=(v.el-m.el)*k; m.aim=m.aim.map((a,i)=>a+(v.aim[i]-a)*k);
      if(Math.abs(v.dist-m.dist)<0.002) this.target=null; this.sale=true; }
    if(this.spinning){ this.cam.az+=0.012; this.sale=true; }
    const change=this.sale;
    if(this.sale){ this.sale=false; this.draw(!change); }
    requestAnimationFrame(()=>this.buckle());
  }
  resize(){
    const c=this.c,dpr=this.dpr||Math.min(window.devicePixelRatio||1,2),W=Math.max(1,c.clientWidth*dpr|0),H=Math.max(1,c.clientHeight*dpr|0);
    if(!c.hidden&&(c.width!==W||c.height!==H)){c.width=W;c.height=H;}
  }
  matrices(){
    const a=this.cam,c=this.c,e=[a.aim[0]+a.dist*Math.sin(a.az)*Math.cos(a.el),a.aim[1]+a.dist*Math.sin(a.el),a.aim[2]+a.dist*Math.cos(a.az)*Math.cos(a.el)];
    // the light: its angle is measured from the camera (it turns with the view), its height from the horizon
    const l=this.r.light, ang=a.az+l.angle/DEG, h=l.height/DEG;
    return {eye:e, VP:mult(perspective(FIELD,c.width/c.height,NEAR,FAR),gaze(e,a.aim,[0,1,0])),
      L:norm([Math.sin(ang)*Math.cos(h),Math.sin(h),Math.cos(ang)*Math.cos(h)])};
  }
}

function compiler(g,vs=VS,fs=FS){
  const sh=(t,s)=>{const o=g.createShader(t);g.shaderSource(o,s);g.compileShader(o);if(!g.getShaderParameter(o,g.COMPILE_STATUS))throw new Error(g.getShaderInfoLog(o));return o;};
  const p=g.createProgram(); g.attachShader(p,sh(g.VERTEX_SHADER,vs)); g.attachShader(p,sh(g.FRAGMENT_SHADER,fs)); g.linkProgram(p);
  if(!g.getProgramParameter(p,g.LINK_STATUS)) throw new Error(g.getProgramInfoLog(p));
  return p;
}

// Picks the renderer: WebGL2 if it really works here, otherwise software drawing.
export function createViewer(canvas,opts){
  try{ const g=document.createElement("canvas").getContext("webgl2"); if(!g) throw new Error("WebGL2 unavailable"); compiler(g); g.getExtension("WEBGL_lose_context")?.loseContext(); return new Viewer(canvas,opts); }
  catch(e){ console.warn("Software rendering:",e.message); return new SoftwareViewer(canvas,opts); }
}

export class Viewer extends ViewerBase {
  constructor(canvas,opts={}){
    const {keep=false}=opts;
    super(canvas,opts);
    const g=this.g=canvas.getContext("webgl2",{antialias:true,alpha:true,premultipliedAlpha:true,preserveDrawingBuffer:keep});
    if(!g) throw new Error("WebGL2 unavailable");
    const p=this.p=compiler(g);
    this.u={}; for(const n of ["VP","T","aTex","diffuse","L","tint","ref","sort","eye","aMaps","top","LM","LMt","Sh","ShadowNoHead","perMeter","material","aDrawing","Drawing","frame","axis","part_","iris","mouth","OmZ","gaze",
      "key","rimC","sky","ground","omV","omC","omP","marche","rimF","satO","edgeBand","crevices","casting","transm","blush","reflection","expo","satF","materials"]) this.u[n]=g.getUniformLocation(p,n);
    // background and bloom in full screen, floor
    const pe=this.pe=compiler(g,VS_SCREEN,FS_SCREEN); this.ue={}; for(const n of ["S","mode","stage","aspect","f0","f1","halo"]) this.ue[n]=g.getUniformLocation(pe,n);
    const pf=this.pf=compiler(g,VS_END,FS_END); this.uf={}; for(const n of ["C","Z","A","G","dir","mode","stage","aspect","outline","contact","blur","thumbnail","grain","outlinePart","outlineC"]) this.uf[n]=g.getUniformLocation(pf,n);
    const ps=this.ps=compiler(g,VS_FLOOR,FS_FLOOR); this.us={}; for(const n of ["VP","floor","LM","Sh0","perMeter","s0","s1","casting"]) this.us[n]=g.getUniformLocation(ps,n);
    // three depth maps (units 5, 6 and 7: everything, everything but the head, everything seen from further away for the floor), compared by the graphics card
    const po=this.po=compiler(g,VS_SHADOW,FS_SHADOW); this.uo={}; for(const n of ["LM","T","aTex"]) this.uo[n]=g.getUniformLocation(po,n);
    this.shadows=[5,6,7].map(unit=>{ const t=g.createTexture(),f=g.createFramebuffer(); g.activeTexture(g.TEXTURE0+unit); g.bindTexture(g.TEXTURE_2D,t);
      g.texImage2D(g.TEXTURE_2D,0,g.DEPTH_COMPONENT24,SHADOW_SIZE,SHADOW_SIZE,0,g.DEPTH_COMPONENT,g.UNSIGNED_INT,null);
      for(const [k,v] of [[g.TEXTURE_MIN_FILTER,g.LINEAR],[g.TEXTURE_MAG_FILTER,g.LINEAR],[g.TEXTURE_WRAP_S,g.CLAMP_TO_EDGE],[g.TEXTURE_WRAP_T,g.CLAMP_TO_EDGE],[g.TEXTURE_COMPARE_MODE,g.COMPARE_REF_TO_TEXTURE],[g.TEXTURE_COMPARE_FUNC,g.LEQUAL]]) g.texParameteri(g.TEXTURE_2D,k,v);
      g.bindFramebuffer(g.FRAMEBUFFER,f); g.framebufferTexture2D(g.FRAMEBUFFER,g.DEPTH_ATTACHMENT,g.TEXTURE_2D,t,0); g.drawBuffers([g.NONE]); g.readBuffer(g.NONE); g.bindFramebuffer(g.FRAMEBUFFER,null);
      return {t,f,unit}; });
    const raw=g.createSampler(); for(const [k,v] of [[g.TEXTURE_MIN_FILTER,g.NEAREST],[g.TEXTURE_MAG_FILTER,g.NEAREST],[g.TEXTURE_WRAP_S,g.CLAMP_TO_EDGE],[g.TEXTURE_WRAP_T,g.CLAMP_TO_EDGE],[g.TEXTURE_COMPARE_MODE,g.NONE]]) g.samplerParameteri(raw,k,v);
    g.activeTexture(g.TEXTURE9); g.bindTexture(g.TEXTURE_2D,this.shadows[0].t); g.bindSampler(9,raw);   // the same map, read without comparison (thickness crossed)
    g.useProgram(p); g.uniform1i(this.u.Sh,5); g.uniform1i(this.u.ShadowNoHead,6); g.uniform1i(this.u.OmZ,9); g.activeTexture(g.TEXTURE0);
    g.enable(g.DEPTH_TEST); g.clearColor(0,0,0,0);
  }
  texture(url){
    if(!url) return null;
    if(!this.tex.has(url)){ const g=this.g,t=g.createTexture(),e={t,ready:false}; this.tex.set(url,e);
      fetch(url).then(r=>r.blob()).then(b=>createImageBitmap(b,{premultiplyAlpha:"none",colorSpaceConversion:"none"})).then(im=>{
        g.bindTexture(g.TEXTURE_2D,t); g.texImage2D(g.TEXTURE_2D,0,g.RGBA,g.RGBA,g.UNSIGNED_BYTE,im); g.generateMipmap(g.TEXTURE_2D);
        g.texParameteri(g.TEXTURE_2D,g.TEXTURE_MIN_FILTER,g.LINEAR_MIPMAP_LINEAR); e.ready=true; e.average=average(im); this.sale=true; if(this.onTexture) this.onTexture(url,e.average); });
    }
    return this.tex.get(url);
  }
  // Adds or replaces a model in the scene; options: skin (texture URL), tint, hair.
  async placer(key,url,opts={}){
    let d=await readGLB(url); const g=this.g;
    let m=this.models.get(key);
    const isNew=!m||m.url!==url;
    let smooth=null;
    if(isNew){
      if(m) this.release(m);
      if(opts.hair){ if(!d.smooth) d.smooth=smoothing(d); smooth=d.smooth; d=smooth.d; }
      const vao=g.createVertexArray(); g.bindVertexArray(vao);
      const buf=(a,loc,n)=>{const b=g.createBuffer();g.bindBuffer(g.ARRAY_BUFFER,b);g.bufferData(g.ARRAY_BUFFER,a,g.DYNAMIC_DRAW);g.enableVertexAttribArray(loc);g.vertexAttribPointer(loc,n,g.FLOAT,false,0,0);return b;};
      m={url,vao,bpos:buf(d.pos,0,3),bnor:buf(d.nor,1,3),buv:buf(d.uv,2,2),brepos:key==="head"?buf(d.pos,4,3):null,d,prims:d.prims.map(p=>{const e=g.createBuffer();g.bindBuffer(g.ELEMENT_ARRAY_BUFFER,e);g.bufferData(g.ELEMENT_ARRAY_BUFFER,p.indices,g.DYNAMIC_DRAW);return {...p,ebo:e,n:p.indices.length};})};
      if(!m.brepos){ g.bindBuffer(g.ARRAY_BUFFER,m.bpos); g.enableVertexAttribArray(4); g.vertexAttribPointer(4,3,g.FLOAT,false,0,0); }   // outside the head, the rest position is simply the position
      m.smooth=smooth;
      g.bindVertexArray(null); this.models.set(key,m);
    }
    m.skin=opts.skin||null; m.tint=opts.tint||null; m.hair=!!opts.hair; m.head=key==="head"; m.body=key==="body";
    if(isNew){ m.top=height(d.pos); m.bottom=bottom(d.pos); if(key!=="body") await this.maps(key,url.replace(/[^/]+$/,"maps.bin")); if(m.head){ m.iris=this.irisBoxes(d); m.mouth=this.mouthBox(d); await this.drawing(m,url.replace(/[^/]+$/,"face-shade")); await this.expressions(m,url.replace(/[^/]+$/,"expressions")); } }
    this.sale=true; return m;
  }
  // Render maps of a model (maps.py): four bytes per vertex; without a file, everything open and nothing glossy.
  async maps(key,url){
    const a=await fetch(url).then(r=>r.ok?r.arrayBuffer():null).catch(()=>null), m=this.models.get(key), g=this.g; if(!m) return;
    g.bindVertexArray(m.vao);
    if(m.bcartes){ g.deleteBuffer(m.bcartes); m.bcartes=null; }
    let o=a&&new Uint8Array(a);
    if(o&&m.smooth&&o.length===m.smooth.n*4) o=m.smooth.average(o,4,Uint8Array,false);
    if(o&&o.length===m.d.pos.length/3*4){
      m.bcartes=g.createBuffer(); g.bindBuffer(g.ARRAY_BUFFER,m.bcartes); g.bufferData(g.ARRAY_BUFFER,o,g.STATIC_DRAW);
      g.enableVertexAttribArray(3); g.vertexAttribPointer(3,4,g.UNSIGNED_BYTE,true,0,0);
    } else g.disableVertexAttribArray(3);
    g.bindVertexArray(null); this.sale=true;
  }
  // Box of each iris (material 目), at +x then at −x: centre x, y, half-width, half-height; without an iris, zeros.
  irisBoxes(d){
    const b=[[1e9,-1e9,1e9,-1e9],[1e9,-1e9,1e9,-1e9]];
    for(const p of d.prims) if(p.name==="目") for(const i of p.indices){ const x=d.pos[3*i],y=d.pos[3*i+1],e=b[x>=0?0:1]; e[0]=Math.min(e[0],x); e[1]=Math.max(e[1],x); e[2]=Math.min(e[2],y); e[3]=Math.max(e[3],y); }
    return new Float32Array(b.flatMap(e=>e[1]<e[0]?[0,0,0,0]:[(e[0]+e[1])/2,(e[2]+e[3])/2,(e[1]-e[0])/2,(e[3]-e[2])/2]));
  }
  // Mouth at rest (material 口线, the lip line): centre x, y, z and half-width; without the line, zeros.
  mouthBox(d){
    let x0=1e9,x1=-1e9,y=0,z=-1e9,n=0;
    for(const p of d.prims) if(p.name==="口线") for(const i of p.indices){ const x=d.pos[3*i]; x0=Math.min(x0,x); x1=Math.max(x1,x); y+=d.pos[3*i+1]; z=Math.max(z,d.pos[3*i+2]); n++; }
    return new Float32Array(n?[(x0+x1)/2,y/n,z,(x1-x0)/2]:[0,0,0,0]);
  }
  // Head expressions (expressions.py): by name, the vertices affected and their offsets.
  async expressions(m,url){
    m.expr=new Map(); m.expression=null;
    const j=await fetch(url+".json").then(r=>r.ok?r.json():null).catch(()=>null); if(!j||j.vertices!==m.d.pos.length/3) return;
    const b=await fetch(url+".bin").then(r=>r.ok?r.arrayBuffer():null).catch(()=>null); if(!b) return;
    for(const e of j.expressions) m.expr.set(e.name,{idx:new Uint32Array(b,e.start,e.n),offset:new Float32Array(b,e.start+4*e.n,3*e.n),gaze:e.gaze||null});
  }
  // Gives the head an expression (null: rest). Returns the available names.
  express(name){
    const m=this.models.get("head"); if(!m||!m.expr) return [];
    const e=m.expr.get(name), a=new Float32Array(m.d.pos), g=this.g;
    if(e) for(let k=0;k<e.idx.length;k++){ const i=3*e.idx[k]; a[i]+=e.offset[3*k]; a[i+1]+=e.offset[3*k+1]; a[i+2]+=e.offset[3*k+2]; }
    g.bindBuffer(g.ARRAY_BUFFER,m.bpos); g.bufferData(g.ARRAY_BUFFER,a,g.DYNAMIC_DRAW); m.expression=e?name:null; this.sale=true;
    return [...m.expr.keys()];
  }
  // Drawn face map (maps.py): image and frame next to the head; without it, the per-vertex map.
  async drawing(m,url){
    const j=await fetch(url+".json").then(r=>r.ok?r.json():null).catch(()=>null); if(!j) return;
    const im=await fetch(url+".png").then(r=>r.blob()).then(b=>createImageBitmap(b,{premultiplyAlpha:"none",colorSpaceConversion:"none"})).catch(()=>null); if(!im) return;
    const g=this.g,t=g.createTexture(); g.activeTexture(g.TEXTURE8); g.bindTexture(g.TEXTURE_2D,t); g.texImage2D(g.TEXTURE_2D,0,g.RGBA,g.RGBA,g.UNSIGNED_BYTE,im);
    for(const [k,v] of [[g.TEXTURE_MIN_FILTER,g.LINEAR],[g.TEXTURE_MAG_FILTER,g.LINEAR],[g.TEXTURE_WRAP_S,g.CLAMP_TO_EDGE],[g.TEXTURE_WRAP_T,g.CLAMP_TO_EDGE]]) g.texParameteri(g.TEXTURE_2D,k,v);
    g.activeTexture(g.TEXTURE0); m.drawing={t,frame:j.frame,axis:j.axis||0}; this.sale=true;
  }
  remove(key){ const m=this.models.get(key); if(m){this.release(m);this.models.delete(key);this.sale=true;} }
  release(m){ const g=this.g; g.deleteVertexArray(m.vao); [m.bpos,m.bnor,m.buv,m.brepos,m.bcartes,...m.prims.map(p=>p.ebo)].forEach(b=>b&&g.deleteBuffer(b)); if(m.drawing) g.deleteTexture(m.drawing.t); }
  positions(key,a){ const m=this.models.get(key); if(!m) return; const g=this.g; if(m.smooth&&a.length===m.smooth.n*3) a=m.smooth.positions(a); g.bindBuffer(g.ARRAY_BUFFER,m.bpos); g.bufferData(g.ARRAY_BUFFER,a,g.DYNAMIC_DRAW); m.top=height(a); m.bottom=bottom(a); this.sale=true; }
  vertices(key,a,n){ const m=this.models.get(key); if(!m) return; this.positions(key,a); const g=this.g; g.bindBuffer(g.ARRAY_BUFFER,m.bnor); g.bufferData(g.ARRAY_BUFFER,n,g.DYNAMIC_DRAW); }
  // Removes the listed faces (numbered across the primitives); true removes them all.
  mask(key,faces){ const m=this.models.get(key); if(!m) return; const g=this.g,s=faces===true?null:new Set(faces); let base=0;
    for(const p of m.prims){ const src=p.indices,n=src.length/3,o=new Uint32Array(s?src.length:0); let k=0;
      if(s) for(let i=0;i<n;i++) if(!s.has(base+i)){o[k++]=src[3*i];o[k++]=src[3*i+1];o[k++]=src[3*i+2];}
      base+=n; g.bindBuffer(g.ELEMENT_ARRAY_BUFFER,p.ebo); g.bufferData(g.ELEMENT_ARRAY_BUFFER,o.subarray(0,k),g.DYNAMIC_DRAW); p.n=k; }
    this.sale=true; }
  // Depth maps of the cast shadows, framed on what the camera looks at. Eyes, lashes and eyebrows cast no shadow.
  wear(L){
    const g=this.g,a=this.cam,r=Math.max(0.3,a.dist*0.34),ci=a.aim;
    // on the face, the hair's shadow seen from a lower light: under the real one, a thick fringe covers the whole top of the face
    const Lt=norm([L[0]*0.6,0.2,L[2]]);
    const [LM,LMt]=[L,Lt].map(l=>mult(ortho(r,0.5,5.5),gaze([ci[0]+l[0]*3,ci[1]+l[1]*3,ci[2]+l[2]*3],ci,[0,1,0])));
    const LMs=mult(ortho(FLOOR_RADIUS,0.5,6.5),gaze([L[0]*3.5,0.8+L[1]*3.5,L[2]*3.5],[0,0.8,0],[0,1,0]));
    g.useProgram(this.po); g.uniform1i(this.uo.T,0); g.viewport(0,0,SHADOW_SIZE,SHADOW_SIZE); g.disable(g.CULL_FACE); g.activeTexture(g.TEXTURE0);
    this.shadows.forEach((o,i)=>{
      g.bindFramebuffer(g.FRAMEBUFFER,o.f); g.clear(g.DEPTH_BUFFER_BIT); g.uniformMatrix4fv(this.uo.LM,false,[LM,LMt,LMs][i]);
      for(const m of this.models.values()){
        if(i===1&&m.head) continue;
        g.bindVertexArray(m.vao);
        for(const p of m.prims){
          if(i<2&&m.head&&!p.skin) continue;
          const x=this.texture((m.skin&&p===m.prims[0])?m.skin:p.image);
          g.uniform1i(this.uo.aTex,x&&x.ready?1:0); if(x&&x.ready) g.bindTexture(g.TEXTURE_2D,x.t);
          g.bindBuffer(g.ELEMENT_ARRAY_BUFFER,p.ebo); g.drawElements(g.TRIANGLES,p.n,g.UNSIGNED_INT,0);
        }
      }
    });
    g.bindFramebuffer(g.FRAMEBUFFER,null); g.bindVertexArray(null);
    return {LM,LMt,LMs,perMeter:1/(2*r)};
  }
  draw(sameShadows=false){
    const size=this.c.width+"x"+this.c.height; this.resize();
    const g=this.g,c=this.c,R=this.r,{VP,L,eye}=this.matrices();
    if(!sameShadows||size!==c.width+"x"+c.height) this.casts=null;
    const casting=R.shadow.casts?R.shadow.castStrength:0;
    const casts=casting>0||(R.shadow.casts&&R.floor.active)?(this.casts||(this.casts=this.wear(L))):null;
    const end=this.buffers();   // the scene goes through a buffer, for the finish
    g.bindFramebuffer(g.FRAMEBUFFER,end.scene);
    g.viewport(0,0,c.width,c.height); g.clear(g.COLOR_BUFFER_BIT|g.DEPTH_BUFFER_BIT);
    g.bindVertexArray(null); g.useProgram(this.pe); g.disable(g.DEPTH_TEST); g.uniform3fv(this.ue.f0,R.background.center); g.uniform3fv(this.ue.f1,R.background.border); g.uniform1i(this.ue.mode,0); g.uniform1f(this.ue.aspect,c.width/c.height); g.drawArrays(g.TRIANGLES,0,3); g.enable(g.DEPTH_TEST);
    g.useProgram(this.p); g.uniformMatrix4fv(this.u.VP,false,VP);
    if(casts){ g.uniformMatrix4fv(this.u.LM,false,casts.LM); g.uniformMatrix4fv(this.u.LMt,false,casts.LMt); g.uniform1f(this.u.perMeter,casts.perMeter); }
    g.uniform3fv(this.u.L,L); g.uniform1i(this.u.T,0);
    { const t=this.models.get("head"), r=t&&t.expression&&t.expr&&t.expr.get(t.expression)?.gaze, b=t&&t.iris;   // the expression's gaze: applies to the head and what surrounds it (hair)
      g.uniform4f(this.u.gaze,r?r.glow||0:0,r?(r.shadow||0)*R.skin.shadowEyes:0,b?(b[1]+b[5])/2:0,b?Math.max(b[3],b[7]):0); }
    // the render settings
    { const l=R.light, o=R.shadow, u=this.u;
      g.uniform3fv(u.key,l.color.map(x=>x*l.force)); g.uniform3fv(u.rimC,l.rim); g.uniform1f(u.rimF,l.forceRim); g.uniform3fv(u.sky,l.sky); g.uniform3fv(u.ground,l.ground);
      g.uniform3fv(u.omV,o.tint); g.uniform3fv(u.omC,o.tintHair); g.uniform3fv(u.omP,R.skin.tint); g.uniform4f(u.marche,o.threshold,o.softness,R.skin.threshold,R.skin.softness);
      g.uniform1f(u.satO,o.saturation); g.uniform1f(u.edgeBand,o.edgeBand); g.uniform1f(u.crevices,o.crevices); g.uniform1f(u.casting,casts?casting:0);
      g.uniform1f(u.transm,R.skin.transmission); g.uniform1f(u.blush,R.skin.blush); g.uniform1f(u.reflection,R.material.hair); g.uniform1i(u.materials,R.material.clothes?1:0);
      g.uniform1f(u.expo,R.image.exposure); g.uniform1f(u.satF,R.image.saturation); }
    g.uniform3fv(this.u.eye,eye);
    g.vertexAttrib4f(3,1,0,1,1); g.activeTexture(g.TEXTURE0);
    g.disable(g.CULL_FACE);
    for(const m of this.models.values()){
      g.bindVertexArray(m.vao); g.uniform1i(this.u.aMaps,m.bcartes?1:0); g.uniform1f(this.u.top,m.top||0);
      g.uniform1i(this.u.aDrawing,m.drawing?1:0); g.uniform1i(this.u.Drawing,8); if(m.drawing){ g.activeTexture(g.TEXTURE8); g.bindTexture(g.TEXTURE_2D,m.drawing.t); g.uniform4fv(this.u.frame,m.drawing.frame); g.uniform1f(this.u.axis,m.drawing.axis); g.activeTexture(g.TEXTURE0); }
      g.uniform4fv(this.u.iris,m.iris||new Float32Array(8)); g.uniform4fv(this.u.mouth,m.mouth||new Float32Array(4));
      const te=m.tint; g.uniform4f(this.u.tint,te?te.r:0,te?te.g:0,te?te.b:0,te?1:0); g.uniform3f(this.u.ref,te?te.ref:1,0,0);
      for(const p of m.prims){
        const u=(m.skin&&p===m.prims[0])?m.skin:p.image, x=this.texture(u);
        g.uniform1i(this.u.aTex,x&&x.ready?1:0); if(x&&x.ready){g.activeTexture(g.TEXTURE0);g.bindTexture(g.TEXTURE_2D,x.t);}
        g.uniform1i(this.u.sort,p.face&&m.head?4:(p.skin?1:(m.hair?2:(m.head?3:0))));
        g.uniform1i(this.u.part_,m.head?(p.name==="白目"?1:(p.name==="目"?2:(p.name==="齿"?3:(/^(口舌|口|舌)$/.test(p.name)?4:0)))):0);
        g.uniform1i(this.u.material,p.material); g.uniform4fv(this.u.diffuse,p.diffuse);   // texture × material colour (MMD convention): VRoid hair, eyebrows and lashes are grey textures tinted by this colour
        g.bindBuffer(g.ELEMENT_ARRAY_BUFFER,p.ebo); g.drawElements(g.TRIANGLES,p.n,g.UNSIGNED_INT,0);
      }
    }
    g.bindVertexArray(null);
    if(R.floor.active) this.floor(VP,casts?casts.LMs:null,casts?casting:0);
    this.finish(end); if(R.image.halo>0) this.glow(R.image.halo); this.grain();
  }
  // Finish buffers, at the size of the view: the scene (antialiased), its colour and depth read back, two working images.
  buffers(){
    const g=this.g,w=this.c.width,h=this.c.height; let x=this.end;
    if(x&&x.w===w&&x.h===h) return x;
    if(x){ [x.tc,x.tz,x.ta,x.tb,x.tl,x.tm].forEach(t=>g.deleteTexture(t)); [x.rc,x.rz].forEach(t=>g.deleteRenderbuffer(t)); [x.scene,x.lit,x.fa,x.fb,x.fl,x.fm].forEach(f=>g.deleteFramebuffer(f)); }
    const tex=(inner,format,type,filter)=>{ const t=g.createTexture(); g.bindTexture(g.TEXTURE_2D,t); g.texImage2D(g.TEXTURE_2D,0,inner,w,h,0,format,type,null);
      for(const [k,v] of [[g.TEXTURE_MIN_FILTER,filter],[g.TEXTURE_MAG_FILTER,filter],[g.TEXTURE_WRAP_S,g.CLAMP_TO_EDGE],[g.TEXTURE_WRAP_T,g.CLAMP_TO_EDGE]]) g.texParameteri(g.TEXTURE_2D,k,v); return t; };
    const buffer=(color,depth)=>{ const f=g.createFramebuffer(); g.bindFramebuffer(g.FRAMEBUFFER,f); g.framebufferTexture2D(g.FRAMEBUFFER,g.COLOR_ATTACHMENT0,g.TEXTURE_2D,color,0);
      if(depth) g.framebufferTexture2D(g.FRAMEBUFFER,g.DEPTH_ATTACHMENT,g.TEXTURE_2D,depth,0); return f; };
    const rb=format=>{ const b=g.createRenderbuffer(); g.bindRenderbuffer(g.RENDERBUFFER,b); g.renderbufferStorageMultisample(g.RENDERBUFFER,Math.min(4,g.getParameter(g.MAX_SAMPLES)),format,w,h); return b; };
    g.activeTexture(g.TEXTURE0);
    x=this.end={w,h,rc:rb(g.RGBA8),rz:rb(g.DEPTH_COMPONENT24),tc:tex(g.RGBA8,g.RGBA,g.UNSIGNED_BYTE,g.LINEAR),tz:tex(g.DEPTH_COMPONENT24,g.DEPTH_COMPONENT,g.UNSIGNED_INT,g.NEAREST),
      ta:tex(g.RGBA8,g.RGBA,g.UNSIGNED_BYTE,g.LINEAR),tb:tex(g.RGBA8,g.RGBA,g.UNSIGNED_BYTE,g.LINEAR),
      tl:tex(g.RGBA8,g.RGBA,g.UNSIGNED_BYTE,g.LINEAR),tm:tex(g.RGBA8,g.RGBA,g.UNSIGNED_BYTE,g.LINEAR)};   // tl, tm: the outline and its smoothing
    x.scene=g.createFramebuffer(); g.bindFramebuffer(g.FRAMEBUFFER,x.scene);
    g.framebufferRenderbuffer(g.FRAMEBUFFER,g.COLOR_ATTACHMENT0,g.RENDERBUFFER,x.rc); g.framebufferRenderbuffer(g.FRAMEBUFFER,g.DEPTH_ATTACHMENT,g.RENDERBUFFER,x.rz);
    x.lit=buffer(x.tc,x.tz); x.fa=buffer(x.ta); x.fb=buffer(x.tb); x.fl=buffer(x.tl); x.fm=buffer(x.tm);
    g.bindFramebuffer(g.FRAMEBUFFER,null); g.bindTexture(g.TEXTURE_2D,null);
    return x;
  }
  // Finish: contact shadows, outline, then depth of field, from the buffered scene to the view.
  finish(x){
    const g=this.g,w=x.w,h=x.h,I=this.r.image,T=this.r.outline;
    g.bindFramebuffer(g.READ_FRAMEBUFFER,x.scene); g.bindFramebuffer(g.DRAW_FRAMEBUFFER,x.lit); g.blitFramebuffer(0,0,w,h,0,0,w,h,g.COLOR_BUFFER_BIT|g.DEPTH_BUFFER_BIT,g.NEAREST);
    g.bindVertexArray(null); g.useProgram(this.pf); g.disable(g.DEPTH_TEST); g.viewport(0,0,w,h);
    g.uniform1i(this.uf.C,0); g.uniform1i(this.uf.Z,10); g.uniform1i(this.uf.A,11); g.uniform2f(this.uf.stage,1/w,1/h); g.uniform1f(this.uf.aspect,w/h);
    g.uniform1f(this.uf.contact,this.r.shadow.contact); g.uniform1f(this.uf.blur,I.blur); g.uniform1f(this.uf.thumbnail,I.thumbnail); g.uniform1f(this.uf.grain,I.grain);
    g.uniform3fv(this.uf.outlineC,T.color); g.uniform1f(this.uf.outlinePart,T.partColor); g.uniform1f(this.uf.outline,0);
    g.activeTexture(g.TEXTURE10); g.bindTexture(g.TEXTURE_2D,x.tz); g.activeTexture(g.TEXTURE11); g.bindTexture(g.TEXTURE_2D,x.ta); g.activeTexture(g.TEXTURE0);
    const passed=(mode,source,aim)=>{ g.bindFramebuffer(g.FRAMEBUFFER,aim); g.uniform1i(this.uf.mode,mode); g.bindTexture(g.TEXTURE_2D,source); g.drawArrays(g.TRIANGLES,0,3); };
    g.activeTexture(g.TEXTURE11); g.bindTexture(g.TEXTURE_2D,null); g.activeTexture(g.TEXTURE0);   // the first pass writes into this image: it does not read it
    passed(0,x.tc,x.fa);
    g.activeTexture(g.TEXTURE11); g.bindTexture(g.TEXTURE_2D,x.ta); g.activeTexture(g.TEXTURE0);
    g.uniform1i(this.uf.G,12);
    if(T.active&&T.opacity>0){
      // outline width: wider in close-up; in image pixels
      // from afar it thins and lightens instead of keeping its width in pixels (it grew thick relative to the character)
      const dist=this.cam.dist, raw=0.0011/(2*Math.tan(FIELD/2)*dist)*1000, e=Math.min(1.9,Math.max(0.5,raw))*h/900*T.thickness;
      g.uniform1f(this.uf.outline,T.opacity*Math.min(1,Math.max(0.55,0.35+0.65*raw)));
      g.activeTexture(g.TEXTURE12); g.bindTexture(g.TEXTURE_2D,null); g.activeTexture(g.TEXTURE0);
      passed(4,x.tc,x.fl); g.uniform2f(this.uf.dir,e/w,0); passed(5,x.tl,x.fm); g.uniform2f(this.uf.dir,0,e/h); passed(5,x.tm,x.fl);
      g.activeTexture(g.TEXTURE12); g.bindTexture(g.TEXTURE_2D,x.tl); g.activeTexture(g.TEXTURE0);
    }
    passed(1,x.tc,x.fb); passed(2,x.tb,null);
    g.enable(g.DEPTH_TEST);
  }
  // Softened edges and light grain, last, over the bloom.
  grain(){
    const I=this.r.image; if(I.thumbnail<=0&&I.grain<=0) return;
    const g=this.g; g.useProgram(this.pf); g.uniform1i(this.uf.mode,3); g.disable(g.DEPTH_TEST); g.enable(g.BLEND); g.blendFunc(g.DST_COLOR,g.SRC_COLOR);
    g.drawArrays(g.TRIANGLES,0,3); g.disable(g.BLEND); g.enable(g.DEPTH_TEST);
  }
  // Floor: disc of light at foot height, with the character's shadow (LMs: its map; casting: shadow strength, 0 for none).
  floor(VP,LMs,casting){
    const g=this.g; let y=1e9; for(const m of this.models.values()) if(m.bottom<y) y=m.bottom; if(y>1e8) return;
    g.useProgram(this.ps); g.uniformMatrix4fv(this.us.VP,false,VP); if(LMs) g.uniformMatrix4fv(this.us.LM,false,LMs); g.uniform1f(this.us.floor,y-0.002);
    g.uniform1i(this.us.Sh0,7); g.uniform1f(this.us.perMeter,1/(2*FLOOR_RADIUS)); g.uniform1f(this.us.casting,LMs?casting:0);
    g.uniform3fv(this.us.s0,this.r.floor.shadow); g.uniform3fv(this.us.s1,this.r.floor.light);
    g.enable(g.BLEND); g.blendFunc(g.ONE,g.ONE_MINUS_SRC_ALPHA); g.depthMask(false);
    g.drawArrays(g.TRIANGLE_STRIP,0,4);
    g.depthMask(true); g.disable(g.BLEND);
  }
  // Bloom: the drawn image is copied, its bright areas reduced to a quarter, blurred twice, then added.
  glow(force){
    const g=this.g,w=this.c.width,h=this.c.height,W=Math.max(1,w>>2),H=Math.max(1,h>>2); let x=this.halo;
    g.activeTexture(g.TEXTURE0);
    if(!x||x.w!==w||x.h!==h){
      if(x){ [x.src,x.a.t,x.b.t].forEach(t=>g.deleteTexture(t)); [x.a.f,x.b.f].forEach(f=>g.deleteFramebuffer(f)); }
      const tex=(w,h)=>{ const t=g.createTexture(); g.bindTexture(g.TEXTURE_2D,t); g.texImage2D(g.TEXTURE_2D,0,g.RGBA8,w,h,0,g.RGBA,g.UNSIGNED_BYTE,null);
        for(const [k,v] of [[g.TEXTURE_MIN_FILTER,g.LINEAR],[g.TEXTURE_MAG_FILTER,g.LINEAR],[g.TEXTURE_WRAP_S,g.CLAMP_TO_EDGE],[g.TEXTURE_WRAP_T,g.CLAMP_TO_EDGE]]) g.texParameteri(g.TEXTURE_2D,k,v); return t; };
      const aim=()=>{ const t=tex(W,H),f=g.createFramebuffer(); g.bindFramebuffer(g.FRAMEBUFFER,f); g.framebufferTexture2D(g.FRAMEBUFFER,g.COLOR_ATTACHMENT0,g.TEXTURE_2D,t,0); return {t,f}; };
      x=this.halo={w,h,src:tex(w,h),a:aim(),b:aim()}; g.bindFramebuffer(g.FRAMEBUFFER,null);
    }
    g.bindTexture(g.TEXTURE_2D,x.src); g.copyTexSubImage2D(g.TEXTURE_2D,0,0,0,0,0,w,h);
    g.useProgram(this.pe); g.uniform1i(this.ue.S,0); g.uniform1f(this.ue.halo,force); g.disable(g.DEPTH_TEST);
    const passed=(mode,source,aim,stage)=>{ g.bindFramebuffer(g.FRAMEBUFFER,aim); g.uniform1i(this.ue.mode,mode); g.uniform2fv(this.ue.stage,stage); g.bindTexture(g.TEXTURE_2D,source); g.drawArrays(g.TRIANGLES,0,3); };
    g.viewport(0,0,W,H);
    passed(1,x.src,x.a.f,[1/w,1/h]);
    for(const k of [1,2.2]){ passed(2,x.a.t,x.b.f,[k/W,0]); passed(2,x.b.t,x.a.f,[0,k/H]); }
    g.viewport(0,0,w,h); g.enable(g.BLEND); g.blendFunc(g.ONE,g.ONE);
    passed(3,x.a.t,null,[0,0]);
    g.disable(g.BLEND); g.enable(g.DEPTH_TEST);
  }
}
// Same scene without WebGL: depth buffer, textures, two tones, hair tint. Of the render, it applies the direction
// and colour of the light, the shadow tints, thresholds and softnesses, the exposure and the background.
export class SoftwareViewer extends ViewerBase {
  constructor(canvas,opts={}){ super(canvas,opts); this.logiciel=true; this.x=canvas.getContext("2d",{willReadFrequently:true}); }
  texture(url){
    if(!url) return null;
    if(!this.tex.has(url)){ const e={ready:false}; this.tex.set(url,e);
      fetch(url).then(r=>r.blob()).then(b=>createImageBitmap(b,{premultiplyAlpha:"none",colorSpaceConversion:"none"})).then(im=>{
        const k=Math.min(1,1024/Math.max(im.width,im.height)),w=Math.max(1,im.width*k|0),h=Math.max(1,im.height*k|0),o=new OffscreenCanvas(w,h),x=o.getContext("2d",{willReadFrequently:true});
        x.drawImage(im,0,0,w,h); e.px=x.getImageData(0,0,w,h).data; e.w=w; e.h=h; e.ready=true; e.average=average(im); this.sale=true; if(this.onTexture) this.onTexture(url,e.average); });
    }
    return this.tex.get(url);
  }
  async placer(key,url,opts={}){
    const d=await readGLB(url); let m=this.models.get(key);
    if(!m||m.url!==url){ m={url,d,pos:d.pos,prims:d.prims.map(p=>({...p,idx:p.indices}))}; this.models.set(key,m); }
    m.skin=opts.skin||null; m.tint=opts.tint||null; m.hair=!!opts.hair; this.sale=true; return m;
  }
  remove(key){ if(this.models.delete(key)) this.sale=true; }
  maps(){}
  positions(key,a){ const m=this.models.get(key); if(m){m.pos=a;this.sale=true;} }
  vertices(key,a,n){ const m=this.models.get(key); if(m){m.pos=a;m.nor=n;this.sale=true;} }
  mask(key,faces){ const m=this.models.get(key); if(!m) return; const s=faces===true?null:new Set(faces); let base=0;
    for(const p of m.prims){ const src=p.indices,n=src.length/3,o=new Uint32Array(s?src.length:0); let k=0;
      if(s) for(let i=0;i<n;i++) if(!s.has(base+i)){o[k++]=src[3*i];o[k++]=src[3*i+1];o[k++]=src[3*i+2];}
      base+=n; p.idx=o.subarray(0,k); }
    this.sale=true; }
  draw(){
    this.resize();
    const c=this.c,W=c.width,H=c.height,R=this.r;
    if(!this.im||this.im.width!==W||this.im.height!==H){ this.im=this.x.createImageData(W,H); this.z=new Float32Array(W*H); }
    const px=this.im.data,Z=this.z; Z.fill(Infinity);
    // background: the render's gradient
    { const f0=R.background.center,f1=R.background.border,as=W/H;
      for(let y=0;y<H;y++) for(let x=0;x<W;x++){ const u=(x+0.5)/W-0.5,v=(1-(y+0.5)/H)-0.58,d=Math.hypot(u*as,v),t0=Math.min(1,d/0.8),t=t0*t0*(3-2*t0),q=4*(y*W+x);
        px[q]=(f0[0]+(f1[0]-f0[0])*t)*255; px[q+1]=(f0[1]+(f1[1]-f0[1])*t)*255; px[q+2]=(f0[2]+(f1[2]-f0[2])*t)*255; px[q+3]=255; } }
    const {VP,L}=this.matrices();
    // the two tones, in sRGB: shade = tint × exposure, lit = light × exposure; one step per kind of surface
    const ex=R.image.exposure, cl=R.light.color.map(x=>x*R.light.force);
    const ton=(om,s,d)=>({o:om.map(x=>enc1(x*ex)),l:cl.map(x=>enc1(x*ex)),m:[s-d,s+d]});
    const TONES={skin:ton(R.skin.tint,R.skin.threshold,R.skin.softness), hair:ton(R.shadow.tintHair,R.shadow.threshold-0.03,2*R.shadow.softness), other:ton(R.shadow.tint,R.shadow.threshold,R.shadow.softness)};
    for(const m of this.models.values()){
      const P=m.pos,Nn=m.nor||m.d.nor,U=m.d.uv,n=P.length/3,C=new Float32Array(n*4);
      for(let i=0;i<n;i++){ const x=P[3*i],y=P[3*i+1],z=P[3*i+2];
        C[4*i]=VP[0]*x+VP[4]*y+VP[8]*z+VP[12]; C[4*i+1]=VP[1]*x+VP[5]*y+VP[9]*z+VP[13]; C[4*i+2]=VP[2]*x+VP[6]*y+VP[10]*z+VP[14]; C[4*i+3]=VP[3]*x+VP[7]*y+VP[11]*z+VP[15]; }
      const te=m.tint;
      for(const p of m.prims){
        const u=(m.skin&&p===m.prims[0])?m.skin:p.image, t=this.texture(u), aT=!!(t&&t.ready), D=p.diffuse, I=p.idx;
        const T=p.skin?TONES.skin:(m.hair?TONES.hair:TONES.other), O=T.o, Lc=T.l, M0=T.m[0], M1=T.m[1];
        for(let f=0;f<I.length;f+=3){ const a=I[f],b=I[f+1],e=I[f+2],wa=C[4*a+3],wb=C[4*b+3],wc=C[4*e+3];
          if(wa<=0.005||wb<=0.005||wc<=0.005) continue;
          const ax=(C[4*a]/wa+1)*W/2,ay=(1-C[4*a+1]/wa)*H/2,bx=(C[4*b]/wb+1)*W/2,by=(1-C[4*b+1]/wb)*H/2,cx=(C[4*e]/wc+1)*W/2,cy=(1-C[4*e+1]/wc)*H/2;
          const A=(bx-ax)*(cy-ay)-(by-ay)*(cx-ax); if(A===0) continue;
          const face=A<0; // counter-clockwise order once the vertical axis is flipped
          const x0=Math.max(0,Math.floor(Math.min(ax,bx,cx))),x1=Math.min(W-1,Math.ceil(Math.max(ax,bx,cx))),y0=Math.max(0,Math.floor(Math.min(ay,by,cy))),y1=Math.min(H-1,Math.ceil(Math.max(ay,by,cy)));
          const za=C[4*a+2]/wa,zb=C[4*b+2]/wb,zc=C[4*e+2]/wc,ia=1/wa,ib=1/wb,ic=1/wc;
          for(let y=y0;y<=y1;y++) for(let x=x0;x<=x1;x++){
            const X=x+0.5,Y=y+0.5,l0=((bx-X)*(cy-Y)-(by-Y)*(cx-X))/A,l1=((cx-X)*(ay-Y)-(cy-Y)*(ax-X))/A,l2=1-l0-l1;
            if(l0<0||l1<0||l2<0) continue;
            const zz=l0*za+l1*zb+l2*zc,o=y*W+x; if(zz>=Z[o]) continue;
            const w=l0*ia+l1*ib+l2*ic,p0=l0*ia/w,p1=l1*ib/w,p2=l2*ic/w;
            let r=1,g=1,bl=1,al=1;
            if(aT){ const uu=p0*U[2*a]+p1*U[2*b]+p2*U[2*e],vv=p0*U[2*a+1]+p1*U[2*b+1]+p2*U[2*e+1],sx=Math.min(t.w-1,(uu-Math.floor(uu))*t.w|0),sy=Math.min(t.h-1,(vv-Math.floor(vv))*t.h|0),q=(sy*t.w+sx)*4;
              r=t.px[q]/255;g=t.px[q+1]/255;bl=t.px[q+2]/255;al=t.px[q+3]/255; if(al<0.4) continue; }
            r*=D[0];g*=D[1];bl*=D[2];
            if(te){ const v=Math.max(r,g,bl),vc=Math.max(te.r,te.g,te.b),sc=vc>0?1-Math.min(te.r,te.g,te.b)/vc:0,v2=Math.min(v*vc/te.ref,1),d=Math.max(vc,1e-5);
              r=v2*te.r/d*sc+v2*(1-sc);g=v2*te.g/d*sc+v2*(1-sc);bl=v2*te.b/d*sc+v2*(1-sc); }
            let nx=p0*Nn[3*a]+p1*Nn[3*b]+p2*Nn[3*e],ny=p0*Nn[3*a+1]+p1*Nn[3*b+1]+p2*Nn[3*e+1],nz=p0*Nn[3*a+2]+p1*Nn[3*b+2]+p2*Nn[3*e+2];
            const ln=Math.hypot(nx,ny,nz)||1,sg=face?1:-1; nx*=sg/ln;ny*=sg/ln;nz*=sg/ln;
            const li=(nx*L[0]+ny*L[1]+nz*L[2])*0.5+0.5,t0=Math.max(0,Math.min(1,(li-M0)/(M1-M0))),s=t0*t0*(3-2*t0);
            const R2=r*(O[0]+(Lc[0]-O[0])*s),G2=g*(O[1]+(Lc[1]-O[1])*s),B2=bl*(O[2]+(Lc[2]-O[2])*s);
            Z[o]=zz; const q2=o*4; px[q2]=Math.min(255,R2*255);px[q2+1]=Math.min(255,G2*255);px[q2+2]=Math.min(255,B2*255);px[q2+3]=255;
          }
        }
      }
    }
    this.x.putImageData(this.im,0,0);
  }
}
// Mean value (max R,G,B) of the visible pixels: reference for the hair tint.
function average(im){
  const c=new OffscreenCanvas(48,48),x=c.getContext("2d",{willReadFrequently:true}); x.drawImage(im,0,0,48,48);
  const d=x.getImageData(0,0,48,48).data; let s=0,n=0;
  for(let i=0;i<d.length;i+=4) if(d[i+3]>128){ s+=Math.max(d[i],d[i+1],d[i+2])/255; n++; }
  return n?s/n:0.5;
}
