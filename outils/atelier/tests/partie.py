"""A real session on Chimera's page: start the server and Firefox, choose, assemble, wait.

    ATELIER_CACHE=… python3 -I outils/atelier/tests/partie.py \\
        --selection selection.json --sortie folder [--nom essai-1] [--port 8732] [--delai 3600]

`selection.json`: {"tete": "Jane", "coiffure": "Anby", "pieces": ["…"], "couleur": "#7a4a2c" or null}.
Everything goes through the page (clicks on the tiles and on Assemble). Returns the model folder; writes
into `sortie`: <nom>-apercu.png (the page), <nom>-planche.png, <nom>.json (final state, page alerts).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, ".."))
from pilote_firefox import Pilote  # noqa: E402
import commun as C  # noqa: E402


def lancer_firefox(port_m):
    prof = tempfile.mkdtemp(prefix="firefox-", dir=os.environ.get("TMPDIR") or None)
    with open(os.path.join(prof, "user.js"), "w") as f:
        f.write('user_pref("security.sandbox.content.level", 0);\nuser_pref("marionette.port", %d);\nuser_pref("webgl.force-enabled", true);\n' % port_m)
    p = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", prof], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p, prof


def attendre_serveur(port, delai=30):
    fin = time.time() + delai
    while True:
        try:
            return json.load(urllib.request.urlopen("http://127.0.0.1:%d/api/catalogue" % port, timeout=10))
        except OSError:
            if time.time() > fin:
                raise
            time.sleep(0.5)


def cliquer(pi, css):
    if not pi.js("const e=document.querySelector(arguments[0]); if(!e) return false; e.click(); return true;", css):
        raise LookupError("nothing to click: %s" % css)


def choisir(pi, onglet, ident):
    cliquer(pi, '#onglets [data-o="%s"]' % onglet)
    pi.attendre("document.querySelector('#grille [data-id=%s]')" % json.dumps(ident))
    pi.js("const e=document.querySelector('#grille [data-id=' + JSON.stringify(arguments[0]) + ']'); if(!e.classList.contains('choisie')) e.click();", ident)


def partie(pi, port, sel, cat, sortie, nom, delai):
    onglet = {p["id"]: p["onglet"] for p in cat["pieces"]}
    manque = [i for i in sel["pieces"] if i not in onglet]
    if manque:
        raise LookupError("missing from the catalogue: %s" % ", ".join(manque))
    pi.aller("http://127.0.0.1:%d/" % port)
    pi.attendre("document.querySelector('#grille .tuile') && document.querySelector('#selection li')", 180)
    while pi.js("const b=document.querySelector('#selection [data-r]'); if(!b) return false; b.click(); return true;"):
        pass
    choisir(pi, "visage", sel["tete"])
    choisir(pi, "coiffure", sel["coiffure"])
    for i in sel["pieces"]:
        choisir(pi, onglet[i], i)
    cliquer(pi, '#onglets [data-o="coiffure"]')
    cliquer(pi, '#pastilles [title=%s]' % json.dumps(sel["couleur"]) if sel["couleur"] else '#pastilles [data-c="0"]')   # 0: original colour, any language
    pi.attendre("document.body.dataset.couches==='pret'", 300, 1)
    time.sleep(3)    # time to draw the image (the software preview is slow)
    alertes = pi.js("return document.querySelector('#alertes').innerText")
    apercu = pi.capture(os.path.join(sortie, nom + "-apercu.png"))
    cliquer(pi, "#assembler")
    pi.attendre("document.querySelector('#resultat img') || /Failed|Échec/.test(document.querySelector('#etape').textContent)", delai, 3)
    etape = pi.js("return document.querySelector('#etape').textContent")
    fin = pi.js("return {html:document.querySelector('#resultat').innerText, vrm:(document.querySelector('#resultat a[download]')||{}).href||null, img:(document.querySelector('#resultat img')||{}).src||null}")
    bilan = dict(selection=sel, alertes_page=alertes, etape=etape, resultat=fin["html"], vrm=fin["vrm"], apercu=apercu)
    if fin["img"]:
        id_ = fin["img"].split("/data/assemblages/")[1].split("/")[0]
        d = os.path.join(C.CACHE, "assemblages", id_)
        shutil.copy(os.path.join(d, "planche.png"), os.path.join(sortie, nom + "-planche.png"))
        bilan["dossier"] = os.path.join(d, "modele")
        bilan["etat"] = C.lire_json(os.path.join(d, "etat.json"))
    C.ecrire_json(os.path.join(sortie, nom + ".json"), bilan)
    return bilan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--nom", default="partie")
    p.add_argument("--port", type=int, default=8732)
    p.add_argument("--delai", type=float, default=3600)
    a = p.parse_args()
    sel = C.lire_json(a.selection)
    sel.setdefault("couleur", None)
    os.makedirs(a.sortie, exist_ok=True)
    port_m = a.port + 1000
    env = dict(os.environ)
    srv = subprocess.Popen([sys.executable, "-I", os.path.join(ICI, "..", "serveur.py"), "--port", str(a.port)], env=env)
    ff, prof = None, None
    try:
        cat = attendre_serveur(a.port)
        ff, prof = lancer_firefox(port_m)
        pi = Pilote(port=port_m)
        pi.taille(1500, 1000)
        b = partie(pi, a.port, sel, cat, a.sortie, a.nom, a.delai)
        pi.fermer()
        print(json.dumps({k: b.get(k) for k in ("etape", "dossier", "vrm", "alertes_page", "resultat")}, ensure_ascii=False, indent=1))
    finally:
        if ff:
            ff.terminate()
            try:
                ff.wait(10)
            except subprocess.TimeoutExpired:
                ff.kill()
        srv.terminate()
        if prof:
            shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
