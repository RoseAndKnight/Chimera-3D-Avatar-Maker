"""A real session on Chimera's page: start the server and Firefox, choose, assemble, wait.

    CHIMERA_CACHE=… python3 -I tools/workshop/tests/session.py \\
        --selection selection.json --output folder [--name trial-1] [--port 8732] [--timeout 3600]

`selection.json`: {"head": "Jane", "hairstyle": "Anby", "parts": ["…"], "color": "#7a4a2c" or null}.
Everything goes through the page (clicks on the tiles and on Assemble). Returns the model folder; writes
into `output`: <name>-preview.png (the page), <name>-sheet.png, <name>.json (final state, page alerts).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from firefox_driver import Driver  # noqa: E402
import common as C  # noqa: E402


def launch_firefox(port_m):
    depth = tempfile.mkdtemp(prefix="firefox-", dir=os.environ.get("TMPDIR") or None)
    with open(os.path.join(depth, "user.js"), "w") as f:
        f.write('user_pref("security.sandbox.content.level", 0);\nuser_pref("marionette.port", %d);\nuser_pref("webgl.force-enabled", true);\n' % port_m)
    p = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", depth], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p, depth


def wait_server(port, delay=30):
    end = time.time() + delay
    while True:
        try:
            return json.load(urllib.request.urlopen("http://127.0.0.1:%d/api/catalog" % port, timeout=10))
        except OSError:
            if time.time() > end:
                raise
            time.sleep(0.5)


def click(pi, css):
    if not pi.js("const e=document.querySelector(arguments[0]); if(!e) return false; e.click(); return true;", css):
        raise LookupError("nothing to click: %s" % css)


def choose(pi, tab, ident):
    click(pi, '#tabs [data-o="%s"]' % tab)
    pi.wait("document.querySelector('#grid [data-id=%s]')" % json.dumps(ident))
    pi.js("const e=document.querySelector('#grid [data-id=' + JSON.stringify(arguments[0]) + ']'); if(!e.classList.contains('chosen')) e.click();", ident)


def region(pi, port, sel, cat, output, name, delay):
    tab = {p["id"]: p["tab"] for p in cat["parts"]}
    missing = [i for i in sel["parts"] if i not in tab]
    if missing:
        raise LookupError("missing from the catalogue: %s" % ", ".join(missing))
    pi.go("http://127.0.0.1:%d/" % port)
    pi.wait("document.querySelector('#grid .tile') && document.querySelector('#selection li')", 180)
    while pi.js("const b=document.querySelector('#selection [data-r]'); if(!b) return false; b.click(); return true;"):
        pass
    choose(pi, "face", sel["head"])
    choose(pi, "hairstyle", sel["hairstyle"])
    for i in sel["parts"]:
        choose(pi, tab[i], i)
    click(pi, '#tabs [data-o="hairstyle"]')
    click(pi, '#dots [title=%s]' % json.dumps(sel["color"]) if sel["color"] else '#dots [data-c="0"]')   # 0: original colour, any language
    pi.wait("document.body.dataset.layers==='ready'", 300, 1)
    time.sleep(3)    # time to draw the image (the software preview is slow)
    warnings = pi.js("return document.querySelector('#warnings').innerText")
    preview = pi.capture(os.path.join(output, name + "-preview.png"))
    click(pi, "#assemble")
    pi.wait("document.querySelector('#result img') || /Failed|Échec/.test(document.querySelector('#step').textContent)", delay, 3)   # fr-ok: the page's French failure label
    step = pi.js("return document.querySelector('#step').textContent")
    end = pi.js("return {html:document.querySelector('#result').innerText, vrm:(document.querySelector('#result a[download]')||{}).href||null, img:(document.querySelector('#result img')||{}).src||null}")
    assessment = dict(selection=sel, warnings_page=warnings, step=step, result=end["html"], vrm=end["vrm"], preview=preview)
    if end["img"]:
        id_ = end["img"].split("/data/assemblies/")[1].split("/")[0]
        d = os.path.join(C.CACHE, "assemblies", id_)
        shutil.copy(os.path.join(d, "sheet.png"), os.path.join(output, name + "-sheet.png"))
        assessment["folder"] = os.path.join(d, "model")
        assessment["state"] = C.read_json(os.path.join(d, "state.json"))
    C.write_json(os.path.join(output, name + ".json"), assessment)
    return assessment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True)
    p.add_argument("--output", dest="output", required=True)
    p.add_argument("--name", dest="name", default="region")
    p.add_argument("--port", type=int, default=8732)
    p.add_argument("--timeout", dest="delay", type=float, default=3600)
    a = p.parse_args()
    sel = C.read_json(a.selection)
    sel.setdefault("color", None)
    os.makedirs(a.output, exist_ok=True)
    port_m = a.port + 1000
    env = dict(os.environ)
    srv = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "..", "server.py"), "--port", str(a.port)], env=env)
    ff, depth = None, None
    try:
        cat = wait_server(a.port)
        ff, depth = launch_firefox(port_m)
        pi = Driver(port=port_m)
        pi.size(1500, 1000)
        b = region(pi, a.port, sel, cat, a.output, a.name, a.delay)
        pi.close()
        print(json.dumps({k: b.get(k) for k in ("step", "folder", "vrm", "warnings_page", "result")}, ensure_ascii=False, indent=1))
    finally:
        if ff:
            ff.terminate()
            try:
                ff.wait(10)
            except subprocess.TimeoutExpired:
                ff.kill()
        srv.terminate()
        if depth:
            shutil.rmtree(depth, ignore_errors=True)


if __name__ == "__main__":
    main()
