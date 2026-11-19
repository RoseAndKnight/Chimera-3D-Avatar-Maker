"""Builds Chimera's free base body, from start to finish.

    python3 -I tools/free_base/build.py [--sources <folder>] [--output <folder>] [--no-sheets]

 1. downloads, if missing, MakeHuman's base mesh (base.obj), its licence, its eyes and the shape targets
    of silhouette.json, from MakeHuman's public repository at a fixed version (CC0, see PROVENANCE.md);
 2. lays our grid on its skin (trial.py, one minute);
 3. gives it the silhouette, the skeleton, the A-pose, the weights and a head (body.py);
 4. writes the PMX with MMD bones (pmx_mmd.py): <output>/body.pmx, the base body to give to Chimera
    (setting workshop.body);
 5. writes the free face and hairstyle (free_face.py): <output>/head.pmx and hairstyle.pmx, which Chimera
    offers as long as no game model is in place;
 6. renders check contact sheets (view_body.py).
Defaults: sources in {data}/makehuman, output in {data}/free-base. Nothing comes from a game.
"""
import argparse
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import makehuman as MH        # noqa: E402
import settings as R          # noqa: E402
from free_face import EYES  # noqa: E402

VERSION = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"      # makehumancommunity/makehuman repository, on 15/11/2026
ADDRESS = "https://raw.githubusercontent.com/makehumancommunity/makehuman/" + VERSION + "/"
FILES = {"base.obj": "makehuman/data/3dobjs/base.obj", "LICENSE.md": "LICENSE.md"}


def certificates():
    """The certificate authorities: Python's, otherwise certifi's, otherwise the system's (the Python
    shipped with Blender finds none on its own)."""
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats()["x509_ca"]:
        return ctx
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    for f in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/cert.pem"):
        if os.path.isfile(f):
            return ssl.create_default_context(cafile=f)
    return ctx


def download(sources, log=print):
    """The MakeHuman files missing from sources; returns the path of base.obj."""
    wanted = dict(FILES)
    for c in MH.doses():
        wanted[os.path.join("targets", c)] = "makehuman/data/targets/" + c
    for c in EYES.values():                              # the eyes of the free head (free_face.py)
        wanted[c] = "makehuman/data/" + c
    ctx = None
    for local, distant in wanted.items():
        path = os.path.join(sources, local)
        if os.path.isfile(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log("download: " + distant)
        ctx = ctx or certificates()
        with urllib.request.urlopen(ADDRESS + distant, timeout=60, context=ctx) as r, open(path + ".tmp", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(path + ".tmp", path)
    return os.path.join(sources, "base.obj")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = R.path("paths.data")
    p.add_argument("--sources", default=os.path.join(data, "makehuman"))
    p.add_argument("--output", dest="output", default=os.path.join(data, "free-base"))
    p.add_argument("--no-sheets", dest="without_sheets", action="store_true")
    a = p.parse_args(argv)
    obj = download(a.sources)
    os.makedirs(a.output, exist_ok=True)
    steps = [["trial.py", obj, a.output], ["body.py", obj, a.output], ["pmx_mmd.py", a.output],
              ["free_face.py", a.sources, a.output]]
    if not a.without_sheets:
        steps.append(["view_body.py", a.output])
    for e in steps:
        print("—", e[0], flush=True)
        subprocess.run([sys.executable, "-I", os.path.join(HERE, e[0])] + e[1:], check=True)
    shutil.copyfile(os.path.join(HERE, "PROVENANCE.md"), os.path.join(a.output, "PROVENANCE.md"))
    body = os.path.join(a.output, "body.pmx")
    print("base body: %s (with head.pmx and hairstyle.pmx: the free face and hairstyle)" % body)
    if os.path.realpath(body) != os.path.realpath(R.path("workshop.body")):
        print("Chimera reads another body (%s). For this one: \"atelier\": {\"corps\": \"%s\", \"cache\": <a cache of its own>} "
              "in settings.json" % (R.path("workshop.body"), body))
    else:
        print("Next: python3 -I tools/workshop/catalog.py prepare, then bin/chimera")

if __name__ == "__main__":
    main()
