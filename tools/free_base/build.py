"""Builds Chimera's free base body, from start to finish.

    python3 -I tools/free_base/build.py [--sources <folder>] [--output <folder>] [--no-sheets]

 1. downloads, if missing, MakeHuman's base mesh (base.obj), its licence, its eyes and the shape targets
    of silhouette.json, from MakeHuman's public repository at a fixed version (CC0, see PROVENANCE.md);
 2. lays our grid on its skin (trial.py, one minute);
 3. gives it the silhouette, the skeleton, the A-pose, the weights and a head (body.py);
 4. writes the PMX with MMD bones (pmx_mmd.py): <sortie>/corps.pmx, the base body to give to Chimera
    (setting workshop.body);
 5. writes the free face and hairstyle (free_face.py): <sortie>/tete.pmx and coiffure.pmx, which Chimera
    offers as long as no game model is in place;
 6. renders check contact sheets (view_body.py).
Defaults: sources in {data}/makehuman, output in {data}/base-libre. Nothing comes from a game.
"""
import argparse
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI, os.path.join(ICI, "..")]
import makehuman as MH        # noqa: E402
import settings as R          # noqa: E402
from free_face import YEUX  # noqa: E402

VERSION = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"      # makehumancommunity/makehuman repository, on 15/11/2026
ADRESSE = "https://raw.githubusercontent.com/makehumancommunity/makehuman/" + VERSION + "/"
FICHIERS = {"base.obj": "makehuman/data/3dobjs/base.obj", "LICENSE.md": "LICENSE.md"}


def certificats():
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


def telecharger(sources, journal=print):
    """The MakeHuman files missing from sources; returns the path of base.obj."""
    voulus = dict(FICHIERS)
    for c in MH.doses():
        voulus[os.path.join("targets", c)] = "makehuman/data/targets/" + c
    for c in YEUX.values():                              # the eyes of the free head (free_face.py)
        voulus[c] = "makehuman/data/" + c
    ctx = None
    for local, distant in voulus.items():
        chemin = os.path.join(sources, local)
        if os.path.isfile(chemin):
            continue
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        journal("download: " + distant)
        ctx = ctx or certificats()
        with urllib.request.urlopen(ADRESSE + distant, timeout=60, context=ctx) as r, open(chemin + ".tmp", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(chemin + ".tmp", chemin)
    return os.path.join(sources, "base.obj")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    donnees = R.chemin("paths.data")
    p.add_argument("--sources", default=os.path.join(donnees, "makehuman"))
    p.add_argument("--output", dest="sortie", default=os.path.join(donnees, "base-libre"))
    p.add_argument("--no-sheets", dest="sans_planches", action="store_true")
    a = p.parse_args(argv)
    obj = telecharger(a.sources)
    os.makedirs(a.sortie, exist_ok=True)
    etapes = [["trial.py", obj, a.sortie], ["body.py", obj, a.sortie], ["pmx_mmd.py", a.sortie],
              ["free_face.py", a.sources, a.sortie]]
    if not a.sans_planches:
        etapes.append(["view_body.py", a.sortie])
    for e in etapes:
        print("—", e[0], flush=True)
        subprocess.run([sys.executable, "-I", os.path.join(ICI, e[0])] + e[1:], check=True)
    shutil.copyfile(os.path.join(ICI, "PROVENANCE.md"), os.path.join(a.sortie, "PROVENANCE.md"))
    corps = os.path.join(a.sortie, "corps.pmx")
    print("base body: %s (with tete.pmx and coiffure.pmx: the free face and hairstyle)" % corps)
    if os.path.realpath(corps) != os.path.realpath(R.chemin("workshop.body")):
        print("Chimera reads another body (%s). For this one: \"atelier\": {\"corps\": \"%s\", \"cache\": <a cache of its own>} "
              "in settings.json" % (R.chemin("workshop.body"), corps))
    else:
        print("Next: python3 -I tools/workshop/catalog.py prepare, then bin/chimera")

if __name__ == "__main__":
    main()
