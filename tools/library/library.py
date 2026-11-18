"""Chimera's ready-made library: install it (any user), build or check it (maintainer).

    python3 -I tools/library/library.py install [SOURCE] [--replace] [--keep-archives]
        SOURCE: the package's address (https://…/resolve/main) or a local folder; default: library.url
        of the settings (settings.default.json). Installs into the data folder of the settings.
    python3 -I tools/library/library.py build <installation> <output> --statuses <statuses.json> [--repo owner/name]
    python3 -I tools/library/library.py check <output> --statuses <statuses.json>

README.md of this folder: what the package holds and why.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _option(a, nom, defaut=None):
    if nom in a:
        i = a.index(nom)
        v = a[i + 1]
        del a[i:i + 2]
        return v
    return defaut


def main(a):
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    commande, a = a[0], list(a[1:])
    if commande == "install":
        import installation as I
        remplacer, garder = "--replace" in a, "--keep-archives" in a
        a = [x for x in a if x not in ("--replace", "--keep-archives")]
        source = a[0] if a else I.R.valeur("library.url", None)
        if not source:
            sys.exit("no package address: give one, or set library.url in settings.json")
        if not I._distante(source):
            source = os.path.abspath(os.path.expanduser(source))
        try:
            I.installer(source, remplacer=remplacer, garder=garder)
        except I.Erreur as e:
            sys.exit("install: %s" % e)
        return 0
    if commande in ("build", "check"):
        import package as P
        statuts = _option(a, "--statuses")
        repo = _option(a, "--repo", "Rose-and-Knight/chimera-vroid-library")
        if not statuts:
            sys.exit("--statuses <statuses.json> is needed (status of each VRoid Hub model: originale, fan…)")
        if commande == "build":
            P.construire(a[0], a[1], statuts, repo=repo)
            return 1 if P.verifier(a[1], statuts) else 0
        return 1 if P.verifier(a[0], statuts) else 0
    sys.exit("unknown command: %s (install, build, check)" % commande)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
