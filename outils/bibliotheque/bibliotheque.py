"""Chimera's ready-made library: install it (any user), build or check it (maintainer).

    python3 -I outils/bibliotheque/bibliotheque.py installer [SOURCE] [--remplacer] [--garder-archives]
        SOURCE: the package's address (https://…/resolve/main) or a local folder; default: bibliotheque.adresse
        of the settings (reglages.defaut.json). Installs into the data folder of the settings.
    python3 -I outils/bibliotheque/bibliotheque.py construire <installation> <output> --statuts <statuts.json> [--repo owner/name]
    python3 -I outils/bibliotheque/bibliotheque.py verifier <output> --statuts <statuts.json>

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
    if commande == "installer":
        import installation as I
        remplacer, garder = "--remplacer" in a, "--garder-archives" in a
        a = [x for x in a if x not in ("--remplacer", "--garder-archives")]
        source = a[0] if a else I.R.valeur("bibliotheque.adresse", None)
        if not source:
            sys.exit("no package address: give one, or set bibliotheque.adresse in reglages.json")
        if not I._distante(source):
            source = os.path.abspath(os.path.expanduser(source))
        try:
            I.installer(source, remplacer=remplacer, garder=garder)
        except I.Erreur as e:
            sys.exit("installer: %s" % e)
        return 0
    if commande in ("construire", "verifier"):
        import paquet as P
        statuts = _option(a, "--statuts")
        repo = _option(a, "--repo", "Rose-and-Knight/chimera-vroid-library")
        if not statuts:
            sys.exit("--statuts <statuts.json> is needed (status of each VRoid Hub model: originale, fan…)")
        if commande == "construire":
            P.construire(a[0], a[1], statuts, repo=repo)
            return 1 if P.verifier(a[1], statuts) else 0
        return 1 if P.verifier(a[0], statuts) else 0
    sys.exit("unknown command: %s (installer, construire, verifier)" % commande)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
