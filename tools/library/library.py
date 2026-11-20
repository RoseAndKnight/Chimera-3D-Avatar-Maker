"""Chimera's ready-made library: install it (any user), build or check it (maintainer).

    python3 -I tools/library/library.py install [SOURCE] [--replace] [--keep-archives]
        SOURCE: the package's address (https://…/resolve/main) or a local folder; default: library.url
        of the settings (settings.default.json). Installs into the data folder of the settings.
    python3 -I tools/library/library.py build <installation> <output> --statuses <statuses.json> [--repo owner/name]
    python3 -I tools/library/library.py check <output> --statuses <statuses.json>
    python3 -I tools/library/library.py migrate <data folder> [--settings <settings.json>] [--check]
        converts a data folder of format 1 (French names, before 19/11/2026) to format 2, in place
        (migrate.py says what it does); --check lists what still has a format-1 name

README.md of this folder: what the package holds and why.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _option(a, name, default=None):
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return v
    return default


def main(a):
    if not a or "-h" in a or "--help" in a:
        print(__doc__)
        return 0
    command, a = a[0], list(a[1:])
    if command == "install":
        import installation as I
        replace, keep = "--replace" in a, "--keep-archives" in a
        a = [x for x in a if x not in ("--replace", "--keep-archives")]
        source = a[0] if a else I.R.value("library.url", None)
        if not source:
            sys.exit("no package address: give one, or set library.url in settings.json")
        if not I._remote(source):
            source = os.path.abspath(os.path.expanduser(source))
        try:
            I.install(source, replace=replace, keep=keep)
        except I.Error as e:
            sys.exit("install: %s" % e)
        return 0
    if command in ("build", "check"):
        import package as P
        statuses = _option(a, "--statuses")
        repo = _option(a, "--repo", "Rose-and-Knight/chimera-vroid-library")
        if not statuses:
            sys.exit("--statuses <statuses.json> is needed (status of each VRoid Hub model, as written in the maintainer's file)")
        if command == "build":
            P.build(a[0], a[1], statuses, repo=repo)
            return 1 if P.check(a[1], statuses) else 0
        return 1 if P.check(a[0], statuses) else 0
    if command == "migrate":
        import migrate as M
        return M.main(a)
    sys.exit("unknown command: %s (install, build, check, migrate)" % command)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
