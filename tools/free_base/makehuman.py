"""MakeHuman's base body (base.obj, CC0) read as a shape reference: the skin (group `body`),
and the position of the joints (centres of the `joint-*` cubes).

Nothing else is taken: the `helper-*` groups (skirt, tights, hair, eyes, teeth) are left out.
"""
import json
import os

import numpy as np


def doses():
    """The shape targets of silhouette.json: {path under makehuman/data/targets: dose}."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "silhouette.json"), encoding="utf-8") as f:
        return json.load(f)["targets"]


def targets_per_default(path):
    """The same, as files: in "targets" next to base.obj, at the same path. {file: dose}."""
    d = os.path.join(os.path.dirname(os.path.abspath(path)), "targets")
    return {os.path.join(d, n): w for n, w in doses().items()}


def read(path, targets=None):
    """Returns V (n, 3) in file units, the faces of the body group (index lists), the centres of the
    joints {name: (3,)}. targets: {.target file: dose} added to the vertices (the joint cubes
    follow); None: those of silhouette.json; {}: the neutral mesh."""
    V, groups, g = [], {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for l in lines:
        if l.startswith("v "):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("g "):
            g = l.split()[1]
            groups.setdefault(g, [])
        elif l.startswith("f "):
            groups[g].append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
    V = np.array(V)
    for c, w in (targets_per_default(path) if targets is None else targets).items():
        with open(c, encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            if l.strip() and not l.startswith("#"):
                i, x, y, z = l.split()
                V[int(i)] += w * np.array([float(x), float(y), float(z)])
    joints = {k[6:]: V[np.unique(np.concatenate(f))].mean(0) for k, f in groups.items() if k.startswith("joint-")}
    return V, groups["body"], joints, groups


if __name__ == "__main__":
    import sys
    V, F, J, Gr = read(sys.argv[1])
    ids = np.unique(np.concatenate([np.array(f) for f in F]))
    B = V[ids]
    print("body: %d vertices, %d faces (%s sides)" % (len(ids), len(F), sorted(set(len(f) for f in F))))
    print("box", B.min(0).round(3), B.max(0).round(3))
    for k in sorted(J):
        print("%-24s %s" % (k, J[k].round(3)))
