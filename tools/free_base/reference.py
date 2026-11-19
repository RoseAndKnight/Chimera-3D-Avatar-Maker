"""The MakeHuman body presented to the grid (tools/rebuilt_body) as a shape reference.

Same interface as rebuilt_body/reference.Reference: V (cm, y up, front = -z, left = +x), F
(triangles), f_body, f_main, f_skull, os (MMD names cited by the grid), weights_os, landmarks.

Nothing comes from a game: the surface is the `body` group of base.obj (CC0), the joints are the
centres of the `joint-*` cubes of the same file. MakeHuman faces +z, in decimetres: z is flipped
(the body is symmetric: left stays at +x) and units become centimetres, feet on the ground.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("rebuilt_body", "atlas", "head", "rebuilt_head"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
sys.path.insert(0, HERE)
import makehuman as MH  # noqa: E402

# MMD name cited by the grid -> MakeHuman joint
OS = {"上半身": "spine-3", "下半身": "pelvis", "首": "neck", "頭": "head",
      "腕": "shoulder", "ひじ": "elbow", "手首": "hand", "足": "upper-leg", "ひざ": "knee", "足首": "ankle",
      "つま先": "foot-2",
      "親指０": "finger-1-1", "親指１": "finger-1-2", "親指２": "finger-1-3"}
for _n, _k in (("人", 2), ("中", 3), ("薬", 4), ("小", 5)):
    for _i, _c in enumerate("１２３"):
        OS["%s指%s" % (_n, _c)] = "finger-%d-%d" % (_k, _i + 1)


class Reference:
    def __init__(self, path, targets=None):
        """targets: see makehuman.read (None: the silhouette of silhouette.json; {}: the neutral mesh)."""
        V, faces, J, _ = MH.read(path, targets)
        ground = J["ground"][1]
        to = lambda p: (np.asarray(p, float) - [0, ground, 0]) * [10.0, 10.0, -10.0]   # noqa: E731
        ids = np.unique(np.concatenate([np.array(f) for f in faces]))
        loc = -np.ones(len(V), int)
        loc[ids] = np.arange(len(ids))
        self.V = to(V[ids])
        Q = loc[np.array(faces)]
        Q = Q[:, ::-1]                                  # z flipped: the face winding too
        self.quads_source = Q
        self.F = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
        self.os = {}
        for name, j in OS.items():
            if name in ("上半身", "下半身", "首", "頭"):
                self.os[name] = to(J[j])
            else:
                self.os["左" + name] = to(J["l-" + j])
                self.os["右" + name] = to(J["r-" + j])
        self.joints = {k: to(p) for k, p in J.items()}
        c = self.V[self.F].mean(1)
        all_ = np.arange(len(self.F))
        body = np.ones(len(self.F), bool)
        self.f_main = {}
        for side, sg in (("G", "左"), ("D", "右")):
            a, p = self.os[sg + "ひじ"], self.os[sg + "手首"]
            u = (p - a) / np.linalg.norm(p - a)
            of_side = np.sign(c[:, 0]) == np.sign(p[0])
            near = np.linalg.norm(c - p, axis=1) < 30.0                 # the hand, not the thigh it brushes against
            self.f_main[side] = all_[((c - p) @ u > -3.0) & of_side & near]
            body &= ~(((c - p) @ u > 3.0) & of_side & near)
        self.f_body = all_[body]
        self.f_skull = np.zeros(0, int)
        self._arm = self._part_arm()

    def _part_arm(self):
        """1 for a vertex closer to an arm (shoulder-elbow-wrist-fingers) than to the trunk and legs:
        stands in for the bone weights that the grid reads on a weighted reference."""
        o = self.os

        def dist(P, a, b):
            d = b - a
            t = np.clip(((P - a) @ d) / (d @ d), 0, 1)
            return np.linalg.norm(P - (a + t[:, None] * d), axis=1)
        out = {}
        for sg, s in (("左", 1.0), ("右", -1.0)):
            ch = [o[sg + "腕"], o[sg + "ひじ"], o[sg + "手首"], o[sg + "中指３"]]
            db = np.min([dist(self.V, a, b) for a, b in zip(ch[:-1], ch[1:])], 0)
            ep = (o["首"] + o[sg + "腕"]) / 2
            tr = [(o["下半身"], o["首"]), (o["首"], o["頭"]), (ep, o[sg + "足"]), (o[sg + "足"], o[sg + "ひざ"]),
                  (o[sg + "ひざ"], o[sg + "足首"]), (o["首"], ep)]
            dt = np.min([dist(self.V, a, b) for a, b in tr], 0)
            out[sg] = ((db < dt) & (self.V[:, 0] * s > 0)).astype(float)
        return out

    def weight_os(self, names):
        sides = {n[0] for n in names}
        assert len(sides) == 1 and all(any(m in n for m in ("腕", "ひじ", "手")) for n in names), names
        return self._arm[sides.pop()]

    def unfold(self, *a, **k):
        return 0


def landmarks(R, log=print):
    """The grid's landmark heights for this body. Its skin is closed (the head is part of it): the
    neck stops a little below its narrowest section, before the underside of the chin."""
    import cut as CP
    import grid as QD
    o = R.os
    V = R.V[np.unique(R.F[R.f_body])]
    middle = V[np.abs(V[:, 0]) < 0.5]
    bottom = float(middle[middle[:, 1] > 0.5 * o["左足"][1]][:, 1].min())
    ys = np.arange(o["首"][1] - 4.0, o["頭"][1] + 4.0, 0.25)
    per = []
    for y in ys:
        t = (y - o["首"][1]) / (o["頭"][1] - o["首"][1])
        z = float(o["首"][2] + np.clip(t, -0.5, 1.5) * (o["頭"][2] - o["首"][2])) - 0.3
        r = CP.outline_around(R.V, R.F[R.f_body], np.array([0.0, y, z]), [0, -1, 0], [0, 0, -1])
        per.append(CP.length(r[0]) if r else np.nan)
    per = np.array(per)
    k = int(np.nanargmin(per))
    log("neck: narrowest section at y = %.1f cm (%.1f cm around)" % (ys[k], per[k]))
    for y, p in zip(ys[::4], per[::4]):
        log("   y %.1f  girth %.1f" % (y, p))
    return {"y_crotch": bottom + 1.8, "y_pelvis": float(o["上半身"][1]) + 5.8,
            "neck_sweep": (float(ys[k]), float(o["首"][1]) - 8.0),
            "y_neck_default": float(o["首"][1]) - 0.1, "y_top": float(ys[k]) - 1.0,      # under the chin
            "toe_perimeter_max": 10.0}      # the big toe of this body is 8.6 cm around


if __name__ == "__main__":
    R = Reference(sys.argv[1])
    print("vertices", len(R.V), "triangles", len(R.F), "box", R.V.min(0).round(1), R.V.max(0).round(1))
    print("body", len(R.f_body), "hands", {k: len(v) for k, v in R.f_main.items()})
    for k in ("上半身", "首", "頭", "左腕", "左ひじ", "左手首", "左足", "左ひざ", "左足首"):
        print(k, R.os[k].round(1))
