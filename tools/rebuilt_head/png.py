"""Read and write an 8-bit PNG (RGB or RGBA, non-interlaced) with no module other than zlib."""
import struct
import zlib

import numpy as np


def lire(chemin):
    d = open(chemin, "rb").read()
    assert d[:8] == b"\x89PNG\r\n\x1a\n"
    p, idat = 8, b""
    while p < len(d):
        n, t = struct.unpack(">I4s", d[p:p + 8])
        c = d[p + 8:p + 8 + n]
        if t == b"IHDR":
            w, h, prof, typ, _, _, entrelace = struct.unpack(">IIBBBBB", c)
            assert prof == 8 and typ in (2, 6) and not entrelace, (prof, typ, entrelace)
        elif t == b"IDAT":
            idat += c
        p += 12 + n
    k = 3 if typ == 2 else 4
    brut = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * k)
    out = np.zeros((h, w * k), np.uint8)
    prec = np.zeros(w * k, np.int32)
    for y in range(h):
        f, l = brut[y, 0], brut[y, 1:].astype(np.int32)
        if f == 0:
            r = l
        elif f == 2:
            r = (l + prec) & 255
        elif f == 1:
            r = l.reshape(w, k).cumsum(0).ravel() & 255
        else:
            r = np.zeros(w * k, np.int32)
            for x in range(w * k):
                a = r[x - k] if x >= k else 0
                b = prec[x]
                c = prec[x - k] if x >= k else 0
                if f == 3:
                    pr = (a + b) // 2
                else:
                    q = a + b - c
                    pa, pb, pc = abs(q - a), abs(q - b), abs(q - c)
                    pr = a if pa <= pb and pa <= pc else b if pb <= pc else c
                r[x] = (l[x] + pr) & 255
        out[y] = r
        prec = r
    return out.reshape(h, w, k)


def ecrire(chemin, img):
    """img: (h, w, 3 or 4) uint8."""
    h, w, k = img.shape
    brut = np.zeros((h, 1 + w * k), np.uint8)
    brut[:, 1:] = img.reshape(h, w * k)

    def bloc(t, c):
        return struct.pack(">I", len(c)) + t + c + struct.pack(">I", zlib.crc32(t + c) & 0xffffffff)
    with open(chemin, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + bloc(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2 if k == 3 else 6, 0, 0, 0))
                + bloc(b"IDAT", zlib.compress(brut.tobytes(), 6)) + bloc(b"IEND", b""))
