"""Image reading without Blender or PIL, to sample a texture outside Blender:
8-bit PNG (grey, RGB, RGBA, palette, non-interlaced) and uncompressed 24- or 32-bit TGA
(the two formats of the atlas face textures). Returns an (h, w, 4) array of floats in
0-1, first row at the top; ValueError for any other format."""
import struct
import zlib

import numpy as np

CANAUX = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def lire(chemin):
    if chemin.lower().endswith(".tga"):
        return _tga(chemin)
    return _png(chemin)


def _tga(chemin):
    with open(chemin, "rb") as fh:
        d = fh.read()
    id_long, typ, w, h, prof, desc = d[0], d[2], *struct.unpack("<HH", d[12:16]), d[16], d[17]
    if typ != 2 or prof not in (24, 32) or d[1]:
        raise ValueError("%s: TGA of type %d, %d bits not supported" % (chemin, typ, prof))
    c = prof // 8
    px = np.frombuffer(d, np.uint8, w * h * c, 18 + id_long).reshape(h, w, c)
    px = px[..., [2, 1, 0] + ([3] if c == 4 else [])]  # BGR(A) -> RGB(A)
    if c == 3:
        px = np.concatenate([px, np.full((h, w, 1), 255, np.uint8)], -1)
    if not desc & 0x20:  # origin at the bottom
        px = px[::-1]
    return np.ascontiguousarray(px).astype(np.float32) / 255


def _png(chemin):
    with open(chemin, "rb") as fh:
        d = fh.read()
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("%s: not a PNG" % chemin)
    i, idat, pal, trns = 8, [], None, None
    while i < len(d):
        n, typ = struct.unpack(">I4s", d[i:i + 8])
        corps = d[i + 8:i + 8 + n]
        if typ == b"IHDR":
            w, h, prof, coul, _, _, entrelace = struct.unpack(">IIBBBBB", corps)
        elif typ == b"PLTE":
            pal = np.frombuffer(corps, np.uint8).reshape(-1, 3)
        elif typ == b"tRNS":
            trns = np.frombuffer(corps, np.uint8)
        elif typ == b"IDAT":
            idat.append(corps)
        elif typ == b"IEND":
            break
        i += 12 + n
    if prof != 8 or entrelace:
        raise ValueError("%s: %d-bit or interlaced PNG not supported" % (chemin, prof))
    c = CANAUX[coul]
    brut = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, 1 + w * c)
    filtres, lignes = brut[:, 0], brut[:, 1:].astype(np.int32)
    px = np.zeros((h, w * c), np.int32)
    prec = np.zeros(w * c, np.int32)
    for y in range(h):
        x, f = lignes[y], filtres[y]
        if f == 0:
            r = x
        elif f == 2:
            r = (x + prec) & 255
        elif f == 1:  # cumulative sum per channel
            r = np.cumsum(x.reshape(w, c), 0).reshape(-1) & 255
        else:
            r = np.empty_like(x)
            for k in range(w * c):
                a = r[k - c] if k >= c else 0
                b = prec[k]
                if f == 3:
                    r[k] = (x[k] + (a + b) // 2) & 255
                else:
                    cc = prec[k - c] if k >= c else 0
                    p = a + b - cc
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - cc)
                    r[k] = (x[k] + (a if pa <= pb and pa <= pc else b if pb <= pc else cc)) & 255
        px[y] = r
        prec = r
    px = px.reshape(h, w, c).astype(np.uint8)
    if coul == 3:
        a = np.full(256, 255, np.uint8)
        if trns is not None:
            a[:len(trns)] = trns
        idx = px[..., 0]
        px = np.concatenate([pal[idx], a[idx][..., None]], -1)
    elif coul == 0:
        px = np.concatenate([px, px, px, np.full_like(px, 255)], -1)
    elif coul == 4:
        px = np.concatenate([px[..., :1].repeat(3, -1), px[..., 1:]], -1)
    elif coul == 2:
        px = np.concatenate([px, np.full(px.shape[:2] + (1,), 255, np.uint8)], -1)
    return px.astype(np.float32) / 255
