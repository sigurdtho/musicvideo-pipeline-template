"""r906c deep QA: contact sheets with many frames per clip (the user: at least 16, more for action), optional zoom box.

  python -m pipeline.clipqa <clip_id> [--n 16] [--zoom x,y,w,h]   -> qc/r906c/clips/<clip_id>.jpg (4 columns)
The zoom box (source pixels of the 1280x720 clip) adds a second sheet with the box enlarged on every sampled frame.
"""
import sys

import cv2
import numpy as np

from pipeline import manifest
from pipeline.manifest import ROOT

OUT = ROOT / "qc" / "r906c" / "clips"


def frames(path, idx):
    cap, out, i, want = cv2.VideoCapture(str(path)), {}, 0, set(idx)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    cap.release()
    return [out[k] for k in idx if k in out], i


def sheet(ims, labels, w=480):
    cells = []
    for im, lab in zip(ims, labels):
        h = int(im.shape[0] * w / im.shape[1])
        c = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(c, lab, (6, 20), 0, 0.55, (0, 255, 255), 2)
        cells.append(c)
    while len(cells) % 4:
        cells.append(np.zeros_like(cells[0]))
    return np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)])


def main(argv):
    cid = argv[0]
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 16
    a = manifest.get(cid)
    p = ROOT / a["meta"]["files"]["fps16"]
    _, total = frames(p, [])
    idx = [round(k * (total - 1) / (n - 1)) for k in range(n)]
    ims, _ = frames(p, idx)
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [f"{cid.split(':', 1)[1]} f{k} {k / 16:.2f}s" for k in idx]
    cv2.imwrite(str(OUT / f"{cid.replace(':', '_')}.jpg"), sheet(ims, labels), [cv2.IMWRITE_JPEG_QUALITY, 82])
    if "--zoom" in argv:
        x, y, w, h = map(int, argv[argv.index("--zoom") + 1].split(","))
        z = [im[y:y + h, x:x + w] for im in ims]
        cv2.imwrite(str(OUT / f"{cid.replace(':', '_')}_zoom.jpg"), sheet(z, labels, 320), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(OUT / f"{cid.replace(':', '_')}.jpg")


if __name__ == "__main__":
    main(sys.argv[1:])
