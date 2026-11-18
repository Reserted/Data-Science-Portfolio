"""Dataset portraits for the hyperparameter posts: many real images per class, not one.

    python blog/diagrams/datasets.py <data-dir>

<data-dir> holds the raw files torchvision itself downloads (no torch needed here):
    kmnist-train-imgs.npz, kmnist-train-labels.npz   codh.rois.ac.jp/kmnist
    EuroSAT.zip                                     huggingface.co/datasets/torchgeo/eurosat
    usps.bz2                                        csie.ntu.edu.tw/~cjlin/libsvmtools
Writes figures/dataset-*.png into each post's folder.
"""
import bz2
import io
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA, BLOG = Path(sys.argv[1]), Path(__file__).parent.parent
BG, INK, DIM = "#fcfcfb", "#1d232b", "#5b6470"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"]})
rng = np.random.default_rng(0)


def grid(rows, labels, path, cmap=None, cell=0.62, note=None):
    """rows: list of image lists (one row per class). labels: one string per row."""
    n = len(rows[0])
    fig, axes = plt.subplots(len(rows), n, figsize=(cell * n + 1.6, cell * len(rows) + (0.4 if note else 0)),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.06})
    for r, (imgs, label) in enumerate(zip(rows, labels)):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10, labelpad=8)
    if note:
        fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=9, color=DIM)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(BLOG))


def pick(y, k, n):
    return rng.choice(np.flatnonzero(y == k), n, replace=False)


# Kuzushiji-MNIST: twelve writers' versions of each character, plus the class average.
x = np.load(DATA / "kmnist-train-imgs.npz")["arr_0"]
y = np.load(DATA / "kmnist-train-labels.npz")["arr_0"]
kana = "おきすつなはまやれを"
roma = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
rows = [[x[i] for i in pick(y, k, 12)] + [x[y == k].mean(0)] for k in range(10)]
grid(rows, [f"{k}  {r}" for k, r in zip(kana, roma)], BLOG / "learning-rate/figures/dataset-kmnist.png",
     cmap="gray", note="Left: the modern character it stands for. Last column: the average of all 6,000 training images of that class.")

# EuroSAT (RGB): ten 64 x 64 Sentinel-2 patches per land-use class.
z = zipfile.ZipFile(DATA / "EuroSAT.zip")
names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
classes = sorted({n.split("/")[1] for n in names})
rows, labels = [], []
for c in classes:
    files = [n for n in names if n.split("/")[1] == c]
    rows.append([np.asarray(Image.open(io.BytesIO(z.read(files[i])))) for i in rng.choice(len(files), 10, replace=False)])
    labels.append(f"{c}  ({len(files):,})")
grid(rows, labels, BLOG / "batch-size/figures/dataset-eurosat.png", cell=0.9,
     note="Ten random patches per class. In brackets: images in the class (27,000 total). Each patch covers 640 m × 640 m.")

# USPS: libsvm text, label 1..10 = digit 0..9, 256 pixels in [-1, 1].
lines = bz2.open(DATA / "usps.bz2", "rt").read().split("\n")
lines = [ln.split() for ln in lines if ln.strip()]
uy = np.array([int(float(ln[0])) - 1 for ln in lines])
ux = np.array([[float(t.split(":")[1]) for t in ln[1:]] for ln in lines]).reshape(-1, 16, 16)
rows = [[ux[i] for i in pick(uy, k, 14)] for k in range(10)]
grid(rows, [f"{k}  ({(uy == k).sum():,})" for k in range(10)], BLOG / "dropout/figures/dataset-usps.png",
     cmap="gray", note=f"Fourteen random training images per digit. In brackets: images of that digit ({len(uy):,} total).")

# One USPS digit with every pixel value printed: 256 numbers are the whole image.
img = ux[pick(uy, 5, 1)[0]]
fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
for (r, c), v in np.ndenumerate(img):
    ax.text(c, r, f"{v + 0:.1f}".replace("-0.0", "0.0").replace("-1.0", "−1").replace("1.0", "1").replace("-", "−"), ha="center", va="center",
            fontsize=6.2, color=INK if v > 0 else "#c9d1da")
ax.set_xticks(np.arange(-.5, 16), minor=True), ax.set_yticks(np.arange(-.5, 16), minor=True)
ax.grid(which="minor", color=DIM, lw=0.3), ax.tick_params(which="both", length=0, labelsize=7, colors=DIM)
ax.set_xticks(range(0, 16, 3)), ax.set_yticks(range(0, 16, 3))
path = BLOG / "dropout/figures/dataset-usps-pixels.png"
fig.savefig(path, dpi=160, bbox_inches="tight")
print("wrote", path.relative_to(BLOG))
