import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

b_dir = Path(__file__).parent

def holes(r):
    img = np.zeros((r.image.shape[0] + 2, r.image.shape[1] + 2))
    img[1:-1, 1:-1] = r.image
    img = np.logical_not(img)
    return np.max(label(img)) - 1

def lines(r):
    img = r.image
    h, w = img.shape
    v = (np.sum(img, 0) / h == 1).sum()
    h = (np.sum(img, 1) / w == 1).sum()
    return v, h

def sym(r):
    img = r.image.astype(int)
    mid = img.shape[1] // 2
    left = img[:, :mid]
    right = np.fliplr(img[:, mid:])
    n = min(left.shape[1], right.shape[1])
    return np.mean(np.abs(left[:, :n] - right[:, :n]))

def feat(r):
    h, w = r.image.shape
    cy, cx = r.centroid_local
    cy /= h
    cx /= w
    v, hln = lines(r)

    return np.array([
        r.area / r.image.size,
        cy,
        cx,
        r.perimeter / r.image.size,
        holes(r),
        v / w,
        hln / h,
        r.eccentricity,
        h / w,
        sym(r)
    ])

def classify(r, temp):
    f = feat(r)
    best = ""
    best_d = 1e18

    for k in temp:
        d = np.linalg.norm(temp[k] - f)
        if d < best_d:
            best_d = d
            best = k

    return best

tmpl_img = imread("alphabet-small.png")[:, :, :-1]
tmpl_bin = tmpl_img.sum(2) != 765
tmpl_lbl = label(tmpl_bin)
tmpl_props = regionprops(tmpl_lbl)

symbols = ["8", "0", "A", "B", "1", "W", "X", "*", "/", "-"]

templates = {}
for r, s in zip(tmpl_props, symbols):
    templates[s] = feat(r)

img = imread("alphabet.png")[:, :, :-1]
bin_img = img.mean(2) > 0
lbl = label(bin_img)
props = regionprops(lbl)

res = {}


out_dir = b_dir / "results"
out_dir.mkdir(exist_ok=True)
n = len(props)
cols = 5
rows = math.ceil(n / cols)

plt.figure(figsize=(cols * 3, rows * 3))

for i, r in enumerate(props):
    s = classify(r, templates)
    res[s] = res.get(s, 0) + 1

    plt.subplot(rows, cols, i + 1)
    plt.imshow(r.image, cmap="binary")
    plt.title(s)
    plt.axis("off")


print("\nРЕЗУЛЬТАТ:\n")
for k, v in sorted(res.items()):
    print(f"{k} -> {v}")

plt.tight_layout()
plt.savefig(out_dir / f"{i}_{s}.png")
plt.show()
