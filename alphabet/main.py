import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
from skimage.io import imread
from skimage.measure import label, regionprops

def holes(obj):
    img = obj.image
    h, w = img.shape
    temp = np.zeros((h + 2, w + 2))
    temp[1:-1, 1:-1] = img
    temp = np.logical_not(temp)
    lbl = label(temp)
    return np.max(lbl) - 1

def lines(obj):
    img = obj.image
    h, w = img.shape
    vert = (np.sum(img, axis=0) / h == 1).sum()
    horiz = (np.sum(img, axis=1) / w == 1).sum()
    return vert, horiz

def sim(obj, tr=False):
    img = obj.image
    if tr:
        img = img.T
    h, w = img.shape
    top = img[:h // 2]
    bottom = img[h // 2 + 1:] if h % 2 else img[h // 2:]
    bottom = bottom[::-1]
    return (top == bottom).sum() / (top == bottom).size

def find_symbol(obj):
    h = holes(obj)
    if h == 2:
        v, _ = lines(obj)
        return "B" if v / obj.image.shape[1] > 0.2 else "8"

    if h == 1:
        v, _ = lines(obj)
        if sim(obj) > 0.7:
            return "D" if v / obj.image.shape[1] > 0.1 else "O"
        else:
            return "P" if v > 0 else "A"

    img = obj.image

    if img.sum() / img.size > 0.95:
        return "-"

    h, w = img.shape
    k = min(h, w) / max(h, w)

    if k > 0.9:
        return "*"

    v, _ = lines(obj)

    if v > 1:
        return "1"

    if sim(obj) > 0.82 and sim(obj, True) > 0.82:
        return "X"

    if sim(obj, True) > 0.8:
        return "W"

    return "/"

img = imread("symbols.png")[:, :, :3]
bw = img.mean(axis=2) > 0
lbl = label(bw)
objs = regionprops(lbl)
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

res = {}

n = len(objs)
cols = 5
rows = math.ceil(n / cols)
plt.figure(figsize=(cols * 3, rows * 3))

for i, obj in enumerate(objs):
    s = find_symbol(obj)
    res[s] = res.get(s, 0) + 1

    plt.subplot(rows, cols, i + 1)
    plt.imshow(obj.image, cmap="binary")
    plt.title(s, fontsize=4)
    plt.axis("off")

print("Символов:", len(objs))
print("\nСловарь:\n")
for k in sorted(res):
    print(k, "->", res[k])

plt.tight_layout()
plt.savefig(out_dir / "all_symbols.png")
plt.show()

# with open("result.txt", "w", encoding="utf-8") as f:
#    f.write(f"symbols: {len(objs)}\n")
#    for k in sorted(res):
#        f.write(f"{k} -> {res[k]}\n")
