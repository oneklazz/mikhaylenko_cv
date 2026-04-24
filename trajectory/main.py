import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import ndimage

zip_name = "motion.zip"
extract_dir = "motion_unzipped"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_name, "r") as z:
        z.extractall(extract_dir)

def gnum(name):
	return int(name.stem.split("_")[1])

folder = Path(extract_dir) / "out"
files = sorted(list(folder.glob("*.npy")), key=gnum)
coords_all = []

for f in files:
	img = np.load(f)
	lab, cnt = ndimage.label(img)
	centers = []

	for i in range(1, cnt + 1):
		m = lab == i
		y, x = ndimage.center_of_mass(m)
		centers.append((x, y))

	coords_all.append(centers)

first = coords_all[0]
tracks = [[p] for p in first]
prev = first

for k in range(1, len(coords_all)):
	cur = coords_all[k]
	new_prev = [None] * len(prev)

	for i, (px, py) in enumerate(prev):
		best = None
		best_d = float("inf")

		for j, (cx, cy) in enumerate(cur):
			d = np.hypot(cx - px, cy - py)
			if d < best_d:
				best_d = d
				best = (cx, cy)

		tracks[i].append(best)
		new_prev[i] = best

	prev = new_prev

for t in tracks:
	xs = [p[0] for p in t]
	ys = [p[1] for p in t]
	plt.plot(xs, ys)

plt.savefig("trajectory.png", dpi=200)
plt.show()
