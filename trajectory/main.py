import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

zip_name = "motion.zip"
out_dir = "motion_unzipped/out"

if not os.path.exists(out_dir):
    with zipfile.ZipFile(zip_name, "r") as z:
        z.extractall("motion_unzipped")

files = sorted(os.listdir(out_dir))
xs = []
ys = []
for f in files:
    frame = np.load(os.path.join(out_dir, f))

    y, x = np.where(frame > 0)

    if len(x) == 0:
        continue

    xs.append(x.mean())
    ys.append(y.mean())

plt.plot(xs, ys, "-o")
plt.gca().invert_yaxis()
plt.grid()

plt.savefig("trajectory.png", dpi=200)
plt.show()
