import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import (opening,dilation,closing,erosion)

files = [
    "files/wires1.npy",
    "files/wires2.npy",
    "files/wires3.npy",
    "files/wires4.npy",
    "files/wires5.npy",
    "files/wires6.npy"
]

struct = np.ones((3, 1))

for f in files:
    print("\n--------------------")
    print(f)
    img = np.load(f)
    proc = opening(img, footprint=struct)
    labeled = label(proc)
    wires_count = labeled.max()
    print(f"wires: {wires_count}")

    for n in range(1, wires_count + 1):
        mask = (labeled == n)
        parts = label(mask)
        print(f"Wire = {n}, parts = {parts.max()}")

    if f == "files/wires3.npy":
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(proc)

        plt.show()
