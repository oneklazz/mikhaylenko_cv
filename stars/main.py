import numpy as np
from skimage.measure import label
from skimage.morphology import opening

img = np.load("data/stars.npy")
clean = opening(img)
labeled = label(clean)

print(labeled.max())
