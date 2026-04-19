import numpy as np
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
  return (1 - t) * v0 + t * v1

size = 100
image = np.zeros((size, size, 3), dtype="uint8")

color1 = np.array([0, 128, 255])
color2 = np.array([255, 128, 0])

for i in range(size):
  for j in range(size):
    t = (i + j) / (2 * (size - 1))
    image[i, j] = lerp(color1, color2, t)

plt.imshow(image)
plt.show()
