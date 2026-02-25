
import os
import numpy as np
from skimage import io

img = io.imread('maps/100.png', 1) * 255
print(f"Min: {np.min(img)}, Max: {np.max(img)}, Mean: {np.mean(img)}")
unique, counts = np.unique(img.astype(int), return_counts=True)
sorted_counts = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]
print("Top values:", sorted_counts)
