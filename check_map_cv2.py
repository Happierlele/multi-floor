
import cv2
import numpy as np

img = cv2.imread('maps/100.png', cv2.IMREAD_UNCHANGED)
print(f"Shape: {img.shape}")
print(f"Dtype: {img.dtype}")
print(f"Unique: {np.unique(img)}")

img_gray = cv2.imread('maps/100.png', cv2.IMREAD_GRAYSCALE)
print(f"Gray Unique: {np.unique(img_gray)}")
