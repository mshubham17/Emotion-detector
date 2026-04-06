import cv2
import numpy as np

img = cv2.imread("angry.png")

# 1. What shape is this image?
print("Shape:", img.shape)           # (height, width, 3)
print("Data type:", img.dtype)       # uint8 — values 0 to 255
print("Total values:", img.size)     # height × width × 3

# 2. Read one pixel (row 100, column 200)
pixel = img[100, 200]
print("Pixel at [100,200]:", pixel)  # [Blue, Green, Red]  ← OpenCV is BGR!

# 3. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Grayscale shape:", gray.shape)  # (height, width) — no 3rd dimension

# 4. Show the image in a window (press any key to close)
cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()