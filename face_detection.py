import os

import cv2
import numpy as np

from detection import skin_detection, hair_detection

ROOT_DIR = 'dataset/'

# Load an color image in grayscale


image = cv2.imread(os.path.join(ROOT_DIR, 'test.png'))
image = cv2.resize(image, (500, 300))
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# ------------------ HSV skin detection --------------------------

# create NumPy arrays from the boundaries
min_HSV = np.array([0, 58, 30], dtype="uint8")
max_HSV = np.array([33, 255, 255], dtype="uint8")

skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
skinHSV = cv2.bitwise_and(image, image, mask=skinRegionHSV)
res2, thr2 = cv2.threshold(skinHSV, 12, 255, cv2.THRESH_BINARY)

# ------------------ HSI skin detection --------------------------
skin = skin_detection(image)
# show the images
res1, thr1 = cv2.threshold(skin, 12, 255, cv2.THRESH_BINARY)

# ---- HSI hair detection --------


hair_hsi = hair_detection(image)
# res3, thr3 = cv2.threshold(hair_hsi, 12, 255, cv2.THRESH_BINARY)


gray = cv2.cvtColor(skinHSV, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 12, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output


componentMask = (labels == 0).astype("uint8") * 255
# show our output image and connected component mask
cv2.imshow(f"Connected Component {0}", componentMask)
cv2.waitKey(0)

cv2.imshow("image", skinHSV)
cv2.waitKey(0)
