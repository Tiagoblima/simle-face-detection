import itertools
import os

import cv2
import numpy as np

from detection import skin_detection, hair_detection
import warnings

warnings.filterwarnings("ignore")
ROOT_DIR = 'dataset/'


# Load an color image in grayscale

def generate_component_mask(src_image):
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 12, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    componentMask = (labels == 0).astype("uint8") * 255
    # show our output image and connected component mask
    return componentMask


def create_rectangle(src_image, mask, color=(255, 0, 0)):
    horizontal, vertical = np.where(mask == 255)
    print(horizontal, vertical)
    h_bound_upper = horizontal[0]
    h_bound_lower = horizontal[-1]

    v_bound_lower = vertical[-1]

    h_bound_lower -= h_bound_lower - v_bound_lower

    start_point = (h_bound_upper, h_bound_upper)

    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (h_bound_lower, h_bound_lower)
    print(start_point, end_point)
    # Blue color in BGR

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    return cv2.rectangle(src_image, start_point, end_point, color, thickness)


def hsv_face_detection(src_image):
    image_hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    # create NumPy arrays from the boundaries
    min_hsv = np.array([0, 58, 30], dtype="uint8")
    max_hsv = np.array([33, 255, 255], dtype="uint8")

    skin_region_hsv = cv2.inRange(image_hsv, min_hsv, max_hsv)
    skin_hsv = cv2.bitwise_and(src_image, src_image, mask=skin_region_hsv)

    skin_hsv_component = generate_component_mask(skin_hsv)

    rect_image = create_rectangle(src_image.copy(), skin_hsv_component)
    return rect_image


image = cv2.imread(os.path.join(ROOT_DIR, 'TD_RGB_E_1.jpg'))
image = cv2.resize(image, (500, 300))

cv2.imshow("image", image)
cv2.waitKey(0)

# ------------------ HSV skin detection --------------------------


hsv_detected_image = hsv_face_detection(image)
cv2.imshow(f"FACE HSV", hsv_detected_image)
cv2.waitKey(0)

# ------------------ HSI skin detection --------------------------
skinHSI = skin_detection(image)
# show the images

skinHSIComponent = generate_component_mask(skinHSI)
rect_image_hsi = create_rectangle(image.copy(), skinHSIComponent, (0, 255, 0))
cv2.imshow(f"skinHSI", rect_image_hsi)
cv2.waitKey(0)

cv2.imwrite('skin.png', skinHSI)


cv2.waitKey(0)
