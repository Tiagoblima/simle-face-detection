import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

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

    component_mask = (labels == 0).astype("uint8") * 255
    # show our output image and connected component mask

    return component_mask


def create_rectangle(src_image, mask, color=(255, 0, 0)):
    comp1, comp2 = np.where(mask == 255)
    start_point = (min(comp2), min(comp1))
    end_point = (max(comp2), max(comp1))
    thickness = 2

    return cv2.rectangle(src_image.copy(), start_point, end_point, color, thickness)


def hsv_skin_detection(src_image):
    image_hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    # create NumPy arrays from the boundaries
    min_hsv = np.array([0, 58, 30], dtype="uint8")
    max_hsv = np.array([33, 255, 255], dtype="uint8")

    skin_region_hsv = cv2.inRange(image_hsv, min_hsv, max_hsv)
    skin_hsv = cv2.bitwise_and(src_image, src_image, mask=skin_region_hsv)

    return skin_hsv


def quantization(image):
    # skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
    image = image.copy()
    (h, w) = image.shape[:2]
    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=2)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    return image


def demo():
    image = cv2.imread(os.path.join(ROOT_DIR, 'TD_RGB_E_4.jpg'))
    image = cv2.resize(image, (500, 300))

    cv2.imshow("image", image)
    cv2.waitKey(0)

    # ------------------ HSV skin detection --------------------------

    hsv_detected_image = hsv_skin_detection(image)
    cv2.imshow(f"FACE HSV", hsv_detected_image)
    cv2.waitKey(0)

    # ------------------ HSI skin detection --------------------------
    image_hsi = hsi_face_detection(image)
    cv2.imshow(f"skinHSI", image_hsi)
    cv2.waitKey(0)

    cv2.imwrite('skin.png', image_hsi)

    cv2.waitKey(0)
