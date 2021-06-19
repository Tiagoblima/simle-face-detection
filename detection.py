import cv2
import numpy as np

from face_detection import kmeans_quantization
from util import RGB2HSI


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()


def skin_segmentation(r_row, g_row, h_row):
    skin_mask = np.zeros(shape=r_row.shape, dtype=np.uint8)
    for i, (value_r, value_g, value_h) in enumerate(zip(r_row, g_row, h_row)):
        f2 = -0.776 * (value_r ** 2) + 0.5601 * value_r + 0.18
        f1 = -1.376 * (value_r ** 2) + 1.0743 * value_r + 0.2
        w = ((value_r - 0.33) ** 2 + (value_g - 0.33) ** 2)

        if f1 > value_g > f2 and w > 0.001 and (value_h > 240 or value_h <= 20):
            skin_mask[i] = 1
        else:
            skin_mask[i] = 0

    return skin_mask


def hair_segmentation(r_row, g_row, b_row, h_row, i_row):
    hair_mask = np.zeros(shape=r_row.shape, dtype=np.uint8)
    for i, (value_r, value_g, value_b, value_h, value_i) in enumerate(zip(r_row, g_row, b_row, h_row, i_row)):

        if value_i < 80 and (((value_b - value_g) < 15) or ((value_b - value_r) < 15)) or (20 < value_h <= 40):
            hair_mask[i] = 1
        else:
            hair_mask[i] = 0

    return hair_mask


def hsi_hair_detection(img):
    R, B, G = cv2.split(img)

    print(R.shape, G.shape, B.shape)

    imgHSI = RGB2HSI(img)

    H = imgHSI[:, :, 0]

    I = imgHSI[:, :, 2]

    mask = np.zeros(shape=img.shape[:2])
    for j in range(R.shape[0]):
        mask[j, :] = hair_segmentation(R[j], G[j], B[j], H[j], I[j])

    hair = cv2.bitwise_and(img, img, mask=np.array(mask, dtype="uint8"))

    return hair


def hsv_skin_detection(src_image):
    image_hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    # create NumPy arrays from the boundaries
    min_hsv = np.array([0, 58, 30], dtype="uint8")
    max_hsv = np.array([33, 255, 255], dtype="uint8")

    skin_region_hsv = cv2.inRange(image_hsv, min_hsv, max_hsv)
    skin_hsv = cv2.bitwise_and(src_image, src_image, mask=skin_region_hsv)

    return skin_hsv


def hsi_skin_detection(img):
    R, G, B = cv2.split(img)

    print(R.shape, G.shape, B.shape)

    # STEP A. Skin Detection
    # Normalizing colors

    r = np.nan_to_num(R / (G + R + B))
    g = np.nan_to_num(G / (R + G + B))

    # imshow_components(labels)

    imgHSI = RGB2HSI(img)

    H = imgHSI[:, :, 0]

    mask = np.zeros(shape=img.shape[:2])
    for j, (r_row_, g_row_) in enumerate(zip(r, g)):
        mask[j, :] = skin_segmentation(r_row_, g_row_, H[j])

    skin = cv2.bitwise_and(img, img, mask=np.array(mask, dtype="uint8"))
    return np.array(skin, dtype="uint8")


def hsi_quantization(src_image, windows_shape=(5, 5)):
    quant_img = src_image.copy()
    row, col = np.indices(img.shape[:2])
    slide_row = np.lib.stride_tricks.sliding_window_view(row, windows_shape)
    slide_col = np.lib.stride_tricks.sliding_window_view(col, windows_shape)

    for srow, scol in zip(slide_row, slide_col):
        for wrow, wcol in zip(srow, scol):

            if np.sum(src_image[wrow, wcol] == 0) > 12:
                quant_img[wrow, wcol] = np.zeros(src_image[wrow, wcol].shape)

    return quant_img


def component_labeling(src_image):
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 12, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S, connectivity=8)
    (numLabels, labels, stats, centroids) = output

    component_mask = (labels == 0).astype("uint8") * 255
    # show our output image and connected component mask

    return component_mask


path_to_image = 'samples/TD_RGB_E_22.jpg'
img = cv2.imread(path_to_image, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (400, 500))

hair_pipe = [hsi_hair_detection, hsi_quantization, component_labeling]
skin_pipe = [hsi_skin_detection, hsi_quantization, component_labeling]


def find_corners(src_image):
    X, Y = np.where(src_image == 255)
    if len(X) == 0:
        min_x = 0
        max_x = 0
    else:
        min_x, max_x = min(X), max(X)
    if len(Y) == 0:
        min_y = 0
        max_y = 0
    else:
        min_y, max_y = min(X), max(Y)


    top_left = (min_x, min_y)
    bottom_right = (max_y, max_x)
    top_right = (max_x, min_y)
    bottom_left = (min_x, max_y)
    return top_left, bottom_right, top_right, bottom_left


def hsi_face_detection(src_image):
    hair = src_image.copy()
    for fun in hair_pipe:
        hair = fun(hair)

    skin = src_image.copy()
    for fun in skin_pipe:
        skin = fun(skin)

    top_left, bottom_right, top_right, bottom_left = find_corners(skin)
    print(top_left, bottom_right, top_right, bottom_left)
    skin_square = cv2.rectangle(src_image.copy(), top_left, bottom_right, (255, 0, 0), 2)

    top_left, bottom_right, top_right, bottom_left = find_corners(hair)
    print(top_left, bottom_right, top_right, bottom_left)

    hair_square = cv2.rectangle(skin_square, top_left, bottom_right, (0, 255, 0), 2)

    return hair_square


print(img.shape[:2])

quant_img = hsi_face_detection(img)

cv2.imshow("images", np.hstack([cv2.resize(quant_img, (400, 500))]))
cv2.waitKey(0)
