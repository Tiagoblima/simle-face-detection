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
    (B, G, R) = cv2.split(img)

    print(R.shape, G.shape, B.shape)

    imgHSI = RGB2HSI(img.copy())

    H = imgHSI[:, :, 0]

    I = imgHSI[:, :, 2]

    mask = np.zeros(shape=img.shape[:2])
    for j in range(R.shape[0]):
        mask[j, :] = hair_segmentation(R[j], G[j], B[j], H[j], I[j])

    hair = cv2.bitwise_and(img, img, mask=np.array(mask, dtype="uint8"))

    return hair


def hsv_skin_detection(src_image):
    src_image = src_image.copy()
    image_hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    # create NumPy arrays from the boundaries
    min_hsv = np.array([0, 58, 30], dtype="uint8")
    max_hsv = np.array([33, 255, 255], dtype="uint8")

    skin_region_hsv = cv2.inRange(image_hsv, min_hsv, max_hsv)
    skin_hsv = cv2.bitwise_and(src_image, src_image, mask=skin_region_hsv)

    return skin_hsv


def hsi_skin_detection(img):
    (B, G, R) = cv2.split(img)

    # print(R.shape, G.shape, B.shape)

    # STEP A. Skin Detection
    # Normalizing colors

    r = np.nan_to_num(R / (R + G + B))
    g = np.nan_to_num(G / (R + G + B))

    # imshow_components(labels)

    imgHSI = RGB2HSI(img.copy())

    H = imgHSI[:, :, 0]

    mask = np.zeros(shape=img.shape[:2])
    for j, (r_row_, g_row_) in enumerate(zip(r, g)):
        mask[j, :] = skin_segmentation(r_row_, g_row_, H[j])

    skin = cv2.bitwise_and(img, img, mask=np.array(mask, dtype="uint8"))
    return np.array(skin, dtype="uint8")


def quantization(src_image, windows_shape=(5, 5)):
    quant_img = src_image.copy()
    quant_img = kmeans_quantization(quant_img)

    row, col = np.indices(src_image.shape[:2])
    slide_row = np.lib.stride_tricks.sliding_window_view(row, windows_shape)
    slide_col = np.lib.stride_tricks.sliding_window_view(col, windows_shape)

    for srow, scol in zip(slide_row, slide_col):
        for wrow, wcol in zip(srow, scol):

            if np.sum(src_image[wrow, wcol] == 0) > 12:
                quant_img[wrow, wcol] = np.zeros(src_image[wrow, wcol].shape)

    return quant_img


def component_labeling(src_image):
    gray = cv2.cvtColor(src_image.copy(), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S, connectivity=8)
    (numLabels, labels, stats, centroids) = output

    component_mask = (labels == 0).astype("uint8") * 255
    # show our output image and connected component mask

    return component_mask


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


def hsi_hair_preprocess(src_image):
    hair_pipe = [hsi_hair_detection, quantization, component_labeling]
    hair_ = src_image.copy()
    for fun in hair_pipe:
        hair_ = fun(hair_)
    return hair_


def hsi_skin_preprocess(src_image):
    skin_pipe = [hsi_skin_detection, quantization, component_labeling]

    skin = src_image.copy()
    for fun in skin_pipe:
        skin = fun(skin)

    return skin


def hsv_preprocess(src_image):
    skin_pipe = [hsv_skin_detection, quantization, component_labeling]

    skin = src_image.copy()
    for fun in skin_pipe:
        skin = fun(skin)

    return skin


def draw_square(src_image, mask_image, color=(255, 0, 0), label="square", org=(50, 50)):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1

    # Line thickness of 2 px
    thickness = 1

    top_left, bottom_right, top_right, bottom_left = find_corners(mask_image)
    print(top_left, bottom_right, top_right, bottom_left)
    skin_square = cv2.rectangle(src_image.copy(), top_left, bottom_right, color, 2)

    if top_left != (0, 0):
        image = cv2.putText(skin_square, label, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
    else:
        image = skin_square

    return image


def hsi_face_detection(src_image):
    skin = hsi_skin_preprocess(src_image.copy())
    hair = hsi_hair_preprocess(src_image.copy())

    # print(top_left, bottom_right, top_right, bottom_left)
    image = draw_square(src_image, skin, label="hsi skin square")
    image = draw_square(image, hair, label="hsi hair square", color=(0, 255, 0), org=(20, 20))
    return image


def hsv_face_detection(src_image):
    skin = hsv_preprocess(src_image.copy())

    # print(top_left, bottom_right, top_right, bottom_left)
    image = draw_square(src_image, skin, label="hsv skin square")

    return image


def compare_detection(src_image):
    skin_hsi = hsi_skin_preprocess(src_image)
    skin_hsv = hsv_preprocess(src_image)

    hsi_square = draw_square(src_image, skin_hsi, color=(255, 0, 0), label="hsi skin preprocess", org=(50, 50))
    hsv_square = draw_square(hsi_square, skin_hsv, color=(0, 255, 0), label="hsv skin preprocess", org=(20, 20))
    return hsv_square


src_ = cv2.imread('samples/TD_RGB_E_10.jpg')
src_ = cv2.resize(src_, (400, 500))

hsi_skin_pipe = [(hsi_skin_detection, 'hsi_skin_detection'),
                 (quantization, 'hsi_quantization'),
                 (component_labeling, 'component_labeling')]

hsi_hair_pipe = [(hsi_hair_detection, 'hsi_hair_detection'),
                 (quantization, 'hsi_quantization'),
                 (component_labeling, 'component_labeling')]

hsv_skin_pipe = [(hsv_skin_detection, 'hsv_skin_detection'),
                 (quantization, 'hsi_quantization'),
                 (component_labeling, 'component_labeling')]

# hsi = hsi_hair_detection(src_)
# # hsv = hsv_skin_detection(src_)
#
# # hsi = quantization(hsi)
# # hsv = hsi_quantization(hsv)
#
# # hsi = component_labeling(hsi)
# # hsv = component_labeling(hsv)
# cv2.imwrite('case2/hair/hair_detection_hsv-hsi.png', np.hstack([hsi]))
