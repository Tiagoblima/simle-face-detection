from util import RGB2HSI
import numpy as np
import cv2


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
        w = ((value_r - 0.33) ** 2 + (value_g - 0.33) ** 2) > 0.001
        f1 = -1.376 * (value_r ** 2) + 1.074 * value_r + 0.2
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


def hair_detection(img):
    B, G, R = cv2.split(img)

    print(R.shape, G.shape, B.shape)

    imgHSI = RGB2HSI(img)

    H = imgHSI[:, :, 0]

    I = imgHSI[:, :, 2]

    mask = np.zeros(shape=img.shape[:2])
    for j in range(R.shape[0]):
        mask[j, :] = hair_segmentation(R[j], G[j], B[j], H[j], I[j])

    hair = cv2.bitwise_and(img, img, mask=np.array(mask, dtype="uint8"))


    return hair


def skin_detection(img):
    B, G, R = cv2.split(img)

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
    return skin
