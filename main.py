import argparse
import os

import cv2
import numpy as np

from detection import compare_detection, hsv_face_detection, hsi_face_detection

# from face_detection import generate_component_mask, hsv_skin_detection, quantization, create_rectangle

ROOT_DIR = 'dataset/'

parser = argparse.ArgumentParser(description='Process some image.')

parser.add_argument('--path_to_image', metavar='N', type=str,
                    help='The path to the image you want the face detected', required=True)

parser.add_argument('--mode', metavar='N', type=str,
                    help='The mode that the api must use. It must be hsi or hsv image skin detection.',
                    required=True)

args = parser.parse_args()
img = cv2.imread(args.path_to_image)

img = cv2.resize(img, (500, 400))

if args.mode.lower() == 'hsv':

    final_image = hsv_face_detection(img)
elif args.mode.lower() == 'hsi':
    final_image = hsi_face_detection(img)
else:
    final_image = compare_detection(img)
name_out = args.path_to_image.split('/')[-1].split('.')[0] + '_pro_' + args.mode + '.png'
root_save = os.path.join('detection_out', name_out)
cv2.imshow("images", np.hstack([cv2.resize(final_image, (400, 500))]))
cv2.imwrite(root_save, np.hstack([cv2.resize(final_image, (400, 500))]))
cv2.waitKey(0)

cv2.destroyAllWindows()
