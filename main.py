import argparse
import os
import uuid

import cv2
import numpy as np

from detection import hsi_skin_detection
from face_detection import generate_component_mask, hsv_skin_detection, quantization, create_rectangle

ROOT_DIR = 'dataset/'

parser = argparse.ArgumentParser(description='Process some image.')

parser.add_argument('--path_to_image', metavar='N', type=str,
                    help='The path to the image you want the face detected', required=True)

parser.add_argument('--mode', metavar='N', type=str,
                    help='The mode that the api must use. It must be hsi or hsv image skin detection.', required=True)

parser.add_argument('--show_steps', action='store_true',
                    help='Shows the detection step by step.', required=False)

args = parser.parse_args()
img = cv2.imread(args.path_to_image)

img = cv2.resize(img, (500, 400))

if args.mode.lower() == 'hsv':
    pipe = [('hsv_skin_detection', hsv_skin_detection)]
elif args.mode.lower() == 'hsi':
    pipe = [('hsi_skin_detection', hsi_skin_detection)]
else:
    raise ValueError("Mode must be hsv or hsi")

pipe.extend([('quantization', quantization), ('component_mask', generate_component_mask)])
result_mask = img.copy()
for label, fun in pipe:
    result_mask = fun(result_mask)
    if args.show_steps:
        cv2.imshow(label, np.hstack([result_mask]))
        cv2.waitKey(0)

detected_image = create_rectangle(img, result_mask)
OUT_DIR = 'detection_out'
image_name = os.path.split(args.path_to_image)[-1].split('.')[0]
cv2.imwrite(f"{OUT_DIR}/{image_name}_detected.jpg", detected_image)
cv2.imshow("images", np.hstack([img, detected_image]))
cv2.waitKey(0)
cv2.destroyAllWindows()
