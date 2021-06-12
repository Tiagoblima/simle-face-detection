import os
import uuid

import cv2
import numpy as np

from detection import hsi_skin_detection
from face_detection import generate_component_mask, hsv_skin_detection, quantization, create_rectangle

ROOT_DIR = 'dataset/'

img = cv2.imread(os.path.join(ROOT_DIR, '1/TD_RGB_E_1.jpg'), cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (500, 400))

hsi_pipe = [hsi_skin_detection, quantization, generate_component_mask]
hsv_pipe = [hsv_skin_detection, quantization, generate_component_mask]

result_mask = img.copy()
for fun in hsi_pipe:
    result_mask = fun(result_mask)

detected_image = create_rectangle(img, result_mask)
OUT_DIR = 'detection_out'
cv2.imwrite(f"{OUT_DIR}/detected_{str(uuid.uuid1())[:4]}.jpg", detected_image)
cv2.imshow("images", np.hstack([img, detected_image]))
cv2.waitKey(0)
cv2.destroyAllWindows()
