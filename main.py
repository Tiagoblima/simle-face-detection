import numpy as np
import cv2
import os

from sklearn.cluster import MiniBatchKMeans

from detection import skin_detection
from util import RGB2HSI

ROOT_DIR = 'dataset/'

img = cv2.imread(os.path.join(ROOT_DIR, 'test.png'), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (500, 400))
skin = skin_detection(img)
# skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
(h, w) = skin.shape[:2]
# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning
image = cv2.cvtColor(skin, cv2.COLOR_BGR2LAB)
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
# display the images and wait for a keypress
cv2.imwrite('img1.png', np.hstack([img, skin]))


cv2.waitKey(0)
# cv2.imshow("images", np.hstack([img, skinHSV]))
# cv2.waitKey(0)
# cv2.waitKey(0)
cv2.destroyAllWindows()
