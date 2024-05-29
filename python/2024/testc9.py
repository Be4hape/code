import numpy as np
import matplotlib.pyplot as plt
import cv2

im = plt.imread("me_regray.jpg").astype(np.uint8)

#임의의 ROI 크기, height,width 50,50. 이동거리 25,25
roi_h, roi_w = 50, 50  # 예시로 50x50 크기
move_y, move_x = 25, 25     # 예시로 25 픽셀씩 이동

#im, height, width
img_h, img_w = im.shape

#sliding window, 각 ROI의 히스토그램 계산

roi_hist = []
for y in range(0, img_h - roi_h + 1, move_y):
    for x in range(0, img_w - roi_w + 1, move_x):
        roi = im[y:y + roi_h, x:x + roi_w]
        hist = np.histogram(roi.flatten(), bins=256, range=(0, 256))[0]
        roi_hist.append(hist)

#Euclidean distance matrix 계산
num_rois = len(roi_hist)
euclid = np.zeros((num_rois, num_rois))

for i in range(num_rois):
    for j in range(num_rois):
        euclid[i, j] = np.linalg.norm(roi_hist[i] - roi_hist[j])

#OpenCV compareHist()를 사용한 히스토그램 비교
cv_matrix = np.zeros((num_rois, num_rois))
for i in range(num_rois):
    for j in range(num_rois):
        cv_matrix[i, j] = cv2.compareHist(roi_hist[i].astype(np.float32), roi_hist[j].astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)

plt.subplot(1, 2, 1)
plt.imshow(euclid, interpolation='nearest')

plt.subplot(1, 2, 2)
plt.imshow(cv_matrix, interpolation='nearest')

plt.tight_layout()
plt.show()
