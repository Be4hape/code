import numpy as np
import matplotlib.pyplot as plt
import cv2

im = plt.imread("me_regray.jpg").astype(np.uint8)

#임의의 ROI 크기, height,width 50,50. 이동거리 25,25
roi_h, roi_w = 60, 60  
move_y, move_x = 60, 60

#im, height, width
img_h, img_w = im.shape


#sliding window, 각 ROI의 히스토그램 계산
roi_hist = []
for y in range(0, img_h - roi_h + 1, move_y):
    for x in range(0, img_w - roi_w + 1, move_x):
        roi = im[y:y + roi_h, x:x + roi_w]
        hist = np.histogram(roi.flatten(), bins=256, range=(0, 256))[0]
        roi_hist.append(hist)

#10개의 hist
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.plot(roi_hist[i])
    plt.xlim([0, 256])

plt.tight_layout()
plt.show()
