import numpy as np
import cv2
import matplotlib.pylab as plt

img = cv2.imread('lena_gray.bmp', cv2.IMREAD_GRAYSCALE)

img_f = img.astype(np.float32)
# img_norm = ((img_f -img_f.min()) * (255) / (img_f.max() - img_f.min())).astype(np.uint8)

img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 256])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 256])

cv2.imshow('Before', img)
# cv2.imshow('Manual', img_norm)
cv2.imshow('cv2.normalize()', img_norm2)

hists = {'Before' : hist, 'cv2.normalize()':hist_norm2}

for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()
