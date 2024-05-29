import numpy as np
import cv2
import matplotlib.pyplot as plt

#image 가져오기, hist,bins 선언
im = plt.imread("lena_gray.bmp").astype(np.uint8)
hist, bins = np.histogram(im.flatten(), bins = 256, range = [0,256])

rows, cols = im.shape[:2]
hist = cv2.calcHist([im], [0], None, [256], [0, 256])

img3 = cv2.equalizeHist(im)
hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])


cv2.imshow('Before', im)
cv2.imshow('cv2.equalizeHist()', img3)
hists = {'Before':hist,'cv2.equalizeHist()':hist3}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
    plt.show()
