import numpy as np
import matplotlib.pyplot as plt
import cv2

im = plt.imread("lena_gray.bmp")

im = 255 - im
y = 255 - im

plt.subplot(1,2,1)
plt.imshow(im, cmap = 'gray')

plt.subplot(1,2,2)
plt.plot(im, y)
plt.xlim(0,255)
plt.ylim(0,255)

plt.show()
