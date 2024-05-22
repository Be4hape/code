import numpy as np
import matplotlib.pyplot as plt
import  cv2

c = 30

im = plt.imread("lena_gray.bmp")
y = c * np.log(im + 1)

plt.subplot(1,2,1)
plt.imshow(y, cmap = 'gray')

plt.subplot(1,2,2)
plt.plot(im, y)
plt.xlim(0, 255)
plt.ylim(0,255)

plt.show()
