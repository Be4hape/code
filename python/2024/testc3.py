import numpy as np
import matplotlib.pyplot as plt

k1 = 80
k2 = 180

im = plt.imread("lena_gray.bmp")
im = im.astype(np.uint8)

# LUT 생성
T = np.arange(256)
T[k1:k2+1] = 255

# LUT 적용
J = T[im]

plt.subplot(121)
plt.imshow(J, 'gray')

plt.subplot(122)
plt.plot(np.arange(256), T)
plt.show()
