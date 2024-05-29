import numpy as np
import cv2
import matplotlib.pyplot as plt

#image 가져오기, hist,bins 선언
im = plt.imread("lena_gray.bmp").astype(np.uint8)
hist, bins = np.histogram(im.flatten(), bins = 256, range = [0,256])

#노멀라이징, cumsum > 누적합
cdf = hist.cumsum()
cdf_norm = cdf * float(hist.max()) / cdf.max()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)

#equal 선언, histogram을 생성할 때, 변수 2개 설정 유의. ,_
im_equal = cdf[im]
im_equal_hist, _ = np.histogram(im_equal.flatten(), bins = 256, range = (0,256))

plt.subplot(221)
plt.imshow(im, cmap='gray')

plt.subplot(222)
plt.plot(hist)

plt.subplot(223)
plt.imshow(im_equal, cmap='gray')

plt.subplot(224)
plt.plot(im_equal_hist)

plt.tight_layout()
plt.show()
