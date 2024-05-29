import numpy as np
import matplotlib.pyplot as plt

#이미지 리딩 및 im의 histogram 생성
im = plt.imread("lena_gray.bmp").astype(np.uint8)
im_hist, im_bins = np.histogram(im.flatten(), bins=256, range=(0,256))
imf = im.astype(np.float32)

##In = (I - Min) * (Maxn - Minn) / (Max-Min) + Minn, Min = 0, Max = 255, 노멀라이징 연산
#((img_f -img_f.min()) * (255) / (img_f.max() - img_f.min())).astype(np.uint8)
cdf_norm = (imf - imf.min()) * 255 / (imf.max() - imf.min())

#norm_im의 histogram 생성
nim_hist, nim_bins = np.histogram(cdf_norm.flatten(), bins = 256, range = (0,256))

plt.subplot(221)
plt.imshow(im, 'gray')

plt.subplot(222)
plt.plot(im_bins[:-1], im_hist)

plt.subplot(223)
# plt.imshow(norm_im, 'gray')
plt.imshow(cdf_norm, 'gray')

plt.subplot(224)
plt.plot(nim_bins[:-1], nim_hist)

plt.show()
