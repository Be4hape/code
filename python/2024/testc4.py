import numpy as np
import matplotlib.pyplot as plt

#이미지 리딩 및 im의 histogram 생성
im = plt.imread("lena_gray.bmp").astype(np.uint8)
im_hist, im_bins = np.histogram(im.flatten(), bins=256, range=(0,256))


#256빈공간 생성, 값을 누적시킬 csum, for 256까지 cdf에 더한 값 대입
cdf = np.zeros(256)
csum = 0
ttl_pixels = im.size

for i in range(256):
    csum += im_hist[i]
    cdf[i] = csum

 #In = (I - Min) * (Maxn - Minn) / (Max-Min) + Minn, Min = 0, Max = 255, 노멀라이징 연산
cdf_norm = cdf * 255 / ttl_pixels


#빈 im크기의 norm_im 생성, cdf에 누적한 값 cdf_norm값을 norm_im에 대입
norm_im = np.zeros_like(im)
for i in range(256):
    #boolean indexing, im==i 가 True일 때, cdf_norm값을 norm_im에 대입
    norm_im[im == i] = cdf_norm[i]


#norm_im의 histogram 생성
nim_hist, nim_bins = np.histogram(norm_im.flatten(), bins = 256, range = (0,256))

plt.subplot(221)
plt.imshow(im, 'gray')

plt.subplot(222)
plt.plot(im_bins[:-1], im_hist)

plt.subplot(223)
plt.imshow(norm_im, 'gray')

plt.subplot(224)
plt.plot(nim_bins[:-1], nim_hist)

plt.show()
