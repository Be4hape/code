import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 및 그레이스케일로 변환
im = plt.imread("lena_gray.bmp").astype(np.uint8)

# 원본 이미지의 히스토그램 계산
im_hist, im_bins = np.histogram(im.flatten(), bins=256, range=[0, 256])

# 누적 분포 함수(CDF) 계산
cdf = np.zeros(256)
csum = 0
ttl_pixels = im.size

for i in range(256):
    csum += im_hist[i]
    cdf[i] = csum

cdf_norm = cdf * 255 / ttl_pixels

# 새로운 이미지 생성 및 맵핑
equal_im = np.interp(im.flatten(), np.arange(256), cdf_norm).reshape(im.shape).astype(np.uint8)

# 균등화된 이미지의 히스토그램 계산
nim_hist, nim_bins = np.histogram(equal_im.flatten(), bins=256, range=[0, 256])

# 결과 이미지와 히스토그램을 하나의 Figure에 출력
plt.figure(figsize=(14, 7))

plt.subplot(221)
plt.imshow(im, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(222)
plt.plot(im_bins[:-1], im_hist, color='black')
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(223)
plt.imshow(equal_im, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(224)
plt.plot(nim_bins[:-1], nim_hist, color='black')
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
