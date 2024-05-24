import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 및 데이터 타입 변환
im = plt.imread("lena_gray.bmp")
if im.ndim == 3:  # 이미지가 컬러라면 그레이스케일로 변환
    im = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])
im = im.astype(np.uint8)

# 원본 이미지의 히스토그램 계산
im_hist, im_bins = np.histogram(im.flatten(), bins=256, range=[0, 256])

# 히스토그램 평활화 수행
cdf = im_hist.cumsum()  # 누적 분포 함수 계산
cdf_normalized = cdf * 255 / cdf[-1]  # CDF 정규화
equalized_im = np.interp(im.flatten(), im_bins[:-1], cdf_normalized)  # 평활화된 값으로 매핑
equalized_im = equalized_im.reshape(im.shape).astype(np.uint8)

# 평활화된 이미지의 히스토그램 계산
eq_hist, eq_bins = np.histogram(equalized_im.flatten(), bins=256, range=[0, 256])

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
plt.imshow(equalized_im, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(224)
plt.plot(eq_bins[:-1], eq_hist, color='black')
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
