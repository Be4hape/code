import numpy as np
import matplotlib.pyplot as plt

def otsu_threshold(im):
    # 이미지 히스토그램 계산
    hist, bins = np.histogram(im.flatten(), bins=256)

    # 픽셀 수 계산
    total_pixels = im.shape[0] * im.shape[1]

    # 임의의 임계값 T 설정
    T = 100

    # 클래스 간 분산 계산
    weight_background = np.cumsum(hist)
    weight_foreground = total_pixels - weight_background
    mean_background = np.cumsum(hist * bins[:-1]) / np.maximum(weight_background, 1)  # 가중치가 0인 경우에는 1로 대체
    mean_foreground = (np.sum(hist * bins[:-1]) - mean_background * weight_background) / np.maximum(weight_foreground, 1)  # 가중치가 0인 경우에는 1로 대체
    variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

    # 최적의 임계값 결정
    threshold_idx = np.argmax(variance)
    threshold = bins[threshold_idx]

    # 임의의 T값을 오츠 메소드를 사용하여 최적의 T값으로 대체
    if T != threshold:
        T = threshold

    # 이진화된 이미지 생성
    result = np.zeros_like(im)
    result[im > T] = 1

    return result

# 이미지 파일 리스트
image_files = ["AA1_1.jpg", "AA1_2.jpg", "AA1_3.jpg"]

# 이미지 개수 구하기
num_im = len(image_files)

# figure 창에서 출력할 이미지 개수 지정
# 이미지 개수만큼, 4개의 공간(원본, 히스토그램, 이진화 결과, 히스토그램 순)
fig, axes = plt.subplots(num_im, 4)

# 이미지 개수만큼 돌아가는 for문
for i in range(num_im):
    # 각각의 이미지 파일 불러오기
    imf = image_files[i]
    # 이미지 배열로 변환
    im = plt.imread(imf)

    # 원본 이미지 출력
    axes[i, 0].imshow(im, cmap='gray')
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')

    # 원본 이미지의 히스토그램 출력
    axes[i, 1].hist(im.flatten(), bins=256, color='gray')
    axes[i, 1].set_title('Histogram')
    axes[i, 1].set_xlim(0, 255)

    # Otsu의 이진화를 적용한 이미지 생성
    im_th = otsu_threshold(im)

    # 이진화된 이미지 출력
    axes[i, 2].imshow(im_th, cmap='gray')
    axes[i, 2].set_title('Otsu Thresholded Image')
    axes[i, 2].axis('off')

    # 이진화된 이미지의 히스토그램 출력
    axes[i, 3].hist(im_th.flatten(), bins=256, color='black')
    axes[i, 3].set_title('Otsu Thresholded Histogram')
    axes[i, 3].set_xlim(0, 1)  # 픽셀 값이 0 또는 1로 정규화되었으므로 범위를 0과 1로 지정

plt.tight_layout()
plt.show()
