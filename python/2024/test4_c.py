import numpy as np
import matplotlib.pyplot as plt

def threshold_otsu(im):
    # 히스토그램 계산
    hist, bins = np.histogram(im, bins=256, range=[0, 256])

    # 전체 픽셀 수 계산
    total_pixels = im.size

    # 초기값 설정
    max_var = 0
    optimal_threshold = 0

    # 0부터 255까지 임계값을 시도하여 최적의 임계값 찾기
    for threshold in range(256):
        # 클래스 1과 클래스 2의 픽셀 수 계산
        n1 = np.sum(hist[:threshold])
        n2 = np.sum(hist[threshold:])

        # 클래스 1과 클래스 2의 평균 계산
        m1 = np.sum(np.arange(threshold) * hist[:threshold]) / n1 if n1 > 0 else 0
        m2 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / n2 if n2 > 0 else 0

        # 클래스 간 분산 계산
        var = n1 * n2 * ((m1 - m2) ** 2)

        # 최대 분산을 가지는 임계값 찾기
        if var > max_var:
            max_var = var
            optimal_threshold = threshold

    return optimal_threshold

# 이미지 파일 리스트
image_files = ["AA1_1.jpg", "AA1_2.jpg", "AA1_3.jpg"]

# 이미지 개수 구하기
num_im = len(image_files)

# 이미지 개수, 4개의 공간(원본, 히스토그램, 이진화 결과, 히스토그램 순)
fig, axes = plt.subplots(num_im, 4)

# 이미지 개수만큼 돌아가는 for문
for i in range(num_im):
    # 각각의 이미지 파일 불러오기
    imf = image_files[i]
    # 이미지 배열로 변환
    im = plt.imread(imf)

    # 원본 이미지
    axes[i, 0].imshow(im, cmap='gray')

    # 원본 이미지 히스토그램
    axes[i, 1].hist(im.flatten(), bins=256)
    axes[i, 1].set_xlim(0, 256)

    # 이진화 적용
    im_th = (im > threshold_otsu(im)).astype(np.uint8)  # 이진화된 이미지 생성

    # 이진화된 이미지
    axes[i, 2].imshow(im_th, cmap='gray')

    # 이진화된 이미지의 히스토그램
    axes[i, 3].hist(im_th.flatten(), bins=256)
    axes[i, 3].set_xlim(-1, 2)  #0과 1로 이진화 > -1은 0인 값들을 보기 위함

plt.tight_layout()
plt.show()
