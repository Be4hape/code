import numpy as np
import matplotlib.pyplot as plt

def threshold_otsu(im):
    # 히스토그램 계산
    hist, bins = np.histogram(im, bins=256, range=[0, 256])
    bins = np.delete(bins, -1)

    # 전체 픽셀 수 계산
    total_pixels = im.size

    # 초기값 설정
    max_var = 0
    optimal_threshold = 0

    # 초기 변수 설정
    pi = hist / np.sum(hist)
    mg = np.sum(bins * pi)
    nk_old = 0
    T_new = 0

    # 0부터 255까지 임계값을 시도하여 최적의 임계값 찾기
    for T in range(1, 256):
        p1 = np.sum(pi[:T])
        mk = np.sum(bins[:T] * pi[:T])

        if p1 == 0 or p1 == 1:
            continue

        ob = (mg * p1 - mk)**2 / (p1 * (1 - p1))
        og = np.sum((bins - mg)**2 * pi)

        nk = ob / og

        if (nk_old < nk and not np.isinf(nk)):
            T_new = T
            nk_old = nk

    optimal_threshold = T_new
    print(T_new)

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
    if im.ndim == 3:
        im = np.mean(im, axis=2)  # 이미지를 그레이스케일로 변환

    # 원본 이미지
    axes[i, 0].imshow(im, cmap='gray')
    axes[i, 0].set_title("Original Image")

    # 원본 이미지 히스토그램
    axes[i, 1].hist(im.flatten(), bins=256, range=[0, 256])
    axes[i, 1].set_xlim(0, 256)
    axes[i, 1].set_title("Histogram")

    # 이진화 적용
    im_th = (im > threshold_otsu(im)).astype(np.uint8)  # 이진화된 이미지 생성

    # 이진화된 이미지
    axes[i, 2].imshow(im_th, cmap='gray')
    axes[i, 2].set_title("Binarized Image")

    # 이진화된 이미지의 히스토그램
    axes[i, 3].hist(im_th.flatten(), bins=256, range=[0, 256])
    axes[i, 3].set_xlim(-1, 2)  # 0과 1로 이진화 > -1은 0인 값들을 보기 위함
    axes[i, 3].set_title("Binarized Histogram")

plt.tight_layout()
plt.show()
