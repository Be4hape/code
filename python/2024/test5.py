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

    # 이진화된 이미지 생성
    im_th = (im > optimal_threshold).astype(np.uint8)
    
    return im_th


image_files = ["AA1_1.jpg", "AA1_2.jpg", "AA1_3.jpg"]
num_im = len(image_files)
fig, axes = plt.subplots(num_im, 8, figsize=(16, 6))  # 그림 크기 조정

for i, image_file in enumerate(image_files):
    # 이미지 읽기
    im = plt.imread(image_file)
    nim = np.copy(im)

    # 이미지 영역 설정 및 히스토그램, 이진화 처리
    regions = [
        nim[:100, :66], nim[:100, 66:133], nim[:100, 133:],
        nim[100:, :66], nim[100:, 66:133], nim[100:, 133:]
    ]
    imths = [threshold_otsu(region) for region in regions]

    # 이미지 영역과 히스토그램 표시
    for j, (region, imth) in enumerate(zip(regions, imths)):
        axes[i, j+1].imshow(region, cmap='gray')
        axes[i, j+1].axis('off')
        axes[i, j+7].imshow(imth, cmap='gray')
        axes[i, j+7].axis('off')

    # 분할 라인 표시
    if i == 0:
        nim[100,:] = 0
        nim[:, 66] = 0
        nim[:, 133] = 0
    elif i == 1:
        nim[70,:] = 0
        nim[:, 55] = 0
        nim[:, 133] = 0
    elif i == 2:
        nim[60,:] = 0
        nim[:, 60] = 0
        nim[:, 150] = 0
    axes[i, 0].imshow(nim, cmap='gray')
    axes[i, 0].axis('off')

plt.tight_layout()
plt.show()
