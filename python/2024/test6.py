import numpy as np
import matplotlib.pyplot as plt

# 오츠 메소드 함수 정의
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
    
    return optimal_threshold, im_th

# AA1-3 그림 파일 읽어들이기
im = plt.imread("AA1_3.jpg")

# 이미지를 numpy 배열로 변환하여 저장
im_array = np.array(im)

# 이미지를 확인하기 위해 플로팅
plt.imshow(im_array, cmap='gray')
plt.axis('off')
plt.show()

# 이미지를 흑백 이미지로 변환
im_gray = np.mean(im_array, axis=(0, 1))  # 각 픽셀의 RGB 값의 평균을 계산하여 흑백 이미지 생성


# 이미지를 1차원으로 펼쳐서 픽셀 값을 데이터로 사용
pixels = im_gray.flatten()

# 픽셀의 수
n_pixels = len(pixels)

# x 값 생성 (픽셀의 인덱스)
x_values = np.arange(n_pixels)

# 최소 자승법을 사용하여 각 픽셀 위치에서의 조명의 변화를 추정
A = np.vstack([x_values, np.ones(n_pixels)]).T
m, c = np.linalg.lstsq(A, pixels, rcond=None)[0]

# 추정된 조명의 변화
illumination_change = m * x_values + c

# 조명 보정된 이미지 생성
illumination_corrected_image = im_gray - illumination_change.reshape(im_gray.shape)

# 결과 출력
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(im_array, cmap='gray')  # 원본 이미지 출력
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(illumination_corrected_image, cmap='gray')  # 조명 보정된 이미지 출력
axes[1].set_title('Illumination Corrected Image')
axes[1].axis('off')

# Otzu를 적용하여 이진화
threshold = threshold_otsu(illumination_corrected_image)[0]  # threshold_otsu 함수에서 첫 번째 반환값은 임계값
binary_image = (illumination_corrected_image > threshold).astype(np.uint8)

axes[2].imshow(binary_image, cmap='gray')  # 이진화된 이미지 출력
axes[2].set_title('Binary Image after Otsu')
axes[2].axis('off')

plt.show()
