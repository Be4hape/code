import matplotlib.pyplot as plt
import numpy as np

# Otsu's 이진화 함수 정의
def otsu_thresholding(image):
    # 이미지 히스토그램 계산
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    bins_centers = (bins[:-1] + bins[1:]) / 2

    # 전체 픽셀 수
    total_pixels = image.shape[0] * image.shape[1]

    # 클래스 간 분산의 초기값 설정
    initial_var = 0

    # 초기 임계값 설정
    threshold = 0

    # 각 픽셀 값을 임계값으로 사용하여 클래스 간 분산을 계산하고 최적의 임계값 찾기
    for t in bins_centers:
        # 클래스 1
        class1_hist = hist[:int(t)]
        class1_weights = np.sum(class1_hist) / total_pixels
        class1_mean = np.sum(class1_hist * bins_centers[:int(t)]) / (np.sum(class1_hist) + 1e-10)

        # 클래스 2
        class2_hist = hist[int(t):]
        class2_weights = np.sum(class2_hist) / total_pixels
        class2_mean = np.sum(class2_hist * bins_centers[int(t):]) / (np.sum(class2_hist) + 1e-10)

        # 클래스 간 분산 계산
        var_between = class1_weights * class2_weights * (class1_mean - class2_mean) ** 2

        # 최적의 임계값 업데이트
        if var_between > initial_var:
            initial_var = var_between
            threshold = t

    # Otsu의 이진화 적용
    binary_image = (image > threshold).astype(np.uint8) * 255

    return binary_image, hist, bins

# 이미지 파일 이름들
image_files = ["AA1_1.jpg", "AA1_2.jpg", "AA1_3.jpg"]

# 결과를 저장할 리스트 생성
result_images = []
result_histograms = []
result_binaries = []

# 각 이미지에 대해 Otsu의 이진화 적용
for filename in image_files:
    # 이미지 불러오기
    image = plt.imread(filename)

    # Otsu의 이진화 적용
    binary_image, binary_hist, binary_bins = otsu_thresholding(image)

    # 결과 저장
    result_images.append(image)
    result_histograms.append((np.histogram(image.flatten(), bins=256, range=(0, 256))))
    result_binaries.append((binary_image, binary_hist, binary_bins))

# 결과를 3x4 행렬에 출력
plt.figure(figsize=(12, 9))

for i in range(len(image_files)):
    # 원본 이미지 출력
    plt.subplot(3, 4, i*4+1)
    plt.imshow(result_images[i], cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # 히스토그램 출력
    plt.subplot(3, 4, i*4+2)
    plt.bar(result_histograms[i][1][:-1], result_histograms[i][0], width=1.0, color='gray')
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    # 이진화된 이미지 출력
    plt.subplot(3, 4, i*4+3)
    plt.imshow(result_binaries[i][0], cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')

    # 이진화된 이미지의 히스토그램 출력
    plt.subplot(3, 4, i*4+4)
    plt.bar(result_binaries[i][2][:-1], result_binaries[i][1], width=1.0, color='gray')
    plt.title("Binary Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
