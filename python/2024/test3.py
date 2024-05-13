import matplotlib.pyplot as plt
import numpy as np

# Global Thresholding 함수 정의
def global_thresholding(image):
    # 이미지 히스토그램 계산
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    
    # 히스토그램을 기반으로 전역 임계값 설정
    threshold = 130  # 임계값은 임의로 설정됨, 실제로는 적절한 값으로 설정해야 함
    
    # 임계값을 기준으로 이진화 처리
    binary_image = (image > threshold).astype(np.uint8) * 255
    
    return binary_image, hist, bins

# AA1-1 이미지 불러오기
image1 = plt.imread("AA1_1.jpg")
# AA1-2 이미지 불러오기
image2 = plt.imread("AA1_2.jpg")
# AA1-3 이미지 불러오기
image3 = plt.imread("AA1_3.jpg")

# 이미지들을 하나의 배열로 합치기
images = np.stack([image1, image2, image3], axis=0)

# 이미지와 이진화된 이미지, 히스토그램을 저장할 리스트 생성
result_images = []
result_binaries = []
result_histograms = []

# 이미지들에 대해 Global Thresholding을 수행하고 결과를 리스트에 저장
for image in images:
    binary_image, hist, bins = global_thresholding(image)
    result_images.append(image)
    result_binaries.append(binary_image)
    result_histograms.append((hist, bins))

# 결과를 3x4 행렬에 출력
plt.figure(figsize=(12, 9))

for i in range(3):
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
    plt.imshow(result_binaries[i], cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    
    # 이진화된 이미지의 히스토그램 출력
    plt.subplot(3, 4, i*4+4)
    binary_hist, binary_bins = np.histogram(result_binaries[i].flatten(), bins=256, range=(0, 256))
    plt.bar(binary_bins[:-1], binary_hist, width=1.0, color='gray')
    plt.title("Binary Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
