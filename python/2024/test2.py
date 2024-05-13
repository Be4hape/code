import matplotlib.pyplot as plt
import numpy as np

# AA1-1 이미지 불러오기
image1 = plt.imread("AA1_1.jpg")
# AA1-2 이미지 불러오기
image2 = plt.imread("AA1_2.jpg")
# AA1-3 이미지 불러오기
image3 = plt.imread("AA1_3.jpg")

# 각 영상의 히스토그램 계산
hist1, bins1 = np.histogram(image1.flatten(), bins=256, range=(0, 256))
hist2, bins2 = np.histogram(image2.flatten(), bins=256, range=(0, 256))
hist3, bins3 = np.histogram(image3.flatten(), bins=256, range=(0, 256))

# 히스토그램 시각화
plt.figure(figsize=(10, 10))

# AA1-1 이미지와 히스토그램 subplot
plt.subplot(3, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title("AA1-1 Image")
plt.axis('off')

if len(hist1) > 0:
    plt.subplot(3, 2, 2)
    plt.bar(bins1[:-1], hist1, width=1.0, color='gray')
    plt.title("AA1-1 Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

# AA1-2 이미지와 히스토그램 subplot
plt.subplot(3, 2, 3)
plt.imshow(image2, cmap='gray')
plt.title("AA1-2 Image")
plt.axis('off')

if len(hist2) > 0:
    plt.subplot(3, 2, 4)
    plt.bar(bins2[:-1], hist2, width=1.0, color='gray')
    plt.title("AA1-2 Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

# AA1-3 이미지와 히스토그램 subplot
plt.subplot(3, 2, 5)
plt.imshow(image3, cmap='gray')
plt.title("AA1-3 Image")
plt.axis('off')

if len(hist3) > 0:
    plt.subplot(3, 2, 6)
    plt.bar(bins3[:-1], hist3, width=1.0, color='gray')
    plt.title("AA1-3 Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
