import numpy as np
import matplotlib.pyplot as plt

def threshold_otsu(im):
    #히스토그램 계산
    hist, bins = np.histogram(im, bins=256, range=[0, 256])

    #전체 픽셀 수 계산
    total_pixels = im.size

    #초기값 설정
    max_var = 0
    optimal_threshold = 0

    #0부터 255까지 임계값을 시도하여 최적의 임계값 찾기
    for threshold in range(256):
        #클래스 1과 클래스 2의 픽셀 수 계산
        n1 = np.sum(hist[:threshold])
        n2 = np.sum(hist[threshold:])

        #클래스 1과 클래스 2의 평균 계산
        m1 = np.sum(np.arange(threshold) * hist[:threshold]) / n1 if n1 > 0 else 0
        m2 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / n2 if n2 > 0 else 0

        #클래스 간 분산 계산
        var = n1 * n2 * ((m1 - m2) ** 2)

        #최대 분산을 가지는 임계값 찾기
        if var > max_var:
            max_var = var
            optimal_threshold = threshold

    #이진화된 이미지 생성
    im_th = (im > optimal_threshold).astype(np.uint8)
    
    return im_th

image_files = ["AA1_1.jpg", "AA1_2.jpg", "AA1_3.jpg"]

#이미지 개수 구하기
num_im = len(image_files)

#이미지 개수, 8개의 공간(원본, 히스토그램, 이진화 결과, 히스토그램 순)
fig, axes = plt.subplots(num_im, 8)

for i in range(3):

    #이미지 저장
    imf = image_files[i]

    #이미지 배열화
    im = plt.imread(imf)


    nim = np.copy(im)

    
# im1의 영역을 리스트에 저장
a = [nim[:100, :66], nim[:100, 66:133], nim[:100, 133:],
     nim[100:, :66], nim[100:, 66:133], nim[100:, 133:]]
# im2의 영역을 리스트에 저장
b = [nim[:70, :55], nim[:70, 55:133], nim[:70, 133:],
     nim[70:, :55], nim[70:, 55:133], nim[70:, 133:]]
# im3의 영역을 리스트에 저장
c = [nim[:60, :60], nim[:60, 60:150], nim[:60, 150:],
     nim[60:, :60], nim[60:, 60:150], nim[60:, 150:]]


#im1 이미지 특정 영역 검정색으로 변경
nim[100,:] = 0
nim[:, 66] = 0
nim[:, 133] = 0
axes[0,0].imshow(nim, cmap = 'gray')

#im2 이미지 특정 영역 검정색으로 변경 - 100 > 70, 66 > 55
nim[70,:] = 0
nim[:, 55] = 0
nim[:, 133] = 0
axes[1,0].imshow(nim, cmap = 'gray')

#im3 이미지 특정 영역 검정색으로 변경 - 100 > 60, 66 > 60, 133 > 150
nim[60,:] = 0
nim[:, 60] = 0
nim[:, 150] = 0
axes[2,0].imshow(nim, cmap = 'gray')

plt.show()