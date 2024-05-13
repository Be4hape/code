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
num_im = len(image_files)
fig, axes = plt.subplots(num_im, 8)


im1 = plt.imread(image_files[0])
im2 = plt.imread(image_files[1])
im3 = plt.imread(image_files[2])

nim1 = np.copy(im1)
nim2 = np.copy(im2)
nim3 = np.copy(im3)

#im1의 영역을 리스트에 저장
a = [nim1[:100, :66], nim1[:100, 66:133], nim1[:100, 133:],
     nim1[100:, :66], nim1[100:, 66:133], nim1[100:, 133:]]

# im2의 영역을 리스트에 저장
b = [nim2[:70, :55], nim2[:70, 55:133], nim2[:70, 133:],
     nim2[70:, :55], nim2[70:, 55:133], nim2[70:, 133:]]
# im3의 영역을 리스트에 저장
c = [nim3[:60, :60], nim3[:60, 60:150], nim3[:60, 150:],
     nim3[60:, :60], nim3[60:, 60:150], nim3[60:, 150:]]
    
imth_a = []
imth_b = []
imth_c = []

for j in range(6):
    axes[0, j+1].hist(a[j].flatten(), bins = 256)
    axes[1, j+1].hist(b[j].flatten(), bins = 256)
    axes[2, j+1].hist(c[j].flatten(), bins = 256)

    #리스트에 리스트를 지정하는 방식
    imth_a.append(threshold_otsu(a[j]))
    imth_b.append(threshold_otsu(b[j]))
    imth_c.append(threshold_otsu(c[j]))


# 8번째 열에 이미지 이진화 병합
for i in range(3):
    if i == 0:
        imth = imth_a
    elif i == 1:
        imth = imth_b
    else:
        imth = imth_c
        
    hstack_top = np.hstack(imth[:3])
    hstack_bot = np.hstack(imth[3:])
    combined = np.vstack((hstack_top, hstack_bot))
    axes[i, 7].imshow(combined, cmap='gray')


#1번째 열에 분할 라인 표시
nim1[100,:] = 0
nim1[:, 66] = 0
nim1[:, 133] = 0
axes[0,0].imshow(nim1, cmap = 'gray')

nim2[70,:] = 0
nim2[:, 55] = 0
nim2[:, 133] = 0
axes[1,0].imshow(nim2, cmap = 'gray')

nim3[60,:] = 0
nim3[:, 60] = 0
nim3[:, 150] = 0
axes[2,0].imshow(nim3, cmap = 'gray')

plt.show()