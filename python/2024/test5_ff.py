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

#이미지 저장
imf1 = image_files[0]
imf2 = image_files[1]
imf3 = image_files[2]

#이미지 배열화
im1 = plt.imread(imf1)
im2 = plt.imread(imf2)
im3 = plt.imread(imf3)

nim1 = np.copy(im1)
nim2 = np.copy(im2)
nim3 = np.copy(im3)

#im1의 a,b,c,d,e,f 영역을 슬라이싱하여 리스트에 저장
a1 = im1[:100, :66]
b1 = im1[:100, 66:133]
c1 = im1[:100, 133:]
d1 = im1[100:, :66]
e1 = im1[100:, 66:133]
f1 = im1[100:, 133:]


#im2의 a,b,c,d,e,f 영역을 슬라이싱하여 리스트에 저장 - 100 > 70, 66 > 55
a2 = im2[:70, :55]
b2 = im2[:70, 55:133]
c2 = im2[:70, 133:]
d2 = im2[70:, :55]
e2 = im2[70:, 55:133]
f2 = im2[70:, 133:]


#im3의 a,b,c,d,e,f 영역을 슬라이싱하여 리스트에 저장 - 100 > 60, 66 > 60, 133 > 150
a3 = im3[:60, :60]
b3 = im3[:60, 60:150]
c3 = im3[:60, 150:]
d3 = im3[60:, :60]
e3 = im3[60:, 60:150]
f3 = im3[60:, 150:]



#im1 이미지 특정 영역 검정색으로 변경
nim1[100,:] = 0
nim1[:, 66] = 0
nim1[:, 133] = 0
axes[0,0].imshow(nim1, cmap = 'gray')


#im2 이미지 특정 영역 검정색으로 변경 - 100 > 70, 66 > 55
nim2[70,:] = 0
nim2[:, 55] = 0
nim2[:, 133] = 0
axes[1,0].imshow(nim2, cmap = 'gray')


#im3 이미지 특정 영역 검정색으로 변경 - 100 > 60, 66 > 60, 133 > 150
nim3[60,:] = 0
nim3[:, 60] = 0
nim3[:, 150] = 0
axes[2,0].imshow(nim3, cmap = 'gray')


#subplot 출력, 2열부터 7열까지
axes[0,1].hist(a1.flatten(), bins = 256)
axes[0,2].hist(b1.flatten(), bins = 256)
axes[0,3].hist(c1.flatten(), bins = 256)
axes[0,4].hist(d1.flatten(), bins = 256)
axes[0,5].hist(e1.flatten(), bins = 256)
axes[0,6].hist(f1.flatten(), bins = 256)
axes[1,1].hist(a2.flatten(), bins = 256)
axes[1,2].hist(b2.flatten(), bins = 256)
axes[1,3].hist(c2.flatten(), bins = 256)
axes[1,4].hist(d2.flatten(), bins = 256)
axes[1,5].hist(e2.flatten(), bins = 256)
axes[1,6].hist(f2.flatten(), bins = 256)
axes[2,1].hist(a3.flatten(), bins = 256)
axes[2,2].hist(b3.flatten(), bins = 256)
axes[2,3].hist(c3.flatten(), bins = 256)
axes[2,4].hist(d3.flatten(), bins = 256)
axes[2,5].hist(e3.flatten(), bins = 256)
axes[2,6].hist(f3.flatten(), bins = 256)

# a1, b1, c1을 이진화
im1_th_a = threshold_otsu(a1)
im1_th_b = threshold_otsu(b1)
im1_th_c = threshold_otsu(c1)

# d1, e1, f1을 이진화
im1_th_d = threshold_otsu(d1)
im1_th_e = threshold_otsu(e1)
im1_th_f = threshold_otsu(f1)

# a1, b1, c1을 수평으로 결합하고 이진화
hstack_top_th = np.hstack((im1_th_a, im1_th_b, im1_th_c))

# d1, e1, f1을 수평으로 결합하고 이진화
hstack_bottom_th = np.hstack((im1_th_d, im1_th_e, im1_th_f))

# 두 클래스를 수직으로 결합하여 이진화된 이미지 생성
im1_combined_th = np.vstack((hstack_top_th, hstack_bottom_th))

# 이진화된 이미지를 8번째 열에 출력
axes[0,7].imshow(im1_combined_th, cmap='gray')



# a2, b2, c2을 이진화
im2_th_a = threshold_otsu(a2)
im2_th_b = threshold_otsu(b2)
im2_th_c = threshold_otsu(c2)

# d2, e2, f2을 이진화
im2_th_d = threshold_otsu(d2)
im2_th_e = threshold_otsu(e2)
im2_th_f = threshold_otsu(f2)

# a2, b2, c2을 수평으로 결합하고 이진화
hstack_top_th = np.hstack((im2_th_a, im2_th_b, im2_th_c))

# d2, e2, f2을 수평으로 결합하고 이진화
hstack_bottom_th = np.hstack((im2_th_d, im2_th_e, im2_th_f))

# 두 클래스를 수직으로 결합하여 이진화된 이미지 생성
im2_combined_th = np.vstack((hstack_top_th, hstack_bottom_th))

# 이진화된 이미지를 8번째 열에 출력
axes[1,7].imshow(im2_combined_th, cmap='gray')



# a3, b3, c3을 이진화
im3_th_a = threshold_otsu(a3)
im3_th_b = threshold_otsu(b3)
im3_th_c = threshold_otsu(c3)

# d3, e3, f3을 이진화
im3_th_d = threshold_otsu(d3)
im3_th_e = threshold_otsu(e3)
im3_th_f = threshold_otsu(f3)

# a3, b3, c3을 수평으로 결합하고 이진화
hstack_top_th = np.hstack((im3_th_a, im3_th_b, im3_th_c))

# d3, e3, f3을 수평으로 결합하고 이진화
hstack_bottom_th = np.hstack((im3_th_d, im3_th_e, im3_th_f))

# 두 클래스를 수직으로 결합하여 이진화된 이미지 생성
im3_combined_th = np.vstack((hstack_top_th, hstack_bottom_th))

# 이진화된 이미지를 8번째 열에 출력
axes[2,7].imshow(im3_combined_th, cmap='gray')


plt.show()


