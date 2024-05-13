import numpy as np
import matplotlib.pyplot as plt

def threshold_otsu(im):
    #이미지hist
    hist, bins = np.histogram(im.flatten(), bins=256)

    #전체 픽셀 수
    pixel_ttl = im.shape[0] * im.shape[1]

    #임계값 T
    T = 100

    while True:
        #T를 기준으로 작은값left, 큰 값right
        T_left  = im[im <= T]
        T_right = im[im > T]

        #T_left, T_right 픽셀 수 계산
        w_left  = len(T_left) / pixel_ttl
        w_right = len(T_right) / pixel_ttl

        #left, right 평균 계산
        m_left  = np.mean(T_left)
        m_right = np.mean(T_right)

        #left, right 분산 계산
        #분산 = 전체 1 중에 해당 픽셀이 갖는 확률. 가중치 or 강도
        v_left  = np.mean((T_left - m_left) ** 2)
        v_right = np.mean((T_right - m_right) ** 2)

        #새로운 임계값 계산
        new_T = (m_left + m_right) / 2

        #종료 조건: 새로운 임계값과 이전 임계값의 차이가 작을 때 종료
        if abs(new_T - T) < 0.01:
            break

        #새로운 임계값으로 갱신
        T = new_T
        print(T)

    #이진화된 이미지 생성
    result = np.zeros_like(im)
    result[im >  T] = 1
    result[im <= T] = 0

    print(np.shape(result))

    return result


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

    # 이진화 적용 이미지 생성
    im_th = threshold_otsu(im)

    # 이진화된 이미지
    axes[i, 2].imshow(im_th, cmap='gray')

    # 이진화된 이미지의 히스토그램
    axes[i, 3].hist(im_th.flatten(), bins=256)
    axes[i, 3].set_xlim(-1, 2)  #0과 1로 이진화 > -1은 0인 값들을 보기 위함

plt.tight_layout()
plt.show()
