import numpy as np
import matplotlib.pyplot as plt

def threshold(im):
    # 임의의 값 T
    T = 100
    # 이진화된 이미지를 저장할 배열 생성
    result = np.zeros_like(im)

    # 반복해서 적절한 임계값(T) 찾기
    while True:
        # threshold값 초과인 픽셀은 1로, 이하인 픽셀은 0으로 설정
        result[im > T] = 1
        result[im <= T] = 0
        #np.where(result[im>T],1,0)

        # 적용된 임계값을 기준으로 픽셀 평균값 계산
        new_T = (np.mean(im[im > T]) + np.mean(im[im <= T])) / 2
        # 새로 계산된 임계값과 이전 임계값의 차이가 0.5보다 작으면 반복 종료
        if np.abs(T - new_T) < 0.5:
            break
        T = new_T
        print(T)
    
    return result

# 이미지 파일 리스트
imAs = ["AA1_1.jpg", "AA1_2.jpg", "AA1_3.jpg"]

# 이미지 개수 구하기
num_im = len(imAs)

# figure 창에서 출력할 이미지 개수 지정
# 이미지 개수만큼, 4개의 공간(원본, 이진화 결과 순)
fig, axes = plt.subplots(num_im, 4)

# 이미지 개수만큼 돌아가는 for문
for i in range(num_im):
    # 각각의 이미지 파일 불러와서
    # 배열로 변환
    im = np.array(plt.imread(imAs[i]))

    # 이진화된 이미지 생성
    im_th = threshold(im)

    # 1.원본 이미지 출력
    axes[i, 0].imshow(im, cmap='gray')

    # 2.이미지의 histogram 출력
    axes[i, 1].hist(im.flatten(), bins=256)

    # 3.이진화 결과
    axes[i, 2].imshow(im_th, cmap='gray')

    #4. histogram
    axes[i, 3].hist(im_th.flatten(), bins=256)

plt.tight_layout()
plt.show()
