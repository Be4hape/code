import numpy as np
import matplotlib.pyplot as plt

#image 가져오기
im = plt.imread("lena_gray.bmp").astype(np.uint8)

#height, width는 이런식으로 선언 가능
height, width = im.shape

#build mask, 마스크의 크기는 이미지 가로길이의 1/4
mask = np.zeros_like(im, dtype=np.float32)
x, y = width // 2, height // 2
r = width // 4

#마스크 내의 원형 영역을 1로 설정
Yn, Xn = np.ogrid[:height, :width]
dToc = np.sqrt((Xn - x)**2 + (Yn - y)**2)
mask[dToc <= r] = 1

#f1은 원본, f2는 마스크 적용
f1 = im.astype(np.float32)
f2 = im * mask

#alpha값 리스트 저장, subplot 생성
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
fig, axes = plt.subplots(1, len(alphas) + 1)

#alpha값 개수만큼 for문 반복
for i in range(len(alphas)):
    alpha = alphas[i]
    blend = ((1 - alpha) * f1 + alpha * f2).astype(np.uint8)
    
    axes[i + 1].imshow(blend, 'gray')

#원본
axes[0].imshow(im, 'gray')

plt.tight_layout
plt.show()
