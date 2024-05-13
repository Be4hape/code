import numpy as np
import matplotlib.pyplot as plt

# 이미지 리딩
im = plt.imread("AA1_3.jpg")
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
imf = np.array(im)

# 특정 영역 검정색으로 변경
imf[100,:] = 0
imf[:, 66] = 0
imf[:, 133] = 0

a = imf[:100, :66]
b = imf[:100, 66:133]
c = imf[:100, 133:]
d = imf[100:, :66]
e = imf[100:, 66:133]
f = imf[100:, 133:]

list1 = [a,b,c,d,e,f]

# 원본 이미지 출력
for i in range(3):
    axes[i,0].imshow(imf, cmap='gray')

# 영역에 대한 히스토그램 출력
for i in range(3):
    for j in range(6):  # 6개의 서브플롯을 채웁니다.
        axes[i,j+1].hist(list1[j].flatten(), bins=256)  # 각 영역의 히스토그램을 검정색으로 출력합니다.

# 이진화 히스토그램 출력 (마지막 열)
# 여기에는 이진화 히스토그램 코드를 추가해야 합니다.

plt.tight_layout()
plt.show()








