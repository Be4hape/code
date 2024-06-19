import numpy as np
import matplotlib.pyplot as plt

def threshold_otsu(im):
    # 히스토그램 계산
    hist, bins = np.histogram(im, bins=256, range=[0, 256])
    bins = np.delete(bins, -1)

    # 전체 픽셀 수 계산
    total_pixels = im.size

    # 초기값 설정
    max_var = 0
    optimal_threshold = 0

    # 초기 변수 설정
    pi = hist / np.sum(hist)
    mg = np.sum(bins * pi)
    nk_old = 0
    T_new = 0

    # 0부터 255까지 임계값을 시도하여 최적의 임계값 찾기
    for T in range(1, 256):
        p1 = np.sum(pi[:T])
        mk = np.sum(bins[:T] * pi[:T])

        if p1 == 0 or p1 == 1:
            continue

        ob = (mg * p1 - mk)**2 / (p1 * (1 - p1))
        og = np.sum((bins - mg)**2 * pi)

        nk = ob / og

        if (nk_old < nk and not np.isinf(nk)):
            T_new = T
            nk_old = nk

    optimal_threshold = T_new
    return optimal_threshold

def estimate_illumination_change(image, corners):
    X = []
    Y = []
    for corner in corners:
        for y in range(corner.shape[0]):
            for x in range(corner.shape[1]):
                X.append([x, y, 1])
                Y.append(corner[y, x])
    
    X = np.array(X)
    Y = np.array(Y)
    
    # 역행렬, a,b,c 계산, np.dot
    pinv_X = np.linalg.pinv(X)
    coeffs = np.dot(pinv_X, Y)
    
    return coeffs

image_file = "AA1_3.jpg"

# 이미지 불러오기
im = plt.imread(image_file)
if im.ndim == 3:
    im = np.mean(im, axis=2)  # 이미지를 그레이스케일로 변환
height, width = im.shape

# 네 모서리
top_left = im[:5, :5]
top_right = im[:5, -5:]
bottom_left = im[-5:, :5]
bottom_right = im[-5:, -5:]

# 네 모서리 영역 리스트 저장
corner_coords = [top_left, top_right, bottom_left, bottom_right]

# 조명 변화 추정
coeffs = estimate_illumination_change(im, corner_coords)

# 평면 생성
a, b, c = coeffs
x = np.arange(width)
y = np.arange(height)
X, Y = np.meshgrid(x, y)
Z = a * X + b * Y + c

print(f'a = {coeffs[0]}, b = {coeffs[1]}, c = {coeffs[2]}')

# 평면 시각화
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(Z, cmap='gray')
ax.set_title('Illumination Plane')
plt.show()

# 조명 평면을 원본 이미지에서 빼기
corrected_im = Z - im

# Otsu 이진화 적용
threshold = threshold_otsu(corrected_im)
binary_im = corrected_im > threshold

# 반전된 결과 생성 (배경이 흰색, 중앙 데이터가 검은색)
binary_im_inverted = np.invert(binary_im)

# 결과 시각화
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(im, cmap='gray')
axes[0].set_title('Original Image')

axes[1].imshow(Z, cmap='gray')
axes[1].set_title('Illumination Plane')

axes[2].imshow(corrected_im, cmap='gray')
axes[2].set_title('Corrected Image (Z - im)')

axes[3].imshow(binary_im_inverted, cmap='gray')
axes[3].set_title('Binarized Image after Otsu (Inverted)')

plt.tight_layout()
plt.show()
