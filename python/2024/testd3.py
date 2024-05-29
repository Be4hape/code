import numpy as np
import matplotlib.pyplot as plt

# RGB to HSV 변환 함수
def rgb_to_hsv(img):
    img = img / 255.0  # 스케일 조정
    hsv = np.zeros_like(img)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    
    max_val = np.max(img, axis=2)
    min_val = np.min(img, axis=2)
    delta = max_val - min_val
    
    # Value 계산
    V = max_val
    
    # Saturation 계산
    S = np.where(V != 0, delta / V, 0)
    
    # Hue 계산
    H = np.zeros_like(V)
    mask = delta != 0
    
    mask_R = (mask) & (V == R)
    mask_G = (mask) & (V == G)
    mask_B = (mask) & (V == B)
    
    H[mask_R] = ((G[mask_R] - B[mask_R]) / delta[mask_R]) * 60
    H[mask_G] = ((B[mask_G] - R[mask_G]) / delta[mask_G]) * 60 + 120
    H[mask_B] = ((R[mask_B] - G[mask_B]) / delta[mask_B]) * 60 + 240
    
    H[H < 0] += 360
    H /= 360  # Hue의 범위를 [0, 1]로 조정
    
    hsv[..., 0] = H
    hsv[..., 1] = S
    hsv[..., 2] = V
    
    return hsv

# 이미지 로드 및 변환
img = plt.imread("cat.jpg")
hsv = rgb_to_hsv(img)

# 2차원 히스토그램 계산 (Hue와 Saturation)
hue = (hsv[..., 0] * 255).astype(np.uint8)
saturation = (hsv[..., 1] * 255).astype(np.uint8)
hist2d, xedges, yedges = np.histogram2d(hue.flatten(), saturation.flatten(), bins=256, range=[[0, 256], [0, 256]])

plt.subplot(131)
plt.imshow(img)

plt.subplot(132)
plt.imshow(hist2d.T, origin='lower', cmap='hot', interpolation='nearest')

plt.subplot(133)
plt.imshow(hsv)

plt.tight_layout()
plt.show()
