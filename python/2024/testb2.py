import numpy as np
import cv2
import matplotlib.pyplot as plt

# 체크무늬 패턴 생성
w, h = 800, 600
nw, nh = 40, 30
rows = h // nh
cols = w // nw
ptrn = np.zeros((h, w), dtype=np.uint8)

# 행과 열 반복
for i in range(rows):
    for j in range(cols):
        # 홀수 행과 열에 해당하는 셀의 픽셀을 255(흰색)으로 설정
        if (i + j) % 2 != 0:
            start_row, end_row = i * nh, (i + 1) * nh
            start_col, end_col = j * nw, (j + 1) * nw
            ptrn[start_row:end_row, start_col:end_col] = 255

# 자신의 영상 불러오기 (G 채널만 사용)
image = cv2.imread("1958015.jpg")
g_channel = image[:,:,1]  # G 채널만 가져오기

# G 채널 정규화
g_min = np.min(g_channel)
g_max = np.max(g_channel)
g_channel_normalized = ((g_channel - g_min) / (g_max - g_min)) * 255

# G 채널과 체크무늬 패턴 곱셈 연산
result = np.multiply(g_channel_normalized, ptrn)

# 결과 이미지 저장 (OpenCV 사용)
cv2.imwrite("1958015_2.jpg", result.astype(np.uint8))

# 결과 이미지 출력 (Matplotlib 사용)
plt.imshow(result, cmap='gray')
plt.title('Result')
plt.show()

# 결과 이미지 출력 (OpenCV 사용)
cv2.imshow('Result', result.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
