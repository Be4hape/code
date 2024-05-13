from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

#체크무늬 패턴 생성
w, h = 800, 600
nw, nh = 40, 30
rows = h // nh #20, /만 사용할 경우 for 조건으로 걸 수 없음,float이기 때문
cols = w // nw #20
ptrn = np.zeros((h, w), dtype=np.uint8)

#행과 열 반복 (rows와 cols만큼), 0행 0열 20열 40열 ' ' ' 
for i in range(rows):
    for j in range(cols):
        #짝수 행과 열에 해당하는 셀의 픽셀을 255(흰색)으로 설정
        if (i + j) % 2 != 0:
            #패턴 이미지에서 현재 셀의 픽셀 값을 255(흰색)으로 설정
            #현재 셀의 영역은 행의 시작과 끝, 열의 시작과 끝을 나타냄
            start_row, end_row = i * nh, (i + 1) * nh
            start_col, end_col = j * nw, (j + 1) * nw
            ptrn[start_row:end_row, start_col:end_col] = 255


#이미지 저장
im = Image.fromarray(ptrn)
im.save("1958015_1.jpg")

#Matplotlib
plt.imshow(ptrn, cmap='gray')
plt.show()

#OpenCV를 사용하여 이미지 출력
cv2.imshow('Checkered ptrn', ptrn)
cv2.waitKey(0)
cv2.destroyAllWindows()