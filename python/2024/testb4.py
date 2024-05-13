import cv2
import numpy as np
win_name = 'Alpha blending' # 창 이름
trackbar_name = 'fade' # 트렉바 이름

def onChange(x): 
 alpha = x/100
 dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
 cv2.imshow(win_name, dst)

img1 = cv2.imread('1958015.jpg')
img2 = cv2.imread('1958015_1.jpg')
cv2.imshow(win_name, img1)
cv2.createTrackbar(trackbar_name, win_name, 0, 100, onChange)
cv2.waitKey()
cv2.destroyAllWindows()
