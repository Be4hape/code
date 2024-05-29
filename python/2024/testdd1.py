import numpy as np,math, cv2
import matplotlib.pyplot as plt

#H = 
#   if. V = R.
#       ((G - B) * 60) / S
#   if. V = G
#       ((G - B) * 60) / S + 120
#   if. V = B
#       ((G - B) * 60) / S + 240

#S =
#   if. V != 0
#       V - (min(R,G,B) / V)
#   else. 0

#V = max(R,G,B)

## hsi변환
def calc_hsi(bgr):
    #B,G,R = bgr.astype(float)
    B,G,R = float(bgr[0]),  float(bgr[1]), float(bgr[2])
    bgr_sum = (R+G+B)

    tmp1 = ((R-G) + (R-B)) * 0.5
    tmp2 = math.sqrt((R-G) * (R-G) + (R-B) * (G-B))
    angle = math.acos(tmp1 / tmp2) * (180 / np.pi) if tmp2 else 0

    H = angle if B <= G else 360 - angle
    S = 1.0 - 3 * min([R,G,B]) / bgr_sum if bgr_sum else 0
    I = bgr_sum /3
    
    return (H / 2, S*255, I)

def bgr2hsi(image):
    hsv = [[calc_hsi(pixel) for pixel in row] for row in image]
    return cv2.convertScaleAbs(np.array(hsv))


im = plt.imread('cat.jpg')

im_hsi = bgr2hsi(im)

plt.imshow(im_hsi, 'gray')
plt.show()










