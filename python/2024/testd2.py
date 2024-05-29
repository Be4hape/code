import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('cat.jpg')

#RGB to YCbCr
def rgb_to_ycbcr(img):
    #YCbCr
    #Y  =  0.299R     + 0.587G    + 0.114B
    #Cb = -0.168736R  - 0.331264G + 0.5B
    #Cr =  0.5R       - 0.418688G - 0.081312B
    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.168736, -0.331264, 0.5],
                                 [0.5, -0.418688, -0.081312]])
    img = img.astype(np.float32)
    ycbcr = np.dot(img, transform_matrix.T)
    ycbcr[..., [1, 2]] += 128.0
    return ycbcr

#YCbCr to RGB
def ycbcr_to_rgb(ycbcr):
    #RGB
    #R = Y +     0     + 1.402Cr
    #G = Y - 0.34414Cb - 0.714136Cr
    #B = Y + 1.772Cb   + 0
    inverse_matrix = np.array([[1, 0, 1.402],
                               [1, -0.344136, -0.714136],
                               [1, 1.772, 0]])
    ycbcr[..., [1, 2]] -= 128.0
    rgb = np.dot(ycbcr, inverse_matrix.T)
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)

#
ycbcr = rgb_to_ycbcr(img)
im_r = ycbcr_to_rgb(ycbcr)

# 결과 출력
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(im_r)

plt.show()
