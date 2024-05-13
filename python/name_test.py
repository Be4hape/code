import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imsi8 = mpimg.imread('C:/Users/qkrtl/OneDrive/바탕 화면/me.jpg')

##gray
lum_img = imsi8[:,:,0]
plt.imshow(lum_img, cmap = 'gray')

plt.show()
