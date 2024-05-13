import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d

imsi8 = mpimg.imread('C:/Users/qkrtl/OneDrive/바탕 화면/me_gray.jpg')
Z = imsi8[:,:,0]

x = np.linspace(0, 479, 480)
y = np.linspace(0, 639, 640)

X,Y = np.meshgrid(x, y)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,Z,rstride=10,cstride=10,cmap='viridis',edgecolor='None')
plt.show()
