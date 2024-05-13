import numpy as np
import matplotlib.pyplot as plt

imsi6 = np.zeros((200, 200))

x = np.linspace(0, 199, 200)
y = np.linspace(0, 199, 200)

X, Y = np.meshgrid(x, y)

y1 = (-7 / 6) * X + 140
y2 = (10 / 199) * X + 140
y3 = (150 / 79) * X + ( (-150 *120) / 79 )

size_y1 = y1-Y
size_y2 = y2-Y
size_y3 = y3-Y

A = np.where(size_y1>0, 1, 0)
B = np.where(size_y2<0, 1, 0)
C = np.where(size_y3>0, 1, 0)

##imsi6[B <=0] = 100 
imsi6 = A+B+C
plt.imshow(imsi6)
plt.show()
