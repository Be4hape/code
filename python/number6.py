import numpy as np
import matplotlib.pyplot as plt

imsi6 = np.zeros((200,200))
a = np.arange(200)
b = np.arange(200)

x,y=np.meshgrid(a, b)

area_1 = ((-140/120) * (x - 120)  - y)
area_2 = ((150/79) * (x - 120) - y)
area_3 = ((10/199) * x - (y - 140))

a = np.where(area_1 > 0, 0, 1)
b = np.where(area_2 > 0, 0, 1)
c = np.where(area_3 < 0, 0, 1)
imsi6 = (a + b + c)

plt.imshow(imsi6)
plt.show()
