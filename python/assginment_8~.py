import numpy as np
import matplotlib.pyplot as plt

imsi7 = np.zeros((200, 600))
x = np.linspace(200, 250, 50)


##ㅂ ㅏ ㄱ
imsi7[50:100, 90:91] = 1
imsi7[50:100, 140:141] = 1

imsi7[75:76, 90:140] = 1
imsi7[99:100, 90:140] = 1

imsi7[50 : 110, 160:161] = 1
imsi7[75 : 76, 160:180] = 1

imsi7[130 : 131, 100 : 160] = 1
imsi7[130 : 160, 160 : 161] = 1


##ㅅ ㅣ
imsi7[75:100 , 225:255] = 0
a = imsi7[75:100 , 225:255]
np.fill_diagonal(a, val = 1)
imsi7[50: 120, 300:301] = 1

##ㅗ
imsi7[20 : 40, 380:381] = 1
imsi7[40:41, 350:410] = 1


##ㅕ
imsi7[50:51, 400:420] = 1
imsi7[60:61, 400:420] = 1
imsi7[30:90, 420:421] = 1

##원
figure, axes = plt.subplots()
draw_circle1 = plt.Circle((380, 60), 15, fill = False)

axes.set_aspect(1)
axes.add_artist(draw_circle1)

draw_circle2 = plt.Circle((410, 110), 15, fill = False)

axes.set_aspect(1)
axes.add_artist(draw_circle2)

plt.plot(x, -x+300)
plt.imshow(imsi7)
plt.show()
