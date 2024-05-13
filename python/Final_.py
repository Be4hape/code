import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imsi = mpimg.imread('C:/Users/for/Desktop/영상처리/1958015.jpg')
z = imsi[:,:,0]
y,x,w = imsi.shape
frame = np.zeros((2000, 2000))


dim = len(imsi)

##1958015 * 2 = 30, 즉 30' rotate
sin = np.sin(30 * np.pi / 180)
cos = np.cos(30 * np.pi / 180)

htM = np.array([[cos, -sin], [sin, cos]])

##for i in range(0,799):
##    count = 0
##    for j in range(0,599):
##        count+=1
##        x1 = imsi[i, count]
##        y1 = imsi[count, j]
##
##        X2 = (x1 * cos) - (y1 * sin)
##        Y2 = (x1 * sin) + (y1 * cos)

for i in range(0,599):
    count = 0
    for j in range(0, 799):
        count+=1
        x1 = imsi[i, count]
        y1 = imsi[count, j]

        X2 = (x1 * cos) - (y1 * sin)
        Y2 = (x1 * sin) + (y1 * cos)

frame = frame[int(X2), int(Y2)]

     
print(frame)
plt.imshow(frame, 'gray')
plt.show()
