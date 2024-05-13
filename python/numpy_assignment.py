import numpy as np
import matplotlib.pyplot as plt

empty = np.zeros((5, 5))
full = np.full((5, 5), 255)


first = np.append(empty, full, axis = 0)
second = np.append(full, empty, axis = 0)

imsi1 = np.append(first, second, axis = 1)
imsi2 = np.tile(imsi1, reps = 20)
imsi3 = np.append(imsi2, imsi2, axis = 0)


for i in range(0, 3):
    imsi3 = np.append(imsi3, imsi3, axis = 0)
    
imsisum = np.append(imsi2, imsi2, axis = 0)
imsisum2 = np.append(imsisum, imsisum, axis = 0)

imsi3 = np.zeros((200, 200)) ## 200 X 200크기의 imsi3 생성.(data = 255)

for i in range(0, 20):
    imsi3[ i * 10 : 200 - (i * 10),  i * 10 : 200 - (i * 10)] = (i % 2) ##짝수 = 0, 홀수 = 1
##값이 1이어도 색표현이 가능하기 때문.
    
    if(i >= 8 and i <=10):
        ##imsi3[ i * 10 : 200 - (i * 10),  i * 10 : 200 - (i * 10)] = 0 ##80~100 = 0
        imsi3[ i * 10 : 200 - (i * 10),  i * 10 : 200 - (i * 10)] = 0
    print(i)    

####

imsi1_1 = np.append(full, full, axis = 1)
imsi1_2 = np.append(empty, empty, axis = 0)
imsi1_3 = np.append(imsi1_1, imsi1_1, axis = 0)##10X10 black
imsi1_4 = np.append(imsi1_2, imsi1_2, axis = 1)##10X10 color


imsi2_1 = np.tile(imsi1_3, reps = 20)##10X200 black
imsi2_2 = np.tile(imsi1_4, reps = 20) ##10X200 color
    
imsi3_1 = np.append(imsi2_1, imsi2_2, axis = 0) ##20X200 black+color(cross)


for i in range(0, 3):
    imsi3_1 = np.append(imsi3_1, imsi3_1, axis = 0)

imsisum3 = np.append(imsi2_1, imsi2_2, axis = 0)
imsisum4 = np.append(imsisum3, imsisum3, axis = 0)
imsi3_1 = np.append(imsi3_1, imsisum4, axis = 0)


imsi2_3 = np.append(imsi2_1, imsi2_2, axis = 0)
imsi2_4 = np.append(imsi2_3, imsi2_3, axis = 0)

imsi4 = np.full((200, 200), 0)
imsi4[0:41, 0:41] = 255
imsi4[40:81, 40:81] = 255
imsi4[80:121, 80:121] = 255
imsi4[120:161, 120: 161] = 255
imsi4[160:201, 160:201] = 255

##imsi5 = np.ones((200, 200))
##u = np.linspace(0, 99, 100)
##imsi5[:u, :u] = 255

imsi6 = np.zeros((200, 200))



plt.imshow(imsi3)
plt.show()
    

