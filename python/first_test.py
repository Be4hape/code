##넘파이 테스트
##리스트와 어레이는 다름, 리스트는 연산 불가
import numpy as np
##data = [[0.9526, -0.246, -0.8856], [0.5639, 0.2379, 0.9104]]
##arr1 = np.array(data)
##print(arr1)
##print(arr1*10)
##print(arr1-arr1)


##import numpy as np
##a = np.arange(15).reshape(3,5)
##print(type(a))
##
##b = np.array([6,7,8])
##print(type(b))

##import numpy as np
##arr = np.array([1,2,3,4,5])


##import numpy as np
##arr = np.array(['3.7', '-1.2', '-2.6', '0.5', '12.9', '10.1'])
##arr.astype(np.float32)

##import numpy as np
##a = np.array([2,3,4])
##print(a.dtype)
####b = np.array([1.2, 3.5, 5.1])
####print(b.dtype)
##
##b = np.array([(1.5, 2, 3), (4,5,6)])
##print(b)
##
##c = np.array([[1,2], [3,4]], dtype = complex)
##print(c)
##
##np.zeros((3,4))

##import numpy as np
##np.arange(10,30,5)
##print(np.arange(10,30,5))
##print(np.linspace(0,2,9))
##print(np.arange(24).reshape(2,3,4))
##print(np.linspace(0,99,100).reshape(10,10))

##import numpy as np
##mean = 0
##std = 1
##np.random.normal(mean, std, (2,3))
##a = np.random.randn(2,3)
##
##data = np.random.normal(0,1,10000)
##import matplotlib.pyplot as plt
##plt. hist(data, bins = 100)
##plt.show()
##
##a = np.random.rand(3,2)
##a = np.random.random((3,2))
##data = np.random.rand(10000)
##import matplotlib.pyplot as plt
##plt.hist(data, bins=10)
##print(plt.show())

##import numpy as np
##a = np.random.randint(5,10,size=(2,4))
##print(np.random.random((2,2)))
##print(np.random.randint(0,10,(2,3)))
##print(np.random.random((2,2)))
##print(np.random.randint(0,10,(2,3)))
##np.random.seed(100)
####seed100은 등록번호 설정, 파이썬을 재부팅 해도 그대로 저장되어 있는 상태
##print(np.random.random((2,2)))
##print(np.random.randint(0,10,(2,3)))
##print(np.random.random((2,2)))
##print(np.random.randint(0,10,(2,3)))

##a = np.array([[1,2,3] , [4,5,6]], float)
##print(a)
##print(a.flatten())
##
##a = np.random.randint(1,10, (2,3))
##print(a.ravel())
##
##a1 = np.array([[1,2] , [3,4]])
##a2 = a1.ravel()
##a1[0][0] = 99
##
##print(a1)

##a = np.arange(1,10).reshape(3,3)
##result1 = np.insert(a,1,999)
##result2 = np.insert(a,1,999,axis = 0)
##result3 = np.insert(a,1,999,axis = 1)
##result4 = np.insert(a,1,999,axis = 2)
####axis = 2 > 3차원
##
##print(result1)
##print(result2)
##print(result3)


##a = np.arange(1,10).reshape(3,3)
##result1 = np.delete(a,1)
##result2 = np.delete(a,1, axis = 0)
##result3 = np.delete(a,1, axis = 1





##a = np.array([[1,2], [3,4]], float)
##b = np.array([[5,6], [7,8]], float)
##print(np.concatenate((a,b)))
##print(np.concatenate((a,b), axis =0) == np.vstack((a,b)))
##print(np.concatenate((a,b), axis =1) == np.hstack((a,b)))
##print(np.dstack((a,b)))


##a = np.arange(9).reshape(3,3)
##np.hsplit(a,3) == np.split(a,3,axis=1)
##np.vsplit(a,3) == np.split(a,3,axis = 0)
##c = np.arange(27).reshape(3,3,3)
##np.dsplit(c,3)
##

##for문과 meshgrid의 소요 시간 차이 출력
##import time
##t0 = time.time()
##points = np.arange(-5, 5, 0.01)
##xs, ys = np.meshgrid(points, points)
##z = np.sqrt(xs**2 + ys**2)
##t1 = time.time()
##
##z = np.zeros((points.shape[0], points.shape[0]))
##for ii, xs in enumerate(points):
##    for jj,  ys in enumerate(points):
##        z[ii, jj] = np.sqrt(xs**2 + ys**2)
##        t2 = time.time()
##print(t1 - t0)
##print (t2 - t1)

import matplotlib.pyplot as plt
##x = np.arange(10)
##y = x**2
##
##plt.plot(x,y, ':g')
##plt.show()

##
##x = np.arange(10)
##y = x**2
##y2 = x*5 +2
##
##plt.plot(x,y, 'b', label = 'first')
##plt.plot(x,y2, 'r', label = 'second')
##plt.legend(loc = 'upper right')
##plt.show()
##


##x = np.arange(10)
##
##plt.subplot(2,2,1)
##plt.plot(x, x**2)
##plt.subplot(2,2,2)
##plt.plot(x, x**5)
##plt.subplot(223)
##plt.plot(x, np.sin(x))
##plt.subplot(224)
##plt.plot(x, np.cos(x))
##plt.show()

####3D 그래프 그리기
##from mpl_toolkits.mplot3d import Axes3D
##import matplotlib.pyplot as plt
##
##from matplotlib import cm
##fig = plt.figure()
##ax = fig.add_subplot(111, projection = '3d')
##u = np.linspace(-1, 1, 100)
##x, y = np.meshgrid(u, u)
##z = x**2 +y**2
##ax.plot_surface(x, y, z, rstride = 4, cstride =4, cmap = cm.YlGnBu_r)
##plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('C:/Users/qkrtl/OneDrive/바탕 화면/1115159.jpg')
lum_img = img[:,:,0]
print(img.shape)
print(type(img))
plt.imshow(lum_img)
plt.show()



