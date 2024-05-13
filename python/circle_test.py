import numpy as np
import matplotlib.pyplot as plt

##imsi5 = np.zeros((200, 200))
####x, y = np.mgrid[:200, :200]
##x = np.linspace(0, 199, 200)
##y = np.linspace(0, 199, 200)
##[X, Y] = np.meshgrid(x,y)
##circle = (X- 100)**2 + (Y - 100)**2
##donut  = (circle <= 10000)
##
##plt.imshow(donut)
##plt.show()


##import numpy as np
##import matplotlib.pyplot as plt
##x = np.linspace(30, 70, 1000)
##y = np.linspace(30, 70, 1000)
##X, Y = np.meshgrid(x,y)
##F = (X-50)**2 + (Y-50)**2
##plt.contour(X,Y,F,[50])
##plt.show()


figure, axes = plt.subplots()
draw_circle1 = plt.Circle((0.5, 0.5), 0.3, fill = False)

axes.set_aspect(1)
axes.add_artist(draw_circle1)

draw_circle2 = plt.Circle((0.5, 0.5), 0.5, fill = False)

axes.set_aspect(1)
axes.add_artist(draw_circle2)
plt.title('Circle')
plt.show()
