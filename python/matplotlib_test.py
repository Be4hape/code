import numpy as np
import matplotlib.pyplot as plt

x  = np.arange(10)
y = x**2

y2 = x*5 + 2

plt.plot(x, y, 'b', label = 'first')
plt.legend(loc = 'upper right')
plt.show()
