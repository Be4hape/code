import numpy as np
import matplotlib.pyplot as plt
import time

t1 = time.time()
x = np.arange(0,500, 0.001)
A = 1
f = 10
y = A*np.sin(2*np.pi*f*x)
elapsed1 = time.time() - t1


t2 = time.time()
x = 0
A = 1
f = 10
while x <=500 :
    y = A* np.sin(2*np.pi*f*x)
    x = x+0.001
    
elapsed2 = time.time() - t2
print(elapsed1)
print(elapsed2)
