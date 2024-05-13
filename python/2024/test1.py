import numpy as np
import matplotlib.pyplot as plt

#Open image
imA1 = plt.imread("AA1_1.jpg")
imA2 = plt.imread("AA1_2.jpg")
imA3 = plt.imread("AA1_3.jpg")

#calculate histogram
hist1, bins1 = np.histogram(imA1.flatten(), bins=256, range=(0, 256))
hist2, bins2 = np.histogram(imA2.flatten(), bins=256, range=(0, 256))
hist3, bins3 = np.histogram(imA3.flatten(), bins=256, range=(0, 256))

#imA1
plt.subplot(3, 2, 1)
plt.imshow(imA1, cmap='gray')
plt.title("AA1_1 Image")
#imA1 histogram
plt.subplot(3, 2, 2)
plt.bar(bins1[:-1], hist1)
plt.title("AA1_1 Histogram")


#imA2 
plt.subplot(3, 2, 3)
plt.imshow(imA2, cmap='gray')
plt.title("AA1-2 Image")
#imA2 histogram
plt.subplot(3, 2, 4)
plt.bar(bins2[:-1], hist2)
plt.title("AA1-2 Histogram")


#imA3
plt.subplot(3, 2, 5)
plt.imshow(imA3, cmap='gray')
plt.title("AA1-3 Image")
#imA3 histogram
plt.subplot(3, 2, 6)
plt.bar(bins3[:-1], hist3)
plt.title("AA1-3 Histogram")

plt.tight_layout()
plt.show()
