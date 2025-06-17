import numpy as np
import matplotlib.pyplot as plt

# k 값 1~30
ks = np.arange(1, 31)

# 각 k에 대한 train accuracy (퍼센티지)
acc = np.array([
    77.55, 77.22, 78.68, 77.22, 76.54, 75.98, 77.10, 76.88, 77.22, 77.44,
    77.44, 79.46, 76.77, 78.68, 80.13, 79.57, 79.35, 79.46, 79.46, 80.47,
    78.79, 79.01, 79.91, 79.69, 79.35, 79.46, 79.35, 79.57, 79.35, 78.90
])

# 최적 k 찾아두기
best_k = ks[np.argmax(acc)]
best_acc = acc.max()

# 플롯
plt.figure(figsize=(8, 5))
plt.plot(ks, acc, marker='o', linestyle='-')
plt.scatter([best_k], [best_acc], s=100, color='red', label=f'Optimal k = {best_k}')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Train Accuracy (%)')
plt.title('Train Accuracy vs. k (1–30) in KNN')
plt.xticks(ks)
plt.grid(True)
plt.legend()
plt.show()
