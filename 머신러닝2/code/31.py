import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting 지원

# --- 1. 파일에서 raw 결과 읽어오기 ---
with open('fold_results.txt', 'r', encoding='utf-8') as f:
    raw = f.read()

# --- 2. 정규표현식으로 k, acc_mean, acc_std 추출 ---
pattern = (
    r"k\s*=\s*(\d+)\s*→\s*ACC\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),"
)
ks, acc_means, acc_stds = [], [], []

for line in raw.splitlines():
    m = re.search(pattern, line)
    if m:
        ks.append(int(m.group(1)))
        acc_means.append(float(m.group(2)))
        acc_stds.append(float(m.group(3)))

ks        = np.array(ks)
acc_means = np.array(acc_means)
acc_stds  = np.array(acc_stds)

# --- 3. 3D 산점도 그리기 (z축: Accuracy 표준편차) ---
fig = plt.figure(figsize=(9, 6))
ax  = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    ks,            # x축: k (Number of Folds)
    acc_means,     # y축: Accuracy mean
    acc_stds,      # z축: Accuracy std
    c=acc_means,   # 색상: 평균 Accuracy
    cmap='viridis',
    marker='o',
    s=30,
    alpha=0.8
)

ax.set_xlabel('k (Number of Folds)')
ax.set_ylabel('Accuracy (mean)')
ax.set_zlabel('Accuracy (std)')
ax.set_title('3D Scatter: k vs. ACC_mean vs. ACC_std')

cbar = fig.colorbar(sc, pad=0.1)
cbar.set_label('ACC mean')

plt.tight_layout()
plt.show()
