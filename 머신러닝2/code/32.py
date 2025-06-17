import re
import numpy as np
import matplotlib.pyplot as plt

# --- 1. fold_results.txt에서 raw 결과 읽어오기 ---
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

ks = np.array(ks)
acc_means = np.array(acc_means)
acc_stds = np.array(acc_stds)

# --- 3. 변화율(상대 변화율 %) 계산 ---
rel_delta_acc = np.diff(acc_means) / acc_means[:-1] * 100
rel_delta_std = np.diff(acc_stds) / acc_stds[:-1] * 100
k_mid = (ks[:-1] + ks[1:]) / 2

# --- 4. 2D 변화율 시각화 ---
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# (a) ACC relative change rate
ax[0].plot(k_mid, rel_delta_acc, marker='o', linestyle='-')
ax[0].axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax[0].set_ylabel('Relative Δ ACC (%)')
ax[0].set_title('Relative Change Rate of Accuracy vs. k')

# (b) ACC std relative change rate
ax[1].plot(k_mid, rel_delta_std, marker='s', linestyle='-')
ax[1].axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax[1].set_xlabel('k (Number of Folds)')
ax[1].set_ylabel('Relative Δ ACC std (%)')
ax[1].set_title('Relative Change Rate of Accuracy Std vs. k')

plt.tight_layout()
plt.show()
