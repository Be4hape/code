import re
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Raw KNN CV results (as before) ---
raw = """
n_splits=  2 → ACC=0.7890±0.0132, F1 =0.7078±0.0303
n_splits=  3 → ACC=0.7879±0.0192, F1 =0.6978±0.0405
n_splits=  4 → ACC=0.7935±0.0237, F1 =0.7067±0.0532
n_splits=  5 → ACC=0.7969±0.0186, F1 =0.7129±0.0351
n_splits=  6 → ACC=0.7890±0.0149, F1 =0.6992±0.0337
n_splits=  7 → ACC=0.7879±0.0265, F1 =0.6963±0.0530
n_splits=  8 → ACC=0.7890±0.0273, F1 =0.6996±0.0537
n_splits=  9 → ACC=0.7879±0.0326, F1 =0.6980±0.0552
n_splits= 10 → ACC=0.7867±0.0295, F1 =0.6936±0.0603
n_splits= 11 → ACC=0.7879±0.0348, F1 =0.6961±0.0592
n_splits= 12 → ACC=0.7924±0.0270, F1 =0.7005±0.0558
n_splits= 13 → ACC=0.7925±0.0278, F1 =0.7005±0.0515
n_splits= 14 → ACC=0.7913±0.0472, F1 =0.7002±0.0689
n_splits= 15 → ACC=0.7925±0.0389, F1 =0.7026±0.0605
n_splits= 16 → ACC=0.7959±0.0497, F1 =0.7036±0.0883
n_splits= 17 → ACC=0.7913±0.0471, F1 =0.6963±0.0840
n_splits= 18 → ACC=0.7948±0.0360, F1 =0.7033±0.0629
n_splits= 19 → ACC=0.7946±0.0437, F1 =0.7014±0.0757
n_splits= 20 → ACC=0.7914±0.0528, F1 =0.6980±0.0932
n_splits= 21 → ACC=0.7948±0.0558, F1 =0.7007±0.0912
n_splits= 22 → ACC=0.7948±0.0568, F1 =0.7022±0.0933
n_splits= 23 → ACC=0.7926±0.0557, F1 =0.6986±0.0888
n_splits= 24 → ACC=0.7901±0.0443, F1 =0.6913±0.0907
n_splits= 25 → ACC=0.7937±0.0493, F1 =0.6987±0.0931
n_splits= 26 → ACC=0.7946±0.0556, F1 =0.6989±0.0989
n_splits= 27 → ACC=0.7969±0.0638, F1 =0.7012±0.1109
n_splits= 28 → ACC=0.7948±0.0652, F1 =0.6985±0.1115
n_splits= 29 → ACC=0.7915±0.0553, F1 =0.6951±0.0963
n_splits= 30 → ACC=0.7959±0.0674, F1 =0.7020±0.1094
n_splits= 31 → ACC=0.7949±0.0706, F1 =0.6981±0.1201
n_splits= 32 → ACC=0.7982±0.0720, F1 =0.7025±0.1153
n_splits= 33 → ACC=0.7980±0.0689, F1 =0.6982±0.1299
n_splits= 34 → ACC=0.7968±0.0659, F1 =0.6984±0.1188
n_splits= 35 → ACC=0.7971±0.0663, F1 =0.7000±0.1217
n_splits= 36 → ACC=0.7972±0.0811, F1 =0.7014±0.1335
n_splits= 37 → ACC=0.7979±0.0639, F1 =0.6970±0.1219
n_splits= 38 → ACC=0.7948±0.0684, F1 =0.6951±0.1252
n_splits= 39 → ACC=0.7994±0.0711, F1 =0.7080±0.1154
n_splits= 40 → ACC=0.7958±0.0722, F1 =0.6961±0.1280
n_splits= 41 → ACC=0.7973±0.0834, F1 =0.7006±0.1361
n_splits= 42 → ACC=0.7968±0.0832, F1 =0.6990±0.1425
n_splits= 43 → ACC=0.8006±0.0713, F1 =0.7023±0.1226
n_splits= 44 → ACC=0.8002±0.0771, F1 =0.6923±0.1709
n_splits= 45 → ACC=0.8016±0.0869, F1 =0.7060±0.1426
"""

# --- 2. Parse k, acc_mean, acc_std ---
pattern = r"n_splits=\s*(\d+)\s*→\s*ACC=([0-9.]+)±([0-9.]+),"
ks, acc_means, acc_stds = [], [], []
for line in raw.strip().splitlines():
    m = re.search(pattern, line)
    if m:
        ks.append(int(m.group(1)))
        acc_means.append(float(m.group(2)))
        acc_stds.append(float(m.group(3)))

ks        = np.array(ks)
acc_means = np.array(acc_means)
acc_stds  = np.array(acc_stds)

# --- 3. Calculate relative change rates ---
rel_delta_acc = np.diff(acc_means) / acc_means[:-1] * 100
rel_delta_std = np.diff(acc_stds)  / acc_stds[:-1]  * 100
k_mid         = (ks[:-1] + ks[1:]) / 2

# --- 4. Restrict to n_splits 35~45 ---
mask_mid = (k_mid >= 35) & (k_mid <= 45)
k_mid2         = k_mid[mask_mid]
rel_delta_acc2 = rel_delta_acc[mask_mid]
rel_delta_std2 = rel_delta_std[mask_mid]

# --- 5. Plot 2D graphs for 35 ≤ n_splits ≤ 45 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(k_mid2, rel_delta_acc2, marker='o', linestyle='-')
ax1.axhline(0, color='gray', linestyle='--')
ax1.set_ylabel('Relative Δ ACC (%)')
ax1.set_title('Relative Change Rate of Accuracy (n_splits=35–45)')

ax2.plot(k_mid2, rel_delta_std2, marker='s', linestyle='-')
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_xlabel('n_splits')
ax2.set_ylabel('Relative Δ ACC std (%)')
ax2.set_title('Relative Change Rate of ACC std (n_splits=35–45)')

# x-axis ticks from 35 to 45
ax2.set_xticks(np.arange(35, 46))
ax2.set_xticklabels(np.arange(35, 46), rotation=45)

plt.tight_layout()
plt.show()
