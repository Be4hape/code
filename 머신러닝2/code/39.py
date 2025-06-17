import re
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Raw KNN CV results as provided ---
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
n_splits= 46 → ACC=0.7986±0.0913, F1 =0.7057±0.1302
n_splits= 47 → ACC=0.8013±0.0857, F1 =0.7016±0.1525
n_splits= 48 → ACC=0.8007±0.0909, F1 =0.7012±0.1554
n_splits= 49 → ACC=0.8004±0.0919, F1 =0.7004±0.1575
n_splits= 50 → ACC=0.8027±0.0807, F1 =0.7039±0.1447
n_splits= 51 → ACC=0.8019±0.0750, F1 =0.7002±0.1402
n_splits= 52 → ACC=0.7991±0.0844, F1 =0.6905±0.1705
n_splits= 53 → ACC=0.8004±0.1039, F1 =0.7000±0.1809
n_splits= 54 → ACC=0.8010±0.1021, F1 =0.7025±0.1678
n_splits= 55 → ACC=0.8003±0.0957, F1 =0.6962±0.1664
n_splits= 56 → ACC=0.8036±0.0915, F1 =0.6998±0.1743
n_splits= 57 → ACC=0.8041±0.0926, F1 =0.7016±0.1748
n_splits= 58 → ACC=0.8028±0.0944, F1 =0.6964±0.1819
n_splits= 59 → ACC=0.8023±0.1073, F1 =0.7005±0.1852
n_splits= 60 → ACC=0.8051±0.0955, F1 =0.7099±0.1488
n_splits= 61 → ACC=0.8032±0.0994, F1 =0.7029±0.1739
n_splits= 62 → ACC=0.8021±0.0960, F1 =0.6995±0.1692
n_splits= 63 → ACC=0.8026±0.1044, F1 =0.6914±0.1983
n_splits= 64 → ACC=0.8048±0.1017, F1 =0.7019±0.1720
n_splits= 65 → ACC=0.8041±0.0967, F1 =0.7035±0.1647
n_splits= 66 → ACC=0.8033±0.0958, F1 =0.7044±0.1500
n_splits= 67 → ACC=0.8029±0.0916, F1 =0.7000±0.1543
n_splits= 68 → ACC=0.8022±0.0899, F1 =0.6952±0.1733
n_splits= 69 → ACC=0.8026±0.1003, F1 =0.7042±0.1589
n_splits= 70 → ACC=0.8019±0.0994, F1 =0.7009±0.1607
n_splits= 71 → ACC=0.8021±0.0899, F1 =0.6998±0.1529
n_splits= 72 → ACC=0.8007±0.0940, F1 =0.6957±0.1587
n_splits= 73 → ACC=0.8002±0.1004, F1 =0.6878±0.1961
n_splits= 74 → ACC=0.8023±0.1077, F1 =0.6987±0.1884
n_splits= 75 → ACC=0.8014±0.1079, F1 =0.6983±0.1784
n_splits= 76 → ACC=0.8021±0.1185, F1 =0.7010±0.1929
n_splits= 77 → ACC=0.8024±0.1021, F1 =0.7046±0.1623
n_splits= 78 → ACC=0.8035±0.1107, F1 =0.7041±0.1837
n_splits= 79 → ACC=0.8031±0.1157, F1 =0.7041±0.1790
n_splits= 80 → ACC=0.8003±0.1119, F1 =0.6940±0.1867
n_splits= 81 → ACC=0.8047±0.1070, F1 =0.6981±0.1905
n_splits= 82 → ACC=0.8039±0.1068, F1 =0.6986±0.1869
n_splits= 83 → ACC=0.8043±0.1083, F1 =0.7015±0.1849
n_splits= 84 → ACC=0.8042±0.1047, F1 =0.7000±0.1835
n_splits= 85 → ACC=0.8044±0.1067, F1 =0.7017±0.1823
n_splits= 86 → ACC=0.8041±0.1081, F1 =0.7014±0.1865
n_splits= 87 → ACC=0.8045±0.1147, F1 =0.6967±0.2079
n_splits= 88 → ACC=0.8043±0.1159, F1 =0.6983±0.2026
n_splits= 89 → ACC=0.8046±0.1151, F1 =0.6948±0.2080
n_splits= 90 → ACC=0.8038±0.1141, F1 =0.6968±0.2054
n_splits= 91 → ACC=0.8031±0.1119, F1 =0.6912±0.2062
n_splits= 92 → ACC=0.8024±0.1207, F1 =0.7011±0.1938
n_splits= 93 → ACC=0.8016±0.1283, F1 =0.7011±0.2029
n_splits= 94 → ACC=0.8015±0.1116, F1 =0.6964±0.1814
n_splits= 95 → ACC=0.8049±0.1203, F1 =0.7000±0.1949
n_splits= 96 → ACC=0.8044±0.1214, F1 =0.6964±0.1985
n_splits= 97 → ACC=0.8039±0.1112, F1 =0.6860±0.2135
n_splits= 98 → ACC=0.8034±0.1226, F1 =0.6882±0.2254
n_splits= 99 → ACC=0.8047±0.1172, F1 =0.6864±0.2338
n_splits=100 → ACC=0.8037±0.1142, F1 =0.6853±0.2309
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

ks = np.array(ks)
acc_means = np.array(acc_means)
acc_stds = np.array(acc_stds)

# --- 3. Compute relative change rates (%) ---
rel_delta_acc = np.diff(acc_means) / acc_means[:-1] * 100
rel_delta_std = np.diff(acc_stds)  / acc_stds[:-1]  * 100
k_mid = (ks[:-1] + ks[1:]) / 2

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(k_mid, rel_delta_acc, marker='o', linestyle='-')
axes[0].axhline(0, color='gray', linestyle='--')
axes[0].set_ylabel('Relative Δ ACC (%)')
axes[0].set_title('Relative Change Rate of Accuracy vs. n_splits')

axes[1].plot(k_mid, rel_delta_std, marker='s', linestyle='-')
axes[1].axhline(0, color='gray', linestyle='--')
axes[1].set_xlabel('n_splits')
axes[1].set_ylabel('Relative Δ ACC std (%)')
axes[1].set_title('Relative Change Rate of ACC std vs. n_splits')

# x축 눈금(2~100) 설정
axes[1].set_xticks(ks)                     # 2부터 100까지 모든 n_splits를 눈금으로
axes[1].set_xticklabels(ks, rotation=90, fontsize=6)  # 보기 좋게 90도 회전

plt.tight_layout()
plt.show()
