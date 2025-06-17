import re
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Decision Tree k-fold 결과를 멀티라인 문자열로 붙여넣기 ---
raw = """
k=  2 → ACC=0.7520±0.0031, F1=0.6730±0.0165
k=  3 → ACC=0.7823±0.0141, F1=0.6949±0.0287
k=  4 → ACC=0.7879±0.0172, F1=0.7104±0.0319
k=  5 → ACC=0.7800±0.0259, F1=0.6978±0.0380
k=  6 → ACC=0.7677±0.0172, F1=0.6849±0.0322
k=  7 → ACC=0.7868±0.0389, F1=0.7098±0.0508
k=  8 → ACC=0.7846±0.0185, F1=0.7036±0.0315
k=  9 → ACC=0.7912±0.0333, F1=0.7111±0.0542
k= 10 → ACC=0.7890±0.0349, F1=0.7104±0.0542
k= 11 → ACC=0.7834±0.0339, F1=0.7021±0.0588
k= 12 → ACC=0.7902±0.0407, F1=0.7089±0.0623
k= 13 → ACC=0.7902±0.0255, F1=0.7056±0.0491
k= 14 → ACC=0.7958±0.0374, F1=0.7180±0.0620
k= 15 → ACC=0.7869±0.0370, F1=0.7061±0.0556
k= 16 → ACC=0.7869±0.0375, F1=0.7064±0.0600
k= 17 → ACC=0.7858±0.0431, F1=0.7045±0.0629
k= 18 → ACC=0.7868±0.0498, F1=0.7027±0.0743
k= 19 → ACC=0.7822±0.0395, F1=0.6996±0.0587
k= 20 → ACC=0.7925±0.0524, F1=0.7141±0.0732
k= 21 → ACC=0.7903±0.0594, F1=0.7085±0.0874
k= 22 → ACC=0.7780±0.0578, F1=0.6927±0.0791
k= 23 → ACC=0.7856±0.0620, F1=0.7014±0.0875
k= 24 → ACC=0.7856±0.0596, F1=0.7048±0.0786
k= 25 → ACC=0.7778±0.0638, F1=0.6932±0.0871
k= 26 → ACC=0.7890±0.0557, F1=0.7067±0.0736
k= 27 → ACC=0.7912±0.0656, F1=0.7125±0.0844
k= 28 → ACC=0.7845±0.0781, F1=0.7019±0.1058
k= 29 → ACC=0.7766±0.0770, F1=0.6947±0.0994
k= 30 → ACC=0.7833±0.0678, F1=0.6971±0.0967
"""

# --- 2. 정규표현식으로 k, acc_mean, acc_std 추출 ---
pattern = r"k\s*=\s*(\d+)\s*→\s*ACC\s*=\s*([0-9.]+)±([0-9.]+),"
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

# --- 3. 상대 변화율(%): successive differences ---
rel_delta_acc = np.diff(acc_means) / acc_means[:-1] * 100
rel_delta_std = np.diff(acc_stds)  / acc_stds[:-1]  * 100
k_mid         = (ks[:-1] + ks[1:]) / 2

# --- 4. 2D 변화율 시각화 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax1.plot(k_mid, rel_delta_acc, marker='o', linestyle='-')
ax1.axhline(0, color='gray', linestyle='--')
ax1.set_ylabel('Δ Accuracy (%)')
ax1.set_title('Relative Change Rate of ACC Mean vs. k')

ax2.plot(k_mid, rel_delta_std, marker='s', linestyle='-')
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_xlabel('k (Number of Folds)')
ax2.set_ylabel('Δ ACC Std (%)')
ax2.set_title('Relative Change Rate of ACC Std vs. k')

plt.tight_layout()
plt.show()
