## 교차검증 - logistic R


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# --- 1. 데이터 로드 및 피처/레이블 분리 ---
train_df = pd.read_csv('process1_result.csv')
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_all    = train_df[features].values
y_all    = train_df['Survived'].values

# --- 2. 하이퍼파라미터 & 교차검증 설정 ---
learning_rates = [0.0001, 0.001, 0.01]
num_epochs     = 100
k_folds        = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# --- 3. 시그모이드 정의 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- 4. 교차검증을 통한 학습 & 평가 ---
results = {}  # { lr: { 'accs':[], 'f1s':[] } }
for lr in learning_rates:
    accs = []
    f1s  = []
    print(f"\n=== Learning rate: {lr} ===")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
        # 4.1 Fold별 데이터 분할
        X_train_raw, X_val_raw = X_all[train_idx], X_all[val_idx]
        y_train,     y_val     = y_all[train_idx], y_all[val_idx]

        # 4.2 Fold별 스케일링 (훈련셋으로만 mean/std 계산)
        means = X_train_raw.mean(axis=0)
        stds  = X_train_raw.std(axis=0)
        stds[stds == 0] = 1.0
        X_train = (X_train_raw - means) / stds
        X_val   = (X_val_raw   - means) / stds

        # 4.3 파라미터 초기화
        m, n = X_train.shape
        w = np.zeros(n)
        b = 0.0

        # 4.4 SGD 로지스틱 회귀 학습
        for epoch in range(num_epochs):
            for i in range(m):
                xi, yi = X_train[i], y_train[i]
                z_i     = np.dot(xi, w) + b
                h_i     = sigmoid(z_i)
                delta   = h_i - yi
                w      -= lr * delta * xi
                b      -= lr * delta

        # 4.5 검증셋 평가
        z_val = X_val.dot(w) + b
        preds = (sigmoid(z_val) >= 0.5).astype(int)
        acc   = accuracy_score(y_val, preds)
        f1    = f1_score(y_val, preds)

        accs.append(acc)
        f1s .append(f1)
        print(f" Fold {fold:>2d} → ACC: {acc:.4f},  F1: {f1:.4f}")

    # 4.6 Fold 결과 요약
    acc_mean, acc_std = np.mean(accs), np.std(accs)
    f1_mean,  f1_std  = np.mean(f1s ), np.std(f1s )
    results[lr] = {
        'acc_mean': acc_mean, 'acc_std': acc_std,
        'f1_mean':  f1_mean,  'f1_std':  f1_std
    }
    print(f" → Mean ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f" → Mean F1 : {f1_mean :.4f} ± {f1_std :.4f}")

# --- 5. 최적 학습률 선택 후 전체 데이터로 재학습 (예시) ---
best_lr = max(results, key=lambda x: results[x]['f1_mean'])
print(f"\n최적 학습률은 {best_lr} (F1 mean 기준)입니다.")
w_final = np.zeros(X_all.shape[1])
b_final = 0.0

# 전체 데이터 스케일링
means = X_all.mean(axis=0)
stds  = X_all.std(axis=0)
stds[stds == 0] = 1.0
X_scaled_all = (X_all - means) / stds

# 전체 데이터로 학습
for epoch in range(num_epochs):
    for xi, yi in zip(X_scaled_all, y_all):
        z_i   = np.dot(xi, w_final) + b_final
        h_i   = sigmoid(z_i)
        delta = h_i - yi
        w_final -= best_lr * delta * xi
        b_final -= best_lr * delta

print("\n전체 데이터로 재학습 완료된 w, b 파라미터를 최종 모델로 사용합니다.")
