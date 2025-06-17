## 에폭스는 9부터 17까지 tets1에 대한 정확도가 같으므로, 과적합을 피하기 위해 초기에 잡았던 목표에 대해서
## 에폭스를 9로 지정하기로 한다.
## 이후 교차검증을 k=2 부터 m(전체 데이터) 만큼 돌리는 코드
## 너무 오랜시간이 걸리므로, 200정도에서 임의로 멈춘다.
## 추후에 코드를 돌린다면, 코드를 수정해야 할 것.
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# --- 1. 데이터 로드 & 피처/레이블 분리 ---
train_df = pd.read_csv('process1_result.csv')
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_raw    = train_df[features].values
y        = train_df['Survived'].values

# --- 2. 수동 표준화 함수 ---
def standardize(train, test):
    means = train.mean(axis=0)
    stds  = train.std(axis=0)
    stds[stds == 0] = 1.0
    return (train - means) / stds, (test - means) / stds

# --- 3. 시그모이드 정의 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- 4. 하이퍼파라미터 고정 ---
lr         = 0.001
num_epochs = 9

# --- 5. k-fold 교차검증 (k = 2 ~ N) ---
m = X_raw.shape[0]

for k in range(2, m + 1):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accs = []
    fold_f1s  = []

    for train_idx, val_idx in skf.split(X_raw, y):
        X_tr_raw, y_tr = X_raw[train_idx], y[train_idx]
        X_val_raw, y_val = X_raw[val_idx], y[val_idx]

        # 표준화
        X_tr, X_val = standardize(X_tr_raw, X_val_raw)

        # 파라미터 초기화
        w = np.zeros(X_tr.shape[1])
        b = 0.0

        # SGD 학습
        for _ in range(num_epochs):
            for xi, yi in zip(X_tr, y_tr):
                z_i   = np.dot(xi, w) + b
                delta = sigmoid(z_i) - yi
                w    -= lr * delta * xi
                b    -= lr * delta

        # 예측 & 평가
        preds = (sigmoid(X_val.dot(w) + b) >= 0.5).astype(int)
        fold_accs.append(accuracy_score(y_val, preds))
        fold_f1s .append(f1_score(y_val, preds))

    # Fold별 결과 평균·표준편차 출력
    acc_mean, acc_std = np.mean(fold_accs), np.std(fold_accs)
    f1_mean,  f1_std  = np.mean(fold_f1s ), np.std(fold_f1s )
    print(f"k = {k:3d} → ACC = {acc_mean:.4f} ± {acc_std:.4f},  F1 = {f1_mean:.4f} ± {f1_std:.4f}")
