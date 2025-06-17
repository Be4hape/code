import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# --- 1. 데이터 로드 및 피처/레이블 분리 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features      = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_all         = train_df[features].values
y_all         = train_df['Survived'].values
X_test_raw    = test_df[features].values
passenger_ids = test_df['PassengerId'].values

# --- 2. 시그모이드 정의 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- 3. 표준화 함수 (훈련셋으로만 fit → train/val/test 변환) ---
def standardize(train, other):
    mean = train.mean(axis=0)
    std  = train .std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (other - mean) / std

# --- 4. 하이퍼파라미터 고정 ---
lr         = 0.001   # 학습률
num_epochs = 9       # 에폭 수
k_folds    = 18      # k 값

# --- 5. 18-fold 교차검증 & 평가 ---
skf    = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
accs   = []
f1s    = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
    # 5.1. 분할
    X_tr_raw, y_tr = X_all[train_idx], y_all[train_idx]
    X_val_raw, y_val = X_all[val_idx], y_all[val_idx]

    # 5.2. 표준화
    X_tr, X_val = standardize(X_tr_raw, X_val_raw)

    # 5.3. 파라미터 초기화
    m, n = X_tr.shape
    w = np.zeros(n)
    b = 0.0

    # 5.4. SGD 학습
    for epoch in range(num_epochs):
        for xi, yi in zip(X_tr, y_tr):
            z_i   = np.dot(xi, w) + b
            delta = sigmoid(z_i) - yi
            w    -= lr * delta * xi
            b    -= lr * delta

    # 5.5. 검증
    preds = (sigmoid(X_val.dot(w) + b) >= 0.5).astype(int)
    acc   = accuracy_score(y_val, preds)
    f1    = f1_score(y_val, preds)
    accs.append(acc)
    f1s .append(f1)
    print(f"Fold {fold:2d} — ACC: {acc:.4f}, F1: {f1:.4f}")

# 5.6. Fold별 결과 평균·표준편차
print("\n=== Cross-Validation Results (k=18) ===")
print(f"ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 : {np.mean(f1s ):.4f} ± {np.std(f1s ):.4f}")

# --- 6. 최종 모델: 전체 데이터로 재학습 ---
# 6.1. 전체 훈련 데이터 표준화
X_all_scaled, X_test_scaled = standardize(X_all, X_test_raw)

# 6.2. 파라미터 초기화
w_final = np.zeros(X_all_scaled.shape[1])
b_final = 0.0

# 6.3. SGD 학습
for epoch in range(num_epochs):
    for xi, yi in zip(X_all_scaled, y_all):
        z_i   = np.dot(xi, w_final) + b_final
        delta = sigmoid(z_i) - yi
        w_final -= lr * delta * xi
        b_final -= lr * delta

# --- 7. 테스트 예측 & 제출 파일 저장 ---
z_test      = X_test_scaled.dot(w_final) + b_final
predictions = (sigmoid(z_test) >= 0.5).astype(int)

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    predictions
})
submission.to_csv('submission_k18_logistic_sgd.csv', index=False)
print("\nSaved: submission_k18_logistic_sgd.csv")
