import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics       import accuracy_score, f1_score

# --- 1. 데이터 로드 & 수동 스케일링 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')
features      = ['Sex','SibSp','Parch','Embarked','TicketNumeric']
X = train_df[features].values
y = train_df['Survived'].values
X_test = test_df[features].values
ids    = test_df['PassengerId'].values

# train 데이터로만 평균·표준편차 계산
means = X.mean(axis=0)
stds  = X.std(axis=0)
stds[stds==0] = 1.0

X_scaled      = (X      - means) / stds
X_test_scaled = (X_test - means) / stds

# --- 2. 거리 행렬 계산 함수 ---
def euclid_dist_matrix(A, B):
    # A: (n, d), B: (m, d) → (n, m)
    return np.sqrt(((A[:,None,:] - B[None,:,:])**2).sum(axis=2))

# 전체 train↔train 거리 행렬
D_full = euclid_dist_matrix(X_scaled, X_scaled)

# --- 3. Manual k-NN 예측 함수 ---
def knn_predict_batch(D, y_train, k):
    # D: (n_samples, n_train) 거리 행렬
    # return: 길이 n_samples 예측 벡터
    idx = np.argpartition(D, kth=k, axis=1)[:,:k]
    return np.array([ np.bincount(y_train[row]).argmax() for row in idx ])

# --- 4. 37-fold CV with k=20 ---
k_neighbors = 20
n_splits    = 37
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

accs, f1s = [], []
print(f"--- {n_splits}-Fold CV (k_neighbors={k_neighbors}) ---")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_scaled, y), start=1):
    # train/val 거리 서브행렬
    D_va_tr = D_full[va_idx][:, tr_idx]
    y_tr    = y[tr_idx]

    # 예측 및 평가
    preds = knn_predict_batch(D_va_tr, y_tr, k_neighbors)
    acc   = accuracy_score(y[va_idx], preds)
    f1    = f1_score    (y[va_idx], preds)

    accs.append(acc)
    f1s .append(f1)
    print(f"Fold {fold:2d} — ACC: {acc:.4f}, F1: {f1:.4f}")

# CV 결과 요약
print("\n=== Cross-Validation Results ===")
print(f"ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 : {np.mean(f1s ): .4f} ± {np.std(f1s):.4f}")

# --- 5. 최종 예측: 전체 train↔test 거리로 k-NN 적용 ---
D_test = euclid_dist_matrix(X_test_scaled, X_scaled)
final_preds = knn_predict_batch(D_test, y, k_neighbors)

submission = pd.DataFrame({
    'PassengerId': ids,
    'Survived':    final_preds
})
submission.to_csv('knn_manual_k20_cv37_submission.csv', index=False)
print("\nSaved: knn_manual_k20_cv37_submission.csv")
