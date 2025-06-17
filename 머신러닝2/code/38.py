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

# train 기준으로 평균·표준편차 계산
means = X.mean(axis=0)
stds  = X.std(axis=0)
stds[stds==0] = 1.0
X_scaled     = (X      - means) / stds

# --- 2. 거리 행렬 미리 계산 (Train vs Train) ---
def euclid_dist_matrix(A, B):
    return np.sqrt(((A[:,None,:] - B[None,:,:])**2).sum(axis=2))

D_full = euclid_dist_matrix(X_scaled, X_scaled)

# --- 3. KNN 배치 예측 함수 (k=20 고정) ---
k = 20
def knn_predict_batch(D, y_train, k):
    # D: (n_samples, n_train) 거리 행렬
    idx = np.argpartition(D, kth=k, axis=1)[:, :k]
    # 각 행마다 다수결로 예측
    return np.array([ np.bincount(y_train[row]).argmax() for row in idx ])

# --- 4. n_splits = 2부터 100까지 교차검증 반복 ---
results = {}
for n_splits in range(2, 101):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []

    for train_idx, val_idx in skf.split(X_scaled, y):
        # Fold별 거리 서브셋
        D_va_tr = D_full[val_idx][:, train_idx]
        y_tr    = y[train_idx]

        # 예측 & 평가
        preds = knn_predict_batch(D_va_tr, y_tr, k)
        accs.append( accuracy_score(y[val_idx], preds) )
        f1s .append( f1_score   (y[val_idx], preds) )

    results[n_splits] = {
        'acc_mean': np.mean(accs),
        'acc_std':  np.std(accs),
        'f1_mean':  np.mean(f1s),
        'f1_std':   np.std(f1s)
    }
    print(f"n_splits={n_splits:3d} → "
          f"ACC={results[n_splits]['acc_mean']:.4f}±{results[n_splits]['acc_std']:.4f}, "
          f"F1 ={results[n_splits]['f1_mean']:.4f}±{results[n_splits]['f1_std']:.4f}")

# --- 5. 최적 n_splits 선택 (ACC mean 기준) ---
best_splits = max(results, key=lambda s: results[s]['acc_mean'])
print(f"\n▶ Optimal number of folds: {best_splits} (ACC mean = {results[best_splits]['acc_mean']:.4f})")

# --- 6. 최종 모델(k=20, best_splits)로 테스트 예측 & 저장 ---
# 테스트 데이터 거리 계산
X_test  = test_df[features].values
X_test_scaled = (X_test - means) / stds
D_test = euclid_dist_matrix(X_test_scaled, X_scaled)

final_preds = knn_predict_batch(D_test, y, k)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].values,
    'Survived':    final_preds
})
submission.to_csv('knn_manual_k20_cv_to100.csv', index=False)
print("Saved: knn_manual_k20_cv_to100.csv")
