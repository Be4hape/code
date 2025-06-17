import numpy as np
import pandas as pd

# --- 1. 데이터 로드 & 스케일링 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features      = ['Sex','SibSp','Parch','Embarked','TicketNumeric']
X_train       = train_df[features].values   # (n_train, d)
y_train       = train_df['Survived'].values # (n_train,)
X_test        = test_df[features].values    # (n_test,  d)
passenger_ids = test_df['PassengerId'].values

# 1) zero-centering & unit-variance 스케일링
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0
Z_train = (X_train - means) / stds         # (n_train, d)
Z_test  = (X_test  - means) / stds         # (n_test,  d)

# --- 2. FLD 축(w) 계산 ---
# 2.1 클래스별 평균
mu0 = Z_train[y_train == 0].mean(axis=0)   # (d,)
mu1 = Z_train[y_train == 1].mean(axis=0)   # (d,)

# 2.2 클래스 내 산포 행렬 SW
d   = Z_train.shape[1]
SW  = np.zeros((d, d))
for z, y in zip(Z_train, y_train):
    diff = (z - (mu1 if y == 1 else mu0)).reshape(-1,1)
    SW += diff.dot(diff.T)

# 2.3 판별 벡터 w ∝ SW^{-1}(μ1−μ0)
w_fld = np.linalg.inv(SW).dot(mu1 - mu0)
w_fld /= np.linalg.norm(w_fld)            # 정규화

# --- 3. 1D 투사 ---
z_train = Z_train.dot(w_fld).reshape(-1,1)  # (n_train, 1)
z_test  = Z_test.dot(w_fld).reshape(-1,1)   # (n_test, 1)

# --- 4. 1D KNN 예측 함수 정의 ---
def knn_predict_1d(train_z, train_y, query_z, k=20):
    """1D 거리 기반 KNN 예측"""
    preds = []
    for q in query_z[:,0]:
        # 거리 계산
        dists = np.abs(train_z[:,0] - q)
        # 최근접 k개 인덱스
        idxs = np.argpartition(dists, k)[:k]
        # 다수결 투표
        counts = np.bincount(train_y[idxs])
        preds.append(np.argmax(counts))
    return np.array(preds)

# --- 5. Test1 (Train) 예측 & 평가 ---
k_neighbors = 20
preds_train = knn_predict_1d(z_train, y_train, z_train, k=k_neighbors)

# Accuracy
acc_train = np.mean(preds_train == y_train) * 100

# F1-score (binary 수동 계산)
tp = np.sum((preds_train == 1) & (y_train == 1))
fp = np.sum((preds_train == 1) & (y_train == 0))
fn = np.sum((preds_train == 0) & (y_train == 1))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_train  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
f1_train *= 100

print(f"Test1 Accuracy: {acc_train:.2f}%")
print(f"Test1 F1-score : {f1_train:.2f}%")

# --- 6. Test2 예측 & 제출 파일 생성 ---
preds_test = knn_predict_1d(z_train, y_train, z_test, k=k_neighbors)

submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    preds_test
})
submission_df.to_csv('submission_knn_fld.csv', index=False)
print("Saved: submission_knn_fld.csv")
