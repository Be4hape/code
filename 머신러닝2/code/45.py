## pca to knn

import numpy as np
import pandas as pd

# --- 1. 데이터 로드 & 수동 스케일링 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features      = ['Sex','SibSp','Parch','Embarked','TicketNumeric']
X_train       = train_df[features].values
y_train       = train_df['Survived'].values
X_test        = test_df[features].values
passenger_ids = test_df['PassengerId'].values

# 1. 데이터 zero-centering & unit-variance 스케일링
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0
X_train_scaled = (X_train - means) / stds
X_test_scaled  = (X_test  - means) / stds

# --- 2. PCA 고유벡터 계산 & k_pca=4 주성분 선택 ---
cov_mat      = np.cov(X_train_scaled, rowvar=False)   # (d, d)
eigvals, eigvecs = np.linalg.eigh(cov_mat)            # 공분산 행렬 고유분해
idx          = np.argsort(eigvals)[::-1]              # 고유값 내림차순 인덱스
eigvecs      = eigvecs[:, idx]

k_pca = 4
V_k   = eigvecs[:, :k_pca]    # (d, 4) 주성분 행렬

# --- 3. PCA 투사 ---
X_train_pca = X_train_scaled.dot(V_k)   # (n_train, 4)
X_test_pca  = X_test_scaled.dot(V_k)    # (n_test,  4)

# --- 4. KNN 예측 함수 정의 (k_neighbors=20) ---
def euclid_dist_matrix(A, B):
    # A: (n_a, d), B: (n_b, d)
    # 반환 D: (n_a, n_b) 거리 행렬
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

def knn_predict(train_X, train_y, test_X, k_neighbors=20):
    # 거리 계산
    D = euclid_dist_matrix(test_X, train_X)
    # 각 테스트 샘플에 대해 k_neighbors개의 최근접 이웃 예측
    neighbor_idxs = np.argpartition(D, k_neighbors, axis=1)[:, :k_neighbors]
    preds = np.array([
        np.bincount(train_y[idxs]).argmax()
        for idxs in neighbor_idxs
    ])
    return preds

# --- 5. Test2 예측 & 제출 파일 생성 ---
k_neighbors = 20
preds_test = knn_predict(
    train_X= X_train_pca,
    train_y= y_train,
    test_X = X_test_pca,
    k_neighbors = k_neighbors
)

submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    preds_test
})
submission_df.to_csv('knn_pca_k4_test2.csv', index=False)
print("Saved: knn_pca_k4_test2.csv")
