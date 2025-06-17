## pca to knn, acc and f1 score

import numpy as np
import pandas as pd

# --- 1. 데이터 로드 & 스케일링 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features      = ['Sex','SibSp','Parch','Embarked','TicketNumeric']
X_train       = train_df[features].values
y_train       = train_df['Survived'].values
X_test        = test_df[features].values
passenger_ids = test_df['PassengerId'].values

# 1) zero-mean, unit-variance 스케일링
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0
X_train_scaled = (X_train - means) / stds
X_test_scaled  = (X_test  - means) / stds

# --- 2. PCA 고유벡터 계산 & k_pca=4 주성분 선택 ---
cov_mat         = np.cov(X_train_scaled, rowvar=False)
eigvals, eigvecs= np.linalg.eigh(cov_mat)
idx             = np.argsort(eigvals)[::-1]
V_k             = eigvecs[:, idx[:4]]   # (d,4)

# --- 3. PCA 투사 ---
X_train_pca = X_train_scaled.dot(V_k)
X_test_pca  = X_test_scaled.dot(V_k)

# --- 4. KNN 예측 함수 정의 ---
def euclid_dist_matrix(A, B):
    return np.sqrt(((A[:,None,:] - B[None,:,:])**2).sum(axis=2))

def knn_predict(train_X, train_y, test_X, k_neighbors=20):
    D = euclid_dist_matrix(test_X, train_X)
    neigh = np.argpartition(D, k_neighbors, axis=1)[:,:k_neighbors]
    preds = []
    for idxs in neigh:
        # 다수결
        counts = np.bincount(train_y[idxs])
        preds.append(np.argmax(counts))
    return np.array(preds)

# --- 5. Test1 (Train 데이터) 예측 & 평가 ---
k_neighbors = 20
preds_train = knn_predict(X_train_pca, y_train, X_train_pca, k_neighbors)

# 정확도 계산
accuracy = np.mean(preds_train == y_train)

# F1-score 계산 (binary)
tp = np.sum((preds_train == 1) & (y_train == 1))
fp = np.sum((preds_train == 1) & (y_train == 0))
fn = np.sum((preds_train == 0) & (y_train == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"Test1 Accuracy: {accuracy*100:.2f}%")
print(f"Test1 F1-score : {f1_score*100:.2f}%")

# --- 6. Test2 예측 & 캐글 제출 파일 생성 ---
preds_test = knn_predict(X_train_pca, y_train, X_test_pca, k_neighbors)

submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    preds_test
})
submission_df.to_csv('knn_pca_k4_test2.csv', index=False)
print("Saved: knn_pca_k4_test2.csv")
