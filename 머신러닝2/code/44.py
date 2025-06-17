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

# zero-mean, unit-variance
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0

X_train_scaled = (X_train - means) / stds
X_test_scaled  = (X_test  - means) / stds

# --- 2. PCA 고유벡터 계산 & k=4 주성분 선택 ---
cov_mat      = np.cov(X_train_scaled, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov_mat)
idx          = np.argsort(eigvals)[::-1]
eigvecs      = eigvecs[:, idx]

k = 4
V_k = eigvecs[:, :k]               # (d, 4)

# --- 3. Train/Test2에 PCA 투사 ---
X_train_pca = X_train_scaled.dot(V_k)  # (n_train, 4)
X_test_pca  = X_test_scaled.dot(V_k)   # (n_test,  4)

# --- 4. SGD 로지스틱 회귀 학습 (lr=0.001, epochs=9) ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_sgd(X, y, lr=0.001, epochs=9):
    m, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        for i in range(m):
            xi, yi = X[i], y[i]
            z  = xi.dot(w) + b
            h  = sigmoid(z)
            g  = h - yi
            w -= lr * g * xi
            b -= lr * g
    return w, b

w, b = train_logistic_sgd(X_train_pca, y_train, lr=0.001, epochs=9)

# --- 5. Test2 예측 & 제출 파일 생성 ---
probs_test = sigmoid(X_test_pca.dot(w) + b)
preds_test = (probs_test >= 0.5).astype(int)

submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    preds_test
})
submission_df.to_csv('submission_logistic_pca_k4.csv', index=False)
print("Saved: submission_logistic_pca_k4.csv")
