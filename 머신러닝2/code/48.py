import numpy as np
import pandas as pd

# --- 1. 데이터 로드 & zero-centering·스케일링 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features      = ['Sex','SibSp','Parch','Embarked','TicketNumeric']
X_train       = train_df[features].values   # (n_train, d)
y_train       = train_df['Survived'].values # (n_train,)
X_test        = test_df[features].values    # (n_test, d)
passenger_ids = test_df['PassengerId'].values

# 1) zero-centering & unit-variance 스케일링
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0
Z_train = (X_train - means) / stds         # (n_train, d)
Z_test  = (X_test  - means) / stds         # (n_test,  d)

# --- 2. 클래스별 평균 벡터 계산 ---
mu0 = Z_train[y_train == 0].mean(axis=0)   # (d,)
mu1 = Z_train[y_train == 1].mean(axis=0)   # (d,)

# --- 3. 클래스 내 산포 행렬 SW ---
d = Z_train.shape[1]
SW = np.zeros((d, d))
for z, y in zip(Z_train, y_train):
    diff = (z - (mu1 if y == 1 else mu0)).reshape(-1, 1)
    SW += diff.dot(diff.T)

# --- 4. 클래스 간 산포 행렬 SB ---
diff_mu = (mu1 - mu0).reshape(-1, 1)
SB = diff_mu.dot(diff_mu.T)

# --- 5. 판별 벡터 w 계산 & 정규화 ---
w_fld = np.linalg.inv(SW).dot(mu1 - mu0)   # (d,)
w_fld /= np.linalg.norm(w_fld)

# --- 6. 1차원 투사 ---
z_train = Z_train.dot(w_fld).reshape(-1, 1)  # (n_train, 1)
z_test  = Z_test.dot(w_fld).reshape(-1, 1)   # (n_test, 1)

# --- 7. SGD 로지스틱 회귀 정의 & 학습 (lr=0.001, epochs=9) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_logistic_sgd_1d(z, y, lr=0.001, epochs=9):
    n = z.shape[0]
    w = 0.0   # weight for 1D feature
    b = 0.0   # bias
    for _ in range(epochs):
        for i in range(n):
            zi = z[i,0]
            yi = y[i]
            pred = sigmoid(w * zi + b)
            grad = pred - yi
            w -= lr * grad * zi
            b -= lr * grad
    return w, b

w_log, b_log = train_logistic_sgd_1d(z_train, y_train, lr=0.001, epochs=9)

# --- 8. Test1 (Train) 예측 & 평가 ---
probs_train = sigmoid(w_log * z_train[:,0] + b_log)
preds_train = (probs_train >= 0.5).astype(int)

# 정확도
acc_train = np.mean(preds_train == y_train) * 100

# F1-score (수동 계산)
tp = np.sum((preds_train == 1) & (y_train == 1))
fp = np.sum((preds_train == 1) & (y_train == 0))
fn = np.sum((preds_train == 0) & (y_train == 1))
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
f1_train  = f1_score * 100

print(f"Test1 Accuracy: {acc_train:.2f}%")
print(f"Test1 F1-score : {f1_train:.2f}%")

# --- 9. Test2 예측 & CSV 저장 ---
probs_test = sigmoid(w_log * z_test[:,0] + b_log)
preds_test = (probs_test >= 0.5).astype(int)

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    preds_test
})
submission.to_csv('submission_logistic_fld.csv', index=False)
print("Saved: submission_logistic_fld.csv")
