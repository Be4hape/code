import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 데이터 로드 & 스케일링 ---
train_df = pd.read_csv('process1_result.csv')
features      = ['Sex','SibSp','Parch','Embarked','TicketNumeric']
X_train       = train_df[features].values
y_train       = train_df['Survived'].values

# 1. 데이터 zero centering (평균 제거) 및 unit-variance 스케일링
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds==0] = 1.0
Z = (X_train - means) / stds    # 중앙화·표준화된 데이터

# --- 2. PCA 고유벡터 한 번만 계산 ---
# 2. 공분산 행렬 계산
cov_mat = np.cov(Z, rowvar=False)       # (d, d)

# 3. 고유분해 (eigen decomposition)
eigvals, eigvecs = np.linalg.eigh(cov_mat)

print(eigvals)


# 4. 고유값 내림차순 정렬 (λ1 ≥ λ2 ≥ …)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

# --- 3. k별 PCA 투사 → 복원 → MSE 계산 → SGD 로지스틱 학습 → Test1 정확도 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_sgd(X, y, lr=0.001, epochs=9):
    m, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        for i in range(m):
            xi, yi = X[i], y[i]
            z = xi.dot(w) + b
            h = sigmoid(z)
            delta = h - yi
            w -= lr * delta * xi
            b -= lr * delta
    return w, b

ks = list(range(1, eigvecs.shape[1] + 1))
mses = []
accuracies = []

for k in ks:
    # 5. 상위 k 주성분(고유벡터) 선택
    V_k = eigvecs[:, :k]             # (d, k)

    # 6. 저차원 투사
    T = Z.dot(V_k)                   # (n, k)

    # 7. 데이터 복원
    Z_hat = T.dot(V_k.T)             # (n, d)

    # 8. 재구성 오차(MSE) 계산
    mse = np.mean((Z - Z_hat) ** 2)
    mses.append(mse)

    # --- 이하 Test1 학습 및 평가 (기존 코드) ---
    w, b = train_logistic_sgd(T, y_train, lr=0.001, epochs=9)
    probs = sigmoid(T.dot(w) + b)
    preds = (probs >= 0.5).astype(int)
    acc   = np.mean(preds == y_train) * 100
    accuracies.append(acc)

    print(f"k={k:2d} → MSE={mse:.4f}, Test1 ACC={acc:.2f}%")

# --- 4. k vs. MSE & ACC 그래프 ---
fig, ax1 = plt.subplots(figsize=(8,4))

color = 'tab:blue'
ax1.set_xlabel('Number of PCA components (k)')
ax1.set_ylabel('MSE', color=color)
ax1.plot(ks, mses, marker='o', color=color, label='Reconstruction MSE')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Test1 ACC (%)', color=color)
ax2.plot(ks, accuracies, marker='s', color=color, label='Test1 Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('PCA Components vs. Reconstruction MSE & Accuracy')
plt.show()
