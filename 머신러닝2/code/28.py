##과적합을 피하기 위해, 적절한 에폭에서 멈추기 위해 시각화하는 코드
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 데이터 로드 & 전처리 ---
train_df = pd.read_csv('process1_result.csv')
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_raw    = train_df[features].values
y        = train_df['Survived'].values

# --- 2. 수동 스케일링 (평균0, 표준편차1) ---
means = X_raw.mean(axis=0)
stds  = X_raw.std(axis=0)
stds[stds == 0] = 1.0
X = (X_raw - means) / stds

# --- 3. 시그모이드 & 손실 함수 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, w, b, epsilon=1e-15):
    z = X.dot(w) + b
    h = sigmoid(z)
    return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

# --- 4. 하이퍼파라미터 고정 ---
lr         = 0.001
num_epochs = 40
m, n       = X.shape

# --- 5. 파라미터 초기화 & 학습 ---
w = np.zeros(n)
b = 0.0
costs = []

for epoch in range(1, num_epochs + 1):
    # 샘플 단위 SGD 업데이트
    for xi, yi in zip(X, y):
        z_i   = np.dot(xi, w) + b
        h_i   = sigmoid(z_i)
        delta = h_i - yi
        w    -= lr * delta * xi
        b    -= lr * delta

    # 전체 데이터에 대한 손실 계산
    loss = compute_loss(X, y, w, b)
    costs.append(loss)
    print(f"Epoch {epoch:2d}: loss = {loss:.4f}")

# --- 6. 손실 수렴 그래프 ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), costs, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Convergence (lr=0.001, epochs=20)')
plt.grid(True)
plt.tight_layout()
plt.show()
