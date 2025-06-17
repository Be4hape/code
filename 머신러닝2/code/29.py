import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 데이터 로드 & 전처리 ---
train_df = pd.read_csv('process1_result.csv')
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_raw    = train_df[features].values
y        = train_df['Survived'].values

# --- 2. 수동 표준화 ---
means = X_raw.mean(axis=0)
stds  = X_raw.std(axis=0)
stds[stds == 0] = 1.0
X = (X_raw - means) / stds

# --- 3. 시그모이드 정의 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- 4. 하이퍼파라미터 고정 ---
lr         = 0.001
max_epoch  = 17   # 1부터 17까지
m, n       = X.shape

# --- 5. 파라미터 초기화 & 에폭별 정확도 기록 ---
w = np.zeros(n)
b = 0.0
epochs = []
accs   = []

for epoch in range(1, max_epoch + 1):
    # SGD 업데이트
    for xi, yi in zip(X, y):
        z_i   = np.dot(xi, w) + b
        delta = sigmoid(z_i) - yi
        w    -= lr * delta * xi
        b    -= lr * delta

    # 에폭 종료 후 train(process1) 정확도 계산
    z_all  = X.dot(w) + b
    preds  = (sigmoid(z_all) >= 0.5).astype(int)
    acc    = np.mean(preds == y)

    epochs.append(epoch)
    accs.append(acc)
    print(f"Epoch {epoch:2d}: Train ACC = {acc:.4f}")

# --- 6. Epoch 15~17에 대한 정확도 그래프 ---
# 슬라이스로 15,16,17 에폭에 대응하는 인덱스 14,15,16 선택
plot_epochs = epochs[14:17]
plot_accs   = accs  [14:17]

plt.figure(figsize=(6,4))
plt.plot(plot_epochs, plot_accs, marker='o', linestyle='-')
plt.xticks(plot_epochs)
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.title('Train Accuracy at Epochs 15–17 (lr=0.001)')
plt.grid(True)
plt.tight_layout()
plt.show()
