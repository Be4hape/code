import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & full feature 추출 ---
df = pd.read_csv('process1_result.csv')
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X = df[features].values    # (n_samples, 5)
y = df['Survived'].values  # (n_samples,)

# --- 2. 데이터 셔플 & Train/Test1 분할 (80:20) ---
np.random.seed(42)
idx = np.arange(len(X))
np.random.shuffle(idx)

split = int(0.8 * len(idx))
train_idx = idx[:split]
test1_idx = idx[split:]

X_train = X[train_idx]
y_train = y[train_idx]
X_test1 = X[test1_idx]
y_test1 = y[test1_idx]

# --- 3. Zero‐centering & Unit‐variance 스케일링 (Train 기준) ---
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0

X_train_scaled = (X_train - means) / stds
X_test1_scaled = (X_test1 - means) / stds

# --- 4. 은닉층 1개, 노드 수 27~37 탐색 & Test1 Accuracy 계산 ---
results = []
for h in range(27, 38):  # 27, 28, ..., 37
    mlp = MLPClassifier(
        hidden_layer_sizes=(h,),
        activation='relu',
        solver='adam',
        learning_rate='constant',
        learning_rate_init=0.001,
        alpha=1e-4,
        batch_size=32,
        max_iter=200,
        shuffle=True,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    preds = mlp.predict(X_test1_scaled)
    acc = accuracy_score(y_test1, preds)
    results.append((h, acc))
    print(f"hidden_layer_sizes=({h},)  →  Test1 Accuracy = {acc*100:.2f}%")

# --- 5. 결과 정리 ---
print("\n=== 탐색 결과 요약 ===")
for h, acc in results:
    print(f"노드 {h:2d}개  →  Test1 Accuracy = {acc*100:.2f}%")
