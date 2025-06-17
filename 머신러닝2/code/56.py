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

X_train_s = (X_train - means) / stds
X_test1_s = (X_test1 - means) / stds

# --- 4. alpha(정규화 계수) 후보 리스트 설정 ---
alphas = [0.0001]

# 은닉층 1개, 노드 30 고정, learning_rate_init=0.001 고정
hidden = (30,)
lr_init = 0.005

results = []
for alpha in alphas:
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation='relu',
        solver='adam',
        learning_rate='constant',
        learning_rate_init=lr_init,
        alpha=alpha,
        batch_size=32,
        max_iter=300,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    # Train 학습
    mlp.fit(X_train_s, y_train)
    # Test1 예측
    preds1 = mlp.predict(X_test1_s)
    acc1 = accuracy_score(y_test1, preds1)
    results.append((alpha, acc1))
    print(f"alpha={alpha:.5f}  →  Test1 Accuracy = {acc1*100:.2f}%")

# --- 5. 결과 요약 ---
print("\n=== alpha별 Test1 Accuracy 요약 ===")
for alpha, acc in results:
    print(f"alpha={alpha:.5f}  →  {acc*100:.2f}%")
