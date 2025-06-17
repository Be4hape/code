import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & full feature 추출 ---
df = pd.read_csv('process1_result.csv')

features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_full = df[features].values    # (n_samples, 5)
y_full = df['Survived'].values  # (n_samples,)

# --- 2. 데이터 셔플 & Train/Test1 분할 (80:20) ---
np.random.seed(42)
idx = np.arange(X_full.shape[0])
np.random.shuffle(idx)

split = int(0.8 * len(idx))
train_idx = idx[:split]
test1_idx = idx[split:]

X_train = X_full[train_idx]
y_train = y_full[train_idx]
X_test1 = X_full[test1_idx]
y_test1 = y_full[test1_idx]

# --- 3. Zero‐centering & Unit‐variance 스케일링 (Train 기준) ---
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0

X_train_scaled = (X_train - means) / stds
X_test1_scaled = (X_test1 - means) / stds

# --- 4. 은닉층 3개짜리 MLP: (h1 ≥ h2 ≥ h3) 조합 실험 ---
candidates = [16, 32, 64, 128]
results = []

for h1 in candidates:
    for h2 in candidates:
        if h2 > h1:
            continue
        for h3 in candidates:
            if h3 > h2:
                continue

            mlp = MLPClassifier(
                hidden_layer_sizes=(h1, h2, h3),
                activation='relu',
                solver='adam',
                learning_rate='constant',
                learning_rate_init=0.001,
                alpha=1e-4,
                batch_size=32,
                max_iter=200,
                shuffle=True,
                random_state=42,
                early_stopping=False
            )

            # 4-1. Train 학습
            mlp.fit(X_train_scaled, y_train)

            # 4-2. Test1 예측
            preds = mlp.predict(X_test1_scaled)

            # 4-3. 정확도 계산
            acc = accuracy_score(y_test1, preds)
            results.append((h1, h2, h3, acc))

            print(f"hidden_layer_sizes=({h1}, {h2}, {h3})  →  Test1 Accuracy = {acc*100:.2f}%")

# --- 5. 결과 정리: 상위 5개 조합 출력 ---
results_sorted = sorted(results, key=lambda x: x[3], reverse=True)

print("\n=== 상위 5개 조합 (h1, h2, h3, Accuracy) ===")
for h1, h2, h3, acc in results_sorted[:5]:
    print(f"({h1}, {h2}, {h3})  →  Test1 Accuracy = {acc*100:.2f}%")
