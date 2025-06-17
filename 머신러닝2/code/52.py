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
indices = np.arange(X_full.shape[0])
np.random.shuffle(indices)

split_idx = int(0.8 * len(indices))
train_idx = indices[:split_idx]
test1_idx = indices[split_idx:]

X_train_full = X_full[train_idx]
y_train_full = y_full[train_idx]
X_test1_full = X_full[test1_idx]
y_test1_full = y_full[test1_idx]

# --- 3. Zero-centering + Unit-variance 스케일링 (Train 기준) ---
means = X_train_full.mean(axis=0)    # (5,)
stds  = X_train_full.std(axis=0)     # (5,)
stds[stds == 0] = 1.0

X_train_full_scaled = (X_train_full - means) / stds
X_test1_full_scaled = (X_test1_full - means) / stds

# --- 4. 은닉층 2개일 때 노드 수 조합 & 정확도 저장용 리스트 ---
# 첫 번째 은닉층 후보 노드 수
candidates = [16, 32, 64, 128]

results = []

for h1 in candidates:
    for h2 in candidates:
        # 두 번째 층 노드 수는 첫 층보다 작거나 같은 경우만 시도
        if h2 > h1:
            continue

        mlp = MLPClassifier(
            hidden_layer_sizes=(h1, h2),  # 은닉층 2개: (h1, h2)
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
        mlp.fit(X_train_full_scaled, y_train_full)

        # 4-2. Test1 예측
        preds = mlp.predict(X_test1_full_scaled)

        # 4-3. 정확도 계산
        acc = accuracy_score(y_test1_full, preds)

        results.append((h1, h2, acc))
        print(f"hidden_layer_sizes=({h1},{h2})  →  Test1 Accuracy = {acc*100:.2f}%")

# --- 5. 결과 정리 및 상위 5개 출력 ---
results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

print("\n=== 상위 5개 조합 (h1, h2, Accuracy) ===")
for h1, h2, acc in results_sorted[:5]:
    print(f"({h1}, {h2})  →  Test1 Accuracy = {acc*100:.2f}%")
