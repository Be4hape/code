import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & full feature 추출 ---
df1 = pd.read_csv('process1_result.csv')   # Train + Test1
df2 = pd.read_csv('process2_result.csv')   # Test2 (제출용)

features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X1 = df1[features].values    # (n_samples, 5)
y1 = df1['Survived'].values  # (n_samples,)
X2 = df2[features].values    # (n_test2, 5)
ids2 = df2['PassengerId'].values  # Test2 제출용 ID

# --- 2. Train/Test1 분할 (80:20) ---
np.random.seed(42)
idx_all = np.arange(len(X1))
np.random.shuffle(idx_all)

split = int(0.8 * len(idx_all))
train_idx = idx_all[:split]
test1_idx = idx_all[split:]

X_train = X1[train_idx]
y_train = y1[train_idx]
X_test1 = X1[test1_idx]
y_test1 = y1[test1_idx]

# --- 3. Zero‐centering & Unit‐variance 스케일링 (Train 기준) ---
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0

X_train_s = (X_train - means) / stds
X_test1_s = (X_test1 - means) / stds
X2_s      = (X2 - means) / stds

# --- 4. MLP 모델 정의 (은닉층=1, 노드=32; lr=0.005; alpha=0.0001; max_iter=300) ---
mlp = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation='relu',
    solver='adam',
    learning_rate='constant',
    learning_rate_init=0.005,
    alpha=1e-4,
    batch_size=32,
    max_iter=300,
    shuffle=True,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

# --- 5. Train 학습 ---
mlp.fit(X_train_s, y_train)

# --- 6. Test1 예측 & 정확도 출력 ---
preds_test1 = mlp.predict(X_test1_s)
acc_test1 = accuracy_score(y_test1, preds_test1)
print(f"Full feature MLP → Test1 Accuracy: {acc_test1*100:.2f}%")

# --- 7. Test2 예측 → Submission 파일 생성 ---
preds_test2 = mlp.predict(X2_s)
submission = pd.DataFrame({
    'PassengerId': ids2,
    'Survived':    preds_test2
})
submission.to_csv('submission_mlp_full.csv', index=False)
print("Saved: submission_mlp_full.csv")
