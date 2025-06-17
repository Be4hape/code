import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & full feature 추출 ---
df = pd.read_csv('process1_result.csv')

# full feature로 사용할 피처 이름
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']

# 특성 행렬 X_full, 라벨 벡터 y_full
X_full = df[features].values    # shape: (n_samples, 5)
y_full = df['Survived'].values  # shape: (n_samples,)

# --- 2. 데이터 셔플 & Train/Test1 분할 (80:20) ---
np.random.seed(42)
indices = np.arange(X_full.shape[0])
np.random.shuffle(indices)

# 80%를 train, 20%를 test1로 분할
split_idx = int(0.8 * len(indices))
train_idx = indices[:split_idx]
test1_idx = indices[split_idx:]

X_train_full = X_full[train_idx]
y_train_full = y_full[train_idx]
X_test1_full = X_full[test1_idx]
y_test1_full = y_full[test1_idx]

# --- 3. Zero-centering + Unit-variance 스케일링 (Train 기반) ---
means = X_train_full.mean(axis=0)     # (5,)
stds  = X_train_full.std(axis=0)      # (5,)
stds[stds == 0] = 1.0                 # 혹시 표준편차가 0인 열 방지

X_train_full_scaled = (X_train_full - means) / stds
X_test1_full_scaled = (X_test1_full - means) / stds

# --- 4. 은닉층 1개일 때 뉴런 수 목록 & 정확도 저장용 리스트 ---
hidden_sizes = [16, 32, 64, 128]
results = []

for h in hidden_sizes:
    # MLPClassifier 정의
    mlp = MLPClassifier(
        hidden_layer_sizes=(h,),     # 은닉층 1개, 뉴런 = h
        activation='relu',           # ReLU 활성화
        solver='adam',               # Adam 옵티마이저
        learning_rate='constant',    # 학습률 고정
        learning_rate_init=0.001,    # 초기 학습률 = 0.001 (필요시 변경)
        alpha=1e-4,                  # L2 정규화 계수
        batch_size=32,               # 배치 크기
        max_iter=200,                # 최대 에포크
        shuffle=True,                # 에포크마다 데이터 셔플
        random_state=42,
        early_stopping=False         # (원한다면 True로 켜도 됨)
    )

    # 4-1. Train 데이터 학습
    mlp.fit(X_train_full_scaled, y_train_full)

    # 4-2. Test1 데이터 예측
    preds = mlp.predict(X_test1_full_scaled)

    # 4-3. 정확도 계산
    acc = accuracy_score(y_test1_full, preds)

    # 결과 저장
    results.append((h, acc))

    print(f"hidden_layer_sizes=({h},)  →  Test1 Accuracy = {acc*100:.2f}%")

# --- 5. 결과 정리 ---
print("\n=== 최종 결과 요약 ===")
for h, acc in results:
    print(f"은닉 노드 {h:3d}개  →  Test1 Accuracy = {acc*100:.2f}%")
