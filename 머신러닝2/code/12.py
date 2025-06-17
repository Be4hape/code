import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 데이터 로드 (전처리+엔지니어링 완료된 결과 파일 사용) ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

# --- 2. 상관계수 딕셔너리와 피처 리스트 정의 ---
coefficients = {
    'Sex': 0.54,
    'SibSp': 0.12,
    'Parch': 0.18,
    'Embarked': 0.10,
    'TicketNumeric': -0.24
}
features = list(coefficients.keys())

# --- 3. 피처 및 레이블 분리 ---
X_train = train_df[features].values
y_train = train_df['Survived'].values
X_test  = test_df[features].values
passenger_ids = test_df['PassengerId'].values

# --- 4. 수동 스케일링 (평균 0, 표준편차 1) --- 0~1사이 값으로 정규화 or 표준화
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
# 0으로 나누는 오류 방지
stds[stds == 0] = 1.0

X_train_scaled = (X_train - means) / stds
X_test_scaled  = (X_test  - means) / stds

# --- 5. 온라인 학습용 로지스틱 회귀 (SGD) 구현 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

m, n = X_train_scaled.shape
w = np.zeros(n)
b = 0.0

# --- (5) 여러 학습률에 대해 SGD 학습 & 비용 기록 ---
learning_rates = [0.0001, 0.001, 0.01]   # 비교할 학습률들
num_epochs     = 100                     # 고정 에폭 수
epsilon        = 1e-15

cost_histories = {}  # { lr: [cost_epoch1, cost_epoch2, ...] }

for lr in learning_rates:
    w = np.zeros(n)
    b = 0.0
    cost_list = []

    for epoch in range(1, num_epochs + 1):
        # 샘플 단위 업데이트
        for i in range(m):
            xi, yi = X_train_scaled[i], y_train[i]
            z_i     = np.dot(xi, w) + b
            h_i     = sigmoid(z_i)
            delta   = h_i - yi
            w      -= lr * delta * xi
            b      -= lr * delta

        # 에폭마다 전체 손실 계산
        z_all = X_train_scaled.dot(w) + b
        h_all = sigmoid(z_all)
        cost = -np.mean(
            y_train * np.log(h_all + epsilon) +
            (1 - y_train) * np.log(1 - h_all + epsilon)
        )
        cost_list.append(cost)

    cost_histories[lr] = cost_list

# --- (6) 수렴 구간의 Loss 값만 출력 ---
N = 50
start_epoch = max(1, num_epochs - N + 1)

print(f"\n=== Converged Loss for Last {N} Epochs ===")
for lr, costs in cost_histories.items():
    converged = costs[start_epoch-1:]
    print(f"\nlr = {lr}")
    for epoch_offset, c in enumerate(converged, start=start_epoch):
        print(f"  epoch {epoch_offset:3d}: cost = {c:.4f}")

# --- (6) 학습률별 수렴 곡선 한 화면에 시각화 ---
plt.figure(figsize=(8,5))
for lr, costs in cost_histories.items():
    plt.plot(
        range(1, num_epochs + 1),
        costs,
        label=f'lr = {lr}'
    )

plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Training Loss Convergence for Different Learning Rates')
plt.xlim(1, num_epochs)
# y축 자동 맞춤 또는 수렴부만 보고 싶으면 plt.ylim(...) 추가
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 7. 테스트 데이터 예측 & 결과 저장 ---
z_test      = X_test_scaled.dot(w) + b
h_test      = sigmoid(z_test)
predictions = (h_test >= 0.5).astype(int)

submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    predictions
})
submission_df.to_csv('logistic_sgd_on_custom_scores.csv', index=False)
print("logistic_sgd_on_custom_scores.csv 파일이 성공적으로 저장되었습니다.")

# --- train(process1) 정확도 계산 ---
z_train   = X_train_scaled.dot(w) + b
h_train   = sigmoid(z_train)
pred_train = (h_train >= 0.5).astype(int)
acc_train  = np.mean(pred_train == y_train)
print(f"Test1 Accuracy: {acc_train * 100:.2f}%")
