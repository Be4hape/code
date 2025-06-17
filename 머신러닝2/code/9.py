import numpy as np
import pandas as pd

# --- 1. 데이터 로드 ---
train_df = pd.read_csv('process1_result.csv')   # 전처리+엔지니어링 완료된 학습 데이터
test_df  = pd.read_csv('process2_result.csv')   # 전처리+엔지니어링 완료된 테스트 데이터

# --- 2. 사용할 피처 정의 ---
# Sex           : 성별 (0=남자, 1=여자)
# SibSp         : 형제/배우자 수
# Parch         : 부모/자녀 수
# Embarked      : 탑승 항구 (숫자로 인코딩)
# TicketNumeric : 숫자로 변환된 티켓 번호
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']

# --- 3. 피처 및 레이블 분리 ---
X_train = train_df[features].values
y_train = train_df['Survived'].values
X_test  = test_df[features].values
passenger_ids = test_df['PassengerId'].values

# --- 4. 수동 스케일링 (평균 0, 표준편차 1) ---
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
# 표준편차가 0인 경우 1로 대체하여 0으로 나누는 오류 방지
stds[stds == 0] = 1
X_train_scaled = (X_train - means) / stds
X_test_scaled  = (X_test  - means) / stds

# --- 5. 유클리드 거리 계산 (numpy 브로드캐스트) ---
# 5-1) Train ↔ Train 거리 행렬
D_train = np.sqrt(((X_train_scaled[:, None, :] - X_train_scaled[None, :, :]) ** 2).sum(axis=2))
# 자기 자신과의 거리는 제외하기 위해 무한대로 설정
np.fill_diagonal(D_train, np.inf)

# 5-2) Test2 ↔ Train 거리 행렬
D_test = np.sqrt(((X_test_scaled[:, None, :] - X_train_scaled[None, :, :]) ** 2).sum(axis=2))

# --- 6. k = 1…30에 대해 Train(=process1) 정확도 평가 ---
best_k, best_acc = 1, 0.0
n_train = X_train_scaled.shape[0]

print(" k | train accuracy")
print("---|----------------")
for k in range(1, 31):
    # 각 샘플에 대해 k개의 최근접 이웃 레이블로 예측
    preds = np.empty(n_train, dtype=int)
    for i in range(n_train):
        neigh_idxs = np.argpartition(D_train[i], k)[:k]
        # 다수결 투표
        labels = y_train[neigh_idxs]
        preds[i] = np.bincount(labels).argmax()
    acc = np.mean(preds == y_train)
    print(f"{k:2d} | {acc*100:6.2f}%")
    if acc > best_acc:
        best_acc, best_k = acc, k

print(f"\n▶ 최적 k = {best_k} (train accuracy = {best_acc*100:.2f}%)\n")

# --- 7. 최적 k로 Test2 예측 및 저장 ---
m_test = X_test_scaled.shape[0]
test_preds = np.empty(m_test, dtype=int)
for i in range(m_test):
    neigh_idxs = np.argpartition(D_test[i], best_k)[:best_k]
    labels = y_train[neigh_idxs]
    test_preds[i] = np.bincount(labels).argmax()

submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    test_preds
})
submission_df.to_csv('knn_manual_test2.csv', index=False)
print("knn_manual_test2.csv 파일이 저장되었습니다.")
