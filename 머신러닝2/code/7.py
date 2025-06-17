import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 1. 데이터 로드 (전처리+엔지니어링 완료된 결과 파일 사용) ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

# --- 2. 상관계수 딕셔너리와 피처 리스트 정의 ---
coefficients = {
    #'Pclass': -0.33,
    #'Name': 0.51,
    'Sex': 0.54,
    'SibSp': 0.12,
    'Parch': 0.18,
    'Embarked': 0.10,
    'TicketNumeric': -0.24
    #'Fare': 0.34,
}

# features 는 딕셔너리의 키 리스트
features = list(coefficients.keys())

# --- 3. 피처 및 레이블 분리 ---
X_train = train_df[features].values
y_train = train_df['Survived'].values

X_test        = test_df[features].values
passenger_ids = test_df['PassengerId'].values

# --- 4. 스케일링 (SGD용 전처리 일관성 유지) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- 5. 온라인 학습용 로지스틱 회귀 (SGD) 구현 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

m, n = X_train_scaled.shape
w = np.zeros(n)
b = 0.0

learning_rate = 0.001
num_epochs    = 100
epsilon       = 1e-15

cost_list = []
for epoch in range(1, num_epochs + 1):
    # 인스턴스 단위 업데이트
    for i in range(m):
        xi = X_train_scaled[i]
        yi = y_train[i]
        zi = np.dot(xi, w) + b
        hi = sigmoid(zi)
        dw = (hi - yi) * xi
        db = (hi - yi)
        w  -= learning_rate * dw
        b  -= learning_rate * db

    # 에폭 종료 후 전체 손실 계산
    z_all = np.dot(X_train_scaled, w) + b
    h_all = sigmoid(z_all)
    cost = -np.mean(
        y_train * np.log(h_all + epsilon) +
        (1 - y_train) * np.log(1 - h_all + epsilon)
    )
    cost_list.append(cost)
    print(f"Epoch {epoch}/{num_epochs} — Cost: {cost:.4f}")

# --- 6. 학습 손실 시각화 (선택) ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), cost_list, 'o-', markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Training Cost per Epoch (SGD Logistic Regression)')
plt.grid(True)
plt.show()

# --- 7. 테스트 데이터 예측 & 결과 저장 ---
z_test      = np.dot(X_test_scaled, w) + b
h_test      = sigmoid(z_test)
predictions = (h_test >= 0.5).astype(int)

submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    predictions
})
submission_df.to_csv('logistic_sgd_on_custom_scores.csv', index=False)
print("logistic_sgd_on_custom_scores.csv 파일이 성공적으로 저장되었습니다.")
