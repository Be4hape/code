## 로지스틱 - 인스턴스 방식으로 수정.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 1. 데이터 로드 ---
train_df = pd.read_csv('process1.csv')    # 전처리된 학습 데이터
test_df  = pd.read_csv('process2.csv')    # 전처리된 테스트 데이터

# --- 2. Age 결측치 채우기 (Pclass×Sex 그룹 중앙값) ---
train_df['Age'] = train_df.groupby(['Pclass','Sex'])['Age']\
                           .transform(lambda x: x.fillna(x.median()))
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

test_df['Age'] = test_df.groupby(['Pclass','Sex'])['Age']\
                         .transform(lambda x: x.fillna(x.median()))
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# --- 3. 피처 및 레이블 분리 ---
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
X_train = train_df[features].values
y_train = train_df['Survived'].values
X_test  = test_df[features].values
passenger_ids = test_df['PassengerId'].values

# --- 4. 스케일링 ---
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
    # 샘플 단위로 파라미터 업데이트 (SGD)
    for i in range(m):
        xi = X_train_scaled[i]
        yi = y_train[i]
        zi = np.dot(xi, w) + b
        hi = sigmoid(zi)
        dw = (hi - yi) * xi
        db = (hi - yi)
        w -= learning_rate * dw
        b -= learning_rate * db

    # 한 epoch 후 전체 손실 계산
    z_all = np.dot(X_train_scaled, w) + b
    h_all = sigmoid(z_all)
    cost = -np.mean(y_train * np.log(h_all + epsilon) +
                    (1 - y_train) * np.log(1 - h_all + epsilon))
    cost_list.append(cost)
    print(f"Epoch {epoch}/{num_epochs} — Cost: {cost:.4f}")

# --- 6. 학습 손실 시각화 ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), cost_list, 'o-', markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Training Cost per Epoch (SGD Logistic Regression)')
plt.grid(True)
plt.show()

# --- 7. 테스트 데이터 예측 ---
z_test = np.dot(X_test_scaled, w) + b
h_test = sigmoid(z_test)
predictions = (h_test >= 0.5).astype(int)

# --- 8. 결과 저장 ---
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})
submission_df.to_csv('logistic_sgd_result.csv', index=False)
print("logistic_sgd_result.csv 파일이 성공적으로 저장되었습니다.")
