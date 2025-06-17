import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 1. 데이터 로드 ---
# process1.csv: 학습 데이터 (Survived 포함)
# process2.csv: 테스트 데이터 (PassengerId 포함, Survived 없음)
train_df = pd.read_csv('process1.csv')
test_df  = pd.read_csv('process2.csv')

# --- 2. Age 결측치 채우기 (그룹별 중앙값 사용) ---
# train 데이터: Pclass와 Sex별 중앙값으로 Age 결측치 채움
train_df['Age'] = train_df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
# 혹시 그룹별 채움 후에도 남아있다면 전체 중앙값으로 채우기
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# test 데이터에도 동일하게 적용 (test 데이터에는 라벨이 없지만, Age 전처리는 동일하게 진행)
test_df['Age'] = test_df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# --- 3. 전처리: 선택했던 피처들을 그대로 사용 (전처리 로직 변경 없이) ---
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

X_train = train_df[features].values
y_train = train_df['Survived'].values

X_test = test_df[features].values
passenger_ids = test_df['PassengerId'].values

# --- 4. 스케일링 ---
# KNN이나 로지스틱 회귀와 같이 수치 연산 기반 모델은 피처 스케일링이 중요함.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- 5. 로지스틱 회귀 모델 구현 및 학습 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

m, n = X_train_scaled.shape
w = np.zeros(n)   # 가중치 벡터 초기화
b = 0             # bias 초기화

learning_rate = 0.001
num_iterations = 100000
cost_list = []  # 학습 과정에서의 손실 값을 저장할 리스트

for i in range(num_iterations):
    # 선형 결합 및 시그모이드 함수 적용
    z = np.dot(X_train_scaled, w) + b
    h = sigmoid(z)

    # 크로스 엔트로피 손실 계산 (수치적 안정성을 위해 epsilon 추가)
    epsilon = 1e-15
    cost = -np.mean(y_train * np.log(h + epsilon) + (1 - y_train) * np.log(1 - h + epsilon))
    cost_list.append(cost)

    # gradient(기울기) 계산
    dw = np.dot(X_train_scaled.T, (h - y_train)) / m
    db = np.sum(h - y_train) / m

    # 파라미터 업데이트: 이전 값에서 학습률과 기울기를 곱한 값을 뺌
    w -= learning_rate * dw
    b -= learning_rate * db

    # 1000회마다 cost 출력 (진행 상황 확인)
    if i % 1000 == 0:
        print(f"Iteration {i}, cost: {cost:.4f}")

# --- 6. 학습 과정 시각화 ---
plt.figure(figsize=(10, 6))
# 10회마다 마커를 찍어서 변화를 보기 쉽게 함
plt.plot(range(0, num_iterations, 10), cost_list[::10], 'o-', markersize=2, label="Cost")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations (Logistic Regression)')
plt.grid(True)
plt.legend()
plt.show()

# --- 7. 테스트 데이터에 대해 예측 ---
z_test = np.dot(X_test_scaled, w) + b
h_test = sigmoid(z_test)
predictions = (h_test >= 0.5).astype(int)

# --- 8. 결과 파일 저장 (PassengerId, Survived) ---
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})
submission_df.to_csv('logistic_result_group_median.csv', index=False)
print("logistic_result_group_median.csv 파일이 성공적으로 저장되었습니다.")
