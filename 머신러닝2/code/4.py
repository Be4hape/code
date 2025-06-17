import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
# process1.csv는 라벨이 포함된 학습 데이터, process2.csv는 PassengerId 등과 함께 전처리된 테스트 데이터라고 가정합니다.
train_df = pd.read_csv('process1.csv')
test_df  = pd.read_csv('process2.csv')

# 2. 필요한 열 선택 (전처리된 상태에서 사용했던 피처 그대로 활용)
# 여기서는 예시로 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked'를 사용한다고 가정함
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

X_train = train_df[features].values       # process1의 학습 피처
y_train = train_df['Survived'].values       # 학습 데이터의 라벨

X_test = test_df[features].values         # process2의 테스트 피처
passenger_ids = test_df['PassengerId'].values  # 테스트 데이터의 고유 ID

# 3. 스케일링
# 전처리된 데이터이더라도, KNN은 거리 계산에 민감하므로 학습 데이터의 분포에 맞춰 StandardScaler를 적용합니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4. KNN 알고리즘 직접 구현 (고정 k = 12)
def euclidean_distance(x1, x2):
    """두 벡터 간 유클리드 거리를 계산"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def majority_vote(neighbors):
    """k개의 이웃의 라벨에 대해 다수결 투표로 예측 클래스 결정"""
    values, counts = np.unique(neighbors, return_counts=True)
    return values[np.argmax(counts)]

def knn_predict(train_X, train_y, test_sample, k):
    """학습 데이터(train_X, train_y)를 기준으로 테스트 샘플(test_sample)에 대해 k-NN 예측"""
    distances = []
    for i in range(train_X.shape[0]):
        d = euclidean_distance(test_sample, train_X[i])
        distances.append((d, train_y[i]))
    # 거리가 가까운 순으로 정렬 후 k개 선택
    distances.sort(key=lambda x: x[0])
    k_neighbors = [label for (d, label) in distances[:k]]
    return majority_vote(k_neighbors)

# 5. 테스트 데이터에 대해 예측 수행 (k=12)
k_fixed = 12
predictions = []
for i in range(X_test_scaled.shape[0]):
    pred = knn_predict(X_train_scaled, y_train, X_test_scaled[i], k_fixed)
    predictions.append(pred)

# 6. 결과를 PassengerId, Survived 형식의 CSV 파일로 저장 (Kaggle 제출용)
result_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

result_df.to_csv('test2_result.csv', index=False)
print("test2_result.csv 파일이 성공적으로 저장되었습니다.")
