from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드
data = load_iris()
X, y = data.data, data.target

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN 모델 생성
k = 5  # K 값 설정
knn = KNeighborsClassifier(n_neighbors=k)

# 모델 학습
knn.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = knn.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f"K={k} Accuracy: {accuracy:.2f}")
