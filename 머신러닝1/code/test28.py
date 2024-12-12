import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 데이터 전처리 함수
def preprocess_data(data, is_train=True):
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(float)

    # Title 추출 및 매핑
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    data['Title'] = data['Title'].replace(['Mme'], 'Mrs')
    data['Title'] = data['Title'].replace(
        ['Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Sir', 'Don', 'Capt'], 'Other'
    )
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}
    data['Title'] = data['Title'].map(title_mapping)

    # Cabin Letter 추출
    data['Cabin_Letter'] = data['Cabin'].str[0]
    cabin_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    data['Cabin_Letter'] = data['Cabin_Letter'].map(cabin_mapping).fillna(-1)

    # 학습 데이터와 테스트 데이터 구분
    features = ['Sex', 'Pclass', 'Age', 'Title', 'Cabin_Letter']
    if is_train:
        return data[features], data['Survived']
    else:
        return data[features]

# 데이터 로드
train_data = pd.read_csv('/content/train.csv')
test_data = pd.read_csv('/content/test1.csv')

# 데이터 전처리
X_train, y_train = preprocess_data(train_data, is_train=True)
X_test = preprocess_data(test_data, is_train=False)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 데이터 분할 (학습/검증)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# KNN 모델 정의
knn = KNeighborsClassifier()

# 하이퍼파라미터 튜닝 (Grid Search)
param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_split, y_train_split)

# 최적의 KNN 모델
best_knn = grid_search.best_estimator_

# 검증 데이터 정확도
y_val_pred = best_knn.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.5f}")

# 테스트 데이터 예측
test_predictions = best_knn.predict(X_test)
test_data['Survived'] = test_predictions


# 최적 하이퍼파라미터 출력
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.5f}")
