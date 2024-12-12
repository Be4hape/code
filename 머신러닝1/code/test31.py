import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 데이터 전처리 함수
def preprocess_data(data, is_train=True):
    # 결측치 처리 및 주요 피처 생성
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
    data['Title'] = data['Title'].fillna(5)

    # Cabin Letter 추출
    data['Cabin_Letter'] = data['Cabin'].str[0]
    cabin_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    data['Cabin_Letter'] = data['Cabin_Letter'].map(cabin_mapping).fillna(-1)

    # 필요한 열 반환
    features = ['Sex', 'Pclass', 'Title', 'Cabin_Letter']
    if is_train:
        return data[features], data['Survived']
    else:
        return data[features]



# 데이터 로드
train_data = pd.read_csv('/content/train.csv')
test1_data = pd.read_csv('/content/test1.csv')
test2_data = pd.read_csv('/content/test2.csv')

# 데이터 전처리
X_train, y_train = preprocess_data(train_data, is_train=True)
X_test1 = preprocess_data(test1_data, is_train=False)
X_test2 = preprocess_data(test2_data, is_train=False)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test1_scaled = scaler.transform(X_test1)
X_test2_scaled = scaler.transform(X_test2)

# train 데이터를 분할하여 하이퍼파라미터 튜닝
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# KNN 모델 정의
knn = KNeighborsClassifier()

# 하이퍼파라미터 튜닝 (Grid Search)
param_grid = {
    'n_neighbors': range(1, 29),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_split, y_train_split)

# 최적의 KNN 모델
best_knn = grid_search.best_estimator_

# 검증 데이터 정확도
y_val_pred = best_knn.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.5f}")

# test1 데이터 예측
test1_predictions = best_knn.predict(X_test1_scaled)
test1_data['Survived'] = test1_predictions
test1_data[['PassengerId', 'Survived']].to_csv('/content/test1_results.csv', index=False)

# test2 데이터 예측
test2_predictions = best_knn.predict(X_test2_scaled)
test2_data['Survived'] = test2_predictions
test2_data[['PassengerId', 'Survived']].to_csv('/content/test2_results.csv', index=False)

# 최적 하이퍼파라미터 출력
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.5f}")
