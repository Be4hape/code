import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 데이터 전처리 함수
def preprocess_data(data, is_train=True):
    # 결측치 처리
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(float)

    # 가족 크기 계산
    data['Family_Size'] = data['SibSp'] + data['Parch'] + 1

    # Embarked와 Ticket 기반 그룹화
    def family_group(row, grouped_data):
        embarked = row['Embarked']
        ticket = row['Ticket']
        if pd.isna(embarked) or pd.isna(ticket):
            return -1
        return grouped_data.get((embarked, ticket), -1)

    # Embarked와 Ticket을 기준으로 그룹화
    grouped_data = {}
    data.sort_values(['Embarked', 'Ticket'], inplace=True)
    current_group = 0

    for _, row in data.iterrows():
        key = (row['Embarked'], row['Ticket'])
        if key not in grouped_data:
            grouped_data[key] = current_group
            current_group += 1

    data['Ticket_Embarked_Group'] = data.apply(family_group, axis=1, grouped_data=grouped_data)

    # Fare 그룹화
    data['Fare_Group'] = data.groupby(['Fare']).ngroup()

    # 최종 그룹 생성: Family_Size를 우선적으로 고려
    data['Family_Group'] = data.apply(
        lambda row: row['Ticket_Embarked_Group']
        if row['Family_Size'] > 1  # 가족 크기가 2 이상인 경우
        else row['Fare_Group'],
        axis=1
    )

    # 생존률 계산
    if is_train:
        family_survival_rate = data.groupby('Family_Group')['Survived'].mean()
        family_survival_rate.name = 'Family_Survival_Rate'
        data = data.merge(family_survival_rate, on='Family_Group', how='left')
    else:
        data['Family_Survival_Rate'] = 0.5  # 테스트 데이터는 생존률 기본값으로 설정

    data['Family_Survival_Rate'] = data['Family_Survival_Rate'].fillna(0.5)

    # 사용할 피처 선택
    features = ['Sex', 'Pclass', 'Family_Survival_Rate']
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

# 하이퍼파라미터 Grid Search
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
