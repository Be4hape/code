import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 가족 생존률 계산 함수
def find_family_groups_with_sibsp_parch(data):
    # Step 1: Count 변수 생성 (본인을 포함한 가족 수)
    data['Count'] = data['SibSp'] + data['Parch'] + 1

    # Step 2: 그룹 생성 (SibSp, Parch, 기본 그룹)
    sibsp_group = data[data['SibSp'] > 0].groupby(['Ticket', 'Embarked', 'Fare']).ngroup()
    parch_group = data[data['Parch'] > 0].groupby(['Ticket', 'Embarked']).ngroup()
    default_group = data.groupby(['Ticket', 'Embarked']).ngroup()

    # Step 3: 데이터프레임에 그룹 추가
    data['SibSp_Group'] = sibsp_group.reindex(data.index, fill_value=-1)
    data['Parch_Group'] = parch_group.reindex(data.index, fill_value=-1)
    data['Default_Group'] = default_group.reindex(data.index, fill_value=-1)

    # Step 4: 결측값 처리: SibSp > Parch > Default 순으로 그룹 선택
    data['Family_Group'] = data['SibSp_Group'].where(data['SibSp_Group'] != -1, data['Parch_Group'])
    data['Family_Group'] = data['Family_Group'].where(data['Family_Group'] != -1, data['Default_Group'])

    # Step 5: 생존률 계산 (학습 데이터에서만)
    if 'Survived' in data.columns:
        family_survival_rate = data.groupby('Family_Group')['Survived'].transform('mean')
        data['Family_Survival_Rate'] = family_survival_rate.fillna(0.5)  # 결측값 기본 생존률
    else:
        data['Family_Survival_Rate'] = 0.5  # 테스트 데이터 기본값

    return data

# 데이터 전처리 함수
def preprocess_data(data, is_train=True):
    # 결측치 처리
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

    # 가족 생존률 추가 (학습 데이터에서만 계산)
    data = find_family_groups_with_sibsp_parch(data)

    # 최종 사용할 열 지정
    features = ['Sex', 'Pclass', 'Title', 'Cabin_Letter', 'Family_Survival_Rate']
    if is_train:
        return data[features], data['Survived']
    else:
        return data[features]

# 데이터 로드
train_data = pd.read_csv('/content/train.csv')
test_data = pd.read_csv('/content/test2.csv')

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

# CSV 파일 저장
test_data[['PassengerId', 'Survived']].to_csv('/content/test2_results.csv', index=False)

# 최적 하이퍼파라미터 출력
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.5f}")
