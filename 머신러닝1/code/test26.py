import pandas as pd
import numpy as np

# 생존 여부 예측 함수
def process_and_predict(data, calculate_threshold=True, threshold=None):
    """
    Titanic 데이터를 처리하고 생존 여부를 예측하는 함수.

    Args:
    - data (pd.DataFrame): 입력 데이터프레임.
    - calculate_threshold (bool): 중앙값 기반 임계값을 계산할지 여부.
    - threshold (float, optional): 지정된 임계값.

    Returns:
    - pd.DataFrame: 예측 결과를 포함한 데이터프레임.
    - float: 계산된 또는 사용된 임계값.
    """
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

    # Cabin Letter 추출
    data['Cabin_Letter'] = data['Cabin'].str[0]
    cabin_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    data['Cabin_Letter'] = data['Cabin_Letter'].map(cabin_mapping).fillna(-1)

    # Weighted_Survival_Score 계산
    data['Weighted_Survival_Score'] = (
        data['Sex'] * 0.54
        + data['Pclass'] * -0.36
        + data['Title'] * 0.29
        + data['Cabin_Letter'] * 0.273788
    )

    # 임계값 계산 또는 사용
    if calculate_threshold:
        threshold = np.median(data['Weighted_Survival_Score'])

    # 생존 여부 예측
    data['newsurvive'] = (data['Weighted_Survival_Score'] >= threshold).astype(int)

    return data, threshold

# 학습 데이터 로드 및 처리
train_data = pd.read_csv('/content/train.csv')
train_data, train_threshold = process_and_predict(train_data)

# 테스트 데이터 처리
test1_data = pd.read_csv('/content/test1.csv')
test1_result, _ = process_and_predict(test1_data, calculate_threshold=False, threshold=train_threshold)

test2_data = pd.read_csv('/content/test2.csv')
test2_result, _ = process_and_predict(test2_data, calculate_threshold=False, threshold=train_threshold)

# 결과 저장
test1_result = test1_result.rename(columns={'newsurvive': 'Survived'})
test2_result = test2_result.rename(columns={'newsurvive': 'Survived'})

test1_result[['PassengerId', 'Survived']].to_csv('/content/test1_results.csv', index=False)
test2_result[['PassengerId', 'Survived']].to_csv('/content/test2_results.csv', index=False)





def calculate_accuracy(test_result, train_data):
    """
    테스트 데이터의 예측 결과와 학습 데이터의 실제 값을 비교하여 정확도를 계산.
    """
    # 테스트 데이터의 PassengerId를 학습 데이터와 매칭
    matched = test_result.merge(train_data[['PassengerId', 'Survived']], on='PassengerId', suffixes=('_predicted', '_actual'))

    # 매칭된 데이터 여부 확인
    if len(matched) == 0:
        print("No matching PassengerId found between test_result and train_data.")
        print(f"Test result PassengerIds: {test_result['PassengerId'].tolist()[:5]}")  # 일부 출력
        print(f"Train data PassengerIds: {train_data['PassengerId'].tolist()[:5]}")    # 일부 출력
        return None

    # 정확도 계산
    correct_predictions = (matched['Survived_predicted'] == matched['Survived_actual']).sum()
    accuracy = correct_predictions / len(matched)

    return accuracy

# Test1 정확도 계산
if 'Survived' in train_data.columns:
    test1_accuracy = calculate_accuracy(test1_result, train_data)
    if test1_accuracy is not None:
        print(f"Prediction Accuracy on Test1 Data: {test1_accuracy:.5f}")
    else:
        print("No matching PassengerId for Test1.")

# Test2 정확도 계산
if 'Survived' in train_data.columns:
    test2_accuracy = calculate_accuracy(test2_result, train_data)
    if test2_accuracy is not None:
        print(f"Prediction Accuracy on Test2 Data: {test2_accuracy:.5f}")
    else:
        print("No matching PassengerId for Test2.")

