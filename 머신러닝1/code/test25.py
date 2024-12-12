import pandas as pd
import numpy as np

# 파일 경로
file_path = '/content/train.csv'
data = pd.read_csv(file_path)

### 전처리 ###

# Age 결측치: 중앙값으로 채우기
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
data['Cabin_Letter'] = data['Cabin_Letter'].map(cabin_mapping)

# Cabin Letter 점수를 기반으로 결측치 처리
def assign_cabin_percentile(data, weighted_col='Weighted_Survival_Score'):
    # 1. 퍼센타일 경계 계산 (7구간)
    percentiles = np.linspace(0, 100, 8)  # 0%, 14.28%, ..., 100%
    bins = np.percentile(data[weighted_col].dropna(), percentiles)  # 결측치 제거 후 퍼센타일 계산

    # 2. 중복 제거 (bins must be strictly increasing)
    bins = np.unique(bins)
    if len(bins) <= 1:
        raise ValueError("Not enough unique bins to create intervals. Check your data distribution.")

    # 3. Cabin Letter 매핑
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G'][: len(bins) - 1]  # labels 수는 bins - 1
    data['Cabin_Letter'] = pd.cut(data[weighted_col], bins=bins, labels=labels, include_lowest=True)

    # 숫자 매핑
    cabin_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    data['Cabin_Letter'] = data['Cabin_Letter'].map(cabin_mapping)

    return data

# Weighted_Survival_Score 계산
data['Weighted_Survival_Score'] = (
    data['Sex'] * 0.54 +                  # 성별 가중치
    data['Pclass'] * -0.36 +             # 객실 등급 가중치
    data['Title'] * 0.29 +               # 사회적 지위 가중치
    data['Cabin_Letter'].fillna(-1) * 0.273788  # Cabin Letter 가중치 (결측치는 -1로 대체)
)

# Cabin Letter 결측치 처리
data = assign_cabin_percentile(data, weighted_col='Weighted_Survival_Score')

# Survived와 newsurvive 비교 후 정확도 계산 (나머지 코드 동일)
threshold = np.median(data['Weighted_Survival_Score'])
data['newsurvive'] = (data['Weighted_Survival_Score'] >= threshold).astype(int)
correct_predictions = (data['newsurvive'] == data['Survived']).sum()
accuracy = correct_predictions / len(data)

# 결과 출력
print(f"Prediction Accuracy: {accuracy:.5f}")
