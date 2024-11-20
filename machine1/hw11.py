import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

test1_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\test1.csv')
test2_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\test2.csv')
train_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\train.csv')


### 데이터 전처리


# 필요한 열만 선택
train_data = train_data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test1_data = test1_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# 문자열 데이터를 숫자로 변환
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test1_data['Sex'] = test1_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test1_data['Embarked'] = test1_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Age 결측치 : Pclass와 Sex별 중앙값으로 채우기
age_medians = train_data.groupby(['Pclass', 'Sex'])['Age'].median()

for (pclass, sex), median in age_medians.items():
    train_data.loc[(train_data['Pclass'] == pclass) & (train_data['Sex'] == sex) & (train_data['Age'].isnull()), 'Age'] = median
    test1_data.loc[(test1_data['Pclass'] == pclass) & (test1_data['Sex'] == sex) & (test1_data['Age'].isnull()), 'Age'] = median

# Fare 결측치 : Pclass별 중앙값으로 채우기
fare_medians = train_data.groupby('Pclass')['Fare'].median()

for pclass, median in fare_medians.items():
    train_data.loc[(train_data['Pclass'] == pclass) & (train_data['Fare'].isnull()), 'Fare'] = median
    test1_data.loc[(test1_data['Pclass'] == pclass) & (test1_data['Fare'].isnull()), 'Fare'] = median

# Embarked 결측치 : 최빈값으로 채우기
train_embarked_mode = train_data['Embarked'].mode()[0]
test_embarked_mode = test1_data['Embarked'].mode()[0]

train_data.loc[train_data['Embarked'].isnull(), 'Embarked'] = train_embarked_mode
test1_data.loc[test1_data['Embarked'].isnull(), 'Embarked'] = test_embarked_mode


### 생존자 추정


#female + pclass 1~2 > survival rates high :: accuracy is 0.15..
#생존 여부 추정 조건
# 1. 초기엔 모두 사망자
# 2. 여성+높은등급 = 1
# 3. 10세 이하 어린이 or 80세 이상 노인 = 1
# 4. 1등급 + 비싼요금 = 1
test1_data['Survived'] = 0
test1_data.loc[(test1_data['Sex'] == 1) & (test1_data['Pclass'] <= 2), 'Survived'] = 1  # 여성 + 높은등급
test1_data.loc[(test1_data['Age'] <= 10) | (test1_data['Age'] >= 80), 'Survived'] = 1  # 10세 이하 어린이 or 80세 이상 노인
test1_data.loc[(test1_data['Pclass'] == 1) & (test1_data['Fare'] > 50), 'Survived'] = 1  # 1등급 + 비싼요금


# 결측치 승객 Survived > NaN으로 설정
test1_data.loc[test1_data.isnull().any(axis=1), 'Survived'] = np.nan




### 비교

# Compare Train, Test1 (추정 성공 비율)
train_survived = train_data['Survived']
test1_survived = test1_data['Survived']


# 일치 비율
correct_predictions = (train_survived == test1_survived).sum()
accuracy = correct_predictions / len(train_survived)

print(f"Prediction Accuracy: {accuracy:.5f}")