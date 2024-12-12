##구글링한 코드, 상관관계 예습을 위해 사용

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# data loading
file_path = r'/content/train.csv'
train_data = pd.read_csv(file_path)

# del to PassengerId, Ticket
train_data = train_data.drop(columns=['PassengerId', 'Ticket'], axis=1)


# Sex : 남자=0, 여자=1,
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# Embarked : C:0, Q:1, S:2
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

### 결측치
train_data['Age'].fillna(train_data['Age'].median())

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

train_data['CabinAvailable'] = train_data['Cabin'].notnull().astype(int)

train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_data['Title'] = train_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
train_data['Title'] = train_data['Title'].replace(['Mme'], 'Mrs')
train_data['Title'] = train_data['Title'].replace(['Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Sir', 'Don', 'Capt'], 'Other')

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}
train_data['Title'] = train_data['Title'].map(title_mapping)

train_data = train_data.drop(columns=['Name', 'Cabin'], axis=1)

###

# 상관계수
correlation_matrix = train_data.corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')

# X, Y축 레이블 설정
features = correlation_matrix.columns
plt.xticks(np.arange(len(features)), features, rotation=45, ha='right')
plt.yticks(np.arange(len(features)), features)

# 제목 추가
plt.title("Correlation of Features with Survival")
plt.ylabel("Features")
plt.xlabel("Features")

# 값 표시 (상관계수)
for i in range(len(features)):
    for j in range(len(features)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                 ha='center', va='center', color='black', fontsize=8)

# 상관계수 계산
correlation_matrix = train_data.corr()

# CabinAvailable과 다른 변수 간의 상관계수를 정렬
sorted_correlation = correlation_matrix['CabinAvailable'].sort_values(ascending=False)

# 출력
print("CabinAvailable과 상관계수가 높은 변수들:")
print(sorted_correlation)

plt.tight_layout()
plt.show()

