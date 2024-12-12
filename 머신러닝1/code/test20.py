import pandas as pd
import numpy as np

file_path = '/content/weighted_survival_scores (1).csv'
data = pd.read_csv(file_path)

###전처리

# name 전처리
if 'Title' not in data.columns:
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 1, 'Master': 1, 'Dr': 0, 'Rev': 0}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'].fillna(0, inplace=True)


# cabin 전처리
data['Cabin_Letter'] = data['Cabin'].str[0]
cabin_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
data['Cabin_Letter'] = data['Cabin_Letter'].map(cabin_mapping)



# Cabin의 상관계수를 포함한 생존 가중치
data['Weighted_Survival_Score'] = (
    data['Sex'] * 0.54 +
    data['Pclass'] * -0.36 +
    data['Fare'] * 0.27 +
    data['Title_Mapped'] * 0.29 +
    data['Cabin_Letter'] * -0.19956
)

###

# 메디안값으로 스레시홀딩 값 지정
threshold = np.median(data['Weighted_Survival_Score'])
data['newsurvive'] = (data['Weighted_Survival_Score'] >= threshold).astype(int)

# 정확도
correct_predictions = (data['newsurvive'] == data['Survived']).sum()
accuracy = correct_predictions / len(data)
print(f"Prediction Accuracy: {accuracy:.5f}")