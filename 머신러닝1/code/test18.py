import pandas as pd
import numpy as np

# 데이터 로드
file_path = '/content/weighted_survival_scores.csv'
data = pd.read_csv(file_path)


if 'Title' not in data.columns:
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 1, 'Master': 1, 'Dr': 0, 'Rev': 0}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'].fillna(0)


data['Weighted_Survival_Score'] = (
    data['Sex'] * 0.54 +
    data['Pclass'] * -0.36 +
    data['Fare'] * 0.27 +
    data['Title_Mapped'] * 0.29
)

threshold = np.median(data['Weighted_Survival_Score'])
data['newsurvive'] = (data['Weighted_Survival_Score'] >= threshold).astype(int)

# 정확도 계산
correct_predictions = (data['newsurvive'] == data['Survived']).sum()
accuracy = correct_predictions / len(data)
print(f"Prediction Accuracy: {accuracy:.5f}")
