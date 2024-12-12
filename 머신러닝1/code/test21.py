## for test2, 캐글 submit을 위한 코드

import pandas as pd
import numpy as np

test_file_path = '/content/test2.csv'
test_data = pd.read_csv(test_file_path)

test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 1, 'Master': 1, 'Dr': 0, 'Rev': 0}
test_data['Title'] = test_data['Title'].map(title_mapping).fillna(0)

test_data['Weighted_Survival_Score'] = (
    test_data['Sex'].map({'male': 0, 'female': 1}) * 0.54 +
    test_data['Pclass'] * -0.36 +
    test_data['Fare'].fillna(test_data['Fare'].median()) * 0.27 +
    test_data['Title'] * 0.29
)
threshold = np.median(test_data['Weighted_Survival_Score'])
test_data['Survived'] = (test_data['Weighted_Survival_Score'] >= threshold).astype(int)

# save to 1958015.csv
submission = test_data[['PassengerId', 'Survived']]
submission.to_csv('/content/1958015.csv', index=False)
print("저장완료 > 1958015")
