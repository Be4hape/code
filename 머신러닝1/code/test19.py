## 추후, 정확도를 올리기 위해 cabin과 survival의 상관계수를 구했음.

import pandas as pd

# 데이터 로드
file_path = '/content/weighted_survival_scores.csv'
data = pd.read_csv(file_path)

# cabin문자열의 첫번째 추출
data['Cabin_Letter'] = data['Cabin'].str[0]
cabin_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
data['Cabin_Letter'] = data['Cabin_Letter'].map(cabin_mapping)

correlation = data['Cabin_Letter'].corr(data['Survived'])
print(f"Survived와 Cabin_Letter의 상관계수: {correlation:.5f}")
