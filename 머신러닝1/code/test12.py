import pandas as pd

file_path = '/content/train.csv'  # 파일 경로 수정 필요
train_data = pd.read_csv(file_path)

# title = middle name
train_data['Title'] = train_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
title_counts = train_data['Title'].value_counts()

# 값이 많은 것부터 0 맵핑. 이후 1,2,3, ' ' '
title_mapping = {title: idx for idx, title in enumerate(title_counts.index)}
train_data['Title_Mapped'] = train_data['Title'].map(title_mapping)

# Title_Mapped와 Survived의 상관계수 계산
correlation = train_data[['Title_Mapped', 'Survived']].corr().loc['Title_Mapped', 'Survived']


print(title_mapping)
print(f"\nTitle_Mapped와 survived의 상관계수: {correlation:.5f}")
