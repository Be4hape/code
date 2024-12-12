import pandas as pd

file_path = '/content/train.csv'
train_data = pd.read_csv(file_path)

# name에서 공통적으로 반복되는 타이틀 추출, extract >> A부터Z, a부터z, 이후 .가 오는 단어는 모두 추출
train_data['Title'] = train_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)

# valuecount로 카운팅 값 내림차순
title_counts = train_data['Title'].value_counts()

print("타이틀 카운팅:")
print(title_counts)
