import pandas as pd
import numpy as np

# 1. process1_result.csv 불러오기
df = pd.read_csv('process1_result.csv')

# 2. Pclass별 Age 중앙값 계산
median_by_pclass = df.groupby('Pclass')['Age'].median()

# 3. Age 결측치가 있으면 Pclass별 중앙값으로 채우기
def fill_age(row):
    if pd.isna(row['Age']):
        return median_by_pclass.loc[row['Pclass']]
    else:
        return row['Age']

df['Age'] = df.apply(fill_age, axis=1)

# 4. 혹시 남은 결측치가 있다면 전체 중앙값으로 채우기
df['Age'].fillna(df['Age'].median(), inplace=True)

# 5. 결과 저장 (원본 덮어쓰기 또는 새로운 파일로 저장)
df.to_csv('process1_result.csv', index=False)
