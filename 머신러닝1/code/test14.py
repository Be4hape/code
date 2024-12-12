import pandas as pd

file_path = '/content/updating_final.csv'  # 파일 경로 수정 필요
data = pd.read_csv(file_path)

if 'Sex' in data.columns:
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
numeric_columns = data.select_dtypes(include=['float64', 'int64'])

# Survived와 모든 전처리 한 숫자 데이터들 상관계수
correlation_with_survived = numeric_columns.corr()['Survived']

# 상관관계가 높은 4가지 : Sex, Pclass, fare, title_mapped
selected_features = ['Sex', 'Pclass', 'Fare', 'Title_Mapped']

# 가중합 계산
data['Weighted_Survival_Score'] = data[selected_features].mul(
    correlation_with_survived[selected_features], axis=1
).sum(axis=1)

# savew
output_file = '/content/weighted_survival_scores.csv'
data.to_csv(output_file, index=False)

print(f"saved SURVIVAL SCORE in '{output_file}'")
