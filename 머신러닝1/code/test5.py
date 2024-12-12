import pandas as pd

file_path = '/content/train.csv'  # 파일 경로 수정
train_data = pd.read_csv(file_path)

train_data['Cabin_Letter'] = train_data['Cabin'].dropna().str[0]

filtered_data = train_data[train_data['Cabin_Letter'].isin(['A', 'B', 'C', 'D', 'E', 'F', 'G'])]
sorted_data = filtered_data.sort_values(by=['Cabin_Letter', 'Survived'], ascending=[True, True])

print("A부터 G까지의 Cabin에 따른 Survived 정렬된 데이터:")
print(sorted_data[['Cabin_Letter', 'Survived', 'Pclass', 'Fare', 'Age']])  # 필요한 열만 출력

# 정렬된 데이터를 CSV 파일로 저장
output_path = '/content/sorted_survived_by_cabin.csv'
sorted_data.to_csv(output_path, index=False)
print(f"\n정렬된 데이터가 다음 경로에 저장되었습니다: {output_path}")
