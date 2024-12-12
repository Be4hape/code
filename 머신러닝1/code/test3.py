import pandas as pd

file_path = r'/content/train.csv'
train_data = pd.read_csv(file_path)

# 오름차순
sorted_data = train_data.sort_values(by=['Pclass', 'Fare'], ascending=[True, True])

# save
output_path = r'/content/sorted_titanic_data.csv'
sorted_data.to_csv(output_path, index=False)

print(f"Pclass와 Fare 기준으로 정렬, /content/위치에 저장완료")