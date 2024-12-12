import pandas as pd

file_path = '/content/sorting_updated.csv'
data = pd.read_csv(file_path)

data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
title_counts = data['Title'].value_counts()
title_mapping = {title: idx for idx, title in enumerate(title_counts.index)}
data['Title_Mapped'] = data['Title'].map(title_mapping)


output_file = '/content/updating_final.csv'
data.to_csv(output_file, index=False)

print(f"매핑데이터 '{output_file}'로 저장완료")
