import pandas as pd

file_path = '/content/sorting_updated.csv'
sorting_data = pd.read_csv(file_path)

# check to nan
nan_check = sorting_data.isnull().sum()

print("NaN 값 체크:")
print(nan_check)

# cabin문자열의 첫번째 문자만 추출
sorting_data['Cabin_Letter'] = sorting_data['Cabin'].str[0]  # Cabin의 첫 글자 추출

# cabin 카운트
cabin_count = sorting_data['Cabin_Letter'].value_counts()

# A ~ G
cabin_summary = {cabin: cabin_count.get(cabin, 0) for cabin in ['A', 'B', 'C', 'D', 'E', 'F', 'G']}
cabin_summary['else'] = cabin_count.sum() - sum(cabin_summary.values())

print("Cabin 카운팅 : ")
for cabin, count in cabin_summary.items():
    print(f"{cabin}: {count}")


