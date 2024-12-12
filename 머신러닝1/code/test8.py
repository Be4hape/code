import pandas as pd

file_path = '/content/sorted_titanic_data.csv'
sorting_data = pd.read_csv(file_path)

# 카피본 생성
sorted_copy = sorting_data.copy()

# NaN 분배 > 직접 계산, pclass가 1인 객실은 비교적 정확하겠지만, 2와 3은 부정확할 것임
pclass_nan_distribution = {
    1: {'A': 3, 'B': 13, 'C': 13, 'D': 6, 'E': 5},
    2: {'D': 42, 'E': 42, 'F': 84},
    3: {'E': 120, 'F': 199, 'G': 160}
}

for pclass, cabin_distribution in pclass_nan_distribution.items():
    # pclass이고 Cabin이 NaN인 행 추출
    nan_rows = sorted_copy[(sorted_copy['Pclass'] == pclass) & (sorted_copy['Cabin'].isnull())]

    # 각 Cabin 값과 그에 해당하는 count로 Nan 값을 채운다.
    start_idx = 0
    for cabin, count in cabin_distribution.items():
        end_idx = start_idx + count
        sorted_copy.loc[nan_rows.index[start_idx:end_idx], 'Cabin'] = cabin
        start_idx = end_idx


# Embarked 공백 채우기 > 최빈값(fare의 +-20범위에서)
sorted_copy['Embarked'] = sorted_copy['Embarked'].fillna('S')

# save
output_file_path = '/content/sorting_updated.csv'
sorted_copy.to_csv(output_file_path, index=False)

print(f"{output_file_path}")