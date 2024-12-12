# import pandas as pd

# file_path = '/content/train.csv'
# train_data = pd.read_csv(file_path)

# train_data['Cabin_Letter'] = train_data['Cabin'].dropna().str[0]
# non_nan_data = train_data[train_data['Cabin'].notnull()]
# survivors_count = non_nan_data[non_nan_data['Survived'] == 1].groupby('Cabin_Letter')['Survived'].count()
# deceased_count = non_nan_data[non_nan_data['Survived'] == 0].groupby('Cabin_Letter')['Survived'].count()

# print("Cabin별 생존자와 사망자 수 (NaN이 아닌 경우):")
# all_cabin_letters = sorted(set(non_nan_data['Cabin_Letter']))

# for cabin in all_cabin_letters:
#     survivors = survivors_count.get(cabin, 0)
#     deceased = deceased_count.get(cabin, 0)
#     print(f"{cabin}인 사람 중 생존자: {survivors}")
#     print(f"{cabin}인 사람 중 사망자: {deceased}")
