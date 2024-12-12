import pandas as pd


file_path = '/content/train.csv'
train_data = pd.read_csv(file_path)

# count to Cabin
def count_cabin_values(data, pclass):
    class_data = data[data['Pclass'] == pclass]
    cabin_counts = class_data['Cabin'].dropna().str[0].value_counts()
    cabin_counts = cabin_counts.reindex(['A', 'B', 'C', 'D', 'E', 'F', 'G'], fill_value=0)  # A부터 G까지의 값
    other_counts = class_data['Cabin'].dropna().str[0].value_counts().drop(['A', 'B', 'C', 'D', 'E', 'F', 'G'], errors='ignore').sum()
    nan_counts = class_data['Cabin'].isnull().sum()
    return cabin_counts, other_counts, nan_counts

# for p에 대해, cabin 값 카운트
for pclass in [1, 2, 3]:
    cabin_counts, other_counts, nan_counts = count_cabin_values(train_data, pclass)
    print(f"\nPclass {pclass} Cabin 값 카운트:")
    for cabin, count in cabin_counts.items():
        print(f"{cabin}: {count}")
    print(f"Other: {other_counts}")
    print(f"NaN: {nan_counts}")

