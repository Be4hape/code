import pandas as pd

file_path = '/content/sorted_titanic_data.csv'
sorting_data = pd.read_csv(file_path)

# 60~100의 요금을 낸 사람들 indexing
filtered_fare_data = sorting_data[(sorting_data['Fare'] >= 60) & (sorting_data['Fare'] <= 100)]
embarked_mode = filtered_fare_data['Embarked'].mode()[0]

print(f"{embarked_mode}")

