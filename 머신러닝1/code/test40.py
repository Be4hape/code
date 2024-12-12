import pandas as pd

# Load the Titanic dataset
data = pd.read_csv('train.csv')  # 파일 경로에 맞게 변경

# Fill missing values in 'Embarked' with the mode (most frequent value)
most_frequent_embarked = data['Embarked'].mode()[0]  # Find the mode
data['Embarked'].fillna(most_frequent_embarked, inplace=True)  # Fill missing values

# Display the updated data
print(f"The most frequent value in 'Embarked' is: {most_frequent_embarked}")
print(data['Embarked'].value_counts())
