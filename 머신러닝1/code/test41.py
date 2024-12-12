import pandas as pd
import re

# Load the Titanic dataset
data = pd.read_csv('train.csv')  # 파일 경로에 맞게 변경

# 1. Name의 Middle Name 추출 및 non-survival count 내림차순으로 번호 부여
def extract_middle_name(name):
    match = re.search(r', ([^\.]+)\.', name)
    return match.group(1).strip() if match else 'Unknown'

data['MiddleName'] = data['Name'].apply(extract_middle_name)
non_survival_counts = data[data['Survived'] == 0]['MiddleName'].value_counts()
middle_name_rank = {name: rank for rank, name in enumerate(non_survival_counts.index)}
data['MiddleNameRank'] = data['MiddleName'].map(middle_name_rank).fillna(len(middle_name_rank)).astype(int)

# 2. 남성은 0, 여성은 1로 번호 부여
data['SexEncoded'] = data['Sex'].map({'male': 0, 'female': 1})

# 3. Ticket의 문자만 추출, 숫자만 있는 데이터는 그대로 유지
def preprocess_ticket(ticket):
    if any(char.isalpha() for char in ticket):
        return ''.join([char for char in ticket if char.isalpha()])
    return ticket

data['ProcessedTicket'] = data['Ticket'].apply(preprocess_ticket)

# 4. Embarked 결측치는 최빈값으로 채우고, 가장 많은 데이터부터 내림차순으로 번호 부여
most_frequent_embarked = data['Embarked'].mode()[0]
data['Embarked'].fillna(most_frequent_embarked, inplace=True)
embarked_counts = data['Embarked'].value_counts()
embarked_rank = {embarked: rank for rank, embarked in enumerate(embarked_counts.index)}
data['EmbarkedRank'] = data['Embarked'].map(embarked_rank)

# Display the processed data
print(data[['Name', 'MiddleName', 'MiddleNameRank', 'Sex', 'SexEncoded',
            'Ticket', 'ProcessedTicket', 'Embarked', 'EmbarkedRank']].head())

# Save the processed data to a new CSV file
data.to_csv('process1.csv', index=False)

print("The processed data has been saved as 'process1.csv'.")

# Overwrite the original columns with the processed values
data['Name'] = data['MiddleNameRank']       # Replace 'Name' with 'MiddleNameRank'
data['Sex'] = data['SexEncoded']            # Replace 'Sex' with 'SexEncoded'
data['Ticket'] = data['ProcessedTicket']    # Replace 'Ticket' with 'ProcessedTicket'
data['Embarked'] = data['EmbarkedRank']     # Replace 'Embarked' with 'EmbarkedRank'

# Drop the temporary processed columns
data.drop(columns=['MiddleNameRank', 'SexEncoded', 'ProcessedTicket', 'EmbarkedRank'], inplace=True, errors='ignore')

# Save the updated DataFrame to process1.csv
data.to_csv('process1.csv', index=False)

print("The original columns, including 'Name', have been overwritten with processed values and saved to 'process1.csv'.")

