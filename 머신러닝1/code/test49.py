## test2에 대해, process2를 만드는 코드



import pandas as pd
import re

# Load the Titanic test dataset
test_data = pd.read_csv('test2.csv')  # 파일 경로에 맞게 변경

# 1. Name의 Middle Name 추출 및 non-survival count 내림차순으로 번호 부여 (훈련 데이터 기반)
def extract_middle_name(name):
    match = re.search(r', ([^\.]+)\.', name)
    return match.group(1).strip() if match else 'Unknown'

test_data['MiddleName'] = test_data['Name'].apply(extract_middle_name)
test_data['MiddleNameRank'] = test_data['MiddleName'].map(middle_name_rank).fillna(len(middle_name_rank)).astype(int)

# 2. 남성은 0, 여성은 1로 번호 부여
test_data['SexEncoded'] = test_data['Sex'].map({'male': 0, 'female': 1})

# 3. Ticket의 문자만 추출, 숫자만 있는 데이터는 그대로 유지
def preprocess_ticket(ticket):
    if any(char.isalpha() for char in ticket):
        return ''.join([char for char in ticket if char.isalpha()])
    return ticket

test_data['ProcessedTicket'] = test_data['Ticket'].apply(preprocess_ticket)

# 4. Embarked 결측치는 최빈값으로 채우고, 가장 많은 데이터부터 내림차순으로 번호 부여 (훈련 데이터 기반)
test_data['Embarked'].fillna(most_frequent_embarked, inplace=True)
test_data['EmbarkedRank'] = test_data['Embarked'].map(embarked_rank)

# Display the processed test data
print(test_data[['PassengerId', 'Name', 'MiddleName', 'MiddleNameRank', 'Sex', 'SexEncoded',
                 'Ticket', 'ProcessedTicket', 'Embarked', 'EmbarkedRank']].head())

# Save the processed test data to a new CSV file
test_data.to_csv('process2.csv', index=False)

print("The processed test data has been saved as 'process2.csv'.")

# Overwrite the original columns with the processed values
test_data['Name'] = test_data['MiddleNameRank']       # Replace 'Name' with 'MiddleNameRank'
test_data['Sex'] = test_data['SexEncoded']            # Replace 'Sex' with 'SexEncoded'
test_data['Ticket'] = test_data['ProcessedTicket']    # Replace 'Ticket' with 'ProcessedTicket'
test_data['Embarked'] = test_data['EmbarkedRank']     # Replace 'Embarked' with 'EmbarkedRank'

# Drop the temporary processed columns
test_data.drop(columns=['MiddleNameRank', 'SexEncoded', 'ProcessedTicket', 'EmbarkedRank'], inplace=True, errors='ignore')

# Save the updated test DataFrame to process2.csv
test_data.to_csv('process2.csv', index=False)

print("The original columns, including 'Name', have been overwritten with processed values and saved to 'process2.csv'.")
