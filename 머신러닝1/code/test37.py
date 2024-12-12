import pandas as pd
import re

# Load the Titanic dataset
data = pd.read_csv('train.csv')  # 파일 경로에 맞게 변경

# Function to extract middle names only (ignore titles or parenthetical names)
def extract_middle_name(name):
    match = re.search(r', ([^\.]+)\.', name)
    if match:
        return match.group(1).strip()
    return 'Unknown'

# Apply the function to extract middle names
data['MiddleName'] = data['Name'].apply(extract_middle_name)

# Split data into survival groups and count occurrences of each middle name
middle_name_survival = data.groupby(['MiddleName', 'Survived']).size().unstack(fill_value=0)

# Rename columns for clarity
middle_name_survival.columns = ['Non_Survived_Count', 'Survived_Count']

# Sort by Non_Survived_Count in descending order and reset index
middle_name_survival = middle_name_survival.sort_values(by='Non_Survived_Count', ascending=False).reset_index()

# Assign a new rank/ID to each middle name based on Non_Survived_Count
middle_name_survival['Non_Survived_Rank'] = range(0, len(middle_name_survival))

# Display or save the results
print(middle_name_survival)
