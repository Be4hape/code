import pandas as pd
import re

# Load the Titanic dataset
data = pd.read_csv('train.csv')  # 파일 경로에 맞게 변경

# Function to extract middle names only (ignore titles or parenthetical names)
def extract_middle_name(name):
    # Match patterns for titles or parenthetical names
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

# Sort the results by Survived_Count and Non_Survived_Count
middle_name_survival = middle_name_survival.sort_values(by=['Survived_Count', 'Non_Survived_Count'], ascending=False)

# Display or save the results
print(middle_name_survival)
