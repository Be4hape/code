## 데이터 전처리 (name, sex, ticket, cabin, embarked) 이후
## 상관관계 - 피어슨 방식

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the processed dataset
data = pd.read_csv('process1.csv')

# Convert 'Ticket' to numeric if possible, otherwise set to NaN
data['TicketNumeric'] = pd.to_numeric(data['Ticket'], errors='coerce')

# Select rows where 'TicketNumeric' is not NaN (valid numeric tickets)
numeric_ticket_data = data.dropna(subset=['TicketNumeric'])

# Select all numeric columns for correlation calculation, including 'TicketNumeric'
numeric_columns = numeric_ticket_data.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap (Including Numeric Tickets)', fontsize=16)
plt.show()
