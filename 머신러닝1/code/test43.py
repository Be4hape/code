## 데이터 전처리 (name, sex, ticket, cabin, embarked) 이후
## 상관관계 - 스피어만 방식

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the processed dataset
data = pd.read_csv('process1.csv')

# Convert 'Ticket' to numeric if possible, otherwise set to NaN
data['TicketNumeric'] = pd.to_numeric(data['Ticket'], errors='coerce')

# Select rows where 'TicketNumeric' is not NaN
numeric_ticket_data = data.dropna(subset=['TicketNumeric'])

# Select all numeric columns for correlation calculation, including 'TicketNumeric'
numeric_columns = numeric_ticket_data.select_dtypes(include=['number'])

# Calculate the correlation matrix using Spearman method
correlation_matrix_spearman = numeric_columns.corr(method='spearman')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_spearman, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Spearman Correlation Heatmap (Including Numeric Tickets)', fontsize=16)
plt.show()
