import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train.csv')  # 파일명을 적절히 변경하세요

# Select numeric columns and drop rows with missing values
numeric_data = data.select_dtypes(include=['float64', 'int64'])
numeric_data_clean = numeric_data.dropna()

# Compute the correlation matrix
correlation_matrix = numeric_data_clean.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap (Excluding Missing Values)', fontsize=16)
plt.show()
