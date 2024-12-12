import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = '/content/sorting_updated (1).csv'
data = pd.read_csv(file_path)

# age 결측치 제거
data = data.dropna(subset=['Age'])

# Sex > male : 0, female : 1
if 'Sex' in data.columns:
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# 정수형 칼럼만 선택
numeric_columns = data.select_dtypes(include=['float64', 'int64'])

# correlation
correlation_matrix = numeric_columns.corr()

### hitmap figure
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='RdYlGn', interpolation='none', aspect='auto')

plt.colorbar(label='Correlation Coefficient')

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

# correlation 값 텍스트 추가.
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        value = correlation_matrix.iloc[i, j]
        plt.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize=8)

plt.show()