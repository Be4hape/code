import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = '/content/weighted_survival_scores.csv'
data = pd.read_csv(file_path)

scores = data['Weighted_Survival_Score']

### 시각화
hist, bin_edges = np.histogram(scores, bins=30)

# 꼭짓점
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.figure(figsize=(12, 6))
plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')

# 꼭짓점 연결
plt.plot(bin_centers, hist, marker='o', color='blue', label='Connected Peaks')

plt.xlabel('score')
plt.ylabel('fre')
plt.title('Histogram SURVIVAL SCORE')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
