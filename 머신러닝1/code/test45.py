import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('process1_result.csv')  # 파일 경로 수정 필요

# Survival Score 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(data['SurvivalScore'], bins=50, alpha=0.75, edgecolor='black')
plt.title('Distribution of Survival Score', fontsize=16)
plt.xlabel('Survival Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
