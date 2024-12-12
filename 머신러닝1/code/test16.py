import pandas as pd
import numpy as np

file_path = '/content/weighted_survival_scores.csv'  # 파일 경로 수정 필요
data = pd.read_csv(file_path)

# Weighted_Survival_Score > 인덱싱
scores = data['Weighted_Survival_Score']

# 중앙값
median_value = np.median(scores)
print(f"survival score의중앙값: {median_value}")
