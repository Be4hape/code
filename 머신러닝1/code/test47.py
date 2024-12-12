import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 데이터 로드
data = pd.read_csv('process1_result.csv')  # 파일 경로 수정 필요

# Survival Score의 중앙값 계산
median_value = data['SurvivalScore'].median()

# 예측 결과 추가
data['PredictedSurvival'] = (data['SurvivalScore'] > median_value).astype(int)

# 정확도 계산 (Survived 열이 있는 경우에만)
if 'Survived' in data.columns:
    accuracy = accuracy_score(data['Survived'], data['PredictedSurvival'])
    print(f"Accuracy of Prediction: {accuracy * 100:.5f}%")

# Survival Score 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(data['SurvivalScore'], bins=50, alpha=0.75, edgecolor='black', label='SurvivalScore')
plt.axvline(median_value, color='red', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')
plt.title('Distribution of Survival Score with Median Threshold', fontsize=16)
plt.xlabel('Survival Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()


# 주요 결과 확인
print(data[['Survived', 'PredictedSurvival']].head())
