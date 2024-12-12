import pandas as pd
import numpy as np


file_path = '/content/weighted_survival_scores.csv'
data = pd.read_csv(file_path)

data = data.sort_values(by='Weighted_Survival_Score').reset_index(drop=True)


median_value = np.median(data['Weighted_Survival_Score'])
print(f"Weighted Survival Score의 중앙값 (Threshold): {median_value}")


data['newsurvive'] = (data['Weighted_Survival_Score'] >= median_value).astype(int)


print("newsurvive Distribution:")
print(data['newsurvive'].value_counts())


output_file = '/content/survival_predictions_sorted.csv'
data.to_csv(output_file, index=False)

print(f"survival prediction 저장 '{output_file}'")
