import numpy as np
import matplotlib.pyplot as plt

# 모델별 정확도 데이터
models = ['Threshold', 'KNN', 'Logistic', 'DecisionTree']
test1_scores = [72.95, 80.47, 79.12, 80.70]
test2_scores = [72.488, 77.033, 77.511, 76.555]

# x축 위치 설정
x = np.arange(len(models))

# 선 그래프 그리기 (marker='o', linestyle='-')
plt.figure(figsize=(8, 5))
plt.plot(x, test1_scores, marker='o', linestyle='-', label='Test1')
plt.plot(x, test2_scores, marker='o', linestyle='-', label='Test2')

# x축 눈금 설정 및 라벨 지정
plt.xticks(x, models)
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Performance: Test1 vs Test2 (Line Chart)')
plt.legend()
plt.grid(True)
plt.show()
