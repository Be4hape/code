import numpy as np
import matplotlib.pyplot as plt

# 모델별 Test1/Test2 원본 및 새로운 정확도
baseline_test1 = 72.95
baseline_test2 = 72.488

models = ['KNN', 'Logistic', 'DecisionTree']
test1_scores = [80.47, 79.12, 80.70]
test2_scores = [77.033, 77.511, 76.555]

# 증가량 계산
delta_test1 = [t1 - baseline_test1 for t1 in test1_scores]
delta_test2 = [t2 - baseline_test2 for t2 in test2_scores]

# x축 위치
x = np.arange(len(models))

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(x, delta_test1, marker='o', linestyle='None', label='Δ Test1')
plt.plot(x, delta_test2, marker='o', linestyle='None', label='Δ Test2')

# 레이블 및 설정
plt.xticks(x, models)
plt.xlabel('Model')
plt.ylabel('Increase in Accuracy (%)')
plt.title('Increase in Accuracy over Baseline (Threshold)')
plt.legend(
    loc='upper center'
)
plt.grid(True)
plt.show()
