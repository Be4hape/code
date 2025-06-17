import numpy as np
import matplotlib.pyplot as plt

# 모델별 Test1/Test2 정확도 데이터
models = ['Threshold', 'KNN', 'Logistic', 'DecisionTree']
test1_scores = [72.95, 80.47, 79.12, 80.70]
test2_scores = [72.488, 77.033, 77.511, 76.555]

# x축 위치 및 막대 너비 설정
x = np.arange(len(models))
width = 0.35

# 그래프 그리기
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, test1_scores, width, label='Test1 Accuracy')
ax.bar(x + width/2, test2_scores, width, label='Test2 Accuracy')

# 레이블 및 타이틀
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Performance: Test1 vs Test2')
ax.legend(
    fontsize=8
)
ax.grid(axis='y')

plt.show()
