import matplotlib.pyplot as plt

# 데이터
results = {
    'Test Set': ['Test1', 'Test2'],
    'Previous Result': [0.77666, 0.67703],
    'Current Result': [0.67003, 0.62918],
    'Improved Result': [0.80584, 0.79425]
}

# 그래프 그리기
x = range(len(results['Test Set']))
width = 0.25  # 막대 폭

# 막대 그래프
plt.bar([i - width for i in x], results['Previous Result'], width=width, label='Previous Result', alpha=0.8, color='yellow', edgecolor='black', linewidth=1.5)
plt.bar(x, results['Current Result'], width=width, label='Current Result', alpha=0.8, color='blue', edgecolor='black', linewidth=1.5)
plt.bar([i + width for i in x], results['Improved Result'], width=width, label='Improved Result', alpha=0.8, color='green', edgecolor='black', linewidth=1.5)

# x축 설정
plt.xticks(x, results['Test Set'])

# 그래프 꾸미기
plt.ylabel('Accuracy')
plt.title('Comparison of Results')
plt.legend()

# 그래프 표시
plt.show()
