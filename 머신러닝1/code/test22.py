## 이전결과, 현재 결과 비교하는 히스토그램.

import matplotlib.pyplot as plt

results = {
    'Test Set': ['Test1', 'Test2'],
    'Previous Result': [0.77666, 0.67703],
    'Current Result': [0.67003, 0.62918]
}

x = range(len(results['Test Set']))

plt.bar(x, results['Previous Result'], width=0.25, label='Previous Result', alpha = 1, color = 'yellow', align='center', edgecolor = 'black', linewidth = 1.5)
plt.bar([i + width for i in x], results['Current Result'], width=0.25, label='Current Result', alpha = 0.7, color = 'blue', align='center', edgecolor = 'black', linewidth = 1.5)

plt.xticks([i + width / 2 for i in x], results['Test Set'])
plt.ylabel('accuracy')
plt.title('Result compare')
plt.legend()

plt.show()
