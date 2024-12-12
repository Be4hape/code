import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
data = pd.read_csv('train.csv')  # 파일 경로에 맞게 변경

# Calculate survival and mortality rates by gender
gender_stats = data.groupby('Sex')['Survived'].value_counts(normalize=True).unstack().fillna(0)
gender_stats.columns = ['Mortality_Rate', 'Survival_Rate']

# Plot survival and mortality rates by gender
gender_stats.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 6),
    color=['blue', 'yellow'],
    alpha=0.85,
    edgecolor = 'black'
)

plt.title('Survival Rates by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Rate', fontsize=12)
plt.ylim(0, 1)  # Rate ranges from 0 to 1
plt.legend(['Mortality Rate', 'Survival Rate'], loc='upper right')
plt.xticks(rotation=0)
plt.show()
