import pandas as pd
import matplotlib.pyplot as plt

test1_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\test1.csv')
test2_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\test2.csv')
train_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\train.csv')

#나이는 평균으로, 항구위치는 최빈값으로.
train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

#passengerID, name, Cabin 제거.
train_data = train_data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


###


#seperate Survivors, and non Survivors
survivors = train_data[train_data['Survived'] == 1]
non_survivors = train_data[train_data['Survived'] == 0]

# 1.gender
gender_survival = train_data.groupby('Sex')['Survived'].mean()

plt.figure()
gender_survival.plot(kind='bar')
plt.xlabel('gender')
plt.ylabel('survival rate')


# 2.age
plt.figure()
plt.hist(survivors['Age'].dropna(), alpha=0.5, label='survived', color='black')
plt.hist(non_survivors['Age'].dropna(), alpha=0.3, label='not survived', color='blue')
plt.xlabel('age')
plt.ylabel('count')
plt.legend()


# 3.pclass
pclass_survival = train_data.groupby('Pclass')['Survived'].mean()

plt.figure()
pclass_survival.plot(kind='bar')
plt.xlabel('pclass')
plt.ylabel('survival rate')
plt.xticks(rotation=0)
plt.show()