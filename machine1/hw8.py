import pandas as pd
import matplotlib.pyplot as plt

test1_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\test1.csv')
test2_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\test2.csv')
train_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\train.csv')

#나이는 평균으로, 항구위치는 최빈값으로.
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

#passengerID, name, Cabin 제거.
train_data = train_data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

train_data.hist()
plt.show()