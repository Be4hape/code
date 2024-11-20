import pandas as pd

test1_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\test1.csv')
test2_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\test2.csv')
train_data = pd.read_csv(r'C:\Users\for\Desktop\sc\24-2Graphics\machine1\train.csv')

print(train_data.dtypes)