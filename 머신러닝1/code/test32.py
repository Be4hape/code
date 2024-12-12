## Knn모델 > test1의 결과

import pandas as pd

# 데이터 로드
train_data = pd.read_csv('/content/train.csv')
test1_results = pd.read_csv('/content/test1_results.csv')

# train_data에서 test1에 해당하는 PassengerId의 실제 Survived 값 추출
test1_actual = train_data[train_data['PassengerId'].isin(test1_results['PassengerId'])]

# 예측값과 실제값 병합
comparison = test1_results.merge(test1_actual[['PassengerId', 'Survived']], on='PassengerId', suffixes=('_predicted', '_actual'))

# 정확도 계산
correct_predictions = (comparison['Survived_predicted'] == comparison['Survived_actual']).sum()
accuracy = correct_predictions / len(comparison)

print(f"Accuracy Test1: {accuracy:.5f}")
