import pandas as pd

# 두 모델 결과 불러오기
sub_pca = pd.read_csv('submission_pca_dt10.csv')
sub5    = pd.read_csv('submission_mlp_full.csv')

# PassengerId 기준으로 병합
df = pd.merge(sub_pca, sub5, on='PassengerId',
              suffixes=('_pca','_5feat'))

# 1) 두 예측이 몇 퍼센트 동일한지
same_cnt = (df['Survived_pca'] == df['Survived_5feat']).sum()
total   = len(df)
print(f"Identical predictions: {same_cnt}/{total} ({same_cnt/total*100:.2f}%)")

# 2) 각 모델이 예측한 생존 수
print("PCA+DT survivors :", df['Survived_pca'].sum())
print("5-feat MLP survivors:", df['Survived_5feat'].sum())

# 3) AND 했을 때 survivor 개수
and_cnt = ((df['Survived_pca']==1) & (df['Survived_5feat']==1)).sum()
print("AND intersection survivors:", and_cnt)

# 4) 차이 나는 PassengerId 예시
diff = df[df['Survived_pca'] != df['Survived_5feat']]
print("Disagreements (up to 10 rows):")
print(diff.head(10))
