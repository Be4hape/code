import pandas as pd

# 1) 두 모델의 결과 불러오기
sub_pca = pd.read_csv('submission_pca_dt10.csv')      # PCA+DT 모델
sub5    = pd.read_csv('submission_mlp_full.csv')     # 기존 5-feature MLP 최고 모델

# 2) 병합
df = pd.merge(sub_pca, sub5, on='PassengerId', suffixes=('_pca','_5feat'))

# 3) OR 룰 적용
#    둘 중 하나라도 생존(1)이라 예측하면 1, 둘 다 0일 때만 0
df['Survived'] = ((df['Survived_pca'] == 1) | (df['Survived_5feat'] == 1)).astype(int)

# 4) 제출 파일 생성
df[['PassengerId','Survived']] \
  .to_csv('submission_ensemble_or.csv', index=False)

print("Saved → submission_ensemble_or.csv")
