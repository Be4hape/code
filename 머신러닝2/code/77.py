import pandas as pd

# 1) 두 모델의 결과 불러오기
sub_pca = pd.read_csv('submission_pca_dt10.csv')      # PCA+DT 모델
sub5    = pd.read_csv('submission_mlp_full.csv')  # 기존 5-feature 최고 모델

# 2) 병합 (PassengerId 기준)
df = pd.merge(sub_pca, sub5, on='PassengerId', suffixes=('_pca', '_5feat'))

# 3) AND 룰 적용
#    두 모델이 모두 1(생존)일 때만 1, 아니면 0
df['Survived'] = ((df['Survived_pca'] == 1) & (df['Survived_5feat'] == 1)).astype(int)

# 4) 최종 제출 파일 만들기
final = df[['PassengerId', 'Survived']]
final.to_csv('submission_ensemble_and.csv', index=False)
print("Saved → submission_ensemble_and.csv")
