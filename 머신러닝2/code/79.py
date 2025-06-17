import numpy as np
import pandas as pd

sub_pca = pd.read_csv('submission_pca_dt10.csv')    # PCA+DT 모델
sub5    = pd.read_csv('submission_mlp_full.csv')   # 5-feature MLP 모델

sub_pca = sub_pca.sort_values('PassengerId').reset_index(drop=True)
sub5    = sub5.sort_values('PassengerId').reset_index(drop=True)

w_mlp = 0.51
w_pca = 0.49
votes = w_pca * sub_pca['Survived'].values + w_mlp * sub5['Survived'].values

# 평균 ≥ 0.5 → 1, else 0
final = (votes >= 0.5).astype(int)

submission = pd.DataFrame({
    'PassengerId': sub_pca['PassengerId'],
    'Survived':    final
})
submission.to_csv('submission_weighted_0.49_0.51.csv', index=False)
print("Saved → submission_weighted_0.49_0.51.csv")
