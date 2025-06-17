import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. process1_result.csv 로드 (이미 Age, Sex 등 전처리 완료된 상태)
df = pd.read_csv('process1_result.csv')

# 2. 사용 가능한 숫자형 피처만 추리기 (PassengerId, Survived 제외)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric_cols if c not in ['PassengerId', 'Survived']]

# 3. X (입력), y (클래스)
X = df[features].values
y = df['Survived'].values

# 4. Z-score 표준화 (평균 0, 표준편차 1)
means = X.mean(axis=0)
stds  = X.std(axis=0)
stds[stds == 0] = 1.0
Z = (X - means) / stds

# 5. SW (within-class scatter) 계산
classes = np.unique(y)
d = Z.shape[1]
SW = np.zeros((d, d))
for cls in classes:
    Z_cls = Z[y == cls]
    mean_cls = Z_cls.mean(axis=0)
    diff = Z_cls - mean_cls
    SW += diff.T.dot(diff)

# 6. SB (between-class scatter) 계산
mean_global = Z.mean(axis=0)
SB = np.zeros((d, d))
for cls in classes:
    Z_cls = Z[y == cls]
    n_cls = Z_cls.shape[0]
    mean_cls = Z_cls.mean(axis=0)
    diff_mean = (mean_cls - mean_global).reshape(-1, 1)
    SB += n_cls * diff_mean.dot(diff_mean.T)

# 7. 일반화된 고유값 문제 풀기: inv(SW) * SB
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(SW).dot(SB))

# 8. 고유값을 실수부만 뽑아내고 내림차순 정렬
eigvals_real = np.real(eigvals)
sorted_idx   = np.argsort(eigvals_real)[::-1]
eigvals_sorted = eigvals_real[sorted_idx]

# 9. 고유값 그래프 그리기
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(eigvals_sorted) + 1), eigvals_sorted, marker='o')
plt.title('LDA Eigenvalues')
plt.xlabel('Component Index')
plt.ylabel('Eigenvalue (Descending)')
plt.xticks(range(1, len(eigvals_sorted) + 1))
plt.grid(True)
plt.show()
