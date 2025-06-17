import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & full feature 추출 ---
df1 = pd.read_csv('process1_result.csv')   # Train + Test1 데이터
df2 = pd.read_csv('process2_result.csv')   # Test2 (제출용)

features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X1_full = df1[features].values      # (n_train_all, 5)
y1_full = df1['Survived'].values    # (n_train_all,)
X2_full = df2[features].values      # (n_test2, 5)
ids2    = df2['PassengerId'].values # (n_test2,)

# --- 2. Train/Test1 분할 (numpy로 직접, 80:20) ---
np.random.seed(42)
indices = np.arange(len(X1_full))
np.random.shuffle(indices)

split = int(0.8 * len(indices))
train_idx = indices[:split]
test1_idx = indices[split:]

X_train = X1_full[train_idx]   # (n_train, 5)
y_train = y1_full[train_idx]   # (n_train,)
X_test1 = X1_full[test1_idx]   # (n_test1, 5)
y_test1 = y1_full[test1_idx]   # (n_test1,)

# --- 3. Zero-centering & Unit-variance 스케일링 (Train 기준) ---
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0

X_train_s = (X_train - means) / stds   # (n_train, 5)
X_test1_s = (X_test1 - means) / stds   # (n_test1, 5)
X2_s      = (X2_full - means) / stds   # (n_test2, 5)

# --- 4. PCA 계산 (상위 4개 주성분) ---
cov_mat = np.cov(X_train_s, rowvar=False)         # (5,5) 공분산 행렬
eigvals, eigvecs = np.linalg.eigh(cov_mat)        # 고유값, 고유벡터
idx_sort = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx_sort]                    # 내림차순 정렬

k_pca = 4
V_pca = eigvecs[:, :k_pca]                        # (5,4)

X_train_pca = X_train_s.dot(V_pca)   # (n_train, 4)
X_test1_pca = X_test1_s.dot(V_pca)   # (n_test1, 4)
X2_pca      = X2_s.dot(V_pca)        # (n_test2, 4)

# --- 5. LDA (Fisher’s Linear Discriminant) 계산 ---
# 5.1 클래스별 평균
mu0 = X_train_s[y_train == 0].mean(axis=0)  # (5,)
mu1 = X_train_s[y_train == 1].mean(axis=0)  # (5,)

# 5.2 클래스 내 산포 행렬 SW
d = X_train_s.shape[1]
SW = np.zeros((d, d))
for x_i, y_i in zip(X_train_s, y_train):
    center = mu1 if y_i == 1 else mu0
    diff = (x_i - center).reshape(-1, 1)
    SW += diff.dot(diff.T)

# 5.3 판별 벡터 w_fld
w_fld = np.linalg.inv(SW).dot(mu1 - mu0)   # (5,)
w_fld /= np.linalg.norm(w_fld)

# 5.4 1차원 투사
X_train_fld = X_train_s.dot(w_fld).reshape(-1, 1)  # (n_train, 1)
X_test1_fld = X_test1_s.dot(w_fld).reshape(-1, 1)  # (n_test1, 1)
X2_fld      = X2_s.dot(w_fld).reshape(-1, 1)       # (n_test2, 1)

# --- 6. MLPClassifier 설정 (은닉층=1, 노드=32; lr=0.005; alpha=1e-4; max_iter=300) ---
mlp_params = {
    'hidden_layer_sizes': (32,),
    'activation': 'relu',
    'solver': 'adam',
    'learning_rate': 'constant',
    'learning_rate_init': 0.005,
    'alpha': 1e-4,
    'batch_size': 32,
    'max_iter': 300,
    'shuffle': True,
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 10
}

# --- 7. 함수 정의: MLP 학습→Test1 평가→Test2 제출 ---
def train_evaluate_mlp(X_tr, y_tr, X_t1, y_t1, X_t2, ids_t2, suffix):
    # MLP 모델 생성
    mlp = MLPClassifier(**mlp_params)

    # 1) Train 학습
    mlp.fit(X_tr, y_tr)

    # 2) Test1 예측 및 정확도
    preds_t1 = mlp.predict(X_t1)
    acc_t1 = accuracy_score(y_t1, preds_t1) * 100
    print(f"MLP ({suffix}) → Test1 Accuracy: {acc_t1:.2f}%")

    # 3) Test2 예측 → 제출 파일 생성
    preds_t2 = mlp.predict(X_t2)
    submission_df = pd.DataFrame({
        'PassengerId': ids_t2,
        'Survived':    preds_t2
    })
    submission_df.to_csv(f'submission_mlp_{suffix}.csv', index=False)
    print(f"Saved: submission_mlp_{suffix}.csv\n")

# --- 8. PCA 기반 MLP 실행 ---
train_evaluate_mlp(
    X_tr = X_train_pca, y_tr = y_train,
    X_t1 = X_test1_pca, y_t1 = y_test1,
    X_t2 = X2_pca, ids_t2 = ids2,
    suffix = 'pca'
)

# --- 9. LDA 기반 MLP 실행 ---
train_evaluate_mlp(
    X_tr = X_train_fld, y_tr = y_train,
    X_t1 = X_test1_fld, y_t1 = y_test1,
    X_t2 = X2_fld, ids_t2 = ids2,
    suffix = 'fld'
)
