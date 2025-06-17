import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# --- 1. 데이터 로드 & full feature 추출 ---
df1 = pd.read_csv('process1_result.csv')   # Train + Test1 데이터
df2 = pd.read_csv('process2_result.csv')   # Test2 (제출용)

features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X1_full = df1[features].values      # (n_train_all, 5)
y1_full = df1['Survived'].values    # (n_train_all,)
X2_full = df2[features].values      # (n_test2, 5)
ids2    = df2['PassengerId'].values # (n_test2,)

# --- 2. Train/Test1 분할 (80:20) ---
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
cov_mat = np.cov(X_train_s, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov_mat)
idx_sort = np.argsort(eigvals)[::-1]
V_pca = eigvecs[:, idx_sort[:4]]      # (5,4)

X_train_pca = X_train_s.dot(V_pca)  # (n_train, 4)
X_test1_pca = X_test1_s.dot(V_pca)  # (n_test1, 4)
X2_pca      = X2_s.dot(V_pca)       # (n_test2, 4)

# --- 5. LDA (Fisher’s Linear Discriminant) 계산 ---
mu0 = X_train_s[y_train == 0].mean(axis=0)
mu1 = X_train_s[y_train == 1].mean(axis=0)

d = X_train_s.shape[1]
SW = np.zeros((d, d))
for x_i, y_i in zip(X_train_s, y_train):
    center = mu1 if y_i == 1 else mu0
    diff = (x_i - center).reshape(-1, 1)
    SW += diff.dot(diff.T)

w_fld = np.linalg.inv(SW).dot(mu1 - mu0)
w_fld /= np.linalg.norm(w_fld)

X_train_fld = X_train_s.dot(w_fld).reshape(-1, 1)  # (n_train, 1)
X_test1_fld = X_test1_s.dot(w_fld).reshape(-1, 1)  # (n_test1, 1)
X2_fld      = X2_s.dot(w_fld).reshape(-1, 1)       # (n_test2, 1)

# --- 6. MLPClassifier 공통 파라미터 ---
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

# --- 7. MLP 실행 함수 (Test1 F1-score, Test2 제출) ---
def run_mlp(X_tr, y_tr, X_t1, y_t1, X_t2, ids_t2, label):
    mlp = MLPClassifier(**mlp_params)
    mlp.fit(X_tr, y_tr)
    preds_t1 = mlp.predict(X_t1)
    f1_t1 = f1_score(y_t1, preds_t1) * 100
    print(f"MLP ({label}) → Test1 F1-score: {f1_t1:.2f}%")
    preds_t2 = mlp.predict(X_t2)
    df_sub = pd.DataFrame({
        'PassengerId': ids_t2,
        'Survived':    preds_t2
    })
    df_sub.to_csv(f'submission_mlp_{label}.csv', index=False)
    print(f"Saved submission_mlp_{label}.csv\n")

# --- 8. Full feature MLP 실행 ---
run_mlp(
    X_tr = X_train_s, y_tr = y_train,
    X_t1 = X_test1_s, y_t1 = y_test1,
    X_t2 = X2_s, ids_t2 = ids2,
    label = 'full'
)

# --- 9. PCA 기반 MLP 실행 ---
run_mlp(
    X_tr = X_train_pca, y_tr = y_train,
    X_t1 = X_test1_pca, y_t1 = y_test1,
    X_t2 = X2_pca, ids_t2 = ids2,
    label = 'pca'
)

# --- 10. LDA 기반 MLP 실행 ---
run_mlp(
    X_tr = X_train_fld, y_tr = y_train,
    X_t1 = X_test1_fld, y_t1 = y_test1,
    X_t2 = X2_fld, ids_t2 = ids2,
    label = 'fld'
)
