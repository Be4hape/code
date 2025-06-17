import numpy as np
import pandas as pd

# --- 1. 데이터 로드 & full feature 추출 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_train_full = train_df[features].values    # (n_train, 5)
y_train      = train_df['Survived'].values  # (n_train,)
X_test2_full  = test_df[features].values    # (n_test2, 5)
ids_test2     = test_df['PassengerId'].values

# --- 2. Train/Test1 분할 (80:20, numpy로 직접) ---
np.random.seed(42)
idx = np.arange(len(X_train_full))
np.random.shuffle(idx)

split = int(0.8 * len(idx))
train_idx = idx[:split]
test1_idx = idx[split:]

X_train = X_train_full[train_idx]
y_train = y_train[train_idx]
X_test1 = X_train_full[test1_idx]
y_test1 = y_train_full = train_df['Survived'].values[test1_idx]  # Test1 레이블

# --- 3. 스케일링: Train 기준 zero-centering & unit-variance ---
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0

X_train_s = (X_train - means) / stds
X_test1_s = (X_test1 - means) / stds
X_test2_s = (X_test2_full - means) / stds

# --- 4. PCA ---
# 4.1 공분산 행렬 & 고유분해
cov_mat = np.cov(X_train_s, rowvar=False)                   # (5,5)
eigvals, eigvecs = np.linalg.eigh(cov_mat)
idx_eig = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx_eig]                               # (5,5) 내림차 순

# 4.2 상위 k 주성분 선택 (k=4 예시)
k_pca = 4
V_k = eigvecs[:, :k_pca]                                     # (5,4)

# 4.3 PCA 투사
X_train_pca = X_train_s.dot(V_k)     # (n_train,4)
X_test1_pca = X_test1_s.dot(V_k)     # (n_test1,4)
X_test2_pca = X_test2_s.dot(V_k)     # (n_test2,4)

# --- 5. LDA (Fisher’s Linear Discriminant) ---
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
X_train_fld = X_train_s.dot(w_fld).reshape(-1, 1)  # (n_train,1)
X_test1_fld = X_test1_s.dot(w_fld).reshape(-1, 1)  # (n_test1,1)
X_test2_fld = X_test2_s.dot(w_fld).reshape(-1, 1)  # (n_test2,1)

# --- 6. Logistic Regression (SGD) 구현 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_sgd(X, y, lr=0.001, epochs=100):
    m, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        for i in range(m):
            xi, yi = X[i], y[i]
            z = xi.dot(w) + b
            h = sigmoid(z)
            delta = h - yi
            w -= lr * delta * xi
            b -= lr * delta
    return w, b

def eval_logistic(X_train, y_train, X_eval, y_eval, lr=0.001, epochs=100):
    w, b = train_logistic_sgd(X_train, y_train, lr=lr, epochs=epochs)
    probs = sigmoid(X_eval.dot(w) + b)
    preds = (probs >= 0.5).astype(int)
    acc = np.mean(preds == y_eval) * 100
    return acc, preds

# --- 7. KNN 구현 ---
def euclid_dist_matrix(A, B):
    # (|A|, |B|) pairwise Euclidean distance
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

def knn_predict(train_X, train_y, test_X, k=20):
    D = euclid_dist_matrix(test_X, train_X)  # (n_test, n_train)
    neigh = np.argpartition(D, k, axis=1)[:, :k]  # 각 test 샘플별 k개 이웃 인덱스
    preds = []
    for idxs in neigh:
        counts = np.bincount(train_y[idxs])
        preds.append(np.argmax(counts))
    return np.array(preds)

def eval_knn(X_train, y_train, X_eval, y_eval, k=20):
    preds = knn_predict(X_train, y_train, X_eval, k=k)
    acc = np.mean(preds == y_eval) * 100
    return acc, preds

# --- 8. Decision Tree 구현 ---
def gini(labels):
    classes, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)

def weighted_gini(col, labels, t):
    left  = labels[col <= t]
    right = labels[col > t]
    if len(left) == 0 or len(right) == 0:
        return None
    n = len(labels)
    return (len(left) / n) * gini(left) + (len(right) / n) * gini(right)

def find_best_split(X, y):
    best_gain, best_feat, best_t = 0.0, None, None
    parent = gini(y)
    for feat in range(X.shape[1]):
        vals = np.unique(X[:, feat])
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        for t in thresholds:
            wg = weighted_gini(X[:, feat], y, t)
            if wg is None:
                continue
            gain = parent - wg
            if gain > best_gain:
                best_gain, best_feat, best_t = gain, feat, t
    return best_feat, best_t

def build_tree(X, y, depth=0, max_depth=15):
    if len(np.unique(y)) == 1 or depth == max_depth:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(cnts)]}
    feat, thresh = find_best_split(X, y)
    if feat is None:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(cnts)]}
    mask = X[:, feat] <= thresh
    left  = build_tree(X[mask], y[mask], depth + 1, max_depth)
    right = build_tree(X[~mask], y[~mask], depth + 1, max_depth)
    return {'leaf': False, 'feat': feat, 'thresh': thresh, 'left': left, 'right': right}

def predict_tree(tree, x):
    if tree['leaf']:
        return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

def eval_tree(X_train, y_train, X_eval, y_eval, max_depth=15):
    tree = build_tree(X_train, y_train, max_depth=max_depth)
    preds = np.array([predict_tree(tree, x) for x in X_eval])
    acc = np.mean(preds == y_eval) * 100
    return acc, preds

# --- 9. 알고리즘별 평가 함수 정의 (full/PCA/FLD) ---
def evaluate_all(name, X_tr, y_tr, X_e1, y_e1, X_e2=None, y_e2=None, submission_ids=None):
    print(f"\n---- {name} ----")
    # Logistic
    acc1_log, _ = eval_logistic(X_tr, y_tr, X_e1, y_e1, lr=0.001, epochs=100)
    print(f"Logistic → Test1 ACC: {acc1_log:.2f}%")
    if X_e2 is not None:
        _, preds2_log = eval_logistic(X_tr, y_tr, X_e2, y_e2, lr=0.001, epochs=100)
        if submission_ids is not None:
            pd.DataFrame({'PassengerId': submission_ids, 'Survived': preds2_log})\
              .to_csv(f'submission_logistic_{name}.csv', index=False)
            print(f"Saved submission_logistic_{name}.csv")

    # KNN
    acc1_knn, _ = eval_knn(X_tr, y_tr, X_e1, y_e1, k=20)
    print(f"KNN (k=20) → Test1 ACC: {acc1_knn:.2f}%")
    if X_e2 is not None:
        preds2_knn = knn_predict(X_tr, y_tr, X_e2, k=20)
        if submission_ids is not None:
            pd.DataFrame({'PassengerId': submission_ids, 'Survived': preds2_knn})\
              .to_csv(f'submission_knn_{name}.csv', index=False)
            print(f"Saved submission_knn_{name}.csv")

    # Decision Tree
    acc1_dt, _ = eval_tree(X_tr, y_tr, X_e1, y_e1, max_depth=15)
    print(f"DecisionTree → Test1 ACC: {acc1_dt:.2f}%")
    if X_e2 is not None:
        # 재학습해서 Test2 예측
        tree_full = build_tree(X_tr, y_tr, max_depth=15)
        preds2_dt = np.array([predict_tree(tree_full, x) for x in X_e2])
        if submission_ids is not None:
            pd.DataFrame({'PassengerId': submission_ids, 'Survived': preds2_dt})\
              .to_csv(f'submission_dt_{name}.csv', index=False)
            print(f"Saved submission_dt_{name}.csv")

# --- 10. 평가 실행: Full feature, PCA, LDA 순서 ---
# 10.1 Full feature
evaluate_all(
    name       = 'full',
    X_tr       = X_train_s, y_tr = y_train,
    X_e1       = X_test1_s, y_e1 = y_test1,
    X_e2       = X_test2_s, y_e2 = None,  # Test2 정답 없으므로 y_e2=None
    submission_ids = ids_test2
)

# 10.2 PCA feature
evaluate_all(
    name       = 'pca',
    X_tr       = X_train_pca, y_tr = y_train,
    X_e1       = X_test1_pca, y_e1 = y_test1,
    X_e2       = X_test2_pca, y_e2 = None,
    submission_ids = ids_test2
)

# 10.3 LDA feature (1D)
evaluate_all(
    name       = 'fld',
    X_tr       = X_train_fld, y_tr = y_train,
    X_e1       = X_test1_fld, y_e1 = y_test1,
    X_e2       = X_test2_fld, y_e2 = None,
    submission_ids = ids_test2
)
