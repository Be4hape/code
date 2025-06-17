import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 ---
train_df = pd.read_csv('process1_result.csv')  # Train + Test1 (전처리 완료)
test_df  = pd.read_csv('process2_result.csv')  # Test2 (전처리 완료)

# --- 2. 숫자형 컬럼 중 Train/Test2 공통 컬럼 자동 선택 ---
common_cols = set(train_df.columns).intersection(test_df.columns)
numeric_train_cols = set(train_df.select_dtypes(include=[np.number]).columns)
numeric_test_cols  = set(test_df.select_dtypes(include=[np.number]).columns)

# PassengerId와 Survived는 제외하고, 나머지 숫자형 공통 컬럼만 feature_cols로 사용
feature_cols = [
    c for c in common_cols
    if (c in numeric_train_cols and c in numeric_test_cols)
    and (c not in ('PassengerId', 'Survived'))
]

# (디버깅) 뽑힌 피처 확인
# print("사용할 Full Feature (Train/Test2 공통, 숫자형):", feature_cols)

# --- 3. Train/Label 분리 및 Test2 준비 ---
X_full = train_df[feature_cols].values   # (n_samples, n_features)
y_full = train_df['Survived'].values     # (n_samples,)

X_test2 = test_df[feature_cols].values   # (n_test2, n_features)
ids_test2 = test_df['PassengerId'].values

# --- 4. Hold-out 분할: Train/Test1 (80:20) ---
np.random.seed(42)
indices = np.arange(len(X_full))
np.random.shuffle(indices)

split = int(0.8 * len(indices))
train_idx = indices[:split]
test1_idx = indices[split:]

X_train = X_full[train_idx].copy()
y_train = y_full[train_idx]
X_test1 = X_full[test1_idx].copy()
y_test1 = y_full[test1_idx]

# --- 5. 누락값(NaN) 처리 (훈련 데이터 평균으로 채우기) ---
# X_train, X_test1, X_test2에 NaN이 있을 경우, 각 열별로 "X_train의 평균"으로 대체합니다.
col_means = np.nanmean(X_train, axis=0)  # 각 칼럼의 평균 (NaN 무시)

# train NA 채우기
inds_train_nan = np.isnan(X_train)
X_train[inds_train_nan] = np.take(col_means, np.where(inds_train_nan)[1])

# test1 NA 채우기
inds_test1_nan = np.isnan(X_test1)
X_test1[inds_test1_nan] = np.take(col_means, np.where(inds_test1_nan)[1])

# test2 NA 채우기
inds_test2_nan = np.isnan(X_test2)
X_test2[inds_test2_nan] = np.take(col_means, np.where(inds_test2_nan)[1])

# --- 6. 스케일링: Train 기준으로 평균/표준편차 계산 후 모두 표준화 ---
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0  # 표준편차가 0인 피처는 1로 대체

X_train_s = (X_train - means) / stds
X_test1_s = (X_test1 - means) / stds
X_test2_s = (X_test2 - means) / stds

# --- 7. Logistic Regression (SGD) 함수 정의 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_sgd(X, y, lr=0.001, epochs=9):
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

def predict_logistic(w, b, X):
    probs = sigmoid(X.dot(w) + b)
    return (probs >= 0.5).astype(int)

# --- 8. KNN (k=20) 함수 정의 ---
def euclid_dist_matrix(A, B):
    # A: (n_A, d), B: (n_B, d) → 각 A_i, B_j 사이의 Euclidean 거리 매트릭스 (n_A, n_B)
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

def knn_predict(train_X, train_y, test_X, k=20):
    D = euclid_dist_matrix(test_X, train_X)      # (n_test, n_train) 거리 행렬 계산
    neigh_idxs = np.argpartition(D, k, axis=1)[:, :k]  # 각 테스트 샘플별 k개 이웃 인덱스
    preds = []
    for idxs in neigh_idxs:
        counts = np.bincount(train_y[idxs])
        preds.append(np.argmax(counts))
    return np.array(preds)

# --- 9. Decision Tree (max_depth=15) 함수 정의 ---
def gini(labels):
    classes, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)

def weighted_gini(col, labels, t):
    left  = labels[col <= t]
    right = labels[col >  t]
    if len(left) == 0 or len(right) == 0:
        return None
    n = len(labels)
    return (len(left)/n)*gini(left) + (len(right)/n)*gini(right)

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

def build_tree(X, y, depth=0, max_depth=7):
    if len(np.unique(y)) == 1 or depth == max_depth:
        vals, counts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(counts)]}
    feat, thresh = find_best_split(X, y)
    if feat is None:
        vals, counts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(counts)]}
    mask = X[:, feat] <= thresh
    left  = build_tree(X[mask], y[mask], depth+1, max_depth)
    right = build_tree(X[~mask], y[~mask], depth+1, max_depth)
    return {'leaf': False, 'feat': feat, 'thresh': thresh, 'left': left, 'right': right}

def predict_tree(tree, x):
    if tree['leaf']:
        return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

# --- 10. MLPClassifier 설정 (hidden_layer_sizes=(32,), lr=0.005, alpha=1e-4, max_iter=300) ---
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

# --- 11. 모델 학습 및 Test1(Hold-out) 평가 ---
# 11-1. Logistic
w_log, b_log = train_logistic_sgd(X_train_s, y_train, lr=0.001, epochs=9)
preds_log1  = predict_logistic(w_log, b_log, X_test1_s)
acc_log1    = accuracy_score(y_test1, preds_log1) * 100

# 11-2. KNN (k=20)
preds_knn1  = knn_predict(X_train_s, y_train, X_test1_s, k=20)
acc_knn1    = accuracy_score(y_test1, preds_knn1) * 100

# 11-3. Decision Tree (max_depth=15)
tree = build_tree(X_train_s, y_train, max_depth=7)
preds_dt1 = np.array([predict_tree(tree, x) for x in X_test1_s])
acc_dt1   = accuracy_score(y_test1, preds_dt1) * 100

# 11-4. MLP
mlp = MLPClassifier(**mlp_params)
mlp.fit(X_train_s, y_train)
preds_mlp1 = mlp.predict(X_test1_s)
acc_mlp1   = accuracy_score(y_test1, preds_mlp1) * 100

print("=== Test1 (Hold-out) Accuracy (Full Feature) ===")
print(f"Logistic   : {acc_log1:.2f}%")
print(f"KNN (k=20) : {acc_knn1:.2f}%")
print(f"DecisionT  : {acc_dt1:.2f}%")
print(f"MLP        : {acc_mlp1:.2f}%")

# --- 12. Test2 예측 및 Submission 파일 생성 ---
# 12-1. Logistic Test2 예측
preds_log2 = predict_logistic(w_log, b_log, X_test2_s)
df_sub_log = pd.DataFrame({'PassengerId': ids_test2, 'Survived': preds_log2})
df_sub_log.to_csv('submission_logistic_full33.csv', index=False)

# 12-2. KNN Test2 예측
preds_knn2 = knn_predict(X_train_s, y_train, X_test2_s, k=20)
df_sub_knn = pd.DataFrame({'PassengerId': ids_test2, 'Survived': preds_knn2})
df_sub_knn.to_csv('submission_knn_full33.csv', index=False)

# 12-3. Decision Tree Test2 예측
preds_dt2 = np.array([predict_tree(tree, x) for x in X_test2_s])
df_sub_dt = pd.DataFrame({'PassengerId': ids_test2, 'Survived': preds_dt2})
df_sub_dt.to_csv('submission_dt_full33.csv', index=False)

# 12-4. MLP Test2 예측
preds_mlp2 = mlp.predict(X_test2_s)
df_sub_mlp = pd.DataFrame({'PassengerId': ids_test2, 'Survived': preds_mlp2})
df_sub_mlp.to_csv('submission_mlp_full33.csv', index=False)

print("\nSaved files:")
print("  - submission_logistic_full33.csv")
print("  - submission_knn_full33.csv")
print("  - submission_dt_full33.csv")
print("  - submission_mlp_full33.csv")
