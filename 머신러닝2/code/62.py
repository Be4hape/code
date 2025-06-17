import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & full feature 추출 ---
df = pd.read_csv('process1_result.csv')
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_full = df[features].values    # (n_samples, 5)
y_full = df['Survived'].values  # (n_samples,)

# --- 2. Stratified 5-Fold 분할 (NumPy만 사용) ---
np.random.seed(42)
pos_idx = np.where(y_full == 1)[0]
neg_idx = np.where(y_full == 0)[0]
np.random.shuffle(pos_idx)
np.random.shuffle(neg_idx)

# 양성/음성 각각 5개 fold로 분할
pos_folds = np.array_split(pos_idx, 5)
neg_folds = np.array_split(neg_idx, 5)

folds = [np.concatenate([pos_folds[i], neg_folds[i]]) for i in range(5)]

# --- 3. Logistic SGD 정의 (기존과 동일) ---
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

def predict_logistic(w, b, X):
    probs = sigmoid(X.dot(w) + b)
    return (probs >= 0.5).astype(int)

# --- 4. KNN 정의 (기존과 동일) ---
def euclid_dist_matrix(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

def knn_predict(train_X, train_y, test_X, k=20):
    D = euclid_dist_matrix(test_X, train_X)
    neigh = np.argpartition(D, k, axis=1)[:, :k]
    preds = []
    for idxs in neigh:
        counts = np.bincount(train_y[idxs])
        preds.append(np.argmax(counts))
    return np.array(preds)

# --- 5. Decision Tree 정의 (기존과 동일) ---
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

def build_tree(X, y, depth=0, max_depth=15):
    if len(np.unique(y)) == 1 or depth == max_depth:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(cnts)]}
    feat, thresh = find_best_split(X, y)
    if feat is None:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(cnts)]}
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

# --- 6. MLPClassifier 설정 (지정된 하이퍼파라미터 고정) ---
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

# --- 7. 5-Fold CV 수행 (Accuracy) ---
logistic_accs = []
knn_accs      = []
dt_accs       = []
mlp_accs      = []

for i in range(5):
    val_idx = folds[i]
    train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

    X_tr, y_tr = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]

    # 7-1. 스케일링: Train 기준 평균/표준편차 계산
    means = X_tr.mean(axis=0)
    stds  = X_tr.std(axis=0)
    stds[stds == 0] = 1.0

    X_tr_s  = (X_tr - means) / stds
    X_val_s = (X_val - means) / stds

    # --- Logistic CV ---
    w, b = train_logistic_sgd(X_tr_s, y_tr, lr=0.001, epochs=100)
    preds_log = predict_logistic(w, b, X_val_s)
    acc_log = accuracy_score(y_val, preds_log) * 100
    logistic_accs.append(acc_log)

    # --- KNN CV ---
    preds_knn = knn_predict(X_tr_s, y_tr, X_val_s, k=20)
    acc_knn = accuracy_score(y_val, preds_knn) * 100
    knn_accs.append(acc_knn)

    # --- Decision Tree CV ---
    tree = build_tree(X_tr_s, y_tr, max_depth=15)
    preds_dt = np.array([predict_tree(tree, x) for x in X_val_s])
    acc_dt = accuracy_score(y_val, preds_dt) * 100
    dt_accs.append(acc_dt)

    # --- MLP CV ---
    mlp = MLPClassifier(**mlp_params)
    mlp.fit(X_tr_s, y_tr)
    preds_mlp = mlp.predict(X_val_s)
    acc_mlp = accuracy_score(y_val, preds_mlp) * 100
    mlp_accs.append(acc_mlp)

# --- 8. 5-Fold CV 결과 출력 ---
print("5-Fold CV 평균 Accuracy (Full Feature):")
print(f"Logistic   : {np.mean(logistic_accs):.2f}%")
print(f"KNN (k=20) : {np.mean(knn_accs):.2f}%")
print(f"Decision T : {np.mean(dt_accs):.2f}%")
print(f"MLP        : {np.mean(mlp_accs):.2f}%")
