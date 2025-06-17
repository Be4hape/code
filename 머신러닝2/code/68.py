import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & 피처 선택 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

common = set(train_df.columns).intersection(test_df.columns)
num_tr = set(train_df.select_dtypes(include=[np.number]).columns)
num_te = set(test_df.select_dtypes(include=[np.number]).columns)
feature_cols = [
    c for c in common
    if c in num_tr and c in num_te and c not in ('PassengerId','Survived')
]

X_full   = train_df[feature_cols].values
y_full   = train_df['Survived'].values
X_test2  = test_df[feature_cols].values
ids_test2 = test_df['PassengerId'].values

# --- 2. 결측치 처리 & 스케일링 (Full 기준) ---
col_means = np.nanmean(X_full, axis=0)
# Impute
inds = np.isnan(X_full)
X_full[inds] = np.take(col_means, np.where(inds)[1])
inds2 = np.isnan(X_test2)
X_test2[inds2] = np.take(col_means, np.where(inds2)[1])
# Scale
means = X_full.mean(axis=0)
stds  = X_full.std(axis=0)
stds[stds==0] = 1.0
X_full_s  = (X_full  - means) / stds
X_test2_s = (X_test2 - means) / stds

# --- 3. 모델 구현 함수들 ---

# 3.1 Logistic SGD
def sigmoid(z): return 1 / (1 + np.exp(-z))

def train_logistic_sgd(X, y, lr, epochs):
    m, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        for i in range(m):
            xi, yi = X[i], y[i]
            delta = sigmoid(xi.dot(w)+b) - yi
            w -= lr * delta * xi
            b -= lr * delta
    return w, b

def predict_logistic(model, X):
    w, b = model
    return (sigmoid(X.dot(w)+b) >= 0.5).astype(int)

# 3.2 KNN
def euclid_dist_matrix(A, B):
    return np.sqrt(((A[:,None,:] - B[None,:,:])**2).sum(axis=2))

def knn_predict(train_X, train_y, test_X, k):
    D = euclid_dist_matrix(test_X, train_X)
    idxs = np.argpartition(D, k, axis=1)[:,:k]
    preds = [np.bincount(train_y[i]).argmax() for i in idxs]
    return np.array(preds)

# 3.3 Decision Tree
def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts/counts.sum()
    return 1 - (p**2).sum()

def weighted_gini(col, labels, t):
    l = labels[col <= t]; r = labels[col > t]
    if len(l)==0 or len(r)==0: return None
    n = len(labels)
    return len(l)/n*gini(l) + len(r)/n*gini(r)

def find_best_split(X, y):
    parent = gini(y)
    best = (0, None, None)
    for f in range(X.shape[1]):
        vals = np.unique(X[:,f])
        for t in (vals[:-1]+vals[1:])/2:
            wg = weighted_gini(X[:,f], y, t)
            if wg is None: continue
            gain = parent - wg
            if gain > best[0]:
                best = (gain, f, t)
    return best[1], best[2]

def build_tree(X, y, depth=0, max_depth=7):
    if len(np.unique(y))==1 or depth==max_depth:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf':True, 'pred':vals[cnts.argmax()]}
    feat, thresh = find_best_split(X, y)
    if feat is None:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf':True, 'pred':vals[cnts.argmax()]}
    mask = X[:,feat] <= thresh
    left  = build_tree(X[mask],    y[mask],    depth+1, max_depth)
    right = build_tree(X[~mask],   y[~mask],   depth+1, max_depth)
    return {'leaf':False, 'feat':feat, 'thresh':thresh, 'left':left, 'right':right}

def predict_tree(tree, x):
    if tree['leaf']: return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

# 3.4 Wrappers for CV
def build_log_wrapper(X,y,lr,epochs): return train_logistic_sgd(X,y,lr,epochs)
def pred_log_wrapper(m, X): return predict_logistic(m, X)

def build_knn_wrapper(X,y,k): return (X,y,k)
def pred_knn_wrapper(m, X): return knn_predict(m[0], m[1], X, m[2])

def build_dt_wrapper(X,y,max_depth): return build_tree(X,y,0,max_depth)
def pred_dt_wrapper(m, X): return np.array([predict_tree(m,x) for x in X])

def build_mlp_wrapper(X,y,hidden_layer_sizes,learning_rate_init):
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        max_iter=300, early_stopping=True,
        random_state=42
    )
    mlp.fit(X, y)
    return mlp
def pred_mlp_wrapper(m, X): return m.predict(X)

# --- 4. K-Fold CV 구현 ---
def get_folds(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    sizes = np.full(k, n//k, int)
    sizes[:n%k] += 1
    folds, start = [], 0
    for sz in sizes:
        folds.append(idx[start:start+sz]); start+=sz
    return folds

def k_fold_cv(X, y, build_fn, pred_fn, param_grid, k=5):
    import itertools
    best = {'score':-1, 'params':None}
    folds = get_folds(len(y), k)
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        scores = []
        for i in range(k):
            val_idx = folds[i]
            tr_idx  = np.hstack([folds[j] for j in range(k) if j!=i])
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            model = build_fn(X_tr, y_tr, **params)
            y_pr  = pred_fn(model, X_va)
            scores.append(accuracy_score(y_va, y_pr))
        avg = np.mean(scores)
        print(f"{build_fn.__name__} {params} → CV acc {avg:.4f}")
        if avg > best['score']:
            best.update({'score':avg, 'params':params})
    print("▶ Best:", best, "\n")
    return best

# --- 5. 각 모델별 CV 탐색 ---
best_log = k_fold_cv(
    X_full_s, y_full,
    build_log_wrapper, pred_log_wrapper,
    param_grid={'lr':[0.001,0.005],'epochs':[5,10]},
    k=5
)

best_knn = k_fold_cv(
    X_full_s, y_full,
    build_knn_wrapper, pred_knn_wrapper,
    param_grid={'k':[5,10,15,20]},
    k=5
)

best_dt = k_fold_cv(
    X_full_s, y_full,
    build_dt_wrapper, pred_dt_wrapper,
    param_grid={'max_depth':[3,5,7,9]},
    k=5
)

best_mlp = k_fold_cv(
    X_full_s, y_full,
    build_mlp_wrapper, pred_mlp_wrapper,
    param_grid={'hidden_layer_sizes':[(16,),(32,)],'learning_rate_init':[0.001,0.005]},
    k=5
)

# --- 6. CV 최고 모델 선택 & Test2 예측 ---
results = {
    'logistic': best_log,
    'knn'     : best_knn,
    'dt'      : best_dt,
    'mlp'     : best_mlp
}
best_model = max(results, key=lambda m: results[m]['score'])
print("=== Selected model:", best_model, results[best_model], "===\n")

if best_model == 'logistic':
    w,b = train_logistic_sgd(X_full_s, y_full, **results['logistic']['params'])
    preds = predict_logistic((w,b), X_test2_s)
elif best_model == 'knn':
    params = results['knn']['params']
    preds = knn_predict(X_full_s, y_full, X_test2_s, **params)
elif best_model == 'dt':
    params = results['dt']['params']
    tree = build_tree(X_full_s, y_full, 0, **params)
    preds = np.array([predict_tree(tree, x) for x in X_test2_s])
else:  # mlp
    params = results['mlp']['params']
    mlp = MLPClassifier(
        max_iter=300, early_stopping=True, random_state=42,
        **params
    ).fit(X_full_s, y_full)
    preds = mlp.predict(X_test2_s)

# --- 7. Submission 파일 생성 ---
submission = pd.DataFrame({'PassengerId': ids_test2, 'Survived': preds})
submission.to_csv('final_submission.csv', index=False)
print("Saved → final_submission.csv")
