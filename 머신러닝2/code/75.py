import numpy as np
import pandas as pd

# --- 1. Load data & select features ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features = ['Pclass','Name','Embarked']
X_full = train_df[features].values
y_full = train_df['Survived'].values

X_test2 = test_df[features].values
ids_test2 = test_df['PassengerId'].values

# --- 2. Impute & standardize on train ---
col_means = np.nanmean(X_full, axis=0)
mask_full = np.isnan(X_full)
X_full[mask_full] = np.take(col_means, np.where(mask_full)[1])

mask_t2 = np.isnan(X_test2)
X_test2[mask_t2] = np.take(col_means, np.where(mask_t2)[1])

means = X_full.mean(axis=0)
stds  = X_full.std(axis=0)
stds[stds == 0] = 1.0

X_full_s  = (X_full  - means) / stds
X_test2_s = (X_test2 - means) / stds

# --- 3. PCA → 2 components ---
cov    = np.cov(X_full_s, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
V_pca = eigvecs[:, order[:2]]   # take top-2 eigenvectors

X_full_pca  = X_full_s.dot(V_pca)
X_test2_pca = X_test2_s.dot(V_pca)

# --- 4. Decision Tree (max_depth=10) implementation ---
def gini(labels):
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return 1 - (p**2).sum()

def weighted_gini(col, labels, t):
    left  = labels[col <= t]
    right = labels[col >  t]
    if len(left)==0 or len(right)==0:
        return None
    n = len(labels)
    return len(left)/n * gini(left) + len(right)/n * gini(right)

def find_best_split(X, y):
    parent = gini(y)
    best_gain, best_feat, best_t = 0, None, None
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        for t in thresholds:
            wg = weighted_gini(X[:, f], y, t)
            if wg is None:
                continue
            gain = parent - wg
            if gain > best_gain:
                best_gain, best_feat, best_t = gain, f, t
    return best_feat, best_t

def build_tree(X, y, depth=0, max_depth=10):
    # stopping
    if len(np.unique(y)) == 1 or depth == max_depth:
        vals, cnt = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[cnt.argmax()]}
    feat, thresh = find_best_split(X, y)
    if feat is None:
        vals, cnt = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[cnt.argmax()]}
    mask = X[:, feat] <= thresh
    left  = build_tree(X[mask],    y[mask],    depth+1, max_depth)
    right = build_tree(X[~mask],   y[~mask],   depth+1, max_depth)
    return {'leaf': False, 'feat': feat, 'thresh': thresh, 'left': left, 'right': right}

def predict_tree(tree, x):
    if tree['leaf']:
        return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

# --- 5. Train on full PCA data & predict Test2 ---
tree = build_tree(X_full_pca, y_full, depth=0, max_depth=10)
preds_test2 = np.array([predict_tree(tree, x) for x in X_test2_pca])

# --- 6. Save submission ---
submission = pd.DataFrame({
    'PassengerId': ids_test2,
    'Survived':    preds_test2
})
submission.to_csv('submission_pca_dt10.csv', index=False)
print("Saved → submission_pca_dt10.csv")
