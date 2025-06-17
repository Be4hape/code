import numpy as np
import pandas as pd

# --- 1. Load train & test2, select features ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features = ['Sex', 'Name']  # Spearman-best subset
X_full = train_df[features].values
y_full = train_df['Survived'].values

X_test2 = test_df[features].values
ids_test2 = test_df['PassengerId'].values

# --- 2. Impute missing with train-means & standardize ---
col_means = np.nanmean(X_full, axis=0)
mask = np.isnan(X_full)
X_full[mask] = np.take(col_means, np.where(mask)[1])

mask2 = np.isnan(X_test2)
X_test2[mask2] = np.take(col_means, np.where(mask2)[1])

means = X_full.mean(axis=0)
stds  = X_full.std(axis=0)
stds[stds == 0] = 1.0

X_full_s  = (X_full  - means) / stds
X_test2_s = (X_test2 - means) / stds

# --- 3. Compute LDA direction (1D) ---
mu0 = X_full_s[y_full==0].mean(axis=0)
mu1 = X_full_s[y_full==1].mean(axis=0)
d   = X_full_s.shape[1]

SW = np.zeros((d,d))
for xi, yi in zip(X_full_s, y_full):
    m = mu1 if yi==1 else mu0
    diff = (xi - m).reshape(-1,1)
    SW += diff.dot(diff.T)

w_lda = np.linalg.inv(SW).dot(mu1 - mu0)
w_lda /= np.linalg.norm(w_lda)

# Project both train & test2 onto that 1D axis
X_train_lda = X_full_s.dot(w_lda).reshape(-1,1)
X_test2_lda = X_test2_s.dot(w_lda).reshape(-1,1)

# --- 4. Decision Tree (max_depth=10) implementation ---
def gini(labels):
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return 1 - (p**2).sum()

def weighted_gini(col, labels, thresh):
    left  = labels[col <= thresh]
    right = labels[col >  thresh]
    if len(left)==0 or len(right)==0:
        return None
    n = len(labels)
    return len(left)/n * gini(left) + len(right)/n * gini(right)

def find_best_split(X, y):
    parent = gini(y)
    best_gain, best_feat, best_thresh = 0, None, None
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        thr = (vals[:-1] + vals[1:]) / 2.0
        for t in thr:
            wg = weighted_gini(X[:, f], y, t)
            if wg is None: continue
            gain = parent - wg
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, f, t
    return best_feat, best_thresh

def build_tree(X, y, depth=0, max_depth=10):
    if len(np.unique(y))==1 or depth==max_depth:
        vals, cnt = np.unique(y, return_counts=True)
        return {'leaf':True, 'pred': vals[cnt.argmax()]}
    feat, thresh = find_best_split(X, y)
    if feat is None:
        vals, cnt = np.unique(y, return_counts=True)
        return {'leaf':True, 'pred': vals[cnt.argmax()]}
    mask = X[:, feat] <= thresh
    return {
        'leaf': False,
        'feat': feat,
        'thresh': thresh,
        'left':  build_tree(X[mask],  y[mask],  depth+1, max_depth),
        'right': build_tree(X[~mask], y[~mask], depth+1, max_depth)
    }

def predict_tree(tree, x):
    if tree['leaf']:
        return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

# --- 5. Train on full LDA data & predict Test2 ---
tree = build_tree(X_train_lda, y_full, depth=0, max_depth=10)
preds_test2 = np.array([predict_tree(tree, x) for x in X_test2_lda])

# --- 6. Save submission ---
submission = pd.DataFrame({
    'PassengerId': ids_test2,
    'Survived':    preds_test2
})
submission.to_csv('submission_lda_dt10.csv', index=False)
print("Saved → submission_lda_dt10.csv")
