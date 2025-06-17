import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & 사용할 피처 지정 ---
train_df = pd.read_csv('process1_result.csv')
features = ['Sex','Pclass','Name','Family_Group','Embarked']
n_feat = len(features)

X_full = train_df[features].values
y_full = train_df['Survived'].values

# --- 2. 결측치 처리 & 스케일링 (전체 train 기준) ---
col_means = np.nanmean(X_full, axis=0)
mask_full = np.isnan(X_full)
X_full[mask_full] = np.take(col_means, np.where(mask_full)[1])

means = X_full.mean(axis=0)
stds  = X_full.std(axis=0)
stds[stds==0] = 1.0
X_full_s = (X_full - means) / stds

# --- 3. 5-Fold 인덱스 생성 ---
def get_folds(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    sizes = np.full(k, n//k, int)
    sizes[:n%k] += 1
    folds, start = [], 0
    for sz in sizes:
        folds.append(idx[start:start+sz])
        start += sz
    return folds

folds = get_folds(len(y_full), k=5, seed=42)

# --- 4. Decision Tree 구현 (max_depth 파라미터만 다르게) ---
def gini(labels):
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return 1 - (p**2).sum()

def weighted_gini(col, labels, t):
    l = labels[col<=t]; r = labels[col>t]
    if len(l)==0 or len(r)==0: return None
    n = len(labels)
    return len(l)/n*gini(l) + len(r)/n*gini(r)

def find_best_split(X, y):
    parent = gini(y)
    best = (0, None, None)
    for f in range(X.shape[1]):
        vals = np.unique(X[:,f])
        ths = (vals[:-1] + vals[1:]) / 2
        for t in ths:
            wg = weighted_gini(X[:,f], y, t)
            if wg is None: continue
            gain = parent - wg
            if gain > best[0]:
                best = (gain, f, t)
    return best[1], best[2]

def build_tree(X, y, depth=0, max_depth=7):
    if len(np.unique(y))==1 or depth==max_depth:
        vals, cnt = np.unique(y, return_counts=True)
        return {'leaf':True, 'pred':vals[cnt.argmax()]}
    feat, thresh = find_best_split(X, y)
    if feat is None:
        vals, cnt = np.unique(y, return_counts=True)
        return {'leaf':True, 'pred':vals[cnt.argmax()]}
    mask = X[:,feat] <= thresh
    return {
        'leaf':False, 'feat':feat, 'thresh':thresh,
        'left': build_tree(X[mask],    y[mask],    depth+1, max_depth),
        'right':build_tree(X[~mask],   y[~mask],   depth+1, max_depth)
    }

def predict_tree(tree, x):
    if tree['leaf']:
        return tree['pred']
    branch = 'left' if x[tree['feat']] <= tree['thresh'] else 'right'
    return predict_tree(tree[branch], x)

# --- 5. 모든 부분집합 × depth 리스트 × 5-Fold CV ---
depths = [3, 5, 7, 10, 15]
results = []

for mask in range(1, 1<<n_feat):
    # 부분집합에 해당하는 컬럼 인덱스
    idxs = [i for i in range(n_feat) if (mask>>i)&1]
    X_sub = X_full_s[:, idxs]

    for md in depths:
        cv_scores = []
        for i in range(5):
            val_idx = folds[i]
            trn_idx = np.hstack([folds[j] for j in range(5) if j!=i])
            X_tr, y_tr = X_sub[trn_idx], y_full[trn_idx]
            X_va, y_va = X_sub[val_idx], y_full[val_idx]

            tree = build_tree(X_tr, y_tr, depth=0, max_depth=md)
            preds = np.array([predict_tree(tree, x) for x in X_va])
            cv_scores.append(accuracy_score(y_va, preds))

        results.append({
            'mask': mask,
            'features': [features[i] for i in idxs],
            'max_depth': md,
            'cv_score': np.mean(cv_scores)
        })

# --- 6. DataFrame으로 결과 집계 & 최고 조합 탐색 ---
df_res = pd.DataFrame(results)
best = df_res.loc[df_res['cv_score'].idxmax()]

print("▶ Best subset:", best['features'])
print("▶ Best max_depth:", best['max_depth'])
print("▶ Best CV score:", best['cv_score'])
