import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# --- 0. Gini-기반 수작업 트리 함수 정의 (기존 구현 그대로) ---
def gini(labels):
    classes, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)

def weighted_gini(col, labels, t):
    left  = labels[col <= t]
    right = labels[col >  t]
    if len(left)==0 or len(right)==0:
        return None
    n = len(labels)
    return (len(left)/n)*gini(left) + (len(right)/n)*gini(right)

def find_best_split(X, y):
    best_gain, best_feat, best_t = 0.0, None, None
    parent = gini(y)
    for feat in range(X.shape[1]):
        vals = np.unique(X[:,feat])
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        for t in thresholds:
            wg = weighted_gini(X[:,feat], y, t)
            if wg is None: continue
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
    left  = build_tree(X[mask],  y[mask],  depth+1, max_depth)
    right = build_tree(X[~mask], y[~mask], depth+1, max_depth)
    return {'leaf': False, 'feat': feat, 'thresh': thresh, 'left': left, 'right': right}

def predict(tree, x):
    if tree['leaf']:
        return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict(tree['left'], x)
    else:
        return predict(tree['right'], x)

# --- 1. 데이터 로드 ---
train_df = pd.read_csv('process1_result.csv')
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_all    = train_df[features].values
y_all    = train_df['Survived'].values

# --- 2. k를 2부터 30까지 바꿔가며 교차검증 ---
results = {}
for k in range(2, 31):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accs, f1s = [], []

    for train_idx, val_idx in skf.split(X_all, y_all):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # --- 트리 학습 & 예측 ---
        tree = build_tree(X_tr, y_tr, max_depth=15)
        preds = np.array([predict(tree, x) for x in X_val])

        accs.append( accuracy_score(y_val, preds) )
        f1s.append ( f1_score   (y_val, preds) )

    results[k] = {
        'acc_mean': np.mean(accs),
        'acc_std':  np.std(accs),
        'f1_mean':  np.mean(f1s),
        'f1_std':   np.std(f1s)
    }
    print(f"k={k:2d} → ACC={results[k]['acc_mean']:.4f}±{results[k]['acc_std']:.4f}, "
          f"F1={results[k]['f1_mean']:.4f}±{results[k]['f1_std']:.4f}")

# --- 3. 최적 k 선택 & 최종 모델 재학습 (예시) ---
# optimal_k = max(results, key=lambda k: results[k]['acc_mean'])
# print("Optimal k:", optimal_k)
