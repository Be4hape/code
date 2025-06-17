import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# --- 1. 데이터 로드 & 피처/레이블 분리 ---
train_df = pd.read_csv('process1_result.csv')
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_all    = train_df[features].values
y_all    = train_df['Survived'].values

# --- 2. Gini 기반 수작업 Decision Tree 함수들 ---
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

# --- 3. k=16 고정, 16-fold 교차검증 ---
k = 16
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
accs, f1s = [], []

print(f"--- {k}-Fold Cross-Validation (Decision Tree) ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    # 트리 학습
    tree = build_tree(X_tr, y_tr, max_depth=15)

    # 예측 및 평가
    preds = np.array([predict(tree, x) for x in X_val])
    acc = accuracy_score(y_val, preds)
    f1  = f1_score(y_val, preds)

    accs.append(acc)
    f1s.append(f1)
    print(f"Fold {fold:2d} — ACC: {acc:.4f}, F1: {f1:.4f}")

# 요약
print("\n=== Cross-Validation Results (k=16) ===")
print(f"ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 : {np.mean(f1s ):.4f} ± {np.std(f1s ):.4f}")
