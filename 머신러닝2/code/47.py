## pca to dicision tree

import numpy as np
import pandas as pd

# --- 1. 데이터 로드 & 수동 스케일링 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features      = ['Sex','SibSp','Parch','Embarked','TicketNumeric']
X_train       = train_df[features].values
y_train       = train_df['Survived'].values
X_test        = test_df[features].values
passenger_ids = test_df['PassengerId'].values

# 1) zero-centering & unit-variance 스케일링
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0
X_train_scaled = (X_train - means) / stds
X_test_scaled  = (X_test  - means) / stds

# --- 2. PCA 고유벡터 계산 & k_pca=4 주성분 선택 ---
cov_mat         = np.cov(X_train_scaled, rowvar=False)
eigvals, eigvecs= np.linalg.eigh(cov_mat)
idx             = np.argsort(eigvals)[::-1]
V_k             = eigvecs[:, idx[:4]]   # (d,4)

# --- 3. PCA 투사 ---
X_train_pca = X_train_scaled.dot(V_k)    # (n_train,4)
X_test_pca  = X_test_scaled.dot(V_k)     # (n_test, 4)

# --- 4. Decision Tree 구현 함수들 ---
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

# --- 5. 전체 Train으로 트리 학습 ---
tree = build_tree(X_train_pca, y_train, max_depth=15)

# --- 6. Test1 (Train) 예측 & 평가 ---
preds_train = np.array([predict(tree, x) for x in X_train_pca])
accuracy    = np.mean(preds_train == y_train)

# F1-score 계산 (binary)
tp = np.sum((preds_train == 1) & (y_train == 1))
fp = np.sum((preds_train == 1) & (y_train == 0))
fn = np.sum((preds_train == 0) & (y_train == 1))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"Test1 (Train) Accuracy: {accuracy*100:.2f}%")
print(f"Test1 (Train) F1-score : {f1_score*100:.2f}%")

# --- 7. Test2 예측 & 제출 파일 생성 ---
preds_test = np.array([predict(tree, x) for x in X_test_pca])
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    preds_test
})
submission_df.to_csv('submission_dt_pca_k4.csv', index=False)
print("Saved: submission_dt_pca_k4.csv")
