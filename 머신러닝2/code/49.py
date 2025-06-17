import numpy as np
import pandas as pd

# --- 1. 데이터 로드 & 수동 스케일링 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features      = ['Sex','SibSp','Parch','Embarked','TicketNumeric']
X_train       = train_df[features].values   # (n_train, d)
y_train       = train_df['Survived'].values # (n_train,)
X_test        = test_df[features].values    # (n_test,  d)
passenger_ids = test_df['PassengerId'].values

# 1) zero-centering & unit-variance 스케일링
means = X_train.mean(axis=0)
stds  = X_train.std(axis=0)
stds[stds == 0] = 1.0
Z_train = (X_train - means) / stds         # (n_train, d)
Z_test  = (X_test  - means) / stds         # (n_test,  d)

# --- 2. FLD 축(w) 계산 ---
# 2.1 클래스별 평균
mu0 = Z_train[y_train == 0].mean(axis=0)   # (d,)
mu1 = Z_train[y_train == 1].mean(axis=0)   # (d,)

# 2.2 클래스 내 산포 행렬 SW
d   = Z_train.shape[1]
SW  = np.zeros((d, d))
for z, y in zip(Z_train, y_train):
    diff = (z - (mu1 if y == 1 else mu0)).reshape(-1,1)
    SW += diff.dot(diff.T)

# 2.3 판별 벡터 w ∝ SW^{-1}(μ1−μ0)
w_fld = np.linalg.inv(SW).dot(mu1 - mu0)
w_fld /= np.linalg.norm(w_fld)            # 정규화

# --- 3. 1D 투사 ---
z_train = Z_train.dot(w_fld).reshape(-1,1)  # (n_train, 1)
z_test  = Z_test.dot(w_fld).reshape(-1,1)   # (n_test, 1)

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
    # X.shape[1] == 1
    vals = np.unique(X[:,0])
    thresholds = (vals[:-1] + vals[1:]) / 2.0
    for t in thresholds:
        wg = weighted_gini(X[:,0], y, t)
        if wg is None: continue
        gain = parent - wg
        if gain > best_gain:
            best_gain, best_feat, best_t = gain, 0, t
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

# --- 5. Train 전체로 트리 학습 ---
tree = build_tree(z_train, y_train, max_depth=15)

# --- 6. Test1 (Train) 예측 & 평가 ---
preds_train = np.array([predict(tree, x) for x in z_train])

# Accuracy
acc_train = np.mean(preds_train == y_train) * 100

# F1-score 수동 계산
tp = np.sum((preds_train == 1) & (y_train == 1))
fp = np.sum((preds_train == 1) & (y_train == 0))
fn = np.sum((preds_train == 0) & (y_train == 1))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_train  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
f1_train *= 100

print(f"Test1 Accuracy: {acc_train:.2f}%")
print(f"Test1 F1-score : {f1_train:.2f}%")

# --- 7. Test2 예측 & 제출 파일 생성 ---
preds_test = np.array([predict(tree, x) for x in z_test])
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived':    preds_test
})
submission_df.to_csv('submission_dt_fld.csv', index=False)
print("Saved: submission_dt_fld.csv")
