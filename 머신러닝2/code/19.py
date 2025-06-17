import numpy as np
import pandas as pd

# --- 1. 학습 데이터 로드 & 트리 구축 ---
train_df = pd.read_csv('process1_result.csv')  # 학습용 전처리 완료 파일
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_train = train_df[features].values
y_train = train_df['Survived'].values

# Gini 불순도 계산
def gini(labels):
    classes, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)

# 가중 Gini 계산
def weighted_gini(col, labels, t):
    left  = labels[col <= t]
    right = labels[col >  t]
    if len(left)==0 or len(right)==0:
        return None
    n = len(labels)
    return (len(left)/n)*gini(left) + (len(right)/n)*gini(right)

# 최적 분할 찾기
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

# 재귀 트리 생성 (max_depth=3)
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

# 예측 함수
def predict(tree, x):
    if tree['leaf']:
        return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict(tree['left'], x)
    else:
        return predict(tree['right'], x)

# 트리 학습
tree = build_tree(X_train, y_train, max_depth=15)

# --- 2. 테스트 데이터 로드 & 예측 ---
test_df = pd.read_csv('process2_result.csv')  # 테스트용 전처리 완료 파일
X_test = test_df[features].values
ids    = test_df['PassengerId'].values

# 예측
test_preds = np.array([predict(tree, row) for row in X_test])

# --- 3. 제출 파일 생성 ---
submission = pd.DataFrame({
    'PassengerId': ids,
    'Survived':    test_preds
})
submission.to_csv('dt_manual_depth15_test2.csv', index=False)
print("dt_manual_depth15_test2.csv 파일이 생성되었습니다.")
