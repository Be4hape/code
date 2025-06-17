import numpy as np
import pandas as pd

# --- 1. 데이터 로드 ---
train_df = pd.read_csv('process1_result.csv')   # 전처리+엔지니어링 완료된 학습 데이터

# --- 2. 사용할 피처 리스트 ---
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']

# --- 3. 피처 및 레이블 분리 ---
X_train = train_df[features].values
y_train = train_df['Survived'].values

# --- 4. Gini 불순도 계산 ---
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)

# --- 5. 주어진 임계값으로 분할했을 때 가중 Gini 구하기 ---
def weighted_gini(X_col, y, t):
    left = y[X_col <= t]
    right= y[X_col >  t]
    if len(left)==0 or len(right)==0:
        return None
    n = len(y)
    return (len(left)/n)*gini(left) + (len(right)/n)*gini(right)

# --- 6. 가장 순수도 개선량(∆Gini)이 큰 분할 찾기 ---
def find_best_split(X, y):
    best_gain, best_feat, best_t = 0, None, None
    parent = gini(y)
    for fid in range(X.shape[1]):
        vals = np.unique(X[:,fid])
        # 연속값 split은 인접값 중간을 기준으로
        thr = (vals[:-1] + vals[1:]) / 2.0
        for t in thr:
            wg = weighted_gini(X[:,fid], y, t)
            if wg is None:
                continue
            gain = parent - wg
            if gain > best_gain:
                best_gain, best_feat, best_t = gain, fid, t
    return best_feat, best_t

# --- 7. 재귀적 트리 생성 (max depth = 3) ---
def build_tree(X, y, depth=0, max_depth=3):
    # 리프 조건: 순수 노드 or 최대 깊이
    if len(np.unique(y))==1 or depth==max_depth:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(cnts)]}

    feat, thresh = find_best_split(X, y)
    if feat is None:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(cnts)]}

    mask = X[:,feat] <= thresh
    left = build_tree(X[mask], y[mask], depth+1, max_depth)
    right= build_tree(X[~mask], y[~mask], depth+1, max_depth)
    return {'leaf':False, 'feat':feat, 'thresh':thresh, 'left':left, 'right':right}

# --- 8. 트리 예측 함수 ---
def predict(tree, x):
    if tree['leaf']:
        return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict(tree['left'], x)
    else:
        return predict(tree['right'], x)

# --- 9. 모델 학습 & 평가 ---
tree = build_tree(X_train, y_train, max_depth=3)
preds = np.array([predict(tree, x) for x in X_train])
train_acc = np.mean(preds == y_train)
print(f"Decision Tree (depth=3) Train Accuracy: {train_acc*100:.2f}%")
