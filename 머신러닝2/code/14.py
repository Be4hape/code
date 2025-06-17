import numpy as np
import pandas as pd

# --- 1. 데이터 로드 ---
df = pd.read_csv('process1_result.csv')  # 전처리+엔지니어링 완료된 학습 데이터

# --- 2. 사용할 5개 피처 리스트 ---
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X = df[features].values
y = df['Survived'].values

# --- 3. Gini 불순도 계산 함수 ---
def gini(labels):
    classes, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)

# --- 4. 주어진 피처열과 임계값 t로 나누었을 때의 가중 Gini ---
def weighted_gini(col, labels, t):
    left = labels[col <= t]
    right = labels[col >  t]
    if len(left) == 0 or len(right) == 0:
        return None
    n = len(labels)
    return (len(left)/n)*gini(left) + (len(right)/n)*gini(right)

# --- 5. 최적의 분할(피처, 임계값) 찾기 ---
def find_best_split(X, y):
    best_gain, best_feat, best_t = 0.0, None, None
    parent_gini = gini(y)
    for feat in range(X.shape[1]):
        vals = np.unique(X[:, feat])
        # 연속형 분할 기준: 중간값
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        for t in thresholds:
            wg = weighted_gini(X[:, feat], y, t)
            if wg is None:
                continue
            gain = parent_gini - wg
            if gain > best_gain:
                best_gain, best_feat, best_t = gain, feat, t
    return best_feat, best_t

# --- 6. 깊이 3짜리 결정 트리 생성(재귀) ---
## depth : 15로 변경
def build_tree(X, y, depth=0, max_depth=15):
    # 리프 조건: 한 클래스만 남거나 최대 깊이 도달
    if len(np.unique(y)) == 1 or depth == max_depth:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(cnts)]}
    feat, thresh = find_best_split(X, y)
    if feat is None:
        vals, cnts = np.unique(y, return_counts=True)
        return {'leaf': True, 'pred': vals[np.argmax(cnts)]}
    mask = X[:, feat] <= thresh
    left = build_tree(X[mask], y[mask], depth+1, max_depth)
    right = build_tree(X[~mask], y[~mask], depth+1, max_depth)
    return {'leaf': False, 'feat': feat, 'thresh': thresh, 'left': left, 'right': right}

# --- 7. 예측 함수 ---
def predict(tree, x):
    if tree['leaf']:
        return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict(tree['left'], x)
    else:
        return predict(tree['right'], x)

# --- 8. 트리 학습 & 평가 ---
tree = build_tree(X, y, max_depth=15)
preds = np.array([predict(tree, xi) for xi in X])
accuracy = (preds == y).mean()
print(f"Depth-15 Decision Tree Train Accuracy: {accuracy*100:.2f}%")
