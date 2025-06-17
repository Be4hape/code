import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 학습 데이터 로드 (파일명은 환경에 맞게 수정하세요) ---
df = pd.read_csv('process1_result.csv')

# --- 2. 피처 리스트 & X, y 준비 ---
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X = df[features].values
y = df['Survived'].values

# --- 3. Gini 불순도 계산 ---
def gini(labels):
    classes, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)

# --- 4. 주어진 임계값 t에 대한 가중 Gini ---
def weighted_gini(col, labels, t):
    left  = labels[col <= t]
    right = labels[col >  t]
    if len(left)==0 or len(right)==0:
        return None
    n = len(labels)
    return (len(left)/n)*gini(left) + (len(right)/n)*gini(right)

# --- 5. 최적 (feat, t) 찾기 ---
def find_best_split(X, y):
    best_gain, best_feat, best_t = 0.0, None, None
    parent = gini(y)
    for feat in range(X.shape[1]):
        vals = np.unique(X[:, feat])
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        for t in thresholds:
            wg = weighted_gini(X[:, feat], y, t)
            if wg is None: continue
            gain = parent - wg
            if gain > best_gain:
                best_gain, best_feat, best_t = gain, feat, t
    return best_feat, best_t

# --- 6. max_depth 제약을 둔 재귀 트리 생성 ---
def build_tree(X, y, depth=0, max_depth=3):
    # 리프 조건: 한 클래스만 남거나 최대 깊이 도달
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

# --- 7. 단일 샘플 예측 함수 ---
def predict(tree, x):
    if tree['leaf']:
        return tree['pred']
    if x[tree['feat']] <= tree['thresh']:
        return predict(tree['left'], x)
    else:
        return predict(tree['right'], x)

# --- 8. Depth 1~20마다 Train Accuracy 출력 & 저장 ---
depths = list(range(1, 21))
accuracies = []

for d in depths:
    tree = build_tree(X, y, max_depth=d)
    preds = np.array([predict(tree, xi) for xi in X])
    acc = (preds == y).mean() * 100
    accuracies.append(acc)
    print(f"Depth {d:2d}: Train Accuracy = {acc:.2f}%")

# --- 9. 결과 시각화 ---
plt.figure(figsize=(10, 6))
plt.plot(depths, accuracies, marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('Train Accuracy (%)')
plt.title('Train Accuracy vs. Tree Depth (1–20)')
plt.xticks(depths)
plt.grid(True)
plt.show()
