## 피어슨 기준 - 피쳐pclass, name, embarked

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- A) 데이터 로드 & full feature 정리 ---
df = pd.read_csv('process1_result.csv')
features = ['Pclass','Name','Embarked']
X = df[features].values
y = df['Survived'].values

# --- B) 스케일링 (full 기준) ---
means = X.mean(axis=0)
stds  = X.std(axis=0)
stds[stds==0] = 1.0
X_std = (X - means) / stds

# --- C) PCA → 2개 주성분 ---
cov   = np.cov(X_std, rowvar=False)
eigs, vecs = np.linalg.eigh(cov)
order = np.argsort(eigs)[::-1]
V_pca = vecs[:, order[:2]]
X_pca = X_std.dot(V_pca)

# --- D) LDA → 1차원 Fisher ---
mu0, mu1 = X_std[y==0].mean(0), X_std[y==1].mean(0)
d = X_std.shape[1]
SW = np.zeros((d,d))
for xi, yi in zip(X_std, y):
    m = mu1 if yi==1 else mu0
    diff = (xi-m).reshape(-1,1)
    SW += diff.dot(diff.T)
w_lda = np.linalg.inv(SW).dot(mu1-mu0)
w_lda /= np.linalg.norm(w_lda)
X_lda = X_std.dot(w_lda).reshape(-1,1)

# --- E) 5-Fold 인덱스 (NumPy만) ---
def get_folds(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    sizes = np.full(k, n//k, int); sizes[:n%k]+=1
    folds=[]; st=0
    for sz in sizes:
        folds.append(idx[st:st+sz])
        st+=sz
    return folds

folds = get_folds(len(y), k=5, seed=42)

# --- F) k-Fold CV 루프 & 출력 ---
sets = {
    'Full': X_std,
    'PCA' : X_pca,
    'LDA' : X_lda
}

for name, Xset in sets.items():
    acc_log, acc_knn, acc_dt, acc_mlp = [], [], [], []
    for i, val_idx in enumerate(folds):
        # train/val 인덱스 분리
        trn_idx = np.hstack([folds[j] for j in range(5) if j!=i])
        X_tr, y_tr = Xset[trn_idx], y[trn_idx]
        X_va, y_va = Xset[val_idx],  y[val_idx]

        # 1) Logistic
        w,b = train_logistic_sgd(X_tr, y_tr, lr=0.001, epochs=9)
        acc_log.append(accuracy_score(y_va, predict_logistic((w,b), X_va)))

        # 2) KNN
        acc_knn.append(accuracy_score(y_va, knn_predict(X_tr, y_tr, X_va, k=20)))

        # 3) Decision Tree(depth=10)
        tree = build_tree(X_tr, y_tr, depth=0, max_depth=10)
        preds = np.array([predict_tree(tree,x) for x in X_va])
        acc_dt.append(accuracy_score(y_va, preds))

        # 4) MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(32,),
            learning_rate_init=0.005,
            max_iter=300,
            early_stopping=True,
            random_state=42
        )
        mlp.fit(X_tr, y_tr)
        acc_mlp.append(accuracy_score(y_va, mlp.predict(X_va)))

    # 평균 ACC 계산
    print(f"\n=== {name} set ===")
    print(f"Logistic    : {np.mean(acc_log)*100:.5f}%")
    print(f"KNN (k=20)  : {np.mean(acc_knn)*100:.5f}%")
    print(f"DecisionT   : {np.mean(acc_dt)*100:.5f}%")
    print(f"MLP         : {np.mean(acc_mlp)*100:.5f}%")
