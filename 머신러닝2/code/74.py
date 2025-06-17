import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & Spearman 최적 subset ['Sex','Name'] ---
df = pd.read_csv('process1_result.csv')
features = ['Sex', 'Name']
X = df[features].values
y = df['Survived'].values

# --- 2. 스케일링 (Standardize) ---
means = X.mean(axis=0)
stds  = X.std(axis=0)
stds[stds==0] = 1.0
X_std = (X - means) / stds

# --- 3. PCA (원본 dim=2 → 2 components) & eigenvalue ---
cov    = np.cov(X_std, rowvar=False)
eigvals_pca, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals_pca)[::-1]
eigvals_pca = eigvals_pca[order]
V_pca = eigvecs[:, order[:2]]   # PCA 축 벡터 (2×2)
X_pca = X_std.dot(V_pca)        # 2차원 데이터

# --- 4. LDA (원본 dim=2 → 1 component) & generalized eigenvalue ---
mu0 = X_std[y==0].mean(axis=0)
mu1 = X_std[y==1].mean(axis=0)
d = X_std.shape[1]
SW = np.zeros((d,d))
for xi, yi in zip(X_std, y):
    m = mu1 if yi==1 else mu0
    diff = (xi - m).reshape(-1,1)
    SW += diff.dot(diff.T)
SB = (mu1-mu0).reshape(-1,1).dot((mu1-mu0).reshape(1,-1))
eigvals_lda, _ = np.linalg.eig(np.linalg.inv(SW).dot(SB))
eigvals_lda = np.real(eigvals_lda)
eigvals_lda = np.sort(eigvals_lda)[::-1]
w_lda = np.linalg.inv(SW).dot(mu1 - mu0)
w_lda /= np.linalg.norm(w_lda)
X_lda = X_std.dot(w_lda).reshape(-1,1)  # 1차원 데이터

# --- 5. PCA/LDA 고유값 시각화 ---
plt.figure(figsize=(10,4))

# PCA eigenvalues
plt.subplot(1,2,1)
plt.plot([1,2], eigvals_pca[:2], '-o')
plt.title('PCA Eigenvalues')
plt.xlabel('Component')
plt.ylabel('Eigenvalue')
plt.grid(True)

# LDA eigenvalues
plt.subplot(1,2,2)
plt.bar([1,2], eigvals_lda[:2], alpha=0.7)
plt.title('LDA Generalized Eigenvalues')
plt.xlabel('Discriminant')
plt.ylabel('Eigenvalue (λ)')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 6. 5-Fold CV 준비 & 모델 정의 (이전과 동일) ---
def get_folds(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    sizes = np.full(k, n//k, int); sizes[:n%k]+=1
    folds, st = [], 0
    for sz in sizes:
        folds.append(idx[st:st+sz]); st+=sz
    return folds

folds = get_folds(len(y), 5)

def sigmoid(z): return 1/(1+np.exp(-z))
def train_logistic_sgd(X,y,lr=0.001,epochs=9):
    m,d = X.shape; w=np.zeros(d); b=0.0
    for _ in range(epochs):
        for i in range(m):
            xi, yi = X[i], y[i]
            delta = sigmoid(xi.dot(w)+b) - yi
            w -= lr*delta*xi; b -= lr*delta
    return w,b

def predict_logistic(model,X):
    w,b = model
    return (sigmoid(X.dot(w)+b)>=0.5).astype(int)

def knn_predict(TR_X,TR_y,TE_X,k=20):
    D = np.sqrt(((TE_X[:,None,:]-TR_X[None,:,:])**2).sum(2))
    neigh = np.argpartition(D, k, axis=1)[:,:k]
    return np.array([np.bincount(TR_y[idx]).argmax() for idx in neigh])

def gini(lbl):
    _,cnt = np.unique(lbl,return_counts=True)
    p = cnt/cnt.sum(); return 1-(p**2).sum()

def weighted_gini(col,lbl,t):
    l=lbl[col<=t]; r=lbl[col>t]
    if len(l)==0 or len(r)==0: return None
    n=len(lbl)
    return len(l)/n*gini(l) + len(r)/n*gini(r)

def find_best_split(X,y):
    parent = gini(y); best=(0,None,None)
    for f in range(X.shape[1]):
        vs = np.unique(X[:,f]); ths=(vs[:-1]+vs[1:])/2
        for t in ths:
            wg = weighted_gini(X[:,f],y,t)
            if wg and parent-wg>best[0]:
                best=(parent-wg, f, t)
    return best[1], best[2]

def build_tree(X,y,depth=0,max_depth=10):
    if len(np.unique(y))==1 or depth==max_depth:
        vals,cnt=np.unique(y,return_counts=True)
        return {'leaf':True,'pred':vals[cnt.argmax()]}
    feat,th = find_best_split(X,y)
    if feat is None:
        vals,cnt=np.unique(y,return_counts=True)
        return {'leaf':True,'pred':vals[cnt.argmax()]}
    m = X[:,feat] <= th
    return {
      'leaf':False,'feat':feat,'thresh':th,
      'left': build_tree(X[m],y[m],depth+1,max_depth),
      'right':build_tree(X[~m],y[~m],depth+1,max_depth)
    }

def predict_tree(tree,x):
    if tree['leaf']: return tree['pred']
    branch = 'left' if x[tree['feat']]<=tree['thresh'] else 'right'
    return predict_tree(tree[branch],x)

# --- 7. 5-Fold CV on Full / PCA / LDA ---
sets = {'Full':X_std, 'PCA':X_pca, 'LDA':X_lda}
for name, Xset in sets.items():
    acc = {'log':[], 'knn':[], 'dt':[], 'mlp':[]}
    for i, val in enumerate(folds):
        trn = np.hstack([folds[j] for j in range(5) if j!=i])
        X_tr,y_tr = Xset[trn], y[trn]
        X_va,y_va = Xset[val],  y[val]

        w,b = train_logistic_sgd(X_tr,y_tr)
        acc['log'].append(accuracy_score(y_va, predict_logistic((w,b),X_va)))

        acc['knn'].append(accuracy_score(y_va, knn_predict(X_tr,y_tr,X_va)))

        tree = build_tree(X_tr,y_tr)
        acc['dt'].append(accuracy_score(y_va, [predict_tree(tree,x) for x in X_va]))

        mlp = MLPClassifier(hidden_layer_sizes=(32,),
                            learning_rate_init=0.005,
                            max_iter=300,
                            early_stopping=True,
                            random_state=42)
        mlp.fit(X_tr,y_tr)
        acc['mlp'].append(accuracy_score(y_va, mlp.predict(X_va)))

    print(f"\n=== {name} (PCA dims=2 / LDA dims=1) ===")
    print(f"logistic: {np.mean(acc['log'])*100:.2f}%")
    print(f"knn     : {np.mean(acc['knn'])*100:.2f}%")
    print(f"dt      : {np.mean(acc['dt'])*100:.2f}%")
    print(f"mlp     : {np.mean(acc['mlp'])*100:.2f}%")
