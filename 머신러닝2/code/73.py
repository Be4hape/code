import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. Load data & select Spearman-based best subset ['Sex','Name'] ---
df = pd.read_csv('process1_result.csv')
features = ['Sex', 'Name']
X = df[features].values
y = df['Survived'].values

# --- 2. Standardize ---
means = X.mean(axis=0)
stds  = X.std(axis=0)
stds[stds == 0] = 1.0
X_std = (X - means) / stds

# --- 3. PCA → 2 components (since original dim=2) ---
cov    = np.cov(X_std, rowvar=False)
eigval, eigvec = np.linalg.eigh(cov)
order = np.argsort(eigval)[::-1]
V_pca = eigvec[:, order[:2]]    # (2×2)
X_pca = X_std.dot(V_pca)        # (n,2)

# --- 4. LDA → 1 Fisher component ---
mu0 = X_std[y==0].mean(axis=0)
mu1 = X_std[y==1].mean(axis=0)
d = X_std.shape[1]
SW = np.zeros((d,d))
for xi, yi in zip(X_std, y):
    m = mu1 if yi==1 else mu0
    diff = (xi-m).reshape(-1,1)
    SW += diff.dot(diff.T)
w_lda = np.linalg.inv(SW).dot(mu1-mu0)
w_lda /= np.linalg.norm(w_lda)
X_lda = X_std.dot(w_lda).reshape(-1,1)  # (n,1)

# --- 5. 5-Fold indices (NumPy only) ---
def get_folds(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    sizes = np.full(k, n//k, int); sizes[:n%k]+=1
    folds, start = [], 0
    for sz in sizes:
        folds.append(idx[start:start+sz])
        start += sz
    return folds

folds = get_folds(len(y), k=5, seed=42)

# --- 6. Model definitions (fixed hyperparams) ---
def sigmoid(z): return 1/(1+np.exp(-z))
def train_logistic_sgd(X,y,lr=0.001,epochs=9):
    m,d = X.shape; w=np.zeros(d); b=0.0
    for _ in range(epochs):
        for i in range(m):
            xi, yi = X[i], y[i]
            delta = sigmoid(xi.dot(w)+b) - yi
            w -= lr*delta*xi; b -= lr*delta
    return w,b

def predict_logistic(model, X):
    w,b = model
    return (sigmoid(X.dot(w)+b)>=0.5).astype(int)

def euclid_dist_matrix(A,B):
    return np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(axis=2))

def knn_predict(TR_X,TR_y,TE_X,k=20):
    D = euclid_dist_matrix(TE_X,TR_X)
    neigh = np.argpartition(D, k, axis=1)[:,:k]
    return np.array([np.bincount(TR_y[idx]).argmax() for idx in neigh])

def gini(labels):
    _,cnt = np.unique(labels,return_counts=True)
    p = cnt/cnt.sum(); return 1-(p**2).sum()

def weighted_gini(col,labels,t):
    l=labels[col<=t]; r=labels[col>t]
    if len(l)==0 or len(r)==0: return None
    n=len(labels)
    return len(l)/n*gini(l) + len(r)/n*gini(r)

def find_best_split(X,y):
    parent = gini(y); best = (0,None,None)
    for f in range(X.shape[1]):
        vals = np.unique(X[:,f]); ths = (vals[:-1]+vals[1:])/2
        for t in ths:
            wg = weighted_gini(X[:,f],y,t)
            if wg is None: continue
            gain = parent - wg
            if gain > best[0]:
                best = (gain,f,t)
    return best[1], best[2]

def build_tree(X,y,depth=0,max_depth=10):
    if len(np.unique(y))==1 or depth==max_depth:
        vals,cnt = np.unique(y,return_counts=True)
        return {'leaf':True,'pred':vals[cnt.argmax()]}
    feat,th = find_best_split(X,y)
    if feat is None:
        vals,cnt = np.unique(y,return_counts=True)
        return {'leaf':True,'pred':vals[cnt.argmax()]}
    mask = X[:,feat] <= th
    return {
      'leaf':False,'feat':feat,'thresh':th,
      'left': build_tree(X[mask],y[mask],depth+1,max_depth),
      'right':build_tree(X[~mask],y[~mask],depth+1,max_depth)
    }

def predict_tree(tree,x):
    if tree['leaf']: return tree['pred']
    branch = 'left' if x[tree['feat']]<=tree['thresh'] else 'right'
    return predict_tree(tree[branch],x)

# --- 7. 5-Fold CV on Full / PCA / LDA ---
sets = {
    'Full': X_std,
    'PCA' : X_pca,
    'LDA' : X_lda
}

for name, Xset in sets.items():
    acc_log, acc_knn, acc_dt, acc_mlp = [], [], [], []
    for i, val_idx in enumerate(folds):
        trn_idx = np.hstack([folds[j] for j in range(5) if j!=i])
        X_tr, y_tr = Xset[trn_idx], y[trn_idx]
        X_va, y_va = Xset[val_idx],  y[val_idx]

        # Logistic
        w,b = train_logistic_sgd(X_tr, y_tr)
        acc_log.append(accuracy_score(y_va, predict_logistic((w,b), X_va)))

        # KNN
        acc_knn.append(accuracy_score(y_va, knn_predict(X_tr, y_tr, X_va)))

        # Decision Tree (depth=10)
        tree = build_tree(X_tr, y_tr, max_depth=10)
        preds = np.array([predict_tree(tree,x) for x in X_va])
        acc_dt.append(accuracy_score(y_va, preds))

        # MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(32,),
            learning_rate_init=0.005,
            max_iter=300,
            early_stopping=True,
            random_state=42
        )
        mlp.fit(X_tr, y_tr)
        acc_mlp.append(accuracy_score(y_va, mlp.predict(X_va)))

    print(f"\n=== {name} ===")
    print(f"logistic : {np.mean(acc_log)*100:.5f}%")
    print(f"knn      : {np.mean(acc_knn)*100:.5f}%")
    print(f"dt       : {np.mean(acc_dt)*100:.5f}%")
    print(f"mlp      : {np.mean(acc_mlp)*100:.5f}%")
