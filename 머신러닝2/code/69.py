import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 & 사용할 피처 지정 ---
train_df = pd.read_csv('process1_result.csv')
test_df  = pd.read_csv('process2_result.csv')

features = [
    'Sex','Pclass','Name','Family_Group','Embarked'
]
n_feat = len(features)

X_full  = train_df[features].values
y_full  = train_df['Survived'].values
X_test2 = test_df[features].values
ids_test2 = test_df['PassengerId'].values

# --- 2. 결측치 처리 & 스케일링 ---
col_means = np.nanmean(X_full, axis=0)
mask_full = np.isnan(X_full)
X_full[mask_full] = np.take(col_means, np.where(mask_full)[1])
mask_t2 = np.isnan(X_test2)
X_test2[mask_t2] = np.take(col_means, np.where(mask_t2)[1])

means = X_full.mean(axis=0)
stds  = X_full.std(axis=0)
stds[stds == 0] = 1.0
X_full_s  = (X_full  - means) / stds
X_test2_s = (X_test2 - means) / stds

# --- 3. 5-Fold 인덱스 생성 (NumPy만) ---
def get_folds(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    sizes = np.full(k, n // k, int)
    sizes[:n % k] += 1
    folds, start = [], 0
    for sz in sizes:
        folds.append(idx[start:start+sz])
        start += sz
    return folds

folds = get_folds(len(y_full), k=5, seed=42)

# --- 4. 모델 구현 (기존 코드 그대로) ---
def sigmoid(z): return 1/(1+np.exp(-z))
def train_logistic_sgd(X,y,lr,epochs):
    m,d = X.shape; w=np.zeros(d); b=0.0
    for _ in range(epochs):
        for i in range(m):
            xi, yi = X[i], y[i]
            delta = sigmoid(xi.dot(w)+b)-yi
            w -= lr*delta*xi; b -= lr*delta
    return w,b
def predict_logistic(model, X):
    w,b = model
    return (sigmoid(X.dot(w)+b)>=0.5).astype(int)

def euclid_dist_matrix(A,B):
    return np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(axis=2))
def knn_predict(TR_X,TR_y,TE_X,k):
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
    return len(l)/n*gini(l)+len(r)/n*gini(r)
def find_best_split(X,y):
    parent=gini(y); best=(0,None,None)
    for f in range(X.shape[1]):
        vals=np.unique(X[:,f]); ths=(vals[:-1]+vals[1:])/2
        for t in ths:
            wg=weighted_gini(X[:,f],y,t)
            if wg is None: continue
            gain=parent-wg
            if gain>best[0]: best=(gain,f,t)
    return best[1],best[2]
def build_tree(X,y,depth=0,max_depth=15):
    if len(np.unique(y))==1 or depth==max_depth:
        vals,cnt=np.unique(y,return_counts=True)
        return {'leaf':True,'pred':vals[cnt.argmax()]}
    feat,th=find_best_split(X,y)
    if feat is None:
        vals,cnt=np.unique(y,return_counts=True)
        return {'leaf':True,'pred':vals[cnt.argmax()]}
    mask=X[:,feat]<=th
    return {
      'leaf':False,'feat':feat,'thresh':th,
      'left': build_tree(X[mask],y[mask],depth+1,max_depth),
      'right':build_tree(X[~mask],y[~mask],depth+1,max_depth)
    }
def predict_tree(tree,x):
    if tree['leaf']: return tree['pred']
    branch = 'left' if x[tree['feat']]<=tree['thresh'] else 'right'
    return predict_tree(tree[branch],x)

# --- 5. 모든 부분집합 순회 + 5-Fold CV ---
best = {'mask':0,'model':None,'score':-1}
for mask in range(1, 1<<n_feat):
    idxs = [i for i in range(n_feat) if (mask>>i)&1]
    X_sub  = X_full_s[:, idxs]
    # CV 점수 저장
    cv_scores = {'logistic':[], 'knn':[], 'dt':[], 'mlp':[]}
    for i in range(5):
        val = folds[i]
        trn = np.hstack([folds[j] for j in range(5) if j!=i])
        X_tr,y_tr = X_sub[trn], y_full[trn]
        X_va,y_va = X_sub[val], y_full[val]
        # logistic
        w,b = train_logistic_sgd(X_tr,y_tr,lr=0.001,epochs=9)
        cv_scores['logistic'].append(accuracy_score(y_va,predict_logistic((w,b),X_va)))
        # knn
        cv_scores['knn'].append(accuracy_score(y_va,knn_predict(X_tr,y_tr,X_va,k=20)))
        # dt
        tree = build_tree(X_tr,y_tr,depth=0,max_depth=15)
        preds = np.array([predict_tree(tree,x) for x in X_va])
        cv_scores['dt'].append(accuracy_score(y_va,preds))
        # mlp
        mlp = MLPClassifier(hidden_layer_sizes=(32,),
                            learning_rate_init=0.005,
                            max_iter=300,
                            early_stopping=True,
                            random_state=42)
        mlp.fit(X_tr,y_tr)
        cv_scores['mlp'].append(accuracy_score(y_va,mlp.predict(X_va)))
    # 평균
    mean_scores = {m:np.mean(v) for m,v in cv_scores.items()}
    mtype, mscore = max(mean_scores.items(), key=lambda x:x[1])
    if mscore > best['score']:
        best.update({'mask':mask,'model':mtype,'score':mscore})

# --- 6. 결과 출력 ---
chosen = [features[i] for i in range(n_feat) if (best['mask']>>i)&1]
print("Best subset:", chosen)
print("Model:", best['model'], "CV score:", best['score'])

# --- 7. 최종 재학습 & Test2 예측 ---
idxs = [i for i in range(n_feat) if (best['mask']>>i)&1]
X_final  = X_full_s[:, idxs]
X2_final = X_test2_s[:, idxs]
if best['model']=='logistic':
    w,b = train_logistic_sgd(X_final,y_full,lr=0.001,epochs=9)
    final_preds = predict_logistic((w,b),X2_final)
elif best['model']=='knn':
    final_preds = knn_predict(X_final,y_full,X2_final,k=20)
elif best['model']=='dt':
    tree = build_tree(X_final,y_full,depth=0,max_depth=15)
    final_preds = np.array([predict_tree(tree,x) for x in X2_final])
else:
    mlp = MLPClassifier(hidden_layer_sizes=(32,),
                        learning_rate_init=0.005,
                        max_iter=300,
                        early_stopping=True,
                        random_state=42)
    mlp.fit(X_final,y_full)
    final_preds = mlp.predict(X2_final)

# --- 8. Submission ---
pd.DataFrame({'PassengerId':ids_test2,'Survived':final_preds}) \
  .to_csv('final_submission.csv', index=False)
print("Saved final_submission.csv")
