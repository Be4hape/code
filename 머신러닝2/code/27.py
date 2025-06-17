import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 및 피처/레이블 분리 ---
train_df = pd.read_csv('process1_result.csv')
features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
X_all    = train_df[features].values
y_all    = train_df['Survived'].values

# --- 2. 시그모이드 정의 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- 3. 교차검증 설정 ---
k_folds = 5
skf     = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# --- 4. 에폭스 최적화 (기본 lr=0.001) ---
default_lr  = 0.001
epoch_list  = range(10, 201, 10)
epoch_scores = {}

for num_epochs in epoch_list:
    fold_accs = []
    for train_idx, val_idx in skf.split(X_all, y_all):
        # 데이터 분할
        X_train_raw, X_val_raw = X_all[train_idx], X_all[val_idx]
        y_train,     y_val     = y_all[train_idx], y_all[val_idx]
        # 스케일링
        means = X_train_raw.mean(axis=0)
        stds  = X_train_raw.std(axis=0)
        stds[stds==0] = 1.0
        X_train = (X_train_raw - means) / stds
        X_val   = (X_val_raw   - means) / stds
        # 파라미터 초기화
        m, n = X_train.shape
        w = np.zeros(n); b = 0.0
        # SGD 학습
        for _ in range(num_epochs):
            for xi, yi in zip(X_train, y_train):
                z = np.dot(xi, w) + b
                delta = sigmoid(z) - yi
                w -= default_lr * delta * xi
                b -= default_lr * delta
        # 검증 정확도
        preds = (sigmoid(X_val.dot(w) + b) >= 0.5).astype(int)
        fold_accs.append(accuracy_score(y_val, preds))
    epoch_scores[num_epochs] = np.mean(fold_accs)
    print(f"Epochs={num_epochs:3d} → ACC mean = {epoch_scores[num_epochs]:.4f}")

best_epoch = max(epoch_scores, key=epoch_scores.get)
print(f"\n-- 최적 Epochs: {best_epoch} (ACC={epoch_scores[best_epoch]:.4f})")

# --- 5. 학습률 최적화 (고정 epochs=best_epoch) ---
lr_list    = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
lr_scores  = {}

for lr in lr_list:
    fold_accs = []
    for train_idx, val_idx in skf.split(X_all, y_all):
        X_train_raw, X_val_raw = X_all[train_idx], X_all[val_idx]
        y_train,     y_val     = y_all[train_idx], y_all[val_idx]
        means = X_train_raw.mean(axis=0)
        stds  = X_train_raw.std(axis=0)
        stds[stds==0] = 1.0
        X_train = (X_train_raw - means) / stds
        X_val   = (X_val_raw   - means) / stds
        m, n = X_train.shape
        w = np.zeros(n); b = 0.0
        for _ in range(best_epoch):
            for xi, yi in zip(X_train, y_train):
                z = np.dot(xi, w) + b
                delta = sigmoid(z) - yi
                w -= lr * delta * xi
                b -= lr * delta
        preds = (sigmoid(X_val.dot(w) + b) >= 0.5).astype(int)
        fold_accs.append(accuracy_score(y_val, preds))
    lr_scores[lr] = np.mean(fold_accs)
    print(f"LR={lr:.5f} → ACC mean = {lr_scores[lr]:.4f}")

best_lr = max(lr_scores, key=lr_scores.get)
print(f"\n-- 최적 Learning Rate: {best_lr} (ACC={lr_scores[best_lr]:.4f})")
