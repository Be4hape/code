## 로지스틱회귀 - test1에대한 정확도, 인스턴스 러닝, 올바른 피쳐 사용


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 (전처리+엔지니어링 완료된 학습 데이터) ---
train_df = pd.read_csv('process1_result.csv')

# --- 2. 상관계수 딕셔너리 & 피처 리스트 정의 ---
coefficients = {
    'Sex': 0.54,
    'SibSp': 0.12,
    'Parch': 0.18,
    'Embarked': 0.10,
    'TicketNumeric': -0.24
}
features = list(coefficients.keys())

# --- 3. 피처 및 레이블 분리 ---
X = train_df[features].values
y = train_df['Survived'].values

# --- 4. 스케일링 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. 온라인 SGD 로지스틱 회귀 학습 ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

m, n = X_scaled.shape
w = np.zeros(n)
b = 0.0
lr = 0.001
epochs = 100
eps = 1e-15

for _ in range(epochs):
    for i in range(m):
        xi, yi = X_scaled[i], y[i]
        zi = xi.dot(w) + b
        hi = sigmoid(zi)
        dw = (hi - yi) * xi
        db = (hi - yi)
        w -= lr * dw
        b -= lr * db

# --- 6. test1(test=process1) 정확도 계산 ---
probs = sigmoid(X_scaled.dot(w) + b)
preds = (probs >= 0.5).astype(int)
acc = accuracy_score(y, preds)
print(f"Test1 Accuracy: {acc * 100:.4f}%")
