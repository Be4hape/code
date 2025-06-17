import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. 데이터 준비: process1.csv 파일 로드 및 전처리
# -------------------------------
data = pd.read_csv('process1.csv')

# 예시로 'Survived'를 타겟, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked'] 피처 사용
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
X = data[features].values
y = data['Survived'].values

# 거리 기반 모델이므로 스케일링 적용 (표준화)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# 2. KNN 알고리즘 직접 구현 함수
# -------------------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def majority_vote(neighbors):
    values, counts = np.unique(neighbors, return_counts=True)
    return values[np.argmax(counts)]

def knn_predict(train_X, train_y, test_sample, k):
    distances = []
    for i in range(train_X.shape[0]):
        distance = euclidean_distance(test_sample, train_X[i])
        distances.append((distance, train_y[i]))
    distances.sort(key=lambda x: x[0])
    k_neighbors = [label for (dist, label) in distances[:k]]
    return majority_vote(k_neighbors)

# -------------------------------
# 3. 5-fold (혹은 그 외) 교차검증 평가 함수:
#    여기서는 k 값은 고정(k=12)로 사용
# -------------------------------
def cross_validate_knn_fixed_k(X, y, k, num_folds):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # 무작위 섞기
    fold_size = n_samples // num_folds
    fold_accuracies = []

    # num_folds가 1이면 전체 데이터를 이용해 학습 정확도를 계산
    if num_folds == 1:
        correct = 0
        for i in range(n_samples):
            pred = knn_predict(X, y, X[i], k)
            if pred == y[i]:
                correct += 1
        accuracy = correct / n_samples
        fold_accuracies.append(accuracy)
        print(f"[1-fold] (학습 데이터 자체 평가) Accuracy: {accuracy:.4f}")
        return accuracy  # 단일 값 반환

    # num_folds >= 2인 경우 교차검증 진행
    for fold in range(num_folds):
        start = fold * fold_size
        end = n_samples if fold == num_folds - 1 else (fold + 1) * fold_size

        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        correct = 0
        for i in range(X_val.shape[0]):
            pred = knn_predict(X_train, y_train, X_val[i], k)
            if pred == y_val[i]:
                correct += 1
        accuracy = correct / X_val.shape[0]
        fold_accuracies.append(accuracy)
        print(f"[{num_folds}-fold] Fold {fold+1} accuracy: {accuracy:.4f}")

    avg_accuracy = np.mean(fold_accuracies)
    print(f"[{num_folds}-fold] Average Accuracy: {avg_accuracy:.4f}\n")
    return avg_accuracy

# -------------------------------
# 4. k는 고정 (k=12) 하고, num_folds를 1부터 10까지 변화시켜서 평가
# -------------------------------
k_fixed = 12
fold_results = {}

for num_folds in range(1, 11):
    print(f"===========================")
    print(f"교차검증 fold 개수: {num_folds}")
    avg_acc = cross_validate_knn_fixed_k(X, y, k_fixed, num_folds)
    fold_results[num_folds] = avg_acc

print("========================================")
print("fold 개수에 따른 평균 정확도:")
for folds, acc in fold_results.items():
    print(f"{folds}-fold: Average Accuracy = {acc:.4f}")
