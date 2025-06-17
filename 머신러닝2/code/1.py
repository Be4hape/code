import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler

# 1. process1.csv 파일 로드
data = pd.read_csv('process1.csv')

# 2. 특징과 레이블 분리 (예: 'Survived'가 타겟 변수, 나머지 피처 사용)
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
X = data[features].values
y = data['Survived'].values

# (선택) 3. 스케일링: KNN은 거리 기반이므로 표준화 권장
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

# 4. KNN의 핵심 함수들 직접 구현
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
    # 거리에 따라 정렬 후, k개의 이웃 선택
    distances.sort(key=lambda x: x[0])
    k_neighbors = [label for (dist, label) in distances[:k]]
    return majority_vote(k_neighbors)

# 5. 5-fold 교차검증으로 후보 k 값 평가하는 함수
def cross_validate_knn(X, y, candidate_k_list, num_folds=5):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # 무작위로 섞기
    fold_size = n_samples // num_folds
    k_accuracy = {}

    for k in candidate_k_list:
        fold_accuracies = []
        for fold in range(num_folds):
            start = fold * fold_size
            # 마지막 fold에는 남은 모든 데이터를 포함
            end = n_samples if fold == num_folds - 1 else (fold + 1) * fold_size

            # 현재 fold를 검증셋, 나머지를 학습셋으로 설정
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
        k_accuracy[k] = np.mean(fold_accuracies)
    return k_accuracy

# 6. 후보 k 값 설정 및 교차검증 수행
candidate_k = [3, 5, 7, 9, 11]
accuracy_results = cross_validate_knn(X, y, candidate_k, num_folds=5)

print("각 후보 k 값에 대한 평균 정확도:")
for k, acc in accuracy_results.items():
    print(f"k = {k}: Accuracy = {acc:.4f}")

# 7. 최적의 k 값 선택
optimal_k = max(accuracy_results, key=accuracy_results.get)
print("최적의 k 값:", optimal_k)
