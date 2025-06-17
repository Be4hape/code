import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1) 데이터 로드
train_df = pd.read_csv('process1_result.csv')
test2_df = pd.read_csv('process2_result.csv')

features = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']

X = train_df[features]
y = train_df['Survived']

X_test2 = test2_df[features]
# test2에 Survived 열이 있다면 평가 가능
has_y_test2 = 'Survived' in test2_df.columns
if has_y_test2:
    y_test2 = test2_df['Survived']

# 2) train/validation 분할 (test1)
X_train, X_test1, y_train, y_test1 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) 모델 정의
models = {
    'KNN': KNeighborsClassifier(n_neighbors=20),
    'LogisticRegression': LogisticRegression(max_iter=10),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

# 4) 학습, 평가, 결과 저장
for name, model in models.items():
    # 학습
    model.fit(X_train, y_train)

    # test1 (validation) 정확도
    y_pred1 = model.predict(X_test1)
    acc1 = accuracy_score(y_test1, y_pred1)
    print(f'{name:20s} | test1 accuracy: {acc1:.4f}')

    # test2 예측 및 정확도 (있다면)
    y_pred2 = model.predict(X_test2)
    if has_y_test2:
        acc2 = accuracy_score(y_test2, y_pred2)
        print(f'{name:20s} | test2 accuracy: {acc2:.4f}')
    else:
        print(f'{name:20s} | test2 predictions saved to CSV (no labels)')

    # submission 파일 생성
    submission = pd.DataFrame({
        'PassengerId': test2_df['PassengerId'],
        'Survived':    y_pred2
    })
    submission_filename = f'{name.lower()}_submission.csv'
    submission.to_csv(submission_filename, index=False)
    print(f' → {submission_filename} 생성\n')
