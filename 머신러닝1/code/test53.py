import pandas as pd
from sklearn.metrics import accuracy_score

# 가족 생존률 계산 함수
def find_family_groups_with_sibsp_parch(data):
    # Step 1: 가족 크기 계산
    data['Count'] = data['SibSp'] + data['Parch'] + 1

    # Step 2: 티켓, Fare, Embarked를 기반으로 가족 그룹 판별
    data['Family_Group'] = data.groupby(['Ticket', 'Fare', 'Embarked']).ngroup()

    # Step 3: 가족별로 Count를 분배
    family_counts = data.groupby('Family_Group')['Count'].transform('sum')
    data['Count'] = family_counts - 1  # 가족 크기에서 본인 제외

    # Step 4: 가족 내 여성 비율 계산
    family_female_ratio = data.groupby('Family_Group')['Sex'].transform('mean')
    data['Family_Survival_Rate'] = 0.5 + 0.5 * family_female_ratio

    return data

def calculate_survival_score(data):
    # Define Spearman correlation coefficients
    correlation_coefficients = {
        #'Pclass': -0.33,
        #'Name': 0.51,
        'Sex': 0.54,
        'SibSp': 0.12,
        'Parch': 0.18,
        #'Fare': 0.34,
        'Embarked': 0.10,
        'TicketNumeric': -0.24
    }

    # Rank Transformation for Spearman correlation
    for column in correlation_coefficients.keys():
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].rank()

    # Initialize Survival Score with 0
    data['SurvivalScore'] = 0

    # Calculate Survival Score by summing (column value * correlation coefficient)
    for column, coefficient in correlation_coefficients.items():
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
            data['SurvivalScore'] += data[column] * coefficient

    return data


# Survival Score 업데이트 함수 (가족 생존률 적용)
def update_survival_score_with_family(data):
    # 가족 생존률을 Survival Score에 추가
    data['SurvivalScore'] += data['Family_Survival_Rate']
    return data

# 메인 실행 함수
def main():
    # 데이터 로드
    data = pd.read_csv('process1.csv')  # Input file

    # 성별 숫자로 변환 (0: 남성, 1: 여성)
    data['Sex'] = data['Sex'].map({0: 0, 1: 1})  # 이미 인코딩된 데이터로 가정

    # 가족 그룹 및 생존률 계산 (모든 Ticket 데이터 사용)
    data = find_family_groups_with_sibsp_parch(data)

    # 생존 점수 계산 (Numerical Ticket만 사용)
    data = calculate_survival_score(data)

    # 가족 생존률 적용
    data = update_survival_score_with_family(data)

    # SurvivalScore의 median 값을 기준으로 생존 여부 판단
    median_value = data['SurvivalScore'].median()
    data['PredictedSurvival'] = (data['SurvivalScore'] > median_value).astype(int)

    # 정확도 계산
    if 'Survived' in data.columns:
        accuracy = accuracy_score(data['Survived'], data['PredictedSurvival'])
        print(f"Accuracy of Survival Prediction: {accuracy * 100:.2f}%")
    else:
        print("No 'Survived' column found in the data.")

if __name__ == "__main__":
    main()
