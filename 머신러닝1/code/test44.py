import pandas as pd

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

    # Step 5: 가족 생존률 설정
    # 학습 데이터에서는 Survived를 사용
    if 'Survived' in data.columns:
        family_survival_rate = data.groupby('Family_Group')['Survived'].transform('mean')
        data['Family_Survival_Rate'] = family_survival_rate.fillna(0.5)
    else:
        # 테스트 데이터에서는 여성 비율 기반 생존률 적용
        data['Family_Survival_Rate'] = 0.5 + 0.5 * family_female_ratio

    return data

def calculate_survival_score(data):
    # Define correlation coefficients for high-correlation columns
    correlation_coefficients = {
        'Pclass': -0.33,
        'Name': 0.33,
        'Sex': 0.54,
        'Parch': 0.12,
        'Fare': 0.29,
        'TicketNumeric': -0.12  # Numerical Ticket 값만 사용
    }

    # Ensure Ticket is converted to numeric for Survival Score
    if 'Ticket' in data.columns:
        data['TicketNumeric'] = pd.to_numeric(data['Ticket'], errors='coerce').fillna(0)

    # 스케일링: Normalize (0~1) or Standardize (Z-score)
    if 'Fare' in data.columns:
        data['Fare'] = (data['Fare'] - data['Fare'].mean()) / data['Fare'].std()
    if 'TicketNumeric' in data.columns:
        data['TicketNumeric'] = (data['TicketNumeric'] - data['TicketNumeric'].mean()) / data['TicketNumeric'].std()

    # Initialize Survival Score with 0
    data['SurvivalScore'] = 0

    # Calculate Survival Score by summing (column value * correlation coefficient)
    for column, coefficient in correlation_coefficients.items():
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
            data['SurvivalScore'] += data[column] * coefficient

    # Debug: Print the contribution of each column to Survival Score
    for column, coefficient in correlation_coefficients.items():
        if column in data.columns:
            data[f'{column}_Score'] = data[column] * coefficient

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

    # 가족 그룹 및 생존률 계산 (모든 Ticket 데이터 사용)
    data = find_family_groups_with_sibsp_parch(data)

    # 생존 점수 계산 (Numerical Ticket만 사용)
    data = calculate_survival_score(data)

    # 가족 생존률 적용
    data = update_survival_score_with_family(data)

    # 결과 저장
    data.to_csv('process1_result.csv', index=False)  # Output file

    # 주요 결과 확인
    print(data[['Survived', 'Family_Survival_Rate', 'SurvivalScore']].head())

if __name__ == "__main__":
    main()
