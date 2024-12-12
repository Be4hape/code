import pandas as pd

# Ticket 전처리 함수
def preprocess_ticket(ticket):
    if any(char.isalpha() for char in ticket):
        return ''.join([char for char in ticket if char.isalpha()])
    return ticket

# 데이터 전처리 함수
def preprocess_data(data):
    # Ticket 데이터 전처리
    data['Ticket'] = data['Ticket'].apply(preprocess_ticket)

    # 나머지 기본 전처리
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(float)

    # Name 열: Middle Name 추출 및 랭크 부여
    data['MiddleName'] = data['Name'].str.extract(r', ([^\.]+)\.', expand=False).fillna('Unknown')
    middle_name_counts = data['MiddleName'].value_counts()
    middle_name_rank = {name: rank for rank, name in enumerate(middle_name_counts.index)}
    data['Name'] = data['MiddleName'].map(middle_name_rank).fillna(len(middle_name_rank)).astype(int)

    return data

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
    if 'Survived' in data.columns:
        family_survival_rate = data.groupby('Family_Group')['Survived'].transform('mean')
        data['Family_Survival_Rate'] = family_survival_rate.fillna(0.5)
    else:
        data['Family_Survival_Rate'] = 0.5 + 0.5 * family_female_ratio

    return data

# Survival Score 계산 함수
def calculate_survival_score(data):
    correlation_coefficients = {
        'Pclass': -0.33,
        'Name': 0.33,
        'Sex': 0.54,
        'Parch': 0.12,
        'Fare': 0.29,
        'TicketNumeric': -0.12  # Numerical Ticket 값만 사용
    }

    if 'Ticket' in data.columns:
        data['TicketNumeric'] = pd.to_numeric(data['Ticket'], errors='coerce').fillna(0)

    if 'Fare' in data.columns:
        data['Fare'] = (data['Fare'] - data['Fare'].mean()) / data['Fare'].std()
    if 'TicketNumeric' in data.columns:
        data['TicketNumeric'] = (data['TicketNumeric'] - data['TicketNumeric'].mean()) / data['TicketNumeric'].std()

    data['SurvivalScore'] = 0
    for column, coefficient in correlation_coefficients.items():
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
            data['SurvivalScore'] += data[column] * coefficient

    return data

# Survival Score 업데이트 함수
def update_survival_score_with_family(data):
    data['SurvivalScore'] += data['Family_Survival_Rate']
    return data
def main():
    # 테스트 데이터 로드
    data = pd.read_csv('test2.csv')  # 파일 경로 수정 필요

    # 데이터 전처리
    data = preprocess_data(data)

    # 가족 그룹 및 생존률 계산
    data = find_family_groups_with_sibsp_parch(data)

    # 생존 점수 계산
    data = calculate_survival_score(data)

    # 가족 생존률 적용
    data = update_survival_score_with_family(data)

    # Median 기준으로 생존 여부 예측
    median_value = data['SurvivalScore'].median()
    data['Survived'] = (data['SurvivalScore'] > median_value).astype(int)  # 타이틀 이름 수정

    # PassengerId와 Survived 열만 남김
    submission = data[['PassengerId', 'Survived']]

    # 결과 저장 (캐글 제출 형식)
    submission.to_csv('.csv', index=False)

    # 주요 결과 확인
    print(submission.head())

if __name__ == "__main__":
    main()
