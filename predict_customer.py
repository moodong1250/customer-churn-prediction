import pandas as pd
import joblib

## LogisticRegression 모델

model = joblib.load(
    "logistic_churn_model.joblib"
)

print("저장된 모델 불러오기 완료")

df = pd.read_csv(
    "data/Telco-Customer-Churn.csv"
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

user_input = input(
    "조회할 고객 아이디 : "  # 6840-RESVB
)

customer = df[
    df["customerID"].str.lower()
    == user_input.lower()
]

if customer.empty:
    print("존재하지 않는 고객님입니다.")
    exit()

customer_X = customer.drop(
    columns=["Churn", "customerID"]
)

probability = model.predict_proba(
    customer_X
)

churn_probability = probability[0][1]



print(
    f"{customer['customerID'].values[0]} "
    f"고객님의 예상 이탈 확률은 "
    f"{churn_probability * 100:.2f}% 입니다."
)