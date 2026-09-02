import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# 데이터 불러오기

df = pd.read_csv("data/Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
).fillna(0)


# X / y 분리

X = df.drop(columns=["Churn", "customerID"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# 문자형 / 숫자형 컬럼 구분

categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns


# 전처리 설정

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_cols
    ),
    (
        "num",
        StandardScaler(),
        numeric_cols
    )
])


# 학습 / 테스트 데이터 분리

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 모델 설정

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# GridSearchCV 설정

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1"
)


# 최적 모델 탐색

print("\n[GridSearchCV]")

grid_search.fit(X_train, y_train)

print("가장 좋은 설정:", grid_search.best_params_)
print(f"가장 높은 평균 F1: {grid_search.best_score_ * 100:.2f}%")

best_model = grid_search.best_estimator_


# 최적 모델 저장

joblib.dump(
    best_model,
    "logistic_churn_model.joblib"
)

print("모델 저장 완료")


# 모델 성능 평가

prediction = best_model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

print("\n모델 성능 평가")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix")
print(cm)


# 고객 조회

print("\n고객 조회")

user_input = input("조회할 고객 아이디 : ")

customer = df[
    df["customerID"].str.lower() == user_input.lower()
]

if customer.empty:
    print("존재하지 않는 고객님입니다.")
    exit()


# 실제 고객 상태 확인

churn_status = customer["Churn"].values[0]

if churn_status == "Yes":
    print("현재 비활동중인 고객님입니다.")
else:
    print("현재 활동중인 고객님입니다.")


# 고객 이탈 확률 예측

customer_X = customer.drop(columns=["Churn", "customerID"])

probability = best_model.predict_proba(customer_X)
churn_probability = probability[0][1]

customer_id = customer["customerID"].values[0]

print(
    f"\n{customer_id} 고객님의 예상 이탈 확률은 "
    f"{churn_probability * 100:.2f}% 입니다."
)