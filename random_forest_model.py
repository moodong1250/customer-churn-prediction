import pandas as pd


from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# STEP 1. 데이터 불러오기 및 전처리
# =========================================================

df = pd.read_csv("data/Telco-Customer-Churn.csv")

# TotalCharges를 숫자로 변환
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 결측값을 0으로 처리
df["TotalCharges"] = df["TotalCharges"].fillna(0)


# =========================================================
# STEP 2. X / y 분리
# =========================================================

# X = 모델이 학습할 고객 정보
# customerID와 정답인 Churn은 제외
X = df.drop(
    columns=["Churn", "customerID"]
)

# y = 정답
# No → 0
# Yes → 1
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# =========================================================
# STEP 3. 문자형 / 숫자형 컬럼 구분
# =========================================================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numeric_cols = X.select_dtypes(
    exclude=["object"]
).columns


# =========================================================
# STEP 4. 전처리 설정
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
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
    ]
)


# =========================================================
# STEP 5. 학습 데이터 / 테스트 데이터 분리
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# STEP 6. Pipeline + RandomForestClassifier 
# =========================================================



pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42
            )
        )
    ]
)





# =========================================================
# STEP 7. 모델 학습
# =========================================================

print("\n[모델 학습]")

pipeline.fit(
    X_train,
    y_train
)

print("모델 학습 완료")


# =========================================================
# STEP 8. 테스트 데이터로 모델 성능 평가
# =========================================================

prediction = pipeline.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    prediction
)

cm = confusion_matrix(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction
)

recall = recall_score(
    y_test,
    prediction
)

f1 = f1_score(
    y_test,
    prediction
)


print("\n==============================")
print("       모델 성능 평가")
print("==============================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix")
print(cm)


# =========================================================
# STEP 9. 고객 ID 조회
# =========================================================

print("\n==============================")
print("       고객 조회")
print("==============================")

user_input = input(
    "조회할 고객 아이디 : "
)

customer = df[
    df["customerID"].str.lower()
    == user_input.lower()
]


# 고객이 존재하지 않는 경우
if customer.empty:

    print("존재하지 않는 고객님입니다.")
    exit()


# =========================================================
# STEP 10. 실제 고객의 Churn 상태 확인
# =========================================================

churn_status = customer["Churn"].values[0]

if churn_status == "Yes":

    print("현재 비활동중인 고객님입니다.")

else:

    print("현재 활동중인 고객님입니다.")


# =========================================================
# STEP 11. 고객의 이탈 확률 예측
# =========================================================

customer_X = customer.drop(
    columns=["Churn", "customerID"]
)

probability = pipeline.predict_proba(
    customer_X
)

churn_probability = probability[0][1]


print(
    f"\n{customer['customerID'].values[0]} "
    f"고객님의 예상 이탈 확률은 "
    f"{churn_probability * 100:.2f}% 입니다."
)


