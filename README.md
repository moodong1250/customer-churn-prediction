
고객 정보를 기반으로 이탈 가능성 예측 프로젝트 연습(2026 / 09 / 02)

사용 모델
→ Decision Tree
→ Logistic Regression
→ Random Forest

최종 모델
→ Logistic Regression



############ <Decision Tree> ############

현재 기준 최고 설정은 max_depth=7, min_samples_leaf=20
Cross Validation 평균 F1은 56.68%

############ <LogisticRegression> ############

가장 좋은 설정: {'model__C': 10} // max_iter=1000 고정 기준
가장 높은 평균 F1: 59.89%

############ <RandomForestClassifier> ############

GridSearchCV 평균 F1 = 57.81%

Random Forest 최종 설정  
n_estimators = 100
max_depth = 10
min_samples_leaf = 1
max_features = "sqrt"





| Decision Tree |   Logistic Regression   |
|               |                         |
| 조건을 하나씩 나눔 | 여러 특성의 영향을 동시에 합산 |
| Tree 구조       |       수식 기반           |
| `max_depth` 등이 중요 | 계수(coefficient)가 중요 |
| 판단 경로 확인 가능 | 각 특성이 얼마나 영향을 줬는지 확인 가능 |