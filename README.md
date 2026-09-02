# 고객 이탈 가능성 예측 프로젝트
연습 프로젝트 / 2026.09.02

고객 정보를 기반으로 이탈 가능성을 예측하는 머신러닝 분류 프로젝트

## 사용 모델
- Decision Tree
- Logistic Regression
- Random Forest

## 최종 모델
- Logistic Regression



## Decision Tree

최적 설정
- max_depth = 7
- min_samples_leaf = 20

Cross Validation 평균 F1 Score
- 56.68%


## Logistic Regression

최적 설정
- C = 10
- max_iter = 1000

Cross Validation 평균 F1 Score
- 59.89%

최종 모델로 선정


## Random Forest

최적 설정
- n_estimators = 100
- max_depth = 10
- min_samples_leaf = 1
- max_features = "sqrt"

Cross Validation 평균 F1 Score
- 57.81%



