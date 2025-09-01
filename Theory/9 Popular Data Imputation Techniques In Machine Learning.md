
# Data Imputation

딥러닝을 공부하는 대학생을 위해, **결측치 처리(Data Imputation)** 방법을 쉽고 전문적으로 설명합니다. 이 글을 읽으면 결측치의 종류와 9가지 대표 기법을 이해하고, 실제 파이썬 예제까지 따라 해볼 수 있습니다.

## 1. 결측치란 무엇인가요?

데이터셋에 누락된 값이 있으면 모델이 정상 작동하지 않습니다.
결측치는 크게 세 가지 유형으로 나뉩니다.

- **MCAR (Missing Completely at Random):**
완전 무작위로 사라진 결측치입니다.
- **MAR (Missing at Random):**
다른 변수 값에 따라 결측 확률이 달라집니다.
- **MNAR (Missing Not at Random):**
특정 값 자체의 특성 때문에 결측됩니다.

결측치 유형을 파악해야 적절한 임퓨테이션 기법을 선택할 수 있습니다.

## 2. 데이터 임퓨테이션의 중요성

1. 모델 성능 보장: 누락된 값이 많으면 편향이 생깁니다.
2. 알고리즘 호환성: 대부분의 라이브러리는 결측치를 처리하지 못합니다.
3. 데이터 손실 최소화: 일부 데이터를 삭제하면 학습 효율이 떨어집니다.

## 3. 9가지 대표 임퓨테이션 기법

| 기법 이름 | 설명 |
| :-- | :-- |
| **평균 대체(Mean Imputation)** | 해당 열의 평균값으로 결측치를 채웁니다. |
| **최빈값 대체(Mode Imputation)** | 범주형 데이터에 자주 등장하는 값으로 채웁니다. |
| **이전/다음 값 채우기** | 시계열에서 앞 또는 뒤 값을 그대로 가져와 채웁니다. |
| **최댓값/최솟값 대체** | 극단치로 결측치를 대체합니다. |
| **Hot Deck Imputation** | 같은 그룹에서 무작위로 값을 뽑아 채웁니다. |
| **Cold Deck Imputation** | 외부 데이터(역사 기록 등)에서 값을 가져와 채웁니다. |
| **회귀 대체(Regression)** | 다른 변수로 회귀 모델을 학습하여 예측된 값으로 대체합니다. |
| **KNN 임퓨테이션** | k개의 유사한 샘플 평균으로 결측치를 채웁니다. |
| **다중 대체(Multiple Imputation)** | 여러 개의 임퓨테이션 결과를 합산해 불확실성을 반영합니다. |

## 4. 기법 선택 가이드

1. **데이터 특성 이해:** 연속형 vs 범주형, 분포 형태 확인
2. **결측 패턴 분석:** MCAR인지 MAR인지 MNAR인지 파악
3. **분석 목표 고려:** 변수 간 관계 보존이 중요한지 확인
4. **영향 평가:** 다양한 기법으로 결과를 비교·검증

## 5. 파이썬으로 임퓨테이션 구현하기

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 예시 데이터 준비
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
rng = np.random.RandomState(42)
mask = rng.rand(*X.shape) < 0.1
X_missing = X.copy()
X_missing[mask] = np.nan

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X_missing, y, random_state=42)

# 1) 평균 임퓨테이션
mean_imp = SimpleImputer(strategy='mean')
X_train_mean = mean_imp.fit_transform(X_train)
X_test_mean = mean_imp.transform(X_test)

# 2) 중앙값 임퓨테이션
median_imp = SimpleImputer(strategy='median')
X_train_med = median_imp.fit_transform(X_train)
X_test_med = median_imp.transform(X_test)

# 3) KNN 임퓨테이션
knn_imp = KNNImputer(n_neighbors=5)
X_train_knn = knn_imp.fit_transform(X_train)
X_test_knn = knn_imp.transform(X_test)

# 모델 학습 및 평가 함수
def evaluate(X_tr, X_te):
    model = LinearRegression().fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    return mean_squared_error(y_test, y_pred)

print("MSE Mean:", evaluate(X_train_mean, X_test_mean))
print("MSE Median:", evaluate(X_train_med, X_test_med))
print("MSE KNN:", evaluate(X_train_knn, X_test_knn))
```

짧은 코드로 다양한 임퓨테이션 기법을 적용하고, **MSE**로 성능을 비교할 수 있습니다.

## 6. 임퓨테이션 작업 시 유의사항

- 결측치 비율이 높으면 과대 대체(over-imputation) 주의
- 이상치 처리 후 임퓨테이션을 진행해야 왜곡을 줄일 수 있습니다.
- *민감도 분석(sensitivity analysis)* 을 통해 기법별 결과 변화를 확인하세요.
- 처리 과정을 문서화하여 재현성을 확보해야 합니다.


## 7. 결론

데이터 임퓨테이션은 모델 성능과 신뢰도를 좌우합니다.
결측치 유형에 맞는 기법을 선택하고, 코드로 직접 실습해보세요.
이 가이드를 따라 하면 결측치 문제를 확실히 해결할 수 있습니다.

---
위 내용을 바탕으로, 여러분의 딥러닝 프로젝트에서 결측치를 당당하게 다루시길 바랍니다!
<span style="display:none">[^1]</span>

<div style="text-align: center">⁂</div>

[^1]: https://dataaspirant.com/data-imputation-techniques/

