
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

# 데이터 결측치 처리 가이드: 모델 성능을 높이는 첫걸음

데이터는 우리가 머신러닝·딥러닝 모델에 그대로 넣기엔 종종 불완전합니다. 결측치(빈칸, None, NaN)가 섞여 있으면 학습 과정에서 오류가 발생하거나 성능이 크게 저하됩니다. 이 글에서는 대표적인 결측치 처리 방법을 소개하고, 실제 코드 예시를 통해 직접 실습해보겠습니다.

***

## 1. 왜 결측치 처리가 중요한가요?  
결측치를 그대로 두면  
- 모델이 해당 샘플을 무시하거나  
- 학습이 중단되거나  
- 예측이 왜곡되는 등의 문제가 발생합니다.  

결측치 처리는 단순한 전처리가 아니라 모델 성능과 직결된 핵심 단계입니다.

***

## 2. 기본 통계값 대체: 평균(mean)·중앙값(median)  
### 2.1 장단점  
- 장점: 구현이 쉽고 빠릅니다.  
- 단점: 데이터 분포를 무시합니다. 이상치에 취약합니다.  

### 2.2 코드 예시  
```python
import pandas as pd

# 예시 데이터프레임
df = pd.DataFrame({
    'age': [25, 30, None, 35, 40],
    'salary': [50000, None, 55000, 60000, None]
})

# 평균 대체
df['age_mean'] = df['age'].fillna(df['age'].mean())
df['salary_mean'] = df['salary'].fillna(df['salary'].mean())

# 중앙값 대체
df['age_median'] = df['age'].fillna(df['age'].median())
df['salary_median'] = df['salary'].fillna(df['salary'].median())
```

***

## 3. K-최근접 이웃(KNN) 기반 대체  
KNN은 비슷한 샘플끼리 묶어 결측값을 채웁니다.  
- **장점**: 단순 통계 대체보다 정확할 수 있습니다.  
- **단점**: 계산량이 많고, 이상치에 민감합니다.

### 코드 예시
```python
import pandas as pd
from impyute.imputation.cs import fast_knn

df = pd.DataFrame({
    'X1': [1.0, 2.0, None, 4.0, 5.0],
    'X2': [2.1, None, 3.1, 4.1, 5.1]
})

# K=5로 imputation
np_imputed = fast_knn(df.values, k=5)
df_imputed = pd.DataFrame(np_imputed, columns=df.columns)
```

***

## 4. 다중대체법(MICE: Multiple Imputation by Chained Equations)  
MICE는 결측값을 여러 번 채워 m개의 완성 데이터셋을 만듭니다.  
1) Imputation: 통계 모델로 m개의 데이터 생성  
2) Analysis: 각 데이터셋 분석  
3) Pooling: 결과를 종합해 최종 결과 도출  

### 코드 예시
```python
import pandas as pd
from impyute.imputation.cs import mice

df = pd.DataFrame({
    'A': [1, None, 3, 4, 5],
    'B': [None, 2, 3, None, 5],
    'C': [1, 2, None, 4, None]
})

# MICE imputation
np_imputed = mice(df.values)
df_imputed = pd.DataFrame(np_imputed, columns=df.columns)
```

***

## 5. 딥러닝 기반 대체: Datawig  
Datawig는 딥러닝 모델을 이용해 결측치를 예측합니다.  
- 다양한 입력 피처를 활용해 정확도를 높입니다.

### 코드 예시
```python
import datawig
import pandas as pd

# 결측치가 있는 데이터 준비
df = pd.DataFrame({
    'X_1': [1, 2, 3, 4, 5],
    'X_2': ['a', 'b', 'c', None, 'e'],
    'X_3': [10, 20, None, 40, 50],
    'X_5': [100, None, 300, 400, 500]
})

imputer = datawig.SimpleImputer(
    input_columns=['X_1', 'X_2', 'X_3'],  
    output_column='X_5'                  
)

# 학습
imputer.fit(train_df=df, num_epochs=50)

# 결측치가 있는 행만 추출해 예측
df_null = df[df['X_5'].isnull()]
df_pred = imputer.predict(df_null)
```

***

## 6. 실제 워크플로우 예시  
1. 결측치 분포 확인  
2. 간단한 통계값 대체 시도  
3. 모델 성능 평가  
4. 성능 향상이 필요하면 KNN·MICE·Datawig 적용  
5. 재평가 및 최적 방법 선택  

짧은 실험 사이클을 통해 가장 적절한 방법을 찾는 것이 핵심입니다.

***

## 마무리  
결측치 처리 방법은 단순히 ‘채우기’가 아닙니다. 데이터 특성과 모델 목적에 맞춰 적절한 방법을 선택해야 합니다.  
- 작은 데이터·빠른 실험: 평균·중앙값 대체  
- 복잡한 분포·정밀도 중요: KNN·MICE  
- 풍부한 피처·딥러닝 활용: Datawig  

이제 여러분의 데이터에 맞는 결측치 처리 기법을 적용해보세요. 모델 성능이 한층 더 향상될 것입니다!

[1](https://velog.io/@ssulee0206/Data-Imputation%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B2%B0%EC%B8%A1%EC%B9%98-%EC%B2%98%EB%A6%AC)


## Mean Imputation
평균 대치법(Mean Imputation)은 결측치 처리 방법 중 하나로, 데이터셋에서 결측된 값을 해당 변수의 평균값으로 대체하는 기법입니다.

이 방법의 장점과 단점은 다음과 같습니다:

장점:
간단함: 구현이 쉽고 빠릅니다.

데이터 유지: 결측치를 제거하지 않고 데이터를 보존할 수 있습니다.

단점:
편향 발생: 데이터의 분산이 줄어들어 버릴 수 있으며, 분석 결과에 영향을 미칠 수 있습니다.  
정보 손실: 결측치 패턴이나 이유를 반영하지 못합니다.  
따라서, 평균 대치법은 데이터 특성과 분석 목표에 따라 신중하게 선택해야 합니다.  

## Mode Imputation
모드 보간법(Mode Imputation)은 데이터셋에서 결측값을 처리하기 위한 기법 중 하나입니다. 이 방법은 주어진 변수에서 가장 빈번하게 나타나는 값을 찾아 결측값 대신 사용하는 방식입니다.

모드 보간법의 특징  
단순성: 이 방법은 구현하기 쉽고, 계산이 빠릅니다.
특정 데이터 유형에 적합: 주로 범주형 데이터에 사용됩니다. 예를 들어, 설문 조사 결과에서 성별이나 선호도와 같은 항목에 적용할 수 있습니다.

사용 예시
주어진 데이터: A, B, B, C, (결측값), A
모드 계산: 가장 빈번한 값은 B입니다.
결측값 대체: 결측값을 B로 채웁니다.

모드 보간법은 간단하고 효과적이지만, 모든 상황에 적합하지 않을 수 있다는 점을 유의해야 합니다. 데이터의 특성을 고려하여 적절한 방법을 선택하는 것이 중요합니다.

## Next or Previous Value 
"Next" 및 "Previous" 값의 개념은 다양한 분야에서 사용되는 용어로, 일반적으로 데이터 시퀀스나 배열에서 현재 값의 위치를 기준으로 그 앞이나 뒤에 있는 값을 참조하는 것을 의미합니다.

다음 값 (Next Value): 현재 값 이후의 값을 지칭합니다. 예를 들어, 리스트에서 현재 값이 2일 때 다음 값은 3이 될 것입니다.

이전 값 (Previous Value): 현재 값 이전의 값을 지칭합니다. 예를 들어, 리스트에서 현재 값이 2일 때 이전 값은 1이 될 것입니다.

## Maximum or Minimum Value
최대 또는 최소 값 대체(Maximum or Minimum Value Imputation)는 결측치를 처리하는 방법 중 하나로, 데이터셋 내 결측치가 있는 위치에 해당 변수의 최대 또는 최소 값을 채우는 방식입니다.

이 방법의 장점은 간단하고 빠르게 결측치를 처리할 수 있다는 점입니다. 그러나 데이터의 분포를 왜곡할 수 있는 단점이 있으므로, 사용 시 주의가 필요합니다. 대체 방법을 선택할 때는 데이터의 특성과 분석 목적을 고려해야 합니다.

## Hot Deck Imputation
핫 덱 대체법(Hot Deck Imputation)은 결측치를 처리하는 통계적 기법 중 하나입니다. 이 방법은 결측치가 있는 데이터 포인트를 채우기 위해 유사한 관측치로부터 값을 가져오는 방식입니다.

주요 특징은 다음과 같습니다:

유사성 기반: 결측치를 가진 데이터와 유사한 다른 관측치에서 값을 선택합니다.
샘플링: 대체할 값을 무작위로 선택하여 데이터의 변동성을 유지합니다.
효율성: 간단하고 직관적인 방식으로 계산이 용이합니다.
이 방법은 데이터가 비교적 많고, 유사성이 명확한 경우에 효과적입니다.

## Cold Deck Imputation
콜드 덱 임퓨테이션(Cold Deck Imputation)은 결측값을 처리하기 위한 방법 중 하나로, 이전에 수집된 데이터셋을 활용하여 현재 데이터셋의 결측값을 채우는 기법입니다. 주로 다음과 같이 사용됩니다.

기존 데이터 활용: 이전 연구나 데이터셋에서 얻은 값들을 사용하여 결측치를 보완합니다.

기본 원칙: 같은 변수에 대한 결측값을 비슷한 특성을 지닌 데이터를 이용해 대체하여 더 정확한 분석을 할 수 있도록 합니다.

이 방법은 데이터의 유사성을 기반으로 하므로, 관련성이 높은 데이터셋을 선택하는 것이 중요합니다.

## Regression Imputation
회귀 대체법(Regression Imputation)은 결측값을 가진 변수를 다른 관련 변수들을 이용해 회귀모형으로 예측하여 결측값을 채우는 통계적 기법입니다.

회귀 대체법의 절차
결측값이 있는 변수를 종속 변수로 설정합니다.
결측값을 예측하는 데 사용할 독립 변수들을 선택합니다. 이 변수들은 종속 변수와 강한 상관관계를 갖고, 결측값이 없어야 합니다.
적절한 회귀모델(예: 선형 회귀, 로지스틱 회귀)을 데이터에 적합하여 결측값을 예측합니다.
장점
변수 간의 관계를 반영하여 보다 정확한 결측값 대체가 가능합니다.
원 변수의 분포를 유지하여 데이터의 패턴과 변동성을 보존합니다.
다중 예측 변수를 포함함으로써 편향을 줄이고 교란변수 효과를 보정할 수 있습니다.
추가 고급 기법
복수의 회귀 모델을 결합하는 앙상블 학습을 활용하여 단일 모델의 한계를 극복할 수 있습니다.
비선형 관계를 반영하기 위해 다항 회귀, 스플라인, 일반화 가법 모델(GAM) 등을 사용해 복잡한 패턴을 포착할 수 있습니다.
회귀 대체법은 단순 평균 대체법보다 데이터 특성을 더 잘 반영하여 결측치를 보완하는 효과적인 방법으로 널리 활용됩니다.

### Generalized Additive Model (GAM)
Generalized Additive Model (GAM) 회귀는 통계 및 머신러닝에서 사용되는 모델로, 종속 변수와 독립 변수 간의 관계를 선형 함수가 아닌 부드러운 (smooth) 함수들의 합으로 표현하는 기법입니다. 즉, 전통적인 선형 회귀모델이 변수들과 종속 변수 사이의 선형 관계를 가정하는 반면, GAM은 각 독립 변수에 대해 비선형적인 부드러운 함수 ( f_i(x_i) )를 추정하여 더 유연하게 데이터를 모델링합니다.

주요 특징은 다음과 같습니다:

GAM은 링크 함수 ( g )를 통해 종속 변수의 기대값과 독립 변수들의 부드러운 함수 합과 연결합니다. 예를 들어,

$$g(O_{E}(Y)) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_m(x_m)$$

와 같이 표현됩니다.

각 ( f_i ) 함수는 스플라인(splines), 국소 가중 회귀(loess) 등 비모수적(Non-parametric) 방법으로 추정하며, 이는 변수와 결과 변수 사이의 복잡한 비선형 관계를 포착할 수 있게 합니다.

모델 추정 시 여러 스무더(smoother)를 동시에 추정하며, 페널티를 가한 반복 가중 최소제곱법(Penalized Iteratively Reweighted Least Squares, PIRLS)이나 로컬 스코어링 알고리즘을 활용합니다. 로컬 스코어링은 다양한 스무더를 사용할 수 있는 유연성을 갖지만, 계산 비용이 더 큽니다.

GAM은 기존의 선형 회귀나 GLM(Generalized Linear Models)의 확장형이며, 데이터에 내재된 비선형성이나 복잡한 패턴을 부드러운 함수로 표현하여 더 좋은 적합도를 제공합니다.

따라서 GAM 회귀는 변수와 종속 변수 사이의 관계가 복잡하거나 비선형적일 때, 이를 효과적으로 모델링하고 해석할 수 있는 강력한 기법입니다.

요약하면, GAM 회귀는 선형 회귀를 확장하여 각 독립변수에 대해 부드러운 함수 형태의 효과를 추정하는 통계모델로, 비선형성과 유연성을 갖는 예측 모형을 만드는 데 적합합니다.

## K-nearest Neighbor (KNN) Imputation
K-nearest Neighbor (KNN) Imputation은 데이터에서 결측값을 처리하는 방법 중 하나로, 결측치가 있는 데이터 포인트 주변의 가장 가까운 K개의 이웃 데이터 값을 이용해 결측치를 채우는 기법입니다.

KNN Imputation 원리
KNN은 특정 데이터의 결측치를 채우기 위해 해당 데이터를 기준으로 가장 가까운 K개의 이웃 데이터를 찾습니다.
이 이웃들의 값을 참고하여 결측값을 대체합니다.
범주형 데이터일 경우, 이웃들 중 최빈값을 사용해 결측치를 채웁니다.
연속형 데이터일 경우, 이웃 값들의 중앙값(median) 또는 평균값(mean)을 사용해 결측치를 대체합니다.
KNN Imputation의 특징
단순 평균이나 중앙값 대체보다 데이터의 분포와 이웃 관계를 반영하여 더 정교하고 정확한 결측치 대체가 가능합니다.
비모수적 방법으로, 데이터의 분포 가정이 필요 없습니다.
주로 머신러닝 모델 전처리 단계에서 결측치 문제를 다룰 때 유용하게 쓰입니다.
대표적인 활용 예시
R의 DMwR 패키지의 knnImputation() 함수로 구현 가능하며, K 값을 조절해 이웃 수를 설정할 수 있습니다.
파이썬 scikit-learn의 KNNImputer를 통해 손쉽게 KNN 임퓨테이션 수행이 가능합니다. 이 경우 결측치는 주변 이웃 값들의 근사값으로 대체됩니다.
요약하면, KNN Imputation은 결측치를 채우기 위해 가장 가까운 K개의 이웃 데이터를 활용하여 범주형은 최빈값, 연속형은 중앙값 또는 평균값으로 대체하는 방법으로, 데이터의 특성을 반영해 결측치 처리의 정확도를 높입니다.

## Multiple Imputation
Multiple Imputation(다중 대치법)은 결측치를 처리하는 통계 방법으로, 하나의 값으로 결측치를 채우는 단일 대치법과 달리 여러 개의 대체 데이터셋을 생성하여 각각 통계 분석을 수행한 뒤 결과를 통합하는 기법입니다.

주요 특징과 절차:

여러 데이터셋(m개)을 만들어 각 데이터셋의 결측치를 대치함
각 데이터셋에 대해 표준 통계 분석 실행
분석 결과를 종합(pool)하여 최종 추정값 산출
불확실성을 더 잘 반영하여 편향과 과대추정을 줄임
R에서 대표적인 패키지인 mice는 "Multivariate Imputation by Chained Equations"의 약자로, 변수 간 상관관계를 고려해 연쇄적으로 결측치를 대치하는 방식입니다. mice는 연속형, 이분형, 범주형 변수 모두에 적용 가능하며, 다양한 대치 방법(pmm, rf, cart 등)을 지원합니다.

IBM SPSS의 Multiple Imputation 프로시저 역시 반복적 MCMC 기법 등을 활용해 다양한 결측 패턴에 적합한 대치값을 생성, 여러 데이터셋으로부터 최종 추정치를 도출하는 방식을 제공합니다.

요약하면, Multiple Imputation은 결측치가 있는 데이터를 여러 대체본으로 확장하여 각 대체본에서 분석 후 결과를 통합함으로써 보다 신뢰도 높고 편향 없는 통계 추정을 가능하게 하는 방법입니다. 이러한 점에서 단일 값 대치법보다 우수한 결측치 처리법으로 널리 사용됩니다.

# Reference
- https://dataaspirant.com/data-imputation-techniques/
- https://velog.io/@ssulee0206/Data-Imputation%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B2%B0%EC%B8%A1%EC%B9%98-%EC%B2%98%EB%A6%AC
- https://daebaq27.tistory.com/43
