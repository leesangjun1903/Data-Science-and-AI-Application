# 데이터 결측치 처리: 빈칸을 채우는 것보다 중요한 것

> **대상:** 머신러닝·딥러닝을 처음 배우는 대학생  
> **자료 확인 기준일:** 2026년 10월 16일  
> **개정 범위:** 원문의 결측 원인, 9가지 대체 기법, GAM, 파이썬 실습을 유지하면서 중복을 통합했습니다. 이론적 오류를 바로잡고, 최신 연구와 평가 방법을 보완했습니다.  
> **근거 구분:** 이론·구현 설명에는 논문 및 공식 문서 각주를 붙였습니다. 5절의 연구 결과는 해당 논문의 실험 범위에 한정합니다. 학생·소득 예시와 실습 데이터 생성 과정은 이해를 돕기 위한 교육용 예시입니다.

**결측치 대체는 사라진 정답을 알아내는 마법이 아닙니다. 남아 있는 정보와 가정을 이용해 그럴듯한 값을 추정하는 일입니다.**

예를 들어 학생의 공부 시간과 시험 점수를 기록한 표에서 어떤 학생의 공부 시간이 비어 있다고 해봅시다. 전체 평균을 넣을 수도 있고, 비슷한 학생을 참고할 수도 있습니다. 하지만 어느 쪽이든 실제 공부 시간을 확인한 것은 아닙니다. 이 차이를 기억해야 분석 결과를 지나치게 확신하지 않습니다.

읽는 순서는 **결측 원인 → 분석 목적 → 처리 기법 → 평가 → 최신 연구 → 실습**입니다. 수식은 직관을 확인하는 보충 설명이므로 처음에는 말로 된 설명부터 읽어도 됩니다.

---

## 1. 결측치란 무엇인가요?

### 1.1 결측치와 결측치 대체는 다릅니다

**결측치(missing value)**는 기록되어 있어야 할 값이 관측되거나 저장되지 않은 상태입니다. 파이썬에서는 `NaN`, `None`, `pd.NA` 등으로 표현할 수 있지만, 어떤 표시를 결측으로 인식하는지는 자료형과 도구에 따라 다릅니다. 공백이나 `-999` 같은 코드를 사용했다면 데이터 설명서를 확인하여 명시적으로 처리해야 합니다.[^sk-impute][^pd-missing]

**결측치 대체(imputation)**는 그 빈칸에 추정값을 넣는 방법입니다. 더 넓은 개념인 **결측치 처리**에는 행·열 삭제, 대체, 결측치를 직접 다루는 모델 사용 등이 포함됩니다.[^sk-impute]

예를 들어 “한 달 독서량 0권”은 관측된 값이고, “응답하지 않음”은 결측입니다. 둘을 같은 `0`으로 저장하면 정보를 잃습니다. “해당 없음”처럼 애초에 적용되지 않는 질문도 단순 미응답과 구분해야 합니다.

또한 **결측치가 있으면 모든 모델이 작동하지 않는 것은 아닙니다.** 가령 scikit-learn의 `HistGradientBoostingRegressor`는 `NaN`을 직접 처리합니다. 나무를 나눌 때 결측인 샘플을 어느 쪽으로 보낼지도 학습합니다. 다만 “모델이 실행된다”와 “편향 없이 올바른 분석을 한다”는 별개의 문제입니다.[^histgb]

### 1.2 MCAR·MAR·MNAR: 빈칸의 모양이 아니라 생긴 이유

세 용어는 결측값의 크기나 자료형이 아니라 **값이 누락되는 과정에 대한 가정**을 뜻합니다.[^mechanism]

| 구분 | 쉬운 설명 | 교육용 예시 |
|---|---|---|
| **MCAR: 완전 무작위 결측** | 누락 여부가 관측된 값이나 누락된 값과 무관합니다. | 모든 학생의 설문 항목이 값과 관계없이 같은 확률로 저장 오류를 겪습니다. |
| **MAR: 관측 정보에 조건부인 무작위 결측** | 관측된 정보를 충분히 고려하면, 누락 여부가 여전히 보이지 않는 값에 추가로 의존하지 않습니다. | 학년별로 소득 질문 응답률이 다릅니다. 학년이 기록되어 있고, 같은 학년 안에서는 실제 소득이 응답 여부에 더 영향을 주지 않는다고 가정합니다. |
| **MNAR: 비무작위 결측** | 관측된 정보를 고려해도, 누락 여부가 보이지 않는 값에 의존합니다. | 학년 등 기록된 정보를 고려한 뒤에도 소득이 높은 학생일수록 소득 질문에 덜 응답합니다. |

여기서 MAR의 “무작위”는 “아무 이유 없이 빠졌다”는 뜻이 아닙니다. **관측된 조건 안에서 비교하면 누락 여부가 미관측 값에 더 의존하지 않는다**는 뜻입니다. MCAR는 MAR의 더 제한적인 특별한 경우입니다.[^mechanism]

#### 수식으로 한 번만 확인하기

한 변수 $W$에 결측이 있고, 보조 정보 $Z$는 모두 관측되었다고 합시다. $M=1$은 $W$가 누락되었다는 표시입니다. 이 단순한 상황에서는 다음처럼 표현할 수 있습니다.[^formal]

$$
\begin{aligned}
\text{MCAR:}\quad &P(M=1\mid W,Z)=P(M=1),\\
\text{MAR:}\quad &P(M=1\mid W,Z)=P(M=1\mid Z).
\end{aligned}
$$

MNAR에서는 $Z$를 고려해도 $W$에 대한 의존성이 남습니다. 여러 변수가 동시에 빠지는 일반적인 상황에서는 관측·결측 패턴을 함께 고려해 정의해야 합니다.[^formal]

### 1.3 데이터만 보고 결측 원인을 확정할 수 있나요?

**일반적으로 관측된 데이터만으로 MAR와 MNAR를 구별할 수는 없습니다.** 구별에 필요한 값 자체가 보이지 않기 때문입니다. MCAR를 점검하는 검정도 있지만, 기각하지 못했다는 사실이 MCAR의 증명은 아닙니다.[^formal]

따라서 결측률 표나 시각화는 출발점입니다. 설문 설계, 센서 작동 방식, 기록 절차처럼 **왜 값이 빠졌는지 설명하는 배경 정보**를 함께 살펴봐야 합니다. MNAR가 의심되면 가능한 가정을 바꿔 보며 결론의 민감도를 확인합니다.[^mnar]

---

## 2. 왜 처리하나요? 먼저 목적을 구분하세요

### 2.1 예측, 통계적 추론, 값 복원은 서로 다른 목표입니다

| 목표 | 알고 싶은 것 | 중요한 평가 기준 |
|---|---|---|
| **예측** | “새 학생의 시험 점수를 잘 맞힐 수 있는가?” | 사용하지 않은 데이터에서의 예측 오차 |
| **통계적 추론** | “공부 시간과 점수의 관계는 어느 정도이며, 추정이 얼마나 불확실한가?” | 추정량의 편향, 표준오차, 신뢰구간의 타당성 |
| **결측값 복원** | “비어 있던 공부 시간을 실제 값에 가깝게 채웠는가?” | 숨겨 둔 정답에 대한 복원 오차와 분포의 적절성 |

**빈칸을 더 잘 맞히는 방법이 최종 예측도 항상 더 잘하는 것은 아닙니다.** 반대로 예측이 잘된다고 해서 회귀계수나 신뢰구간까지 올바른 것은 아닙니다. 대체 품질과 최종 예측 성능을 구분한 벤치마크, 그리고 다중 대체의 추론 이론을 함께 봐야 하는 이유입니다.[^benchmark][^pool]

### 2.2 빈칸이 있는 행을 그냥 버리면 안 되나요?

항상 잘못은 아닙니다. MCAR에서는 완전한 행만 분석해도 모집단 평균 같은 추정이 편향되지 않을 수 있지만, 사용할 수 있는 표본과 정보가 줄어듭니다. 반면 특정 유형의 사람만 남는다면 편향이 생길 수 있습니다. 삭제의 타당성은 결측 과정과 추정하려는 대상에 달려 있습니다.[^simple-theory]

정보 손실은 생각보다 클 수 있습니다. 각 칸이 **서로 독립적으로** 10% 확률로 빠지고 열이 20개라면, 완전한 행이 남을 확률은 다음과 같습니다.

$$
(1-0.1)^{20}=0.9^{20}\approx 0.122.
$$

이 가정 아래에서는 칸의 10%만 빠져도 완전한 행은 약 12.2%뿐입니다. 실제 결측은 서로 연관될 수 있으므로 이 수치를 일반적인 법칙으로 쓰지는 마세요.

**“결측률이 몇 % 이하면 안전하다”라는 보편적 기준은 없습니다.** 중요한 것은 비율뿐 아니라 어떤 정보가, 어떤 이유로, 누구에게서 빠졌는지입니다.[^simple-theory]

---

## 3. 원문의 9가지 대표 대체 기법, 정확하게 이해하기

아래 목록은 원문의 학습 순서를 유지한 것입니다. 모든 항목이 동일한 종류의 알고리즘은 아닙니다. 특히 **다중 대체는 여러 대체본을 만들고 분석 결과를 통합하는 틀**이며, 그 안에 회귀나 기증자 기반 방법 등을 사용할 수 있습니다.[^mice-api]

| 방법 | 핵심 생각 | 먼저 기억할 주의점 |
|---|---|---|
| 평균·중앙값 | 같은 열의 대표값을 넣습니다. | 값의 퍼짐과 변수 사이 관계를 왜곡할 수 있습니다. |
| 최빈값 | 가장 자주 나온 범주를 넣습니다. | 흔한 범주가 더 많아집니다. |
| 이전·다음 값 | 시간상 가까운 관측값을 이어 붙입니다. | 실제 예측 시점 이후의 정보를 쓰면 안 됩니다. |
| 최댓값·최솟값 | 특정 극단값을 넣습니다. | 일반적인 첫 선택이 아닙니다. |
| Hot deck | 현재 자료의 적절한 기증자 값을 가져옵니다. | 어떤 관측치를 기증자로 고르는지가 중요합니다. |
| Cold deck | 과거·외부 자료를 참고합니다. | 단위·시점·대상 집단이 맞아야 합니다. |
| 회귀·반복 대체 | 다른 열로 비어 있는 열을 예측합니다. | 예측값만 넣으면 변동성을 충분히 반영하지 못합니다. |
| KNN | 비슷한 관측치의 값을 참고합니다. | 거리의 의미와 변수 스케일이 중요합니다. |
| 다중 대체 | 가능한 대체본을 여러 개 만들어 함께 분석합니다. | 데이터 자체가 아니라 추정값과 불확실성을 통합합니다. |

각 방법의 근거와 한계는 바로 아래에서 설명합니다.

### 3.1 평균 대체와 중앙값 대체

**평균 대체**는 관측된 값의 평균을, **중앙값 대체**는 크기순으로 정렬했을 때 가운데 값을 넣습니다. scikit-learn에서는 `SimpleImputer(strategy="mean")` 또는 `strategy="median"`을 사용합니다.[^simple-api]

직접 계산해 봅시다. 관측값이 `10, 20, 30, 100`이면 평균은 40, 중앙값은 25입니다. 마지막 값이 100에서 1,000으로 바뀌면 평균은 265가 되지만 중앙값은 25로 같습니다. 이 예시는 중앙값이 일부 극단값에 평균보다 덜 민감한 이유를 보여 줍니다.

그러나 중앙값이 이상치에 강하다고 해서 분포를 보존하는 것은 아닙니다. 같은 수를 반복해서 넣으면 원래 자료에 없던 값의 집중이 생깁니다. 평균 대체 역시 관측 평균은 유지하지만, 분산과 변수 간 관계를 왜곡할 수 있습니다.[^simple-theory]

예를 들어 관측값 `10, 20, 30, 40`의 평균은 25입니다. 빈칸 두 개에 25를 넣어도 평균은 25지만, 각 값의 제곱편차를 자료 수로 나눈 분산은 125에서 약 83.3으로 줄어듭니다. **모르는 값을 평균이라고 확정한 결과, 실제보다 덜 흩어진 자료처럼 보이는 것입니다.**

예측 프로젝트에서는 이처럼 단순한 방법도 좋은 비교 출발점이 될 수 있습니다. 복잡한 대체법을 추가할지는 같은 검증 조건에서 판단합니다.[^benchmark]

### 3.2 최빈값 대체: 가장 흔한 범주 넣기

최빈값은 가장 자주 나타나는 값입니다. 전공처럼 순서나 간격이 없는 범주형 변수에는 평균보다 자연스러운 대체값입니다. `SimpleImputer(strategy="most_frequent")`가 이 전략을 제공합니다.[^simple-api]

예를 들어 `A, B, B, C, 결측, B`라면 최빈값은 B입니다. 반면 원문에 등장한 `A, B, B, C, 결측, A`에서는 A와 B가 각각 두 번이므로 **공동 최빈값**입니다. B만 정답이라고 할 수 없고 동률 처리 규칙이 필요합니다.

최빈값을 넣으면 흔한 범주 비율이 더 커집니다. 예측용으로는 `"미상"`이라는 별도 범주를 두는 전략도 비교할 수 있습니다. 이것은 실제 범주를 알아낸 것이 아니라, 범주를 모른다는 사실을 모델에 알려주는 방식입니다.[^simple-api]

### 3.3 이전·다음 값 채우기: 시간 순서가 있을 때만

**전방 채우기(`ffill`)**는 마지막으로 관측한 값을 뒤쪽 빈칸에 복사합니다. **후방 채우기(`bfill`)**는 다음에 관측한 값을 앞쪽 빈칸에 복사합니다.[^pd-fill]

```text
원래 값:       20, 결측, 결측, 23
전방 채우기:   20, 20,   20,   23
후방 채우기:   20, 23,   23,   23
```

이 방법은 “짧은 시간 동안 값이 크게 달라지지 않는다”는 생각에 기대고 있습니다. 오랫동안 끊긴 센서값을 같은 값으로 계속 채우면 변화를 놓칩니다. 시간 사이를 직선 등으로 연결하는 **보간(interpolation)**도 가능하지만, 이것 역시 데이터의 변화 방식에 대한 가정입니다.[^pd-missing]

**예측 시점이 오전 10시라면 오전 11시에야 얻을 정보를 사용할 수 없습니다.** 따라서 후방 채우기나 미래 관측값을 이용한 보간은 그 상황에서 정보 누출입니다. 이미 끝난 기록을 사후 복원하며 뒤쪽 관측값 사용이 허용된 문제와 구분해야 합니다.[^cv]

여러 사람이나 센서가 섞였다면 개체별로 나누고 시간순으로 정렬한 뒤 처리해야 합니다. 서로 다른 사람의 마지막 값을 가져와서는 안 됩니다. 6.3절에서 짧은 예제를 확인합니다.

### 3.4 최댓값·최솟값 대체: 보통의 기본값으로 쓰지 않기

나이를 모르는 사람에게 관측된 최대 나이를 넣는다고 생각해 보세요. 그 사람이 가장 나이가 많다는 근거는 없습니다. 극단값 대체가 간단하다는 사실은 그 값이 타당하다는 근거가 되지 않습니다.

특별한 목적이 없다면 평균·중앙값, 관측치 기반 방법, 모델 기반 방법보다 먼저 선택할 이유가 약합니다. 단순 대체값의 타당성은 데이터가 생성되고 누락된 과정에 비추어 판단해야 합니다.[^simple-theory]

예측 모델에서 `-999` 같은 값을 “미상 표시”로 쓰는 것과 실제 값을 추정하는 것은 다릅니다. 특히 숫자 사이 거리나 크기를 이용하는 모델에서는 이 표시가 진짜 극단값으로 해석될 수 있습니다. 대체값과 함께 결측 여부 표시를 사용하는 방법을 별도로 검증하는 편이 명확합니다.[^simple-api]

또한 센서의 검출 한계보다 낮아 측정되지 않았다는 것은 “완전히 아무 정보도 없다”와 다릅니다. 값이 특정 범위에 있다는 정보를 모델에 반영할 수 있으며, 무조건 최솟값으로 채워야 한다는 뜻은 아닙니다.[^mnar]

### 3.5 Hot deck: 비슷한 기증자의 실제 값 가져오기

**Hot deck**은 현재 조사·데이터 안에서 적절한 관측치인 **기증자(donor)**를 찾아 그 값을 가져오는 방법입니다. 예를 들어 같은 학년과 비슷한 근로 시간을 가진 학생의 소득을 참고할 수 있습니다. 같은 그룹 안에서 무작위로 고르는 방식도 있고, 가까운 기증자를 정하는 방식도 있습니다. 따라서 “반드시 무작위 추출”이라고 정의하면 너무 좁습니다.[^donor]

실제로 관측된 값을 사용하므로 존재하지 않는 범주나 불가능한 수치를 만들 가능성을 줄일 수 있습니다. 그렇다고 모든 변수의 관계나 불확실성이 자동으로 보존되는 것은 아닙니다. 여러 열을 각기 다른 기증자로 채우면 서로 어울리지 않는 조합이 생길 수 있습니다. 기증자 선정과 반복 사용을 점검해야 합니다.[^donor]

### 3.6 Cold deck: 과거·외부 자료 참고하기

**Cold deck**은 이전 조사나 별도 자료처럼 현재 데이터 바깥의 정보를 기증자로 이용합니다. Hot deck과의 핵심 차이는 **참고 자료의 출처**입니다.[^donor]

예를 들어 지난 학기의 기록으로 이번 학기 값을 채우기 전에, 해당 변수가 그 사이 바뀔 수 있는지 생각해야 합니다. 비교 대상의 단위, 정의, 시점, 집단이 맞지 않으면 “관측된 값”이어도 현재 빈칸의 적절한 대체값은 아닐 수 있습니다.

예측 시스템에서는 외부 자료도 **실제 예측 시점에 얻을 수 있는 정보**여야 합니다. 나중에 공개된 통계를 과거 시점 예측에 사용하는 것도 정보 누출이 될 수 있습니다.[^cv]

### 3.7 회귀 대체와 반복 대체

#### 다른 변수로 빈칸을 예측합니다

예를 들어 공부 시간으로 시험 점수를 예측하는 회귀모형을 학습한 뒤, 점수가 빠진 학생의 값을 채울 수 있습니다. 수치형 변수에는 회귀모형을, 범주형 변수에는 범주에 맞는 예측모형을 고려합니다. 여러 열에 결측이 있으면 초기 대체 후 각 열을 번갈아 예측하는 **반복 대체**를 사용할 수 있습니다.[^iterative-api][^mice-api]

따라서 “회귀 대체에 쓰는 모든 설명변수는 원래부터 완전해야 한다”는 제한은 일반적인 정의가 아닙니다. 다만 모델에 값을 입력할 수 있도록 다른 결측을 함께 다루는 절차가 필요합니다.[^iterative-api]

#### 예측값만 넣으면 왜 문제가 되나요?

공부 시간이 같아도 시험 점수는 사람마다 다릅니다. 그런데 회귀모형의 예측값만 넣으면 이런 개인차가 사라집니다. 결정론적 회귀 대체가 원래 분포와 변동성을 자동으로 보존한다는 설명은 잘못입니다.[^simple-theory]

이를 다음과 같이 생각할 수 있습니다.

$$
\text{실제 값}=\text{모델이 설명하는 부분}+\text{모델이 설명하지 못하는 차이}.
$$

예측값만 넣는 것은 뒤의 차이를 모두 0으로 두는 것과 비슷합니다. **확률적 회귀 대체**는 적절한 잔차 분포에서 값을 뽑아 이 차이를 반영하려고 합니다. 통계적 추론에서는 모형을 추정하는 불확실성도 고려해야 하므로, 임의의 잡음을 한 번 넣는 것만으로 충분하지 않습니다.[^stochastic]

#### GAM은 대체법인가요, 예측모형인가요?

**일반화 가법 모형(GAM, Generalized Additive Model)**은 대체에 사용할 수 있는 예측모형의 한 종류입니다. 각 변수의 효과를 직선으로만 표현하지 않고, 부드러운 곡선으로 표현한 뒤 더합니다.[^gam]

기본적인 형태는 다음과 같습니다.

$$
g\!\left(\mathbb{E}[Y\mid X_1,\ldots,X_p]\right)
=\beta_0+f_1(X_1)+\cdots+f_p(X_p).
$$

여기서 $\mathbb{E}[Y\mid X]$는 주어진 입력에서의 평균적인 결과, $g$는 결과의 종류에 맞게 평균을 변환하는 **링크 함수**, $f_j$는 각 변수의 효과를 나타내는 함수입니다. 연속형 결과에 항등 링크를 쓰면 평균을 그대로, 이진 결과에 로짓 링크를 쓰면 확률의 로그 오즈를 표현합니다.[^gam]

“공부 시간을 늘리면 처음에는 점수가 빠르게 오르지만 나중에는 증가 폭이 작아진다” 같은 관계를 표현할 수 있습니다. 기본 가법식에는 변수 간 상호작용이 자동으로 모두 들어 있지 않으며, 필요하면 별도로 포함합니다. GAM도 부적절하게 설정하면 잘못 예측할 수 있고, 평균 예측값만 채우면 앞서 설명한 변동성 문제는 그대로 남습니다.[^gam]

#### 나무 기반 반복 대체도 있습니다

`missForest`는 각 변수를 랜덤 포레스트로 예측하면서 반복적으로 빈칸을 채우는 대표적인 방법입니다. 수치형·범주형이 섞인 자료의 비선형 관계를 다룰 수 있습니다. 다만 나무를 여러 개 사용하거나 대체를 여러 번 반복한다는 사실만으로 3.9절의 다중 대체 분석이 완성되는 것은 아닙니다.[^missforest][^pool]

### 3.8 KNN 대체: 가까운 이웃 참고하기

KNN 대체는 관측된 특징이 비슷한 샘플을 찾아 빈칸을 추정합니다. scikit-learn의 `KNNImputer`는 **대체하려는 열의 값이 관측된 이웃들**을 사용하며, 해당 값의 단순 평균 또는 거리 가중 평균으로 채웁니다. 한 행에서도 어느 열을 채우는지에 따라 참고하는 이웃이 달라질 수 있습니다.[^knn]

예를 들어 비슷한 학생 세 명의 공부 시간이 2시간, 3시간, 4시간이면 단순 평균은 3시간입니다. 하지만 “비슷하다”는 기준에 학년, 과목, 기존 성적 중 무엇을 포함하느냐에 따라 이웃이 달라집니다.

**변수의 스케일을 점검하세요.** 한 열이 나이이고 다른 열이 원 단위 소득이라면, 스케일 조정 없이 거리를 계산할 때 숫자가 큰 소득 차이가 지배할 수 있습니다. 표준화는 이 문제를 줄이는 선택지이며, 학습 자료에서만 통계를 구해야 합니다.[^scaler]

공통으로 관측된 특징이 너무 적으면 거리를 신뢰하기 어렵고, 데이터가 크면 이웃 탐색 비용도 커집니다. 정규분포를 직접 가정하지 않는다는 것이 “아무 가정도 없다”는 뜻은 아닙니다. 가까운 샘플끼리는 결측값도 비슷하리라는 생각에 의존합니다.

**구현상의 중요한 구분:** 범주형에 최빈값을 쓰는 이웃 기반 방법을 별도로 만들 수는 있지만, scikit-learn의 `KNNImputer`가 범주형 최빈값 대체나 중앙값 대체를 자동으로 제공하는 것은 아닙니다. 전공을 0·1·2로 코딩한 후 평균 1.4로 채우는 것은 올바른 범주 복원이 아닙니다.[^knn]

### 3.9 다중 대체와 MICE: 여러 가능한 답을 함께 분석하기

#### 다중 대체의 핵심은 불확실성입니다

빈칸에 3시간만 넣으면 그 값이 확실한 것처럼 보입니다. 반면 적절한 확률모형에서 여러 가능한 값을 뽑으면, 결측 때문에 분석 결과가 얼마나 달라질 수 있는지 살펴볼 수 있습니다.[^mice]

다중 대체의 흐름은 다음과 같습니다.[^pool]

```text
불완전한 데이터
   ├─ 대체본 1 → 분석 → 추정값 Q₁, 그 추정값의 분산 U₁
   ├─ 대체본 2 → 분석 → 추정값 Q₂, 그 추정값의 분산 U₂
   └─ 대체본 m → 분석 → 추정값 Qₘ, 그 추정값의 분산 Uₘ
                         ↓
              추정값과 불확실성을 함께 통합
```

**대체본들을 행 방향으로 이어 붙여 표본 수를 늘리거나, 빈칸별 평균을 낸 한 데이터셋만 분석하는 것과 다릅니다.** 일반적인 통계적 다중 대체에서는 각 대체본을 따로 분석한 뒤 그 결과를 통합합니다.[^pool]

#### MICE와 반복 대체는 같은 말인가요?

**MICE**는 연쇄방정식을 이용한 다중 대체를 가리킵니다. R 패키지 `mice`의 이름은 *Multivariate Imputation by Chained Equations*에서 왔습니다. 각 불완전한 변수에 적절한 조건부 모형을 정하고, 변수들을 번갈아 갱신해 대체본을 만듭니다.[^mice]

예를 들어 공부 시간과 출석률이 함께 빠졌다면, 출석률 등으로 공부 시간을 채우고, 갱신된 공부 시간 등으로 출석률을 채우는 과정을 반복합니다. 통계적 다중 대체를 하려면 이 과정에서 적절한 확률적 변동을 반영하여 여러 대체본을 생성해야 합니다.[^mice-api]

scikit-learn의 `IterativeImputer`는 연쇄적인 다변량 대체를 제공하지만, **기본 설정으로 한 번 실행하면 대체본 하나**를 반환합니다. `max_iter=20`은 갱신 횟수이지 대체본 20개라는 뜻이 아닙니다. `sample_posterior=True`와 서로 다른 난수 시드로 반복 실행하면 여러 대체본을 만들 수 있습니다. 기본 추정기는 가우시안 예측분포를 사용하는 `BayesianRidge`이므로 범주형 자료에 그대로 적용해서는 안 됩니다.[^iterative-api]

#### Rubin의 통합 규칙: 왜 표준오차도 필요할까요?

하나의 계수처럼 스칼라 추정량을 통합한다고 합시다. 대체본이 $m$개이고, 각 추정값과 그 분산이 $Q_k$, $U_k$라면 다음을 계산합니다.[^mi-validity]

$$
\bar Q=\frac{1}{m}\sum_{k=1}^{m}Q_k,\qquad
\bar U=\frac{1}{m}\sum_{k=1}^{m}U_k,
$$

$$
B=\frac{1}{m-1}\sum_{k=1}^{m}(Q_k-\bar Q)^2,\qquad
T=\bar U+\left(1+\frac{1}{m}\right)B.
$$

$\bar U$는 **각 대체본 안에서의 추정 불확실성**, $B$는 **대체본 사이의 결과 차이**, $T$는 두 가지를 합친 전체 분산입니다. 통합 표준오차는 $\sqrt{T}$입니다. 표본도 유한하고 빈칸의 답도 불확실하므로 둘 다 고려하는 것입니다.[^mi-validity]

이 규칙에는 대체모형과 분석모형의 적절성 등 조건이 필요합니다. 신뢰구간을 구할 때는 자유도와 표본 크기를 고려해야 하며, 정확도나 AUC 같은 모든 성능 지표에 위 식을 그대로 적용하면 안 됩니다.[^pool][^pool-parameters]

**다중 대체는 편향을 자동으로 없애는 보증서가 아닙니다.** 적절한 MAR 가정과 모형이 중요한 출발점이며, MNAR를 다루려면 추가 가정이나 민감도 분석 등이 필요합니다.[^mi-validity][^mnar]

---

## 4. 어떤 방법을 고르고, 어떻게 평가하나요?

### 4.1 정답표가 아니라 비교 출발점

아래는 앞 절의 원리와 벤치마크를 바탕으로 정리한 실험 설계 가이드입니다. 특정 기법의 우월성을 보장하는 규칙은 아닙니다.[^benchmark]

| 상황 | 비교할 출발점 | 반드시 점검할 것 |
|---|---|---|
| 일반적인 표 데이터 예측 | 평균·중앙값, 결측 표시 추가, 결측 직접 지원 모델 | 같은 데이터 분할에서의 최종 예측 성능 |
| 변수 사이 관계를 활용할 수 있음 | KNN, 반복 회귀, 나무 기반 대체 | 스케일, 계산 비용, 모형 가정 |
| 범주형과 수치형이 섞임 | 유형별 전처리, 혼합형 지원 방법 | 범주를 연속적인 숫자로 잘못 처리하지 않는지 |
| 센서·시계열 | 짧은 구간의 전방 채우기, 시계열 전용 모델 | 시간 순서, 개체 구분, 실제로 이용 가능한 시점 |
| 계수·신뢰구간을 보고해야 함 | 분석모형에 맞는 다중 대체 | 모형 적합성, 수렴, 통합 규칙 |
| MNAR가 의심됨 | 수집 과정 검토와 민감도 분석 | 추가 가정을 바꿔도 결론이 유지되는지 |

### 4.2 데이터 누출을 먼저 막으세요

**학습에 허용되지 않은 정보를 미리 이용하는 것**이 데이터 누출입니다. 전체 데이터의 평균을 계산한 뒤 학습·테스트로 나누면, 테스트 데이터 정보가 전처리에 이미 들어갑니다.[^leakage]

```text
올바른 순서:
데이터 분리 → 학습 데이터에서 대체·스케일링 학습 → 검증·테스트에는 적용만

교차검증에서도:
각 학습 폴드에서 전처리를 새로 학습 → 해당 검증 폴드에는 적용만
```

`Pipeline`은 대체, 스케일링, 예측모형을 묶어 이 순서를 지키는 데 도움이 됩니다. 미래 예측에는 시간순 분할을, 같은 사람의 반복 기록이 있다면 목적에 맞는 개체 단위 분할을 고려해야 합니다. 파이프라인을 썼더라도 데이터 분할 자체가 잘못되면 누출을 막지 못합니다.[^cv]

### 4.3 대체 정확도와 예측 정확도를 따로 보세요

**복원 평가**에서는 원래 관측된 값의 일부를 가려 정답으로 보관하고, 숨긴 위치에서만 오차를 계산합니다. 보이지 않는 실제 결측값에는 보통 정답이 없으므로 직접 오차를 계산할 수 없습니다. 수치형은 MAE·RMSE, 범주형은 잘못 분류한 비율 등을 사용할 수 있습니다.[^benchmark]

$$
\mathrm{RMSE}_{\mathrm{imp}}
=\sqrt{\frac{1}{|H|}\sum_{(i,j)\in H}(x_{ij}-\hat x_{ij})^2}.
$$

여기서 $H$는 **평가를 위해 숨긴 위치들의 집합**입니다. 단위가 다른 여러 열을 한꺼번에 평가할 때는 열별 지표나 명시적인 정규화 기준이 필요합니다. 큰 단위의 열 하나가 전체 점수를 좌우하지 않게 해야 합니다.

**최종 예측 평가**는 채워진 입력으로 학습한 모델이 새로운 정답 $y$를 얼마나 잘 예측하는지 봅니다. 6.1절의 RMSE는 이 예측 오차이지 결측값 복원 오차가 아닙니다.

**통계적 추론 평가**는 반복 모의실험에서 편향이나 신뢰구간의 포함률을 확인하는 방식이 필요합니다. 한 번 계산된 RMSE나 그럴듯한 히스토그램만으로 추론의 타당성을 입증할 수 없습니다.[^mi-validity]

### 4.4 무작위로 가려 본 실험의 한계

무작위 마스킹은 유용하지만, 그 결과만으로 현실의 MAR·MNAR나 센서의 긴 연속 결측까지 잘 처리한다고 결론 내릴 수는 없습니다. 단일 지점 결측과 긴 구간 결측은 서로 다른 난이도를 가질 수 있습니다. 결측 패턴을 다양하게 평가해야 한다는 점은 최근 시계열·표 데이터 벤치마크에서도 중요하게 다룹니다.[^tsi][^tabimpute]

원래 관측되기 쉬운 값만 골라 가린 실험은, 애초에 잘 관측되지 않는 값에서의 성능을 충분히 보여 주지 못할 수도 있습니다. 따라서 실제 수집 과정에 맞는 마스킹, 여러 난수 시드, 집단별 결과, 학습 때와 사용 때의 결측 패턴 차이를 점검하세요.

---

## 5. 최신 연구는 무엇을 바꾸고 있나요?

> 이 절은 원문에 외부 연구를 추가한 부분입니다. 오래된 기반 연구와 최근 연구를 구분합니다. 논문에서 보고한 성능은 해당 데이터·결측 설정·평가 지표에 대한 결과이며, 여기에서 그 대규모 실험을 재현한 것은 아닙니다.

### 5.1 딥러닝을 “더 복잡한 평균 계산기”로만 생각하지 마세요

딥러닝 기반 대체는 다른 변수나 시간 사이의 복잡한 관계를 학습할 수 있습니다. 예를 들어 **SAITS**는 자기어텐션을 이용해 시계열의 시간적 관계와 특징 사이 관계를 활용합니다. 관측값 일부를 인위적으로 가리는 학습 과제도 사용합니다. 이 연구는 2022년 사전공개 후 2023년 학술지에 발표된 기반 연구입니다.[^saits]

직관적으로는 “어떤 시점과 어떤 측정값을 더 참고해야 할까?”를 학습하는 것입니다. 그러나 정보를 잘 참고한다는 것과, 보이지 않는 정답을 확실하게 아는 것은 다릅니다.

### 5.2 확산 모델: 그럴듯한 완성본을 여러 번 만들기

**조건부 확산 모델**은 관측된 정보를 조건으로 삼아 잡음을 점차 그럴듯한 값으로 바꾸는 방식으로 이해할 수 있습니다. 시계열 대체에 이를 활용한 대표적인 기반 연구가 **CSDI, NeurIPS 2021**입니다. 하나의 고정된 숫자뿐 아니라 가능한 값의 분포를 표현하려는 접근입니다.[^csdi]

직관적으로는 “빈칸의 유일한 정답은 3”이라고 말하기보다 “주어진 정보에서 이런 값들이 가능하다”는 여러 완성본을 만드는 것입니다. 다만 여러 번 샘플링할 수 있다는 이유만으로 통계적 다중 대체의 타당한 신뢰구간이 자동으로 얻어지는 것은 아닙니다. 표본의 다양성과 불확실성의 적절성은 따로 확인해야 합니다.[^mi-validity]

### 5.3 2025–2026년의 대표적인 연구 방향

| 연구와 확인한 시점 | 무엇을 다루나요? | 초보자가 가져갈 교훈 |
|---|---|---|
| **IMAGIC-500, 2025년 6월 사전공개본** | 공개 합성 사회경제 자료로 MCAR·MAR·MNAR, 수치형·범주형 복원, 계산 비용, 최종 예측을 함께 비교하는 벤치마크입니다.[^imagic] | 한 가지 오차만 보지 말고 여러 평가 축을 보세요. 합성자료 결과를 모든 실제 조사에 그대로 일반화하지 마세요. |
| **RefiDiff, AAAI 2026; 학회 논문집 공개일 2026-03-14** | 기계학습의 초기 예측과 확산 기반 보정을 결합합니다. 논문은 혼합형 표 데이터와 여러 결측 설정에서 평가합니다.[^refidiff] | 전통적 예측과 생성형 모델을 결합하는 연구가 진행되고 있습니다. 논문의 MNAR 실험 성과는 MNAR가 일반적으로 해결되었다는 뜻이 아닙니다. |
| **TabImpute / One Pipeline, Many Transformers, 2026-08-28의 arXiv v6** | 합성 표로 사전학습한 트랜스포머와 결측 패턴별 모델을 연구합니다. MissBench에서 42개 데이터셋과 11가지 결측 패턴을 평가합니다.[^tabimpute] | 현재 데이터에서만 모델을 학습하는 방식 외에, 미리 학습한 표 전용 모델을 사용하는 방향도 있습니다. 여기서는 확인한 arXiv 개정본을 근거로 설명합니다. |

이 자료들은 **새로운 후보를 제시하는 근거**이지 “항상 가장 좋은 대체법”을 확정하는 근거는 아닙니다. 이 문서의 권고는 단순 기준선과 새 방법을 **동일한 정보 접근 조건과 평가 절차**에서 비교하라는 것입니다.

---

## 6. 파이썬으로 직접 실습하기

아래 세 예제는 서로 독립적으로 실행할 수 있습니다. 코드는 **Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0** 환경에서 실행 확인했습니다. 이 버전들이 최신이라는 뜻은 아닙니다.

```bash
python -m pip install numpy pandas scikit-learn
```

설치된 버전과 환경에 따라 경고·소수점 결과가 달라질 수 있습니다. 실험을 기록할 때는 실행한 버전을 함께 남기세요. `IterativeImputer`는 확인한 공식 문서에서 실험적 기능으로 안내하므로 활성화용 import가 필요합니다.[^iterative-api]

### 6.1 예측 실습: 교차검증으로 선택하고 테스트는 마지막에

원문의 당뇨병 예제 데이터는 유지하되, 미리 전체 데이터로 표준화된 값을 쓰지 않도록 `scaled=False`로 불러옵니다.[^diabetes]

이 실습은 수치형 대체에 집중하기 위해 숫자로 코딩된 이진 변수 `sex`를 제외합니다. 실제 혼합형 데이터 분석에서 범주형 변수를 무조건 버리라는 뜻은 아닙니다. 범주형은 유형에 맞는 대체와 인코딩을 별도로 적용해야 합니다.

예측모형 `Ridge`는 큰 회귀계수에 벌점을 주는 선형 회귀입니다. 여기서는 대체법을 비교할 공통 모델로 사용합니다. `HistGradientBoostingRegressor` 후보는 대체법뿐 아니라 예측모형도 다르므로 **전체 예측 절차의 비교**라는 점을 기억하세요.[^ridge][^histgb]

```python
"""수치형 입력의 결측치 처리: 교차검증으로 선택한 뒤 테스트는 한 번만 평가."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mask_mcar(X: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """완전한 수치형 배열의 각 칸을 값과 무관하게 숨기는 교육용 함수."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("결측값과 무한대가 없는 2차원 수치형 배열이 필요합니다.")
    if not 0 <= rate < 1:
        raise ValueError("rate는 0 이상 1 미만이어야 합니다.")
    rng = np.random.default_rng(seed)
    masked = X.copy()
    masked[rng.random(X.shape) < rate] = np.nan
    if np.isnan(masked).all(axis=0).any():
        raise ValueError("전부 결측인 열이 생겼습니다. 데이터 또는 결측 설정을 점검하세요.")
    return masked


# 이미 전체 데이터로 표준화된 기본값 대신 원래 단위의 데이터를 불러옵니다.
data = load_diabetes(as_frame=True, scaled=False)
# 이 예제는 수치형 대체만 다룹니다. 숫자로 코딩된 이진 변수는 제외합니다.
X = data.data.drop(columns=["sex"]).to_numpy(dtype=float)
y = data.target.to_numpy(dtype=float)

# 어떤 통계값도 계산하기 전에 먼저 분리합니다.
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
X_train = mask_mcar(X_train_full, rate=0.15, seed=101)
X_test = mask_mcar(X_test_full, rate=0.15, seed=202)

# 각 Pipeline은 교차검증의 학습 폴드 안에서만 전처리를 학습합니다.
candidates = {
    "Mean + Ridge": make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=1.0)
    ),
    "Median + Ridge": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
    ),
    "Median + indicator + Ridge": make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(), Ridge(alpha=1.0)
    ),
    "Scaled KNN + Ridge": make_pipeline(
        StandardScaler(), KNNImputer(n_neighbors=5), Ridge(alpha=1.0)
    ),
    "Iterative single + Ridge": make_pipeline(
        StandardScaler(),
        IterativeImputer(max_iter=30, tol=0.01, random_state=42),
        Ridge(alpha=1.0)
    ),
    # 이 후보는 대체법뿐 아니라 예측 모델도 다릅니다.
    "Native NaN + HistGB": HistGradientBoostingRegressor(
        max_iter=150, max_leaf_nodes=15, learning_rate=0.05, random_state=42
    ),
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, estimator in candidates.items():
    negative_mse = cross_val_score(
        estimator, X_train, y_train, cv=cv,
        scoring="neg_mean_squared_error", n_jobs=1, error_score="raise"
    )
    rmse = np.sqrt(-negative_mse)
    rows.append({
        "method": name,
        "cv_rmse_mean": float(rmse.mean()),
        "cv_rmse_sd": float(rmse.std(ddof=1)),
    })

results = pd.DataFrame(rows).sort_values("cv_rmse_mean")
print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# 테스트 점수가 아니라 교차검증 점수로 선택합니다.
best_name = str(results.iloc[0]["method"])
best_model = clone(candidates[best_name]).fit(X_train, y_train)
predictions = best_model.predict(X_test)
test_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
print(f"\nSelected: {best_name}")
print(f"Test RMSE: {test_rmse:.3f}")
```

#### 왜 이런 순서로 작성했나요?

`StandardScaler`는 관측된 값으로 평균·스케일을 추정하고, 변환 중 `NaN`은 그대로 유지합니다. 따라서 KNN 후보에서는 **스케일을 맞춘 뒤 이웃을 찾는 순서**가 가능합니다. 다른 스케일러가 모두 같은 동작을 하는 것은 아니므로 사용하는 도구의 문서를 확인해야 합니다.[^scaler]

`add_indicator=True`는 결측 여부를 나타내는 추가 열을 만듭니다. 다만 기본적으로 학습 때 결측이 있었던 열에만 표시가 추가됩니다. 이 표시가 언제나 예측 성능을 높이는 것은 아닙니다.[^simple-api]

`cv_rmse_sd`는 교차검증 폴드 점수의 표준편차입니다. 학습 자료가 겹치는 폴드 사이의 변동성이므로, 이를 그대로 독립 표본에서 얻은 신뢰구간처럼 해석하면 안 됩니다.[^cv]

#### 실행 확인 결과

다음은 위에 명시한 환경에서 얻은 한 번의 실행 결과입니다. **일반적인 기법 순위가 아닙니다.**

```text
                    method  cv_rmse_mean  cv_rmse_sd
  Iterative single + Ridge        58.784       3.923
        Scaled KNN + Ridge        59.080       5.256
              Mean + Ridge        60.493       4.393
            Median + Ridge        60.531       4.232
Median + indicator + Ridge        62.189       4.694
       Native NaN + HistGB        65.051       4.920

Selected: Iterative single + Ridge
Test RMSE: 55.425
```

이 실험에서는 반복 단일 대체 후보가 선택되었습니다. 하지만 데이터 분할, 결측을 만든 방식, 난수 시드, 예측모형을 바꾸면 순위가 달라질 수 있습니다. 교차검증에서 선택한 모델의 테스트 점수를 확인한 뒤 그 점수를 기준으로 다시 후보를 조정하면, 테스트 데이터가 사실상 검증 데이터가 됩니다.[^cv]

또한 이 예제는 **각 입력 칸을 15% 확률로 가리는 MCAR 수치형 예측 실습**입니다. MAR·MNAR나 실제 의료 현장에서의 성능을 검증한 실험이 아닙니다. 보관된 `X_train_full`, `X_test_full`의 숨긴 실제 값은 대체모형 학습에 사용하지 않습니다.

### 6.2 추론 실습: 여러 대체본 생성 → 계수 추정 → 불확실성 통합

이번에는 새 사람을 예측하는 문제가 아니라, 합성자료에서 $x_1$의 회귀계수를 추정합니다. 서로 관련된 연속형 변수 두 개와 결과 $y$를 만들고, 관측된 $y$에 따라 입력값이 빠지게 합니다. 따라서 이 코드에서 만든 결측 과정은 $y$를 포함한 정보에 대해 MAR입니다.

**여기서는 관측된 결과변수 $y$를 대체모형에 포함합니다.** 통계적 추론을 위한 다중 대체에서는 분석에 필요한 관계를 보존하도록 결과변수 등을 포함하는 것이 중요합니다. 반면 6.1절처럼 실제 사용 시 정답을 모르는 예측 문제에서는 테스트 정답을 대체에 넣어서는 안 됩니다. 목적이 달라 두 절의 정보 사용 방식도 다릅니다.[^mice][^leakage]

```python
"""연속형 합성자료로 배우는 확률적 연쇄 대체와 Rubin의 통합 규칙."""
from __future__ import annotations

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

rng = np.random.default_rng(42)
n = 600
x1 = rng.normal(size=n)
x2 = 0.5 * x1 + rng.normal(size=n)
y = 1.0 + 2.0 * x1 - 0.8 * x2 + rng.normal(size=n)

# 여기서는 새 사람을 예측하는 것이 아니라 x1의 회귀계수를 추정합니다.
# 관측된 결과변수 y도 대체 모형에 포함합니다.
data = np.column_stack([x1, x2, y])
prob_missing = 0.10 + 0.25 / (1.0 + np.exp(-y / 2.0))
for column in (0, 1):
    data[rng.random(n) < prob_missing, column] = np.nan

m = 20  # 교육용 대체본 수입니다. max_iter와 다른 개념입니다.
estimates = []
within_variances = []
completed_datasets = []

for seed in range(m):
    imputer = IterativeImputer(
        sample_posterior=True, max_iter=15, random_state=1000 + seed
    )
    completed = imputer.fit_transform(data)
    completed_datasets.append(completed)

    # 각 대체본에서 y ~ 1 + x1 + x2를 최소제곱법으로 적합합니다.
    design = np.column_stack([np.ones(n), completed[:, :2]])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("설명변수 사이의 완전한 중복으로 회귀계수를 추정할 수 없습니다.")
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    residual_df = n - design.shape[1]
    if residual_df <= 0:
        raise ValueError("회귀계수의 분산을 추정할 표본이 부족합니다.")
    residual_variance = (residuals @ residuals) / residual_df
    covariance = residual_variance * np.linalg.inv(design.T @ design)

    estimates.append(beta[1])
    within_variances.append(covariance[1, 1])

Q = np.asarray(estimates)
U = np.asarray(within_variances)
Q_bar = float(Q.mean())
U_bar = float(U.mean())
B = float(Q.var(ddof=1))
T = U_bar + (1.0 + 1.0 / m) * B

# 원래 관측된 값은 대체 과정에서 바뀌지 않았는지 확인합니다.
observed = ~np.isnan(data)
assert all(np.allclose(d[observed], data[observed]) for d in completed_datasets)
assert not np.allclose(completed_datasets[0], completed_datasets[1])

print(f"Completed datasets: {len(completed_datasets)}")
print(f"Pooled x1 coefficient: {Q_bar:.3f}")
print(f"Within-imputation variance: {U_bar:.6f}")
print(f"Between-imputation variance: {B:.6f}")
print(f"Pooled standard error: {np.sqrt(T):.3f}")
print("Synthetic population coefficient of x1: 2.000")
```

#### 실행 확인 결과

```text
Completed datasets: 20
Pooled x1 coefficient: 1.991
Within-imputation variance: 0.002225
Between-imputation variance: 0.001089
Pooled standard error: 0.058
Synthetic population coefficient of x1: 2.000
```

핵심은 실제 계수 2에 가까운 숫자가 나왔다는 사실보다, **20개의 대체본에서 계수와 그 분산을 각각 구하고 통합했다**는 점입니다. 실제 자료에서는 모집단의 진짜 계수를 모르므로 이런 직접 비교도 할 수 없습니다.

이 코드는 연속형·선형·정규 오차의 합성자료에서 확률적 연쇄 대체와 통합 계산을 배우기 위한 것입니다. 범주형, 복잡한 상호작용, 반복 측정, 가중치가 있는 조사자료에는 그대로 적용할 수 없습니다. 기본 추정기의 가정과 분석모형의 적합성도 확인해야 합니다.[^iterative-api][^mi-validity]

`m=20`과 `max_iter=15`는 예제 설정이지 보편적인 권장값이 아닙니다. 실제 분석에서는 대체 과정의 수렴과 대체본 수를 늘렸을 때의 안정성, 몬테카를로 오차를 점검해야 합니다. 코드 실행 성공은 신뢰구간의 타당성을 증명하지 않습니다.[^m-number]

### 6.3 시계열 실습: 개체별로, 과거 값만 사용하기

```python
"""센서를 구분하고 과거 관측값만 한 칸까지 전달하는 예제."""
import numpy as np
import pandas as pd

frame = pd.DataFrame({
    "sensor": ["A", "A", "A", "A", "B", "B", "B"],
    "time": [1, 2, 3, 4, 1, 2, 3],
    "value": [20.0, np.nan, np.nan, 23.0, np.nan, 10.0, np.nan],
}).sort_values(["sensor", "time"]).reset_index(drop=True)

# 원래 값과 결측 여부는 따로 보관합니다.
frame["was_missing"] = frame["value"].isna()
frame["filled"] = frame.groupby("sensor", sort=False)["value"].ffill(limit=1)

# 긴 공백이나 이전 관측값이 없는 위치까지 억지로 채우지 않습니다.
assert pd.isna(frame.loc[2, "filled"])  # 센서 A의 연속 결측 두 번째 칸
assert pd.isna(frame.loc[4, "filled"])  # 센서 B의 첫 기록
assert frame.loc[6, "filled"] == 10.0
print(frame.to_string(index=False))
```

실행 결과는 다음과 같습니다.

```text
sensor  time  value  was_missing  filled
     A     1   20.0        False    20.0
     A     2    NaN         True    20.0
     A     3    NaN         True     NaN
     A     4   23.0        False    23.0
     B     1    NaN         True     NaN
     B     2   10.0        False    10.0
     B     3    NaN         True    10.0
```

`limit=1`은 직전 관측값으로 연속 결측을 한 칸까지만 채우도록 제한합니다. 센서 A의 세 번째 시점과 센서 B의 첫 번째 시점은 여전히 결측입니다. 근거가 부족한 값을 억지로 채우지 않은 결과입니다.[^pd-fill]

실제 시스템에서는 한 칸이 몇 초·몇 시간인지도 중요합니다. 기록 간격이 불규칙하면 `limit`만으로 시간 간격을 제한할 수 없습니다. 마지막 관측 이후 경과 시간을 따로 계산해 허용 범위를 정해야 합니다.

---

## 7. 실전에서 꼭 확인할 것

### 7.1 결측 표시와 변수의 뜻을 보존하세요

원본과 결측 마스크를 보관하고, 채운 값을 실제 관측값과 구별할 수 있게 하세요. `0`, 미응답, 측정 실패, 해당 없음을 무턱대고 같은 값으로 합치지 마세요. 한 열 전체가 비어 있다면 그 열의 평균이나 다른 열과의 관계를 현재 관측자료만으로 학습할 수 없습니다. 상수로 열 모양을 유지하는 기능이 있어도 정보를 복원한 것은 아닙니다.[^sk-impute]

### 7.2 이상치 처리가 무조건 먼저인 것은 아닙니다

먼저 단위 오류나 입력 실수인지 확인하세요. 큰 값이 진짜 관측일 수도 있으므로, 무조건 지우는 규칙은 적절하지 않습니다. 어떤 변환·스케일링·대체를 사용할지 목적에 맞게 설계하고, 자료에서 학습하는 임곗값은 학습 폴드 안에서만 정해야 합니다.[^scaler][^leakage]

### 7.3 대체값의 범위와 관계를 확인하세요

나이가 음수이거나 범주형 값이 1.4가 되는 식의 결과는 경고 신호입니다. 수치가 가능 범위 안에 있다는 것만으로 타당성이 입증되는 것도 아닙니다. 여러 변수의 조합, 분포, 중요한 집단의 결과를 함께 확인해야 합니다. 범주나 범위에 맞는 모형, 실제 기증자 값을 이용하는 방법 등을 고려할 수 있습니다.[^stochastic][^mice-api]

### 7.4 결측 표시가 편향을 해결하지는 않습니다

결측 여부 자체가 예측에 도움이 될 수는 있습니다. 하지만 **결측 표시를 추가하는 것만으로 회귀계수의 해석이나 인과효과 추정이 올바르게 되는 것은 아닙니다.** 특히 관찰자료에서 결측 표시 방식이 편향을 일으킬 수 있다는 연구가 있습니다.[^indicator]

또한 수집 방식이 바뀌면, 학습 때 유용했던 “어떤 값이 빠졌는가”라는 정보가 더 이상 같은 의미를 갖지 않을 수 있습니다. 실제 사용 단계에서도 결측 패턴과 예측 성능을 점검해야 합니다.

### 7.5 MNAR에는 가정을 바꾸는 민감도 분석이 필요합니다

예를 들어 “소득 미응답자의 소득은 MAR 모형이 추정한 값보다 더 높을 수 있다”고 생각한다면, 그 차이에 대한 몇 가지 합리적인 시나리오를 정해 결과를 비교할 수 있습니다. 연속형 변수에서는 MAR 기반 대체값을 일정한 $\delta$만큼 이동시키는 **델타 조정**이 한 예입니다.[^mnar]

$\delta$는 데이터가 자동으로 알려주는 정답이 아닙니다. 변수의 단위·범위와 분야 지식을 근거로 정하는 추가 가정입니다. 이 가정을 바꾸면 핵심 결론도 바뀌는지 보고하는 것이 중요합니다.[^mnar]

### 7.6 정답 레이블의 결측은 입력의 결측과 구분하세요

입력 $X$의 결측을 채우는 문제와, 학습 정답 $y$를 모르는 문제는 다릅니다. 이 문서의 예측 실습은 정답이 관측된 상황입니다. 테스트 정답을 모델로 생성한 뒤 그것을 실제 정답처럼 평가하면 모델이 스스로 만든 답을 채점하는 셈입니다.

결측 레이블을 다루는 통계모형이나 준지도학습은 별도 문제이며, 여기서 소개한 입력 대체 코드를 그대로 적용하면 해결되는 것은 아닙니다. 추론용 다중 대체에서 결과변수를 다루는 경우에도 명시적인 모형과 가정이 필요합니다.[^mice]

### 7.7 결과와 가정을 함께 기록하세요

최소한 결측 표시의 의미, 변수별 결측률, 가정한 결측 과정, 데이터 분할, 대체법과 설정, 사용한 정보의 시점, 평가 지표, 난수 시드, 실행 버전을 남기세요. **좋은 결측치 처리는 빈칸이 없어 보이는 표가 아니라, 무엇을 관측했고 무엇을 가정했는지 설명할 수 있는 분석입니다.**

---

## 8. 핵심 정리

결측치를 처리할 때 가장 먼저 물어볼 것은 “어떤 라이브러리를 쓰지?”가 아니라 **“왜 빠졌고, 무엇을 알아내려는가?”**입니다.

예측이 목표라면 단순한 기준선에서 출발해 정보 누출 없이 검증하세요. 계수와 신뢰구간이 목표라면 대체의 불확실성을 분석에 반영하세요. 최신 모델을 쓰더라도 결측 원인에 대한 가정과 평가 설계가 중요하다는 사실은 달라지지 않습니다.[^leakage][^pool][^tabimpute]

**빈칸을 채웠다는 사실과, 그 빈칸의 진짜 값을 알게 되었다는 사실을 혼동하지 않는 것. 이것이 결측치 처리의 출발점입니다.**

---

## 부록 A. 원문에서 바로잡은 주요 내용

| 원문의 표현·구성 | 개정 내용 |
|---|---|
| 결측치가 있으면 모델이 정상 작동하지 않는다. | 모델마다 다릅니다. 결측을 직접 지원하는 모델을 포함했습니다. |
| MAR는 다른 변수에 따라 결측 확률이 달라지는 경우이다. | 관측 정보를 조건으로 하면 미관측 값에 추가로 의존하지 않는다는 조건을 명시했습니다. |
| 결측 유형을 파악한 뒤 방법을 선택하면 된다. | 관측자료만으로 MAR와 MNAR를 일반적으로 구별할 수 없음을 설명했습니다. |
| 평균·중앙값의 단점을 묶어 이상치에 취약하다고 설명한다. | 중앙값이 극단값에 덜 민감한 이유를 직접 계산하는 예시로 구분했습니다. |
| `A, B, B, C, 결측, A`의 최빈값은 B이다. | A와 B가 각각 두 번인 공동 최빈값임을 바로잡았습니다. |
| 회귀 대체는 원 분포와 변동성을 유지한다. | 결정론적 예측값 대체와 확률적 대체를 구분했습니다. |
| GAM 기대값 표기를 `g(O_E(Y))`로 제시한다. | 조건부 평균을 사용하는 $g(\mathbb{E}[Y\mid X])$ 형태로 수정하고 각 기호를 설명했습니다. |
| KNN의 일반적 변형과 scikit-learn 구현을 혼용한다. | `KNNImputer`가 수치형의 평균·거리 가중 평균을 사용한다는 점을 명시했습니다. |
| 다중 대체를 설명하지만 코드에서는 완성 데이터 하나만 만든다. | 여러 확률적 대체본 생성, 개별 분석, 추정값·분산 통합을 코드로 보여 줍니다. |
| 다중 대체는 편향 없는 결과를 보장한다. | 결측 가정과 모형의 적절성에 의존한다는 조건을 명시했습니다. |
| 테스트 점수로 여러 대체법을 비교하는 예제이다. | 교차검증으로 후보를 선택하고 최종 테스트는 선택 후에 평가합니다. |
| 같은 주제의 설명·코드·결론이 반복된다. | 9가지 기법을 중심으로 통합하고, 최신 연구와 실전 평가를 별도 절로 구성했습니다. |
| Datawig를 딥러닝 대체의 대표적인 다음 단계로 제시한다. | 특정 도구의 우월성을 전제하지 않고 연구 접근법과 검증 조건을 설명했습니다. |
| 가이드를 따르면 결측치 문제를 확실히 해결할 수 있다고 결론 낸다. | 추가 정보 없이 알 수 없는 부분과 결과의 불확실성을 명확히 했습니다. |

부록의 원문 표현은 문제점을 요약한 것이며, 모든 항목을 원문 그대로 인용한 것은 아닙니다. 각 수정의 근거는 해당 본문 절의 각주에 연결했습니다.

## 참고문헌 및 공식 문서

기본 개념은 해당 이론을 직접 설명하는 저자 자료와 공식 구현 문서로 확인했습니다. 최근 논문은 사전공개본과 학회·학술지 논문을 구별했습니다. 아래 자료는 2026년 10월 16일 확인했으며, 변동 가능한 웹 문서의 설명과 코드의 실제 실행 환경은 구분해야 합니다.

[^sk-impute]: scikit-learn. **Imputation of missing values**. 단일·반복·KNN 대체, 결측 표시, 전체가 결측인 열의 처리에 관한 공식 사용자 가이드. <https://scikit-learn.org/stable/modules/impute.html>

[^pd-missing]: pandas. **Working with missing data**. 결측 표시와 자료형, 채우기와 보간을 설명하는 공식 사용자 가이드. <https://pandas.pydata.org/docs/user_guide/missing_data.html>

[^histgb]: scikit-learn. **HistGradientBoostingRegressor**. 결측값을 직접 처리하는 방식과 API. <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html>

[^mechanism]: van Buuren, S. (2018). **Flexible Imputation of Missing Data**, 2nd ed., §1.2, “Missing data mechanisms”. 저자가 제공하는 온라인 본문. <https://stefvanbuuren.name/fimd/sec-MCAR.html>

[^formal]: van Buuren, S. **Flexible Imputation of Missing Data**, “Identification concepts”. 결측 과정의 수학적 정의와 MAR·MNAR의 식별 한계. <https://stefvanbuuren.name/fimd/sec-idconcepts.html>

[^mnar]: van Buuren, S. **Flexible Imputation of Missing Data**, “Imputation under nonignorable missingness”. MNAR와 델타 조정 등 민감도 분석. <https://stefvanbuuren.name/fimd/sec-nonignorable.html>

[^benchmark]: Jäger, S., Allhorn, A., & Bießmann, F. (2021). **A Benchmark for Data Imputation Methods**. *Frontiers in Big Data*, 4, 693674. 대체 품질과 최종 예측 성능을 구분하는 벤치마크. <https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2021.693674/full>

[^pool]: mice 개발팀. **Combine estimates by pooling rules — pool**. 대체본별 분석 결과의 통합, 전체 분산, 자유도 관련 공식 문서. <https://amices.org/mice/reference/pool.html>

[^simple-theory]: van Buuren, S. **Flexible Imputation of Missing Data**, “Simple solutions”. 완전 사례 분석과 평균·회귀 등 단순 대체의 한계. <https://stefvanbuuren.name/fimd/sec-simplesolutions.html>

[^mice]: van Buuren, S., & Groothuis-Oudshoorn, K. (2011). **mice: Multivariate Imputation by Chained Equations in R**. *Journal of Statistical Software*, 45(3), 1–67. DOI: 10.18637/jss.v045.i03. <https://www.jstatsoft.org/article/view/v045i03>

[^mice-api]: mice 개발팀. **mice: Multivariate Imputation by Chained Equations**. 변수별 모형, 혼합 자료형, 대체본 수와 반복 횟수를 구분하는 공식 API 문서. <https://amices.org/mice/reference/mice.html>

[^simple-api]: scikit-learn. **SimpleImputer**. 평균·중앙값·최빈값·상수 대체 및 결측 표시 옵션. <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>

[^pd-fill]: pandas. **DataFrame.ffill**. 이전의 유효한 관측값 전달과 연속 결측 개수 제한에 관한 공식 문서. <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html>

[^leakage]: scikit-learn. **Common pitfalls and recommended practices**, “Data leakage”. 학습·테스트 분리와 전처리 파이프라인의 올바른 사용. <https://scikit-learn.org/stable/common_pitfalls.html>

[^donor]: Andridge, R. R., & Little, R. J. A. (2010). **A Review of Hot Deck Imputation for Survey Non-response**. *International Statistical Review*, 78(1), 40–64. Hot deck의 여러 변형과 기증자 선택을 설명하는 논문. <https://pmc.ncbi.nlm.nih.gov/articles/PMC3130338/>

[^iterative-api]: scikit-learn. **IterativeImputer**. 기본 추정기, 확률적 샘플링, `max_iter`, 실험적 API 관련 공식 문서. <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>

[^stochastic]: van Buuren, S. **Flexible Imputation of Missing Data**, “Linear regression”. 예측값 대체, 확률적 회귀, 모형 불확실성의 차이. <https://stefvanbuuren.name/fimd/sec-linearnormal.html>

[^gam]: R `mgcv` 개발팀. **Generalized Additive Models — gam**. 링크 함수, 매끄러운 함수와 벌점 기반 추정에 관한 공식 문서. <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.html>

[^missforest]: Stekhoven, D. J., & Bühlmann, P. (2012). **MissForest—non-parametric missing value imputation for mixed-type data**. *Bioinformatics*, 28(1), 112–118. 아래는 저자들의 2011년 사전공개본이며, 학술지 발표는 2012년입니다. <https://arxiv.org/abs/1105.0828>

[^knn]: scikit-learn. **KNNImputer**. 결측을 고려한 거리와 이웃의 평균·거리 가중 평균을 이용하는 공식 구현. <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>

[^scaler]: scikit-learn. **StandardScaler**. 변수 스케일 조정과 `NaN`의 학습·변환 시 처리. <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>

[^mi-validity]: van Buuren, S. **Flexible Imputation of Missing Data**, “Why and when does it work?”. Rubin의 통합 규칙과 적절한 다중 대체의 조건. <https://stefvanbuuren.name/fimd/sec-whyandwhen.html>

[^pool-parameters]: van Buuren, S. **Flexible Imputation of Missing Data**, “Pooling”. 추정량의 종류와 척도에 맞춘 통합. <https://stefvanbuuren.name/fimd/sec-pooling.html>

[^cv]: scikit-learn. **Cross-validation: evaluating estimator performance**. 검증·테스트의 역할과 개체·시간 구조를 고려한 분할. <https://scikit-learn.org/stable/modules/cross_validation.html>

[^tsi]: Du, W., et al. (2024). **TSI-Bench: Benchmarking Time Series Imputation**. arXiv:2406.12747. 결측 설정과 평가의 차이를 다루는 시계열 벤치마크. <https://arxiv.org/abs/2406.12747>

[^saits]: Du, W., Côté, D., & Liu, Y. (2023). **SAITS: Self-Attention-based Imputation for Time Series**. *Expert Systems with Applications*, 219, 119619. DOI: 10.1016/j.eswa.2023.119619. 아래 사전공개본은 2022년에 처음 제출되었습니다. <https://arxiv.org/abs/2202.08516>

[^csdi]: Tashiro, Y., Song, J., Song, Y., & Ermon, S. (2021). **CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation**. *Advances in Neural Information Processing Systems*, 34. <https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html>

[^imagic]: Sun, S., et al. (2025). **IMAGIC-500: IMputation benchmark on A Generative Imaginary Country (500k samples)**. arXiv:2506.08844, 2025년 6월 사전공개본. 합성 사회경제 자료를 사용하는 벤치마크. <https://arxiv.org/abs/2506.08844>

[^refidiff]: Ahamed, M. A., Ye, Q., & Cheng, Q. (2026). **RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation**. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(24), 19551–19559. 논문집 공개일 2026-03-14. DOI: 10.1609/aaai.v40i24.39034. <https://ojs.aaai.org/index.php/AAAI/article/view/39034>

[^tabimpute]: Feitelberg, J., Saha, D., Choi, K., Ahmad, Z., Agarwal, A., & Dwivedi, R. (2026). **One Pipeline, Many Transformers: Pattern-Specific Imputation Specialists for Tabular Missing Data**. arXiv:2510.02625v6, 2026-08-28 개정본. 초기 버전은 2025년에 제출되었으며, 여기서는 v6의 제목·실험 범위를 사용했습니다. <https://arxiv.org/abs/2510.02625v6>

[^diabetes]: scikit-learn. **load_diabetes**. 예제 데이터와 `scaled` 매개변수의 공식 설명. <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>

[^ridge]: scikit-learn. **Ridge**. L2 정규화를 사용하는 선형 최소제곱 회귀의 공식 설명. <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>

[^m-number]: van Buuren, S. **Flexible Imputation of Missing Data**, “How many imputations?”. 대체본 수와 몬테카를로 변동에 관한 설명. <https://stefvanbuuren.name/fimd/sec-howmany.html>

[^indicator]: Groenwold, R. H. H., et al. (2012). **Missing covariate data in clinical research: when and when not to use the missing-indicator method for analysis**. *CMAJ*, 184(11), 1265–1269. DOI: 10.1503/cmaj.110977. <https://pubmed.ncbi.nlm.nih.gov/22371511/>
