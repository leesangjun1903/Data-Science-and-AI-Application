# 분류 모델 평가 지표 가이드: Classification Report부터 ROC·PR Curve까지

> **대상:** AI와 머신러닝을 처음 배우는 대학생  
> **자료 검토 기준일:** 2026년 10월 15일  
> **개정 범위:** 원문의 1~7장 흐름을 유지하면서 정의와 코드의 오류를 수정했습니다. PR 곡선, 확률 보정, 검증 데이터에서의 임계값 선택은 새로 보강한 내용입니다. 기초 정의는 공식 문서에, 최근 연구의 해석은 본문에 표시한 논문에 근거합니다. 2026년 개정 프리프린트는 동료심사 논문과 구분했습니다.

분류 모델은 이메일이 스팸인지, 사진에 고양이가 있는지처럼 **정해진 범주 중 어디에 속하는지** 예측합니다. 그런데 “몇 개 맞혔는가?”만으로는 모델의 장단점을 충분히 알 수 없습니다. 스팸을 놓치는 실수와 정상 메일을 차단하는 실수는 서로 다르기 때문입니다.

이 문서는 다음 질문을 차례로 다룹니다.

| 알아보고 싶은 것 | 살펴볼 지표 |
|---|---|
| 어떤 종류의 실수를 했는가? | 혼동 행렬 |
| 최종 판정을 얼마나 잘했는가? | Accuracy, Precision, Recall, F1 |
| 각 클래스에서도 잘 작동하는가? | Classification Report, macro·weighted 평균 |
| 양성에 높은 점수를 주는가? | ROC-AUC, PR 곡선, Average Precision |
| 모델이 말한 확률을 믿을 수 있는가? | 확률 보정 곡선, Brier score, Log loss |

이 질문들은 서로 다릅니다. 하나의 지표가 나머지 질문까지 모두 해결해 주지는 않습니다.[^evaluation][^calibration]

## 읽기 전에: 정답, 점수, 최종 예측을 구분하자

이 문서의 기본 설정은 **한 샘플이 두 클래스 중 하나에만 속하는 이진 분류**입니다. `1`을 양성, `0`을 음성으로 정합니다. 여기서 양성은 “관심 있게 찾으려는 클래스”라는 뜻이지, 좋은 결과라는 뜻은 아닙니다. 스팸 탐지에서는 스팸이 양성이고, 7장의 생존 예측에서는 생존이 양성입니다.

| 변수 | 의미 | 스팸 탐지의 예 |
|---|---|---|
| `y_true` | 실제 정답 레이블 | 실제 스팸이면 `1` |
| `y_score` | 양성일수록 커지도록 정의한 모델 점수 | 스팸일 추정 확률 `0.8` |
| `y_pred` | 점수를 기준으로 내린 최종 판정 | 기준을 넘으면 `1` |

예를 들어 **점수가 0.5 이상이면 양성**이라고 정할 수 있습니다. 이 기준을 **임계값(threshold)**이라고 합니다. 같은 점수라도 임계값을 바꾸면 최종 예측이 달라집니다.[^threshold]

> `predict_proba()`는 클래스별 **추정 확률**을, `decision_function()`은 **판정용 점수**를 반환합니다. 점수가 반드시 확률인 것은 아니며, 추정 확률이라고 해서 실제 빈도와 잘 맞는 것도 아닙니다.[^threshold][^calibration]

**코드 실행 방법:** 1~6장의 파이썬 코드는 같은 노트북이나 파일에서 순서대로 실행합니다. 7장의 실전 코드는 별도로 실행할 수 있도록 필요한 가져오기 구문을 모두 포함했습니다. 필요한 패키지는 `numpy`, `pandas`, `scikit-learn`, `matplotlib`입니다.

---

## 1. 혼동 행렬(Confusion Matrix)

혼동 행렬은 **실제 정답과 모델의 최종 예측을 교차해 센 표**입니다. scikit-learn의 `confusion_matrix()`는 **행이 실제 클래스, 열이 예측 클래스**입니다. `labels=[0, 1]`로 순서를 지정하면 다음과 같습니다.[^cm]

| 실제 정답 ↓ / 모델 예측 → | 음성으로 예측: 0 | 양성으로 예측: 1 |
|---|---:|---:|
| 실제 음성: 0 | TN: 정상 메일을 정상으로 판정 | FP: 정상 메일을 스팸으로 오탐 |
| 실제 양성: 1 | FN: 스팸을 정상으로 놓침 | TP: 스팸을 스팸으로 판정 |

약어의 **T/F는 맞았는지**, **P/N은 무엇으로 예측했는지**를 뜻합니다. 예를 들어 FP는 “양성이라고 예측했지만 틀렸다”는 뜻입니다.

### 공통 예제: 메일 100개

다음 숫자는 지표를 설명하기 위해 만든 예제이며, 실제 서비스나 연구의 성능 결과가 아닙니다.

실제 스팸은 20개, 정상 메일은 80개입니다. 모델이 스팸 15개를 잡고 5개를 놓쳤으며, 정상 메일 10개를 스팸으로 잘못 분류했다고 합시다.

| 실제 정답 ↓ / 모델 예측 → | 정상으로 예측 | 스팸으로 예측 | 실제 개수 |
|---|---:|---:|---:|
| 정상 메일 | TN = 70 | FP = 10 | 80 |
| 스팸 | FN = 5 | TP = 15 | 20 |
| 예측 개수 | 75 | 25 | 100 |

```python
import numpy as np
from sklearn.metrics import confusion_matrix

# 앞의 20개는 실제 스팸, 뒤의 80개는 실제 정상 메일입니다.
y_true = np.array([1] * 20 + [0] * 80)
y_pred = np.array([1] * 15 + [0] * 5 + [1] * 10 + [0] * 70)

# labels를 지정해 행과 열의 클래스 순서를 고정합니다.
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(cm)
```

출력:

```text
[[70 10]
 [ 5 15]]
```

`[[TN, FP], [FN, TP]]`라는 해석은 **이진 분류에서 음성, 양성 순서로 배치했을 때** 성립합니다. 클래스 순서가 바뀌면 위치의 의미도 바뀝니다. 세 클래스 이상에서는 각 셀이 “실제 클래스 i를 클래스 j로 예측한 개수”이며, 전체 행렬을 네 값으로 펼칠 수 없습니다.[^cm]

---

## 2. 정확도(Accuracy)

정확도는 **전체 샘플 중 맞힌 비율**입니다.[^basics]

$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

메일 예제에서는 100개 중 85개를 맞혔으므로 다음과 같습니다.

$$
\mathrm{Accuracy}=\frac{15+70}{100}=0.85
$$

```python
from sklearn.metrics import accuracy_score

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
# Accuracy: 0.8500
```

### 정확도가 높아도 중요한 클래스를 전혀 찾지 못할 수 있다

스팸 1개와 정상 메일 99개가 있을 때, **모든 메일을 정상이라고 예측해도 정확도는 99%**입니다. 그러나 스팸을 찾는 능력은 없습니다.

이처럼 클래스별 개수가 크게 다른 상태를 **클래스 불균형(class imbalance)**이라고 합니다. 정확도 자체가 틀린 지표인 것은 아닙니다. 다만 다수 클래스의 결과가 전체 값을 지배할 수 있으므로, 클래스별 성능과 실수의 종류도 함께 확인해야 합니다.[^basics]

**보완 지표: 균형 정확도(Balanced Accuracy).** 각 클래스의 재현율을 구한 뒤 똑같은 비중으로 평균 냅니다. 이진 분류에서 기본 설정인 `adjusted=False`일 때 다음과 같습니다.[^balanced]

$$
\mathrm{Balanced\ Accuracy}
=\frac{1}{2}\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right)
$$

첫 항은 “스팸을 얼마나 잘 찾았는가”, 둘째 항은 “정상 메일을 얼마나 잘 지켰는가”입니다. 둘째 항은 **특이도(specificity)**라고도 합니다. 메일 예제의 균형 정확도는 $(0.75+0.875)/2=0.8125$입니다.

```python
from sklearn.metrics import balanced_accuracy_score

print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
# Balanced accuracy: 0.8125
```

균형 정확도도 오류의 실제 비용까지 자동으로 반영하지는 않습니다. 평가 목적에 맞게 선택해야 합니다.[^flores]

---

## 3. 정밀도(Precision)와 재현율(Recall)

둘을 구분하는 가장 쉬운 방법은 **분모에 누가 들어가는지** 확인하는 것입니다.[^basics]

### 3.1. 정밀도: 모델이 찾았다고 한 것 중 진짜는 몇 개인가?

$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$

메일 예제에서 모델은 25개를 스팸으로 분류했습니다. 그중 진짜 스팸은 15개입니다.

$$
\mathrm{Precision}=\frac{15}{15+10}=0.60
$$

즉, **“스팸이라고 판정한 메일 중 60%가 실제 스팸이다”**라는 뜻입니다. 정밀도가 낮으면 양성 예측에 오탐이 많이 섞여 있다는 뜻입니다.

### 3.2. 재현율: 실제로 찾아야 할 것 중 몇 개를 찾았는가?

$$
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

실제 스팸 20개 중 15개를 찾았으므로 다음과 같습니다.

$$
\mathrm{Recall}=\frac{15}{15+5}=0.75
$$

즉, **“실제 스팸 중 75%를 찾아냈다”**라는 뜻입니다. 재현율은 **민감도(sensitivity)** 또는 **진양성률(TPR)**이라고도 합니다.[^report]

```python
from sklearn.metrics import precision_score, recall_score

print(f"Precision: {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
# Precision: 0.6000
# Recall:    0.7500
```

### 3.3. 무엇을 더 중요하게 봐야 할까?

**양성을 놓치는 실수(FN)가 큰 문제라면 재현율**을, **양성이라고 잘못 판단하는 실수(FP)가 큰 문제라면 정밀도와 오탐 규모**를 주의 깊게 봅니다. 예를 들어 중요한 정상 메일의 자동 차단을 피하려면 스팸 판정의 정밀도가 중요할 수 있습니다. 하지만 “스팸 탐지는 무조건 정밀도”처럼 업무 이름만으로 결정하지는 않습니다. 자동 차단인지, 사람이 다시 검토하는 경고인지에 따라서도 비용이 달라집니다.[^threshold][^flores]

한 지표만 높이는 것은 충분하지 않습니다. 예를 들어 모든 메일을 스팸으로 예측하면 재현율은 100%지만, 정상 메일까지 모두 차단합니다.

> **분모가 0인 경우:** 양성으로 예측한 샘플이 하나도 없으면 정밀도는 수학적으로 정의되지 않습니다. 실제 양성이 하나도 없으면 재현율이 정의되지 않습니다. `zero_division=0`은 이때 보고서에 0을 넣는 **표시·계산 규칙**이지, 정의되지 않은 값이 본래 0이라는 뜻은 아닙니다. 실제 보고에서는 해당 클래스의 샘플 수와 이런 상황을 함께 밝혀야 합니다.[^prfs]

---

## 4. F1 Score

F1은 정밀도와 재현율의 **조화 평균**입니다. 두 값이 모두 높을 때 높아지고, 한쪽이 낮으면 그 영향을 크게 받습니다.[^f1]

$$
F_1=2\frac{\mathrm{Precision}\times\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
=\frac{2TP}{2TP+FP+FN}
$$

메일 예제에서는 다음과 같습니다.

$$
F_1=\frac{2\times15}{2\times15+10+5}
=\frac{30}{45}\approx0.6667
$$

정밀도가 0.9이고 재현율이 0.1이면 산술 평균은 0.5이지만, F1은 0.18입니다. 한쪽만 잘하는 모델에 높은 점수를 주지 않는다는 점을 볼 수 있습니다.

```python
from sklearn.metrics import f1_score

print(f"F1: {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
# F1: 0.6667
```

### F1이 알려주지 않는 것

**F1에는 TN이 직접 들어가지 않습니다.** 따라서 양성 탐지를 요약하는 데 유용할 수 있지만, 음성을 얼마나 잘 지켰는지나 전체 오류 비용을 모두 보여주지는 않습니다. 또한 F1의 정밀도·재현율 대칭성이 “현실에서 FP와 FN의 비용이 같다”는 뜻은 아닙니다.[^f1][^flores]

재현율 쪽에 더 무게를 두는 $F_\beta$도 있습니다. 하지만 $\beta$를 정하는 일과 실제 금전적·운영적 오류 비용을 설정하는 일은 같지 않습니다.[^prfs]

$$
F_\beta=(1+\beta^2)
\frac{\mathrm{Precision}\times\mathrm{Recall}}
{\beta^2\mathrm{Precision}+\mathrm{Recall}}
$$

$\beta>1$이면 재현율 쪽에, $0<\beta<1$이면 정밀도 쪽에 상대적으로 더 무게를 둡니다. 처음에는 F1의 의미를 이해하는 것만으로 충분합니다.

정밀도나 재현율의 분모가 0인 경우와 별도로, F1은 $2TP+FP+FN=0$이면 정의되지 않습니다. 이때도 scikit-learn의 `zero_division` 설정을 확인해야 합니다.[^f1]

---

## 5. Classification Report

scikit-learn의 `classification_report()`는 **각 클래스의 정밀도·재현율·F1과 샘플 수**를 한 번에 보여줍니다. 각 클래스의 행은 그 클래스를 관심 대상으로 놓고, 나머지 클래스를 모두 묶어 비교한 결과입니다.[^report]

원문의 세 클래스 출력 예제는 수치상 맞습니다. 아래에서는 그 출력을 실제로 재현할 수 있도록 입력을 명시했습니다. 1장의 변수와 섞이지 않도록 이름도 구분합니다.

```python
from sklearn.metrics import classification_report

report_y_true = [0, 1, 2, 2, 2]
report_y_pred = [0, 0, 2, 2, 1]

print(classification_report(
    report_y_true,
    report_y_pred,
    labels=[0, 1, 2],
    target_names=["class 0", "class 1", "class 2"],
    digits=2,
    zero_division=0,
))
```

출력:

```text
              precision    recall  f1-score   support

     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3

    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
```

### 5.1. support는 예측 개수가 아니라 실제 개수다

`class 2`의 `support=3`은 실제 정답이 클래스 2인 샘플이 3개라는 뜻입니다. `sample_weight`를 사용하지 않은 이 예제에서 support는 단순 개수입니다. 샘플 가중치를 사용하면 가중치 합으로 해석합니다.[^prfs]

`class 1`의 재현율이 0이므로 이 클래스의 실제 샘플은 하나도 맞히지 못했습니다. 전체 정확도 0.60만 보았다면 놓치기 쉬운 정보입니다.

### 5.2. macro avg: 모든 클래스에 같은 비중을 준다

클래스가 $K$개이고 클래스별 F1이 $F_{1,k}$일 때, scikit-learn의 macro F1은 다음과 같습니다.[^prfs]

$$
\mathrm{Macro\ F1}=\frac{1}{K}\sum_{k=1}^{K}F_{1,k}
$$

예제에서는 반올림 전 클래스별 F1을 평균 내면 약 0.49입니다. 샘플이 적은 클래스도 평균에서 같은 비중을 받습니다. 단, 샘플이 매우 적은 클래스의 지표는 소수의 예측 변화에도 크게 달라질 수 있으므로 support도 함께 봅니다.

> **주의:** macro F1은 **클래스별 F1을 평균 낸 값**입니다. macro precision과 macro recall을 먼저 구한 다음 그 둘의 조화 평균을 낸 값이 아닙니다.[^prfs]

### 5.3. weighted avg: 실제 클래스 개수만큼 비중을 준다

클래스 $k$의 실제 샘플 수를 $n_k$, 전체 수를 $N$이라고 하면 다음과 같습니다.[^prfs]

$$
\mathrm{Weighted\ F1}=\sum_{k=1}^{K}\frac{n_k}{N}F_{1,k}
$$

예제에서 클래스 2의 비중은 $3/5$이므로, weighted F1에는 이 클래스의 결과가 더 크게 반영됩니다. **weighted 평균이 높다고 모든 클래스가 잘 분류되는 것은 아닙니다.**[^report]

### 5.4. micro avg가 보이지 않는 이유

micro 평균은 클래스별 TP·FP·FN을 먼저 모두 합쳐 계산합니다. **한 샘플에 정답 클래스와 예측 클래스가 각각 하나씩 있고, 모든 클래스를 포함해 평가하면** 다음이 성립합니다.[^report][^recall]

$$
\mathrm{Micro\ Precision}
=\mathrm{Micro\ Recall}
=\mathrm{Micro\ F1}
=\mathrm{Accuracy}
$$

이 조건에서는 weighted recall도 accuracy와 같습니다. 그래서 일반적인 단일 레이블 분류 보고서는 중복되는 micro 평균을 생략합니다. 여러 레이블을 동시에 붙이는 **다중 레이블 분류**나 일부 클래스만 평가하는 경우에는 이 등식들을 그대로 적용하면 안 됩니다.[^report][^recall]

---

## 6. ROC Curve와 AUC

1~5장은 주로 **최종 판정 `y_pred`**를 평가했습니다. 이제는 판정을 내리기 전의 **점수 `y_score`**를 살펴봅니다.

### 6.1. ROC 곡선: 판정 기준을 바꿔 가며 놓침과 오탐을 살펴본다

ROC(Receiver Operating Characteristic) 곡선은 임계값을 바꿀 때 다음 두 값이 어떻게 달라지는지 나타냅니다.[^roc]

$$
\mathrm{TPR}=\frac{TP}{TP+FN}=\mathrm{Recall}
$$

$$
\mathrm{FPR}=\frac{FP}{FP+TN}=1-\mathrm{Specificity}
$$

**가로축은 FPR**, **세로축은 TPR**입니다. 같은 오탐률에서 양성을 더 많이 찾을수록, 또는 같은 재현율에서 오탐률이 낮을수록 좋습니다. 따라서 왼쪽 위에 가까운 지점이 바람직합니다.

메일 예제의 현재 판정 기준은 $(\mathrm{FPR},\mathrm{TPR})=(10/80,15/20)=(0.125,0.75)$라는 점 하나에 해당합니다.

> **FPR과 $1-\mathrm{Precision}$은 다릅니다.** FPR의 분모는 실제 음성 전체이고, $1-\mathrm{Precision}=FP/(TP+FP)$의 분모는 양성으로 예측한 전체입니다. 메일 예제에서는 각각 0.125와 0.4로 다릅니다.

### 6.2. ROC-AUC: 양성에 더 높은 점수를 주는 능력

AUC는 일반적으로 “곡선 아래 면적”을 뜻합니다. 어떤 곡선인지 분명히 하기 위해 **ROC-AUC** 또는 **AUROC**라고 쓰겠습니다.

**ROC-AUC의 범위는 0부터 1까지입니다. 원문의 “0.5에서 1.0 사이”는 잘못된 설명입니다.** 0.5는 하한이 아니라 무작위 순위의 기대 성능입니다.[^auc][^evaluation]

| ROC-AUC | 해석 |
|---|---|
| 1.0 | 평가한 데이터에서 모든 양성의 점수가 모든 음성보다 높음 |
| 0.5 | 무작위 순위의 기대값과 같음. 이 값 하나만으로 점수가 완전히 무정보라고 단정할 수는 없음 |
| 0.5 미만 | 양성·음성의 순서가 평균적으로 반대인 경향. 점수 방향, 레이블 지정, 표본 변동 등을 점검 |
| 0.0 | 평가한 데이터에서 모든 양성의 점수가 모든 음성보다 낮음 |

직관적으로는 **양성 하나와 음성 하나를 뽑았을 때 양성에 더 높은 점수를 줄 확률**입니다. 동점은 반만 맞힌 것으로 처리합니다.[^evaluation]

$$
\mathrm{AUROC}
=\Pr(s_+>s_-)+\frac12\Pr(s_+=s_-)
$$

여기서 $s_+$는 양성 샘플의 점수, $s_-$는 음성 샘플의 점수입니다. 따라서 ROC-AUC가 0.9라고 해서 “90%를 올바르게 분류했다”거나 “예측 확률이 90% 정확하다”는 뜻은 아닙니다.[^evaluation][^calibration]

### 6.3. ROC-AUC를 계산할 때는 점수를 넣는다

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 1장의 메일 100개에 대한 설명용 점수입니다.
# 앞의 스팸 20개: 15개는 0.9, 놓친 5개는 0.4
# 뒤의 정상 80개: 오탐 10개는 0.8, 나머지 70개는 0.1
# 이 값들은 확률 보정이 검증된 확률이 아니라 설명용 점수입니다.
y_score = np.array([0.9] * 15 + [0.4] * 5 + [0.8] * 10 + [0.1] * 70)
assert np.array_equal((y_score >= 0.5).astype(int), y_pred)

if np.unique(y_true).size != 2:
    raise ValueError("ROC-AUC를 계산하려면 실제 양성과 음성이 모두 필요합니다.")

fpr, tpr, roc_thresholds = roc_curve(y_true, y_score, pos_label=1)
roc_auc = roc_auc_score(y_true, y_score)
print(f"ROC-AUC: {roc_auc:.4f}")
# ROC-AUC: 0.9688

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random-ranking reference")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()
```

같은 메일 예제의 Accuracy는 0.85인데 ROC-AUC는 약 0.9688입니다. 모순이 아닙니다. 앞의 값은 **정해진 임계값에서의 판정**, 뒤의 값은 **점수의 순위**를 평가하기 때문입니다.

실제 학습 모델에서는 `model.classes_`의 순서를 확인해 **관심 있는 양성 클래스의 점수 열**을 골라야 합니다. `predict_proba(X)[:, 1]`은 두 번째 열이지, 아무 설정에서나 자동으로 원하는 양성 클래스가 되는 열은 아닙니다. 7장에서는 레이블 `1`에 해당하는 열을 직접 찾습니다.[^auc]

0·1로 된 `y_pred`를 ROC-AUC에 넣는 계산도 가능하지만, 점수 순위의 많은 정보가 사라집니다. 보통의 ROC 평가는 임계값 적용 전 점수를 사용합니다. 실제 정답에 한 클래스만 있으면 ROC-AUC는 정의되지 않으므로, 이를 임의로 0이나 0.5로 채우지 않습니다.[^roc][^auc]

### 6.4. 임계값을 바꾸면 모든 지표가 좋아질까?

아닙니다. 고정된 점수와 데이터에서 “점수가 임계값 이상이면 양성”이라고 할 때, 임계값을 낮추면 양성 예측 집합이 커집니다. 따라서 **재현율과 FPR은 감소하지 않습니다.** 하지만 추가된 샘플에 진짜 양성이 얼마나 섞여 있는지에 따라 **정밀도는 내려갈 수도, 올라갈 수도 있습니다.**

다음은 정밀도가 단조롭게 변하지 않는 직접적인 반례입니다. 점수가 0.9, 0.8, 0.7인 세 샘플의 정답이 차례로 1, 0, 1이라고 합시다.

| 임계값: 이상이면 양성 | Precision | Recall |
|---|---:|---:|
| 0.7 | 2/3 | 1.0 |
| 0.8 | 1/2 | 0.5 |
| 0.9 | 1/1 | 0.5 |

임계값을 0.7에서 0.8로 **높였는데 정밀도가 오히려 낮아집니다.** 따라서 “임계값을 올리면 정밀도가 반드시 올라간다”는 설명은 정확하지 않습니다.

**임계값은 검증 데이터에서 선택하고, 테스트 데이터에서는 고정해야 합니다.** 테스트 데이터에서 가장 좋은 F1을 주는 임계값을 고른 뒤 그 F1을 최종 성능으로 보고하면, 평가할 데이터를 선택 과정에 사용한 것이 됩니다.[^threshold][^pitfalls]

또한 임계값만 바꾸고 원래 점수는 그대로 두면 ROC-AUC나 AP는 바뀌지 않습니다. 달라지는 것은 최종 판정에 기반한 정밀도·재현율·F1 등입니다.[^threshold][^ap]

### 6.5. 새로 보강: PR 곡선과 Average Precision

PR(Precision–Recall) 곡선은 **가로축에 재현율**, **세로축에 정밀도**를 놓고 임계값 변화에 따른 결과를 보여줍니다. “얼마나 많이 찾으면서, 찾은 것의 정확성을 유지하는가?”를 살펴보는 데 유용합니다.[^prcurve]

특히 양성이 드문 문제에서는 정밀도가 실제 알림이나 검토 부담을 이해하는 데 중요할 수 있습니다. 다만 **“클래스가 불균형이면 PR-AUC가 ROC-AUC보다 항상 더 좋은 평가 지표다”라고 일반화할 수는 없습니다.**[^mcdermott]

#### 왜 같은 FPR에서도 정밀도가 달라질까?

양성 비율을 $\pi$라고 하면, 정의에서 다음 관계를 유도할 수 있습니다.

$$
\mathrm{Precision}
=\frac{\pi\,\mathrm{TPR}}
{\pi\,\mathrm{TPR}+(1-\pi)\,\mathrm{FPR}}
$$

예를 들어 10,000개 중 실제 양성이 100개이고 TPR이 80%, FPR이 5%라고 합시다. 그러면 TP는 80개지만 FP는 $9,900\times0.05=495$개입니다. 정밀도는 $80/(80+495)\approx13.9\%$입니다.

여기서 FPR 계산이 틀린 것이 아닙니다. **작은 비율의 오탐도 음성의 전체 개수가 많으면 큰 수가 될 수 있다**는 뜻입니다.

클래스별 점수 분포를 그대로 둔 채 양성 비율만 바꾸는 이상적인 조건에서는 모집단 ROC 곡선은 유지되지만 PR 곡선은 달라집니다. 반면 실제 데이터나 학습 과정이 바뀌어 클래스별 점수 분포까지 변하면 ROC-AUC도 바뀔 수 있습니다. “ROC-AUC는 어떤 불균형 변화에도 영향받지 않는다”는 설명 역시 지나친 일반화입니다.[^mcdermott][^mihelich]

#### AP와 사다리꼴 PR-AUC는 같은 계산이 아니다

`average_precision_score()`는 **Average Precision(AP)**을 계산합니다. 재현율을 증가시키는 순서로 곡선의 지점을 놓았을 때 다음과 같이 정밀도를 가중 평균 냅니다.[^ap]

$$
\mathrm{AP}=\sum_n(R_n-R_{n-1})P_n
$$

쉽게 말해 **새로 찾아낸 양성의 비중만큼, 그 지점의 정밀도에 점수를 주는 방식**입니다. 반면 `auc(recall, precision)`은 점들을 직선으로 잇고 사다리꼴 면적을 계산합니다. 보간 방식이 달라 두 값은 일반적으로 같지 않습니다.[^ap]

따라서 보고서에는 막연히 “PR-AUC”라고만 쓰기보다 **“AP, scikit-learn의 `average_precision_score`로 계산”**처럼 계산 방법을 밝혀 두는 편이 명확합니다.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_true, y_score, pos_label=1
)
ap = average_precision_score(y_true, y_score, pos_label=1)
positive_rate = float(np.mean(y_true == 1))
print(f"Average precision (AP): {ap:.4f}")
# Average precision (AP): 0.9167

plt.figure(figsize=(6, 5))
plt.step(pr_recall, pr_precision, where="post", label=f"AP = {ap:.4f}")
plt.axhline(
    positive_rate,
    linestyle="--",
    label=f"Positive-rate reference = {positive_rate:.2f}",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.show()
```

**PR의 무정보 모집단 기준선은 양성 비율 $\pi$입니다.** 양성이 1%인 문제에서 기준 정밀도는 0.5가 아니라 0.01입니다. 단, 유한한 표본에서 무작위로 점수를 매겨 계산한 AP가 매번 정확히 $\pi$가 되는 것은 아닙니다. 양성 비율이 다른 데이터셋의 AP를 숫자만으로 직접 비교하는 것도 주의해야 합니다.[^prcurve][^mcdermott]

#### 최근 연구는 어떻게 해석하면 좋을까?

Mihelich 등의 **ICML 2024** 연구는 높은 ROC-AUC가 높은 PR-AUC를 보장하지 않음을 이론적으로 분석했습니다. McDermott 등의 **NeurIPS 2024** 연구는 불균형만을 근거로 PR-AUC가 언제나 우월하다고 주장할 수 없음을 보여주었습니다. 두 결과를 함께 읽으면, 특정 지표 하나를 무조건 고르기보다 **평가 목적과 운영 조건을 먼저 정해야 한다**는 점을 이해할 수 있습니다.[^mihelich][^mcdermott]

**이 가이드의 적용 제안:** 양성 탐지가 중요한 문제에서는 양성 비율과 ROC-AUC, PR 곡선/AP를 함께 확인하고, 실제 사용할 임계값에서 정밀도·재현율·오탐 개수도 보고합니다. 이것은 위 연구들을 바탕으로 한 실무적 제안이지, 모든 업무에 같은 지표 묶음이 최적이라는 정리는 아닙니다.

### 6.6. 새로 보강: 순위를 잘 매기는 것과 확률을 잘 예측하는 것은 다르다

“양성을 음성보다 앞에 놓는 능력”과 “0.8이라고 말한 예측이 실제로 약 80%의 빈도로 맞는가”는 다릅니다. 후자를 **확률 보정(calibration)**이라고 합니다. 이진 분류에서 양성 확률을 약 0.8로 예측한 샘플들을 모았을 때, 그중 약 80%가 실제 양성이면 그 구간에서 잘 보정된 것입니다.[^calibration]

확률을 의사결정에 사용할 때는 **보정 곡선(calibration curve)**을 확인하고, 확률 예측 전체의 품질을 평가하는 **Brier score**나 **Log loss**도 함께 볼 수 있습니다. 두 손실 모두 작을수록 좋습니다.[^calibration][^brier]

이진 분류의 Brier score는 추정 확률 $p_i$와 실제 정답 $y_i\in\{0,1\}$의 차이를 제곱해 평균 낸 값입니다.

$$
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
$$

**Brier score나 Log loss는 순수한 보정 오차만 재는 지표가 아닙니다.** 보정뿐 아니라 예측의 구별 능력 등도 반영되므로, 손실 하나가 작다는 이유만으로 확률 보정이 더 좋다고 단정하지 않습니다. 확률 보정 모델을 추가로 학습할 때도 테스트 데이터를 사용하면 안 됩니다.[^calibration]

Flores 등의 **NeurIPS 2025** 연구는 임상 의사결정의 맥락에서 확률 보정, 양성 비율 변화, 서로 다른 오류 비용을 함께 고려하는 평가 방식을 제안합니다. 이 결과가 모든 분야에 하나의 새 지표를 쓰라는 뜻은 아니지만, **Accuracy나 ROC-AUC만으로 의사결정의 적절성을 판단하기 어렵다**는 점을 구체화합니다.[^flores]

또한 Sameni의 **2026년 4월 14일 개정 프리프린트**는 ROC·PR 곡선의 기하학적 관계와 운영 지점 선택을 분석하며, 양성으로 판정한 사례를 실제로 처리할 수 있는 수의 제약도 다룹니다. 초보자는 “높은 점수뿐 아니라, 하루에 검토할 수 있는 알림 수 같은 제약도 임계값 선택에 중요하다”는 점을 가져가면 됩니다. 이 문헌은 동료심사 게재를 확인한 연구와 구분해 참고합니다.[^sameni]

### 6.7. 다중 클래스에서는 무엇이 달라질까?

세 클래스 이상인 다중 클래스 분류에서도 Classification Report는 바로 사용할 수 있습니다. 반면 scikit-learn의 `roc_curve()`는 이진 분류용이므로, 클래스별로 “이 클래스 대 나머지”인 **One-vs-Rest(OvR)** 문제를 만들어 곡선을 그립니다.[^roc]

`roc_auc_score()`의 다중 클래스 계산에서는 `multi_class="ovr"` 또는 `"ovo"`를 명시하고, `average="macro"`인지 `"weighted"`인지도 밝혀야 합니다. 입력은 클래스 순서가 일치하고 각 행의 합이 1인 클래스별 확률 행렬입니다. 특히 OvR의 “나머지 클래스” 구성이 달라지면 macro 평균이라도 ROC-AUC가 달라질 수 있습니다.[^auc]

---

## 7. 실전 예시: 타이타닉 생존 예측

이 예제는 생존 여부를 예측하며, **생존 `1`을 양성, 사망 `0`을 음성**으로 정합니다. 여기서 생존 여부는 학습용 분류 문제의 정답이지, 개인에 대한 실제 의사결정을 위한 모델을 만든다는 뜻은 아닙니다.

### 7.1. 원문 코드에서 고쳐야 할 부분

원문은 `Survived`를 제외한 열을 그대로 `LogisticRegression`에 넣었습니다. 일반적인 원본 타이타닉 표에는 문자열과 결측치가 있어, 이런 입력을 그대로 학습에 사용할 수 없습니다. scikit-learn의 공식 타이타닉 예제도 숫자형과 범주형 열을 나누어 전처리합니다.[^mixed]

아래 코드는 **결측치 채우기, 범주를 숫자로 바꾸기, 숫자 크기 조정, 모델 학습**을 하나의 `Pipeline`으로 묶습니다. 이렇게 하면 전처리에 필요한 중앙값이나 평균 등도 학습 데이터에서만 배울 수 있습니다.[^mixed][^pitfalls]

또한 데이터를 다음처럼 나눕니다. 이 비율은 설명을 위한 선택이며 모든 문제의 정답은 아닙니다.

| 데이터 | 예제의 비중 | 용도 |
|---|---:|---|
| 학습 데이터 | 약 60% | 전처리 규칙과 모델 학습 |
| 검증 데이터 | 약 20% | 후보 임계값 중 하나 선택 |
| 테스트 데이터 | 약 20% | 선택을 끝낸 뒤 최종 평가 |

이는 독립적인 샘플을 가정한 입문 예제입니다. 같은 사람·가족·기관에서 나온 샘플이 강하게 관련되어 있거나 시간 순서가 중요한 실제 문제라면, 무작위 분할 대신 그룹 또는 시간 기준 분할을 검토해야 합니다. `stratify`는 클래스 비율을 비슷하게 유지하는 기능이지, 샘플 간 독립성을 보장하는 기능은 아닙니다.[^cv]

### 7.2. 필요한 파일

실행하는 작업 폴더에 정답 열 `Survived`가 포함된 `titanic_train.csv`를 준비합니다. 필요한 열은 `Survived`, `Age`, `Fare`, `SibSp`, `Parch`, `Pclass`, `Sex`, `Embarked`입니다. `Survived`가 없는 제출용 테스트 파일은 이 예제의 입력이 아닙니다.

**첨부된 원문에는 이 CSV가 포함되어 있지 않습니다. 따라서 아래 코드는 실제 타이타닉 성능 수치를 미리 제시하지 않습니다.** 데이터 파일이 없거나 필요한 열이 빠지면 명확한 오류 메시지를 냅니다.

### 7.3. 전처리부터 최종 평가까지

아래 블록 하나를 복사해 실행할 수 있습니다.

```python
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1) 데이터 읽기와 기본 확인
csv_path = Path("titanic_train.csv")
if not csv_path.is_file():
    raise FileNotFoundError(
        f"파일을 찾을 수 없습니다: {csv_path.resolve()}\n"
        "Survived 열이 포함된 titanic_train.csv를 작업 폴더에 넣으세요."
    )

df = pd.read_csv(csv_path)
numeric_features = ["Age", "Fare", "SibSp", "Parch"]
categorical_features = ["Pclass", "Sex", "Embarked"]
features = numeric_features + categorical_features

missing_columns = sorted(set(features + ["Survived"]) - set(df.columns))
if missing_columns:
    raise ValueError(f"필요한 열이 없습니다: {missing_columns}")

# 입력 특성의 결측치와 달리, 정답 레이블은 임의로 채우지 않습니다.
y = pd.to_numeric(df["Survived"], errors="raise")
if y.isna().any() or set(y.unique()) != {0, 1}:
    raise ValueError("Survived에는 결측치 없이 0과 1이 모두 있어야 합니다.")
y = y.astype(int)
if y.value_counts().min() < 5:
    raise ValueError("이 3분할 예제에는 각 클래스에 최소 5개 샘플이 필요합니다.")

X = df[features].copy()
for column in numeric_features:
    X[column] = pd.to_numeric(X[column], errors="raise")

# 2) 학습/검증/테스트 분리: 대략 60% / 20% / 20%
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=11
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.25, stratify=y_trainval, random_state=11
)
for name, labels in [("학습", y_train), ("검증", y_val), ("테스트", y_test)]:
    if labels.nunique() != 2:
        raise ValueError(f"{name} 데이터에 두 클래스가 모두 필요합니다.")

# 3) 숫자형: 중앙값으로 결측치를 채우고, 크기를 표준화합니다.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scaler", StandardScaler()),
])

# 범주형: 최빈값으로 결측치를 채우고, 범주별 0/1 열로 바꿉니다.
# Pclass도 여기서는 1/2/3이라는 범주로 취급합니다.
# 처음 보는 범주는 오류 대신 해당 범주의 인코딩을 모두 0으로 처리합니다.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# 전처리와 모델은 학습 데이터에서만 학습합니다.
model.fit(X_train, y_train)
positive_column = list(model.classes_).index(1)

# 4) 검증 데이터에서만 임계값 선택
# 예제의 선택 목표: 생존 클래스의 F1. 실제 업무의 목표는 별도로 정합니다.
val_score = model.predict_proba(X_val)[:, positive_column]
threshold_candidates = np.linspace(0.0, 1.0, 101)
validation_f1 = np.array([
    f1_score(y_val, (val_score >= threshold).astype(int), zero_division=0)
    for threshold in threshold_candidates
])

# 후보는 오름차순이므로 동률이면 가장 낮은 임계값을 선택합니다.
best_index = int(np.argmax(validation_f1))
chosen_threshold = float(threshold_candidates[best_index])
print(f"선택한 임계값: {chosen_threshold:.2f}")
print(f"선택에 사용한 검증 F1: {validation_f1[best_index]:.4f}")

# 5) 모델과 임계값을 고정한 뒤 테스트 데이터 평가
# 여기서 모델을 다시 학습하거나 테스트 결과로 임계값을 바꾸지 않습니다.
test_proba = model.predict_proba(X_test)
test_score = test_proba[:, positive_column]
test_pred = (test_score >= chosen_threshold).astype(int)

print(f"\n학습/검증/테스트 크기: {len(y_train)}/{len(y_val)}/{len(y_test)}")
print(f"테스트의 실제 생존 비율: {y_test.mean():.4f}")
print("\nConfusion Matrix: 행=실제, 열=예측, 순서=[사망, 생존]")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

print("\nClassification Report")
print(classification_report(
    y_test, test_pred,
    labels=[0, 1],
    target_names=["Dead", "Survived"],
    digits=4,
    zero_division=0,
))

print(f"Accuracy:          {accuracy_score(y_test, test_pred):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, test_pred):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_test, test_score):.4f}")
print(f"AP:                {average_precision_score(y_test, test_score):.4f}")
print(f"Brier score:       {brier_score_loss(y_test, test_score):.4f}")
print(f"Log loss:          {log_loss(y_test, test_proba, labels=model.classes_):.4f}")

# 학습 데이터의 다수 클래스로만 예측하는 단순 기준 모델과도 비교합니다.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
print(f"\n다수 클래스 기준 Accuracy: {accuracy_score(y_test, baseline_pred):.4f}")
```

### 7.4. 코드에서 꼭 이해할 점

**전처리도 학습입니다.** 전체 데이터의 평균이나 중앙값으로 결측치를 채운 후 분할하면, 테스트 데이터의 정보가 학습 과정으로 새어 들어갈 수 있습니다. 이것을 **데이터 누출(data leakage)**이라고 합니다. `Pipeline`을 분할 후의 학습 데이터에 `fit()`하는 이유입니다.[^pitfalls]

**검증에서 선택한 최고 F1은 최종 성능이 아닙니다.** 이 코드는 0.01 간격의 후보 중 검증 F1이 가장 높은 임계값을 선택합니다. 모든 실수 임계값에 대한 이론적 최적값을 찾거나, 미래 데이터에서도 최고 성능임을 보장하는 코드는 아닙니다.[^threshold]

**F1 최대화는 예시 목표입니다.** 실제 업무라면 “재현율을 일정 수준 이상 유지하면서 오탐을 줄이기”, “검토 가능한 알림 수를 넘지 않기” 등으로 목표를 정할 수 있습니다. 모델과 임계값을 선택할 때 쓴 목표, 양성 클래스, 클래스 비율을 함께 기록해야 합니다.[^flores][^sameni]

**모델을 다시 학습하면 임계값도 재검토해야 합니다.** 학습·검증 데이터를 합쳐 재학습하면 점수 분포가 달라질 수 있으므로, 기존 임계값이 그대로 최적이라고 보장할 수 없습니다. 이 예제는 흐름을 명확히 하기 위해 재학습하지 않습니다. 더 체계적인 임계값 선택에는 `TunedThresholdClassifierCV` 같은 교차검증 기반 기능도 사용할 수 있지만, 테스트 데이터는 여전히 마지막 평가용으로 분리해야 합니다.[^threshold][^cv]

**한 번의 분할로 나온 소수점 차이를 과신하지 않습니다.** 평가 데이터와 분할이 바뀌면 결과도 바뀔 수 있습니다. 중요한 비교에서는 샘플의 독립성·그룹 구조에 맞는 교차검증이나 재표본추출로 변동성을 살펴보고, 테스트 데이터 크기와 클래스별 support를 함께 보고합니다. 교차검증 점수의 표준편차 자체를 곧바로 95% 신뢰구간이라고 부르지는 않습니다.[^cv]

마지막으로, **정답이 확인된 샘플만 평가해도 편향이 생길 수 있습니다.** 예를 들어 일부 알림만 사람이 조사했다면, 조사된 사례가 전체를 대표하는지 따져야 합니다. Dervovic와 Cashmore의 **AISTATS 2025** 연구는 평가 시 정답 레이블이 누락되는 문제와 그에 따른 편향을 다룹니다. 지표 공식만 맞는다고 평가 전체가 자동으로 타당해지는 것은 아닙니다.[^missinglabels]

---

## 마무리: 숫자를 읽을 때 함께 물어볼 질문

| 지표 | 핵심 질문 | 함께 확인할 것 |
|---|---|---|
| Confusion Matrix | 무엇을 무엇으로 잘못 예측했는가? | 축, 클래스 순서, 실제 오류 개수 |
| Accuracy | 전체에서 얼마나 맞혔는가? | 클래스 비율과 단순 기준 모델 |
| Precision | 찾았다고 한 것 중 진짜는 얼마나 되는가? | 양성 예측 수와 오탐 개수 |
| Recall | 실제 양성 중 얼마나 찾았는가? | 놓친 양성 수 |
| F1 | 정밀도와 재현율을 함께 요약하면 어떤가? | TN과 실제 비용은 모두 반영하지 않음 |
| Macro / weighted 평균 | 클래스들을 어떤 비중으로 요약했는가? | 클래스별 원래 지표와 support |
| ROC-AUC | 양성에 더 높은 점수를 주는가? | 실제 사용할 임계값에서의 성능 |
| PR 곡선 / AP | 양성을 찾으면서 정밀도를 유지하는가? | 양성 비율과 AP 계산 방식 |
| 보정 곡선 / 확률 손실 | 예측한 확률은 관측 결과와 잘 맞는가? | Brier·Log loss는 순수 보정 오차가 아님 |

위 표는 앞 절들의 요약입니다. **“누구를 양성으로 정했는가, 어떤 실수가 더 중요한가, 어디에서 임계값을 골랐는가”를 먼저 밝힌 뒤 여러 지표를 함께 읽는 것**이 핵심입니다.

### 이해 확인

**질문 1.** 양성이 1개, 음성이 99개일 때 모두 음성으로 예측하면 정확도와 양성 재현율은 얼마일까요?  
**답:** 정확도는 99%, 양성 재현율은 0%입니다. 양성 정밀도는 분모가 0이라 정의되지 않으며, 코드에서 0으로 출력할지는 별도의 설정입니다.

**질문 2.** 점수를 그대로 두고 임계값만 바꾸면 ROC-AUC가 달라질까요?  
**답:** 달라지지 않습니다. 점수의 순위는 그대로이기 때문입니다. 최종 예측에 기반한 정밀도·재현율·F1은 달라질 수 있습니다.

**질문 3.** macro F1은 macro precision과 macro recall의 조화 평균일까요?  
**답:** 아닙니다. scikit-learn에서는 클래스별 F1을 먼저 구한 다음 단순 평균 냅니다.

---

## 개정 내역과 검증 범위

| 구분 | 수정·보강한 내용 |
|---|---|
| 명확한 이론 오류 수정 | ROC-AUC의 범위를 0.5~1.0에서 **0~1**로 수정하고, 0.5의 의미를 구분 |
| 조건과 한계 명시 | 혼동 행렬의 축·레이블 순서, 양성 클래스, 0으로 나누는 경우, F1의 한계 |
| 초보자 설명 보강 | 동일한 메일 100개로 Accuracy·Precision·Recall·F1을 계산하고 실제 출력을 제시 |
| 기존 내용 재현성 개선 | 원문의 올바른 3클래스 Classification Report 출력은 유지하고 입력 배열을 추가 |
| 새로 추가한 평가 내용 | Balanced Accuracy, micro 평균, PR/AP, 확률 보정, 테스트 데이터와 검증 데이터의 구분 |
| 최근 연구 반영 | ROC-AUC와 PR-AUC의 관계, 오류 비용·양성 비율 변화, 운영 제약, 누락된 평가 레이블 |
| 실전 코드 수정 | 문자열·결측치 처리, Pipeline, 층화 3분할, 검증 전용 임계값 선택, 입력 검증 |

**실행 검증:** 1~6장의 예제 코드와 출력 수치는 Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, Matplotlib 3.10.8 환경에서 확인했습니다. 7장은 실제 타이타닉 CSV가 제공되지 않아, 동일한 열 구조와 결측치를 가진 합성 데이터로 코드 실행을 점검했습니다. 이 점검은 실제 타이타닉 데이터에서의 성능 검증을 의미하지 않습니다.

**문헌 범위:** 아래는 이 개정에 직접 사용한 공식 문서와 연구입니다. 모든 최신 연구를 빠짐없이 검토했다는 뜻은 아닙니다. 원문의 개인 블로그 링크는 수정된 이론과 구현의 근거로 사용하지 않았습니다.

## 참고 문헌

### 공식 문서: 정의와 구현

[^evaluation]: scikit-learn. *Metrics and scoring: quantifying the quality of predictions*. 공식 사용자 가이드. https://scikit-learn.org/stable/modules/model_evaluation.html

[^cm]: scikit-learn. *confusion_matrix*. 축의 의미와 `labels` 순서. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

[^basics]: Google Machine Learning Crash Course. *Classification: Accuracy, recall, precision, and related metrics*. 공식 교육 문서, 페이지 갱신일 2026-01-12. https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

[^balanced]: scikit-learn. *balanced_accuracy_score*. 클래스별 재현율의 평균과 조정 옵션. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html

[^f1]: scikit-learn. *f1_score*. F1의 정의와 0으로 나누는 경우. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

[^prfs]: scikit-learn. *precision_recall_fscore_support*. 평균 방식, support, F-beta, 정의되지 않는 지표. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

[^report]: scikit-learn. *classification_report*. 클래스별 보고서와 micro 평균 생략 조건. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

[^recall]: scikit-learn. *recall_score*. weighted recall과 accuracy의 관계. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

[^roc]: scikit-learn. *roc_curve*. 점수 입력과 이진 ROC 계산. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

[^auc]: scikit-learn. *roc_auc_score*. 이진·다중 클래스 ROC-AUC와 클래스 순서. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

[^ap]: scikit-learn. *average_precision_score*. AP와 사다리꼴 보간의 차이. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

[^prcurve]: scikit-learn. *Precision-Recall*. PR 곡선과 해석 예제. https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

[^threshold]: scikit-learn. *Tuning the decision threshold for class prediction*. 임계값 선택과 검증 분리. https://scikit-learn.org/stable/modules/classification_threshold.html

[^calibration]: scikit-learn. *Probability calibration*. 보정 곡선과 확률 손실 해석상의 한계. https://scikit-learn.org/stable/modules/calibration.html

[^brier]: scikit-learn. *brier_score_loss*. 이진 Brier score의 정의. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html

[^pitfalls]: scikit-learn. *Common pitfalls and recommended practices*. 전처리 일관성과 데이터 누출. https://scikit-learn.org/stable/common_pitfalls.html

[^mixed]: scikit-learn. *Column Transformer with Mixed Types*. 타이타닉 데이터의 숫자형·범주형 전처리 예제. https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

[^cv]: scikit-learn. *Cross-validation: evaluating estimator performance*. 평가 분리, 그룹·시계열 분할과 교차검증의 주의점. https://scikit-learn.org/stable/modules/cross_validation.html

### 동료심사 연구: 최근 해석과 평가 쟁점

[^mihelich]: Mihelich, M., Castagnos, F., & Dognin, C. (2024). *Interplay of ROC and Precision-Recall AUCs: Theoretical Limits and Practical Implications in Binary Classification*. ICML 2024, PMLR 235, 35639–35646. https://proceedings.mlr.press/v235/mihelich24a.html

[^mcdermott]: McDermott, M. B., Zhang, H., Hansen, L. H., Angelotti, G., & Gallifant, J. (2024). *A Closer Look at AUROC and AUPRC under Class Imbalance*. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/4df3510ad02a86d69dc32388d91606f8-Abstract-Conference.html

[^flores]: Flores, G., Smith, A. H., Fukuyama, J., & Wilson, A. C. (2025). *Aligning Evaluation with Clinical Priorities: Calibration, Label Shift, and Error Costs*. NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/10c456d2160517581a234dfde15a7505-Abstract-Conference.html

[^missinglabels]: Dervovic, D., & Cashmore, M. (2025). *Model Evaluation in the Dark: Robust Classifier Metrics with Missing Labels*. AISTATS 2025, PMLR 258, 1909–1917. https://proceedings.mlr.press/v258/dervovic25a.html

### 2026년 개정 프리프린트: 동료심사 문헌과 구분

[^sameni]: Sameni, R. *On the Geometry of Receiver Operating Characteristic and Precision-Recall Curves*. arXiv:2504.02169, v3, 2026-04-14 개정. 최초 공개는 2025-04-02이며, 이 개정에서 확인한 출처는 arXiv 프리프린트입니다. https://arxiv.org/abs/2504.02169v3
