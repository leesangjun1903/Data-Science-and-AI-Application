# Imbalanced Data 가이드: 표형(Tabular) 데이터 중심으로

> **핵심은 클래스 수를 똑같이 만드는 것이 아니라, 중요한 사건을 얼마나 잘 찾고 그 과정에서 어떤 실수를 하는지 올바르게 평가하는 것입니다.**

**검토 기준일: 2026년 10월 15일**  
**범위:** 거래 내역처럼 행과 열로 정리된 데이터의 지도학습, 특히 정답이 0과 1인 이진 분류.  
**원문과의 관계:** 기존의 평가 지표 → 샘플링 → 비용/가중치 → 임계값 → 파이프라인 → 딥러닝 흐름을 유지했습니다. 최신 연구로 달라지는 권고는 9절에, 원문의 구체적인 수정 사항은 12절에 구분했습니다. 별도로 표시한 논문 실험을 제외하면, 아래 거래 사례와 숫자는 이해를 돕기 위해 만든 예시입니다.

---

## 1. 핵심 요약

사기 거래가 전체의 1%라면, 모든 거래를 정상이라고 예측해도 정확도는 99%입니다. 하지만 이 모델은 사기를 한 건도 찾지 못합니다. 따라서 불균형 데이터에서는 **정확도 하나로 모델을 평가하면 안 됩니다.**

처음에는 다음 순서로 접근하세요.

1. **무엇을 잘해야 하는지 정합니다.** 사기를 놓치는 것이 더 문제인지, 정상 거래에 경보를 너무 많이 내는 것이 더 문제인지부터 결정합니다.
2. **데이터를 먼저 나누고 기본 모델을 만듭니다.** 불균형을 그대로 둔 모델을 기준선으로 삼습니다. 전처리와 샘플링은 훈련 데이터 안에서만 수행합니다.[^sk-pitfalls][^im-pitfalls]
3. **필요한 방법을 비교합니다.** 클래스 가중치, SMOTE 같은 재표본추출, 서로 다른 모델을 비교하고, 검증 데이터로 경보 기준인 임계값을 정합니다. 모든 기법을 처음부터 함께 쓰는 것이 정답은 아닙니다.[^climb][^threshold]

**기억할 문장:** “데이터를 늘리기 전에 평가를 바로잡고, 기법을 추가하기 전에 기본 모델과 비교한다.”

## 2. 왜 불균형 데이터가 문제인가

### 2.1 먼저 용어부터 이해하기

**표형 데이터**는 엑셀 표처럼 정리된 데이터입니다. 한 행은 거래 한 건, 열은 거래 금액·거래 시간 같은 **특성(feature)** 입니다. 맞혀야 할 사기 여부는 **정답(label)** 입니다.

**클래스(class)** 는 정답의 종류입니다. 정상 거래가 9,900건, 사기 거래가 100건이면 정상은 **다수 클래스**, 사기는 **소수 클래스**입니다. 이 글에서는 관심 사건인 사기를 **양성(1)**, 정상을 **음성(0)** 으로 정합니다. 여기서 양성은 “좋은 결과”라는 뜻이 아닙니다.

### 2.2 같은 99:1이라도 난이도는 다릅니다

불균형 자체와 함께 살펴봐야 할 것이 있습니다. 소수 클래스의 실제 관측 수가 너무 적은지, 두 클래스의 특징이 서로 겹치는지, 정답이 잘못 붙어 있는지입니다. CLIMB의 실험에서도 데이터 품질과 선택한 학습 방법이 중요한 영향을 보였습니다.[^climb]

예를 들어 `정상 990건 / 사기 10건`과 `정상 990,000건 / 사기 10,000건`은 비율이 같습니다. 하지만 후자에는 학습할 수 있는 실제 사기 사례가 훨씬 많습니다. **비율만 보고 난이도나 처방을 결정할 수는 없습니다.**

또한 드문 사례라고 모두 오류는 아닙니다. 사기 자체가 드문 패턴이므로, 이상하게 생겼다는 이유만으로 소수 클래스의 관측을 삭제하는 것은 신중해야 합니다. 먼저 실제 오류인지 확인하는 것을 이 가이드의 데이터 점검 원칙으로 삼습니다.

## 3. 올바른 평가 지표

### 3.1 혼동행렬: 맞힌 것과 틀린 것을 네 칸으로 나누기

다음은 10,000건 중 사기가 100건인 **설명용 예시**입니다.

| 실제 정답 / 모델의 판단 | 사기라고 경보 | 정상이라고 통과 |
|---|---:|---:|
| 실제 사기 100건 | TP = 80: 사기를 찾음 | FN = 20: 사기를 놓침 |
| 실제 정상 9,900건 | FP = 99: 잘못된 경보 | TN = 9,801: 정상을 통과시킴 |

`T/F`는 판단이 맞았는지(True/False), `P/N`은 모델이 양성/음성으로 판단했는지를 나타냅니다. 이 네 칸으로 정밀도, 재현율, F1 등을 계산합니다.[^metrics]

### 3.2 정밀도와 재현율: 서로 다른 두 질문

**정밀도(precision)** 는 “경보를 낸 거래 중 진짜 사기가 얼마나 되는가?”입니다.[^precision]

$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$

예시에서는 $80/(80+99)\approx44.7\%$입니다. 경보 179건을 확인하면 그중 80건이 실제 사기입니다.

**재현율(recall, 민감도)** 은 “실제 사기 중 얼마나 찾아냈는가?”입니다.[^recall]

$$
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

예시에서는 $80/(80+20)=80\%$입니다. 실제 사기 100건 중 80건을 찾았습니다.

두 지표는 같은 뜻이 아닙니다. 경보를 무조건 많이 내면 사기를 더 찾을 수 있지만, 정상 거래까지 경보에 포함될 수 있습니다. 특히 **재현율만 높이는 것**이 목표라면 모든 거래에 경보를 내는 무용한 해법도 가능합니다.

**F1**은 정밀도와 재현율을 조화평균으로 묶은 값입니다. 둘 중 하나만 높을 때는 높은 점수를 받기 어렵습니다.[^f1]

$$
F_1=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
=\frac{2TP}{2TP+FP+FN}
$$

예시에서는 약 $0.573$입니다. 다만 식에 TN이 없고, 실제 오류 비용도 들어 있지 않습니다. 따라서 **F1이 가장 높은 모델과 비용이 가장 낮은 모델은 다를 수 있습니다.** 이는 수식에서 직접 알 수 있는 한계입니다.

참고로 이 모델의 정확도는 $(80+9{,}801)/10{,}000=98.81\%$입니다. 모든 거래를 정상으로 판단한 모델의 99%보다 낮지만, 사기 80건을 찾아냈습니다. 어느 모델이 더 유용한지는 오류 비용과 업무 목적에 달려 있습니다.

### 3.3 ROC와 PR: 경보 기준을 바꾸면 성능이 어떻게 달라질까?

대부분의 분류 모델은 먼저 “사기일 것 같은 정도”를 점수로 출력합니다. **임계값(threshold)** 은 이 점수를 경보 여부로 바꾸는 기준입니다. 예를 들어 점수가 0.2 이상이면 경보를 내도록 정할 수 있습니다.[^threshold]

ROC와 PR 곡선은 임계값을 여러 값으로 바꾸면서 얻은 결과를 모은 것입니다.

| 지표 | 무엇을 보는가? | 함께 기억할 점 |
|---|---|---|
| ROC-AUC | 양성이 음성보다 높은 점수를 받도록 얼마나 잘 구분했는가 | 실제 경보량이나 특정 임계값의 성능을 직접 알려주지는 않음 |
| PR 곡선 | 재현율을 높일 때 정밀도가 어떻게 달라지는가 | 양성 비율에 영향을 받으므로 평가 데이터의 양성 비율도 기록 |
| AP, Average Precision | PR 관계를 하나의 수치로 요약 | `average_precision_score`의 결과이며 사다리꼴 PR 면적과 구분 |

ROC의 가로축은 $\mathrm{FPR}=FP/(FP+TN)$, 세로축은 재현율입니다. 위 예시의 FPR은 1%지만, 정상 거래가 많기 때문에 오경보가 99건입니다. **낮은 FPR도 상당한 업무량이 될 수 있습니다.** ROC-AUC 자체가 불균형 때문에 무효가 되는 것은 아닙니다.[^roc][^auroc-auprc]

PR 곡선 아래 면적은 보통 **PR-AUC 또는 AUPRC**라고 부릅니다. 그런데 scikit-learn의 **AP는 사다리꼴 적분으로 구한 면적과 같은 계산이 아닙니다.** AP는 재현율이 증가한 양을 가중치로 사용해 정밀도를 평균합니다.[^ap]

$$
\mathrm{AP}=\sum_n (R_n-R_{n-1})P_n
$$

여기서 $P_n$은 정밀도, $R_n$은 재현율이며, 재현율이 증가하는 순서로 생각하면 됩니다. 이 가이드의 코드는 결과를 모호한 `pr_auc`가 아니라 **`AP`** 로 표기합니다. AP와 ROC-AUC에는 0/1로 잘라낸 예측값이 아니라 **원래 예측 점수**를 넣습니다.[^ap][^roc]

> **“불균형 데이터에서는 PR-AUC가 ROC-AUC보다 언제나 낫다”는 규칙은 없습니다.** NeurIPS 2024 연구도 이 일반화를 반박합니다. 실제 목적에 맞는 지표를 정하고, 필요한 경우 두 관점을 함께 보세요.[^auroc-auprc]

**조금 더 알아보기 — 양성 비율이 바뀌면 정밀도도 바뀝니다.** 양성 비율을 $\pi$, 재현율을 TPR이라 하면 정의로부터 다음 식을 얻습니다.

$$
\mathrm{Precision}
=\frac{\pi\,\mathrm{TPR}}
{\pi\,\mathrm{TPR}+(1-\pi)\,\mathrm{FPR}}
$$

TPR과 FPR이 그대로여도 $\pi$가 작아지면 정밀도는 낮아질 수 있습니다. 따라서 양성 비율이 다른 데이터셋의 AP를 그대로 비교하거나, 테스트 데이터를 인위적으로 1:1로 만든 뒤 운영 성능이라고 해석하지 않습니다.[^im-pitfalls]

### 3.4 균형정확도, G-mean, IBA는 보조 지표로

**특이도(specificity)** 는 실제 정상 중 정상으로 통과시킨 비율인 $TN/(TN+FP)$입니다. **균형정확도(balanced accuracy)** 는 이진 분류에서 재현율과 특이도의 산술평균입니다. **G-mean**은 두 값의 기하평균입니다.[^metrics][^im-metrics]

$$
\mathrm{Balanced\ accuracy}=\frac{\mathrm{Recall}+\mathrm{Specificity}}{2}
$$

$$
G\text{-mean}=\sqrt{\mathrm{Recall}\times\mathrm{Specificity}}
$$

예시에서는 각각 $0.895$, 약 $0.890$입니다. 두 클래스를 어느 정도 고르게 맞히는지 살펴보는 데 유용하지만, 실제 오류 비용을 대신하지는 않습니다.

**IBA는 Index of Balanced Accuracy**입니다. 다른 성능 지표에 클래스 간 균형을 반영하는 보정 방식을 뜻하며, “지수함수를 사용하는 균형정확도”라는 뜻이 아닙니다. `classification_report_imbalanced`로 G-mean과 IBA 등을 확인할 수 있지만, 입문 단계에서는 정밀도·재현율·AP·혼동행렬부터 이해하면 충분합니다.[^im-metrics]

**실무용 보고의 제안:** 양성 개수와 비율, AP, ROC-AUC, 선택한 임계값의 정밀도·재현율, 경보 건수, 오류 비용을 함께 기록하세요. 지표가 많을수록 좋은 것이 아니라, 각 수치가 어떤 판단에 필요한지 명확해야 합니다.

## 4. 실무 전략 개요: 방법보다 실험 순서가 먼저입니다

### 4.1 훈련·검증·테스트의 역할을 분리하기

**훈련 데이터**는 모델이 공부하는 문제집, **검증 데이터**는 공부 방법과 경보 기준을 정하는 모의고사, **테스트 데이터**는 선택을 모두 끝낸 뒤 보는 최종 시험이라고 생각하면 됩니다. 테스트 결과를 보고 모델이나 임계값을 고르면 더 이상 독립적인 최종 시험이 아닙니다.[^sk-pitfalls]

```text
전체 데이터
  → 훈련 / 검증 / 테스트 분리
  → 훈련 데이터 내부 교차검증으로 모델과 설정 비교
  → 선택한 모델을 훈련 데이터에 학습
  → 검증 데이터로 임계값 선택
  → 모델과 임계값을 고정
  → 테스트 데이터로 최종 평가
```

**교차검증(CV)** 은 훈련 데이터를 여러 묶음인 **폴드(fold)** 로 나누어, 일부로 학습하고 나머지로 평가하는 일을 반복하는 방법입니다. 여기서 평가용 폴드는 모델 선택을 위한 것이며, 위에서 따로 떼어 놓은 최종 테스트와 다릅니다.[^cv]

독립적인 관측이라면 클래스 비율을 대략 유지하는 **계층화 분할(stratified split)** 을 쓸 수 있습니다. 미래 거래를 예측한다면 과거로 학습하고 미래로 평가하는 시간순 분할이 필요합니다. 같은 고객의 반복 관측이 있고 목표가 새 고객에 대한 일반화라면 고객 단위 그룹 분할을 고려합니다. 시간 조건과 그룹 조건이 모두 중요하면 둘 다 반영해야 하며, **무작위 계층화만으로 모든 누출이 해결되지는 않습니다.**[^cv]

### 4.2 전처리와 SMOTE는 어디에서 해야 할까?

숫자형 특성의 단위가 제각각이고 SMOTE를 쓴다면, 기본 순서는 다음과 같습니다.

```text
각 CV 훈련 폴드 안에서:
  결측값 처리 학습 → 스케일링 학습 → SMOTE → 분류기 학습

해당 CV 평가 폴드 / 별도 검증 / 최종 테스트에서는:
  훈련에서 정한 결측값 처리 적용 → 훈련에서 정한 스케일링 적용 → 예측
  ※ SMOTE는 수행하지 않음
```

**스케일링**은 열들의 수치적 크기를 맞추는 전처리입니다. 예를 들어 금액은 수백만 단위이고 거래 횟수는 한 자릿수라면, 그대로 거리를 계산할 때 금액이 훨씬 크게 작용할 수 있습니다. `StandardScaler`는 훈련 데이터의 평균과 표준편차로 값을 변환합니다. SMOTE는 이웃을 거리로 찾으므로, 이 예제에서는 **스케일링을 SMOTE보다 먼저** 합니다.[^scale][^smote-api]

결측값을 채울 중앙값, 스케일링 통계, 특성 선택 규칙처럼 **데이터에서 배우는 것**은 훈련 부분에서만 결정해야 합니다. 전체 데이터에서 SMOTE를 먼저 실행하고 나중에 분할하는 것은 금지합니다. 합성에 쓰인 원본과 이웃의 정보가 평가 쪽으로 섞일 수 있습니다.[^sk-pitfalls][^im-pitfalls]

### 4.3 처음 비교할 후보

이 가이드에서는 **원래 데이터로 학습한 로지스틱 회귀**, **가중치를 준 로지스틱 회귀**, **SMOTE를 쓴 로지스틱 회귀**, **재표본추출 없는 트리 기반 부스팅**을 비교 후보로 제안합니다. 부스팅은 여러 작은 모델을 순차적으로 결합하는 방식입니다.

이는 모든 문제에 대한 최적 조합이 아니라, 방법의 효과를 구분해 보는 출발점입니다. CLIMB의 결과를 특정 모델의 보편적 우승으로 해석하지 말고, 재표본추출 없는 모델과 앙상블도 비교에 포함하라는 근거로 읽는 편이 적절합니다.[^climb]

## 5. 샘플링 방법 한눈에 보기

**재표본추출(resampling)** 은 훈련 데이터를 뽑는 방식을 바꾸는 것입니다. 소수 클래스를 늘리는 것이 **오버샘플링**, 다수 클래스 일부를 줄이는 것이 **언더샘플링**입니다.[^over][^ensemble]

| 방법 | 쉬운 설명 | 주의할 점 |
|---|---|---|
| Random Oversampling | 기존 소수 사례를 다시 뽑아 더 자주 학습 | 새로운 실제 사례가 생기는 것은 아님 |
| SMOTE | 가까운 소수 사례 두 점 사이에 새 점 생성 | 두 점 사이가 실제로 같은 클래스라는 보장은 없음 |
| ADASYN | 주변에 다른 클래스가 많은 소수 사례에 더 많이 생성 | 잘못된 정답이나 이상치에도 집중할 수 있음 |
| Borderline-SMOTE | 경계 근처라고 판단한 소수 사례를 중심으로 생성 | 거리와 경계 판정에 민감 |
| Random Undersampling | 다수 사례 중 일부만 선택 | 유용한 정상 사례를 잃을 수 있음 |
| 균형 샘플링 앙상블 | 다르게 뽑은 데이터로 여러 모델을 학습해 결합 | 단일 샘플링과 별도 후보로 평가 |

방법의 원리와 구현은 imbalanced-learn 문서에 정리되어 있습니다. 아래 세 방법은 새로운 현장 데이터를 수집하는 것이 아니라 **기존 관측을 바탕으로 훈련용 사례를 만드는 방법**입니다.[^over][^ensemble]

### 5.1 SMOTE: 비슷한 소수 사례 사이를 채우기

SMOTE는 **Synthetic Minority Over-sampling Technique**의 약자입니다. 소수 클래스의 한 사례를 골라, 같은 클래스의 가까운 이웃과 그 사이의 점을 만듭니다.[^smote-paper][^smote-api]

$$
x_{\mathrm{new}}=x_i+\lambda(x_{\mathrm{near}}-x_i),
\qquad \lambda\sim U(0,1)
$$

$x_i$는 기준 사례, $x_{\mathrm{near}}$는 선택한 소수 클래스 이웃입니다. $\lambda$는 0과 1 사이에서 무작위로 뽑은 값으로, 두 점 사이에서 얼마나 이동할지를 정합니다.

**숫자 예시:** 설명을 위해 두 개의 연속형 특성만 사용하고, 소수 사례가 $(2,4)$와 $(6,8)$이라고 합시다. $\lambda=0.25$이면,

$$
(2,4)+0.25\{(6,8)-(2,4)\}=(3,5)
$$

새로운 점 $(3,5)$는 두 원래 점 사이에 있습니다. 즉, SMOTE는 없던 사실을 알아내는 것이 아니라 **가까운 사례 사이도 비슷한 사례일 것이라는 가정**을 사용하는 것입니다.

실제로 사용할 때는 두 가지를 확인하세요. `k_neighbors=5`이면 해당 훈련 폴드의 소수 클래스에 자기 자신을 포함해 최소 6개 사례가 있어야 합니다. 또한 합성 목표를 반드시 1:1로 할 필요는 없습니다. 이진 분류에서 `sampling_strategy=0.2`는 합성 후 **소수/다수 개수의 비율을 0.2로** 만든다는 뜻이며, 양성 비율 20%를 뜻하지 않습니다. 양성 비율은 $0.2/1.2\approx16.7\%$입니다.[^smote-api]

일반 SMOTE는 생성의 기준점을 무작위로 고릅니다. 따라서 모든 원본 사례에서 정확히 같은 개수의 합성 표본이 나온다고 설명하면 부정확합니다. 또 클래스가 겹치거나 서로 떨어진 소수 집단을 연결하면 부적절한 합성 사례가 생길 수 있습니다.[^over]

### 5.2 ADASYN: 주변이 더 어려운 곳에 많이 생성하기

ADASYN은 **Adaptive Synthetic Sampling**입니다. SMOTE처럼 새 점을 만들되, **어느 기준점에 몇 개를 배정할지**를 바꿉니다.[^adasyn]

먼저 소수 사례 $x_i$ 주변에서, 두 클래스를 모두 포함해 $K$개의 가까운 이웃을 찾습니다. 그중 다수 클래스의 비율을 $r_i$라고 합니다.

$$
r_i=\frac{x_i\text{의 }K\text{개 이웃 중 다수 클래스 개수}}{K}
$$

전체적으로 새 점을 $G$개 만들 계획이라면,

$$
\hat r_i=\frac{r_i}{\sum_j r_j},
\qquad g_i\approx G\hat r_i
$$

로 각 기준점에 생성량 $g_i$를 배정합니다. 그다음에는 **소수 클래스 이웃과의 보간**으로 새 점을 만듭니다. 난이도를 계산할 때의 이웃과 새 점을 만들 때의 이웃 범위가 다르다는 점이 중요합니다.[^adasyn-code]

**숫자 예시:** 두 소수 사례의 이웃을 각각 5개 살펴봤더니, 다수 클래스가 각각 1개와 4개였다면 $r_1=0.2$, $r_2=0.8$입니다. 총 100개를 만들 때 각각 약 20개와 80개를 배정합니다. 뒤쪽 사례가 더 어려운 곳에 있다고 판단한 것입니다.

하지만 “주변에 정상 사례가 많다”는 것은 정말 어려운 사기일 수도, 정답이 잘못 붙은 사례일 수도 있습니다. **어려움에 집중하는 것이 항상 유리하지는 않습니다.** 모든 $r_i$가 0이면 위 정규화를 할 수 없고, imbalanced-learn 구현은 오류를 냅니다. 생성량도 정수 반올림 때문에 목표와 조금 달라질 수 있습니다.[^adasyn-code]

### 5.3 Borderline-SMOTE: 경계 근처의 소수 사례에 집중하기

Borderline-SMOTE는 소수 클래스 전체가 아니라, **클래스 경계에 가깝다고 판단한 사례**를 생성의 출발점으로 삼습니다.[^borderline]

이진 분류에서 자기 자신을 제외한 주변 이웃 $m$개 중 다수 클래스 개수를 $m_{\mathrm{maj}}$라고 하면, imbalanced-learn 구현의 구분은 다음과 같습니다.[^smote-code]

| 구분 | 조건 | 의미 |
|---|---|---|
| SAFE | $m_{\mathrm{maj}}<m/2$ | 같은 클래스 이웃이 더 많음 |
| DANGER | $m/2\le m_{\mathrm{maj}}<m$ | 이웃의 절반 이상이 다수 클래스이지만 전부는 아님 |
| NOISE | $m_{\mathrm{maj}}=m$ | 이웃 전부가 다수 클래스 |

이때 **NOISE는 알고리즘의 이름 붙이기일 뿐, 실제로 잘못된 데이터임을 증명하지 않습니다.**

예를 들어 $m=6$이면 다수 이웃이 1개인 점은 SAFE, 4개인 점은 DANGER, 6개인 점은 NOISE입니다. Borderline-SMOTE는 이 중 DANGER를 생성의 출발점으로 선택합니다.[^smote-code]

**Borderline-1**은 DANGER 사례와 소수 클래스 이웃 사이에서 다음 보간을 합니다.

$$
x_{\mathrm{new}}=x_i+\lambda(x_{\mathrm{near}}-x_i),
\quad x_i\in\mathrm{DANGER},\quad\lambda\sim U(0,1)
$$

**Borderline-2**는 다른 클래스의 이웃도 고려하며, 다수 클래스 이웃 방향으로 이동할 때 거리를 조절합니다. 그러므로 위의 단순한 소수-소수 보간 설명은 Borderline-1에 해당합니다. 경계 판정용 `m_neighbors`와 합성용 `k_neighbors`도 서로 다른 역할입니다.[^borderline][^smote-code]

**세 방법을 한 문장으로 구분하면:** SMOTE는 소수 사례 주변을 채우고, ADASYN은 어려운 곳에 생성량을 더 배정하며, Borderline-SMOTE는 경계 후보를 생성의 출발점으로 선택합니다.

### 5.4 범주형 데이터에는 일반 SMOTE를 그대로 적용하지 않기

카드 종류를 `1=체크카드, 2=신용카드`로 표현했다고 해서 `1.4`라는 카드 종류가 의미를 갖지는 않습니다. 원-핫 인코딩된 범주 사이의 중간값도 실제 범주와 다를 수 있습니다.

연속형과 범주형이 섞였으면 **SMOTENC**, 모두 범주형이면 **SMOTEN**을 검토할 수 있습니다. SMOTENC에는 어떤 열이 범주형인지 알려줘야 하며, 수치 열의 결측값 처리·스케일링과 범주 열 처리를 나눠 설계합니다. **이 글의 실습 코드는 숫자형 연속 특성용이며, 혼합형 데이터에 그대로 복사하는 예제가 아닙니다.**[^smotenc][^over]

범주를 보존하더라도 나이·금액·기간 사이의 업무 규칙까지 자동으로 만족하는 것은 아닙니다. 합성 행이 실제로 가능한 조합인지 별도로 점검하는 것을 권합니다.

## 6. 비용/가중치 조정의 포인트

### 6.1 클래스 가중치: 중요한 오답에 더 큰 벌점 주기

**손실(loss)** 은 모델이 얼마나 틀렸는지를 학습에 사용할 숫자로 나타낸 것입니다. **클래스 가중치**는 어느 클래스의 오답을 학습 중 더 중요하게 볼지를 바꿉니다.

예를 들어 일부 분류기의 `class_weight="balanced"`는 학습에 전달된 클래스 빈도에 따라 다음 값을 사용합니다.[^weights]

$$
w_c=\frac{N}{K\,N_c}
$$

$N$은 전체 학습 사례 수, $K$는 클래스 수, $N_c$는 클래스 $c$의 사례 수입니다. 정상 9,900건, 사기 100건이면 정상 가중치는 약 $0.505$, 사기 가중치는 $50$입니다. 따라서 사기 사례 하나가 정상 사례 하나보다 상대적으로 99배 큰 가중치를 받습니다.

하지만 **희귀한 정도와 실제 오류 비용은 다릅니다.** 위 가중치는 사기를 놓쳤을 때 실제로 얼마의 손해가 발생하는지 알지 못합니다. 사용자 지정 가중치도 검증으로 비교해야 하며, 라이브러리·모델마다 `class_weight`나 `sample_weight` 지원 방식이 다릅니다.

### 6.2 SMOTE와 `balanced`를 같이 쓰면 어떻게 될까?

SMOTE로 두 클래스 개수를 정확히 같게 만든 뒤 분류기가 `class_weight="balanced"`를 계산하면, 두 클래스의 자동 가중치는 모두 1입니다. 이는 위 식에 $N_c=N/2$를 대입하면 알 수 있습니다. **원문 코드처럼 완전 균형 SMOTE 뒤에 `balanced`를 붙인다고 소수 클래스가 한 번 더 강조되는 것은 아닙니다.**[^weights]

반대로 원래 불균형 비율에서 계산한 큰 가중치를 고정해서 오버샘플링과 함께 쓰면, 소수 클래스의 영향이 중복해서 커질 수 있습니다. 먼저 단독 효과를 비교하고, 조합은 별도 실험으로 검증하세요.

## 7. 임계값 튜닝: 학습한 점수를 실제 행동으로 바꾸기

### 7.1 0.5는 편리한 기본값이지 정답이 아닙니다

확률 점수를 내는 이진 분류기에서 임계값 $t$를 정하면 다음처럼 판단합니다.[^threshold]

$$
\hat y=\begin{cases}
1 & \text{점수}\ge t\\
0 & \text{점수} < t
\end{cases}
$$

같은 점수에 대해 임계값을 낮추면 경보 대상이 늘고 재현율은 줄어들지 않습니다. 정밀도는 나빠질 수 있지만, 유한한 데이터에서는 매번 일정한 방향으로 변한다고 보장되지는 않습니다. **임계값만 바꾸면 정밀도·재현율은 바뀌어도 원래 점수의 AP와 ROC-AUC는 그대로**입니다. 점수의 순서를 바꾸지 않았기 때문입니다.[^threshold]

### 7.2 비용으로 임계값 정하기

사기를 놓친 한 건의 비용을 $C_{FN}$, 정상 거래에 잘못 경보를 낸 한 건의 비용을 $C_{FP}$라고 하겠습니다. 올바른 판단의 비용은 0, 각 오류 유형의 건당 비용은 일정하다고 가정하면,

$$
\mathrm{Cost}(t)=C_{FN}\,FN(t)+C_{FP}\,FP(t)
$$

를 검증 데이터에서 계산해 가장 작은 임계값을 선택할 수 있습니다. 금액마다 손해가 다르면 건별 비용을 사용해야 합니다. 비용 함수의 설계는 실제 업무의 가정을 반영해야 합니다.[^cost-example]

앞의 혼동행렬에서 $C_{FN}=10$, $C_{FP}=1$이면 비용은 $10\times20+99=299$입니다. 전부 정상이라고 판단하면 사기 100건을 놓쳐 비용이 1,000입니다. 정확도는 낮아졌어도 가정한 비용은 줄어든 것입니다.

업무 제약이 있다면 “재현율 **최소** 80%”, “정밀도 **최소** 50%”, “하루 경보 **최대** 100건”처럼 방향을 명확하게 정하세요. 이 수치들은 설명용 예시입니다. 모든 조건을 만족하는 임계값이 없을 수도 있으며, 이때는 모델·업무 용량·목표를 다시 검토해야 합니다.

`TunedThresholdClassifierCV`는 교차검증으로 임계값을 선택하는 도구입니다. **기본 최적화 지표는 비용이 아니라 balanced accuracy**이므로, 비용 최소화가 목적이면 `scoring`에 그 목적을 반영해야 합니다. 미리 학습한 모델을 이용할 때도 모델 학습과 임계값 선택에 같은 데이터를 쓰지 않도록 주의합니다.[^tuned-api]

### 7.3 확률 보정: 점수 0.8은 정말 80%일까?

**확률 보정(calibration)** 은 “0.8이라고 예측한 사례들에서 실제 양성이 약 80% 발생하는가?”를 다루는 문제입니다. 순위를 잘 매기는 것, 확률이 정확한 것, 적절한 경보 기준을 고르는 것은 서로 다른 문제입니다.[^calibration]

재표본추출은 모델이 보는 클래스 비율을 바꾸므로, 출력값을 원래 모집단의 확률로 곧바로 해석하면 위험할 수 있습니다. 대규모 관측 건강 데이터에 대한 2024년 연구에서는 무작위 오버/언더샘플링 후 위험 확률이 과대평가되는 현상이 관찰됐습니다. 이는 해당 연구 범위의 결과이며, 모든 문제에서 샘플링을 금지하라는 뜻은 아닙니다.[^yang]

실제 위험 확률이 필요하다면 원래 운영 분포를 반영하는 데이터로 보정을 평가하세요. `CalibratedClassifierCV`를 이용할 수 있으며, 보정에 쓰이는 예측은 해당 사례로 모델을 학습하지 않은 상태에서 얻어야 합니다. 입문용으로는 아래처럼 역할을 나누면 이해하기 쉽습니다.[^calibration]

```text
모델 훈련 → 별도 데이터로 확률 보정 → 별도 검증으로 임계값 선택 → 최종 테스트
```

데이터가 부족하면 훈련 부분의 교차검증 안에서 보정을 구성할 수 있습니다. 이후에도 임계값 선택과 최종 테스트의 역할은 구분합니다.

**수식으로 이해하기 — 확률이 정확하다면:** $p$가 운영 분포의 잘 보정된 양성 확률이고, 올바른 판단의 비용이 0이며 오류 비용이 일정하고 경보 용량 제한이 없다면, 경보를 낼 때의 기대 비용은 $C_{FP}(1-p)$, 통과시킬 때는 $C_{FN}p$입니다. 두 값을 비교하면,

$$
t^*=\frac{C_{FP}}{C_{FP}+C_{FN}}
$$

를 얻습니다. $C_{FN}=10$, $C_{FP}=1$이면 약 $0.091$입니다. 이는 위 가정에서 직접 유도한 식입니다. **보정되지 않은 점수에 이 숫자를 그대로 적용하라는 뜻은 아닙니다.** 아래 코드는 확률 보정을 수행하지 않고, 검증 데이터에서 실제 오류 건수를 비교해 임계값을 고릅니다.

## 8. 엔드투엔드 파이프라인 예시: sklearn + imbalanced-learn

### 8.1 이 코드가 하는 일

숫자형 합성 데이터에서 전체의 60%를 훈련, 20%를 임계값 검증, 20%를 최종 테스트에 사용합니다. 훈련 데이터 내부의 5폴드 교차검증으로 네 후보를 비교하고, 평균 AP가 가장 높은 모델을 선택합니다. 이후 검증 데이터에서 오류 비용이 가장 작은 임계값을 고릅니다.

이 예제는 **AP로 모델을 고른 뒤 비용으로 임계값을 고르는 교육용 절차**입니다. 가능한 모든 모델 중 최소 비용 모델을 찾는 절차와는 다릅니다. 최종 목적이 비용 최소화라면, 훈련 부분의 내부 검증에서 모델과 임계값을 함께 비용 기준으로 비교하고, 최종 테스트는 여전히 별도로 남겨 두어야 합니다.[^cost-example][^cv]

전처리와 SMOTE는 `imblearn.pipeline.Pipeline` 안에 넣습니다. 이렇게 해야 교차검증의 각 훈련 폴드에서만 합성이 이루어집니다. 평가 지표에는 문자열 `"average_precision"`, `"roc_auc"`를 사용해 원문의 오래된 `needs_proba=True` 사용을 없앴습니다.[^im-pitfalls][^scorer]

**실행 확인 환경:** Python 3.13.5, NumPy 2.3.5, scikit-learn 1.8.0, imbalanced-learn 0.14.1. 이는 이 개정본을 실제로 실행한 환경이며, 각 패키지의 최신 버전이라는 뜻은 아닙니다. 별도 가상환경에서 재현할 때 다음 버전을 설치할 수 있습니다.

```bash
python -m pip install numpy==2.3.5 scikit-learn==1.8.0 imbalanced-learn==0.14.1
python imbalanced_data_example.py
```

아래 코드는 함께 제공한 `imbalanced_data_example.py`와 같습니다.

```python
"""불균형 표형 데이터 실습: 훈련 CV -> 검증 임계값 선택 -> 최종 테스트.

실행: python imbalanced_data_example.py
필요 패키지: numpy, scikit-learn, imbalanced-learn
숫자형 합성 데이터 전용 예제이며, 실제 운영 성능을 보장하지 않습니다.
"""
from __future__ import annotations

import platform

import imblearn
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
C_FN, C_FP = 10.0, 1.0  # 수업용 가정: 사기 누락 1건의 비용 = 오경보 10건의 비용


def make_candidates() -> dict:
    """새 파이프라인을 생성합니다. SMOTE와 자동 가중치는 별도로 비교합니다."""
    def logistic_pipeline(class_weight=None, use_smote=False):
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),  # 거리 계산을 위해 SMOTE보다 먼저 실행
        ]
        if use_smote:
            steps.append(("smote", SMOTE(
                sampling_strategy=0.2, k_neighbors=5, random_state=SEED
            )))  # 합성 후 소수/다수 비율 0.2. 양성 비율 20%라는 뜻은 아님.
        steps.append(("model", LogisticRegression(
            class_weight=class_weight, max_iter=2000
        )))
        return Pipeline(steps)

    return {
        "logistic": logistic_pipeline(),
        "weighted_logistic": logistic_pipeline(class_weight="balanced"),
        "smote_logistic": logistic_pipeline(use_smote=True),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.07,
                l2_regularization=1.0, random_state=SEED,
            )),
        ]),
    }


def select_cost_threshold(
    y_true: np.ndarray, scores: np.ndarray,
    c_fn: float = C_FN, c_fp: float = C_FP,
) -> float:
    """검증 세트의 모든 서로 다른 점수 경계에서 FN/FP 비용을 비교합니다."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    if y_true.ndim != 1 or scores.ndim != 1 or y_true.shape != scores.shape:
        raise ValueError("정답과 점수는 길이가 같은 1차원 배열이어야 합니다.")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("검증 데이터에는 0과 1 클래스가 모두 있어야 합니다.")
    if not np.isfinite(scores).all():
        raise ValueError("점수에 NaN 또는 무한대가 포함되어 있습니다.")
    if not (np.isfinite(c_fn) and np.isfinite(c_fp) and c_fn > 0 and c_fp > 0):
        raise ValueError("오류 비용은 유한한 양수여야 합니다.")

    # ROC의 비율을 실제 오류 건수로 바꿉니다. 중간 경계를 생략하지 않습니다.
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    n_pos, n_neg = np.sum(y_true == 1), np.sum(y_true == 0)
    fp = np.rint(fpr * n_neg).astype(int)
    fn = np.rint((1.0 - tpr) * n_pos).astype(int)
    costs = c_fn * fn + c_fp * fp
    # thresholds는 내림차순. 비용이 같으면 높은 임계값(적은 경보)을 선택합니다.
    # 첫 경계 inf는 모두 음성, 마지막 경계는 모두 양성인 경우도 포함합니다.
    return float(thresholds[np.argmin(costs)])


def report(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """사전에 고정한 임계값에서 운영 지표를 계산합니다."""
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "alerts": int(predicted.sum()),
        "cost_per_1000": float(1000 * (C_FN * fn + C_FP * fp) / len(y_true)),
    }


def main() -> None:
    print("Versions:", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scikit-learn": sklearn.__version__, "imbalanced-learn": imblearn.__version__,
    })
    X, y = make_classification(
        n_samples=15000, n_features=20, n_informative=8, n_redundant=4,
        weights=[0.98, 0.02], flip_y=0.001, class_sep=1.0, random_state=SEED,
    )
    # 먼저 분할합니다. 테스트 데이터는 마지막 평가까지 사용하지 않습니다.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED + 1
    )  # 전체 데이터의 훈련/검증/테스트 비율 = 60/20/20
    for name, labels in [("train", y_train), ("validation", y_val)]:
        print(name, {"n": len(labels), "positives": int(labels.sum()),
                     "positive_rate": float(labels.mean())})

    # CV(교차검증)는 훈련 세트 안에서만 수행합니다.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(X_train, y_train))  # 후보마다 동일한 분할 사용
    if any(np.sum(y_train[train_ids] == 1) <= 5 for train_ids, _ in splits):
        raise ValueError("각 CV 훈련 폴드의 양성이 6개 이상이어야 SMOTE(k=5)를 쓸 수 있습니다.")
    candidates, mean_ap = make_candidates(), {}
    for name, model in candidates.items():
        result = cross_validate(
            model, X_train, y_train, cv=splits,
            scoring={"ap": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1, error_score="raise",
        )
        mean_ap[name] = float(result["test_ap"].mean())
        print(name, {
            "CV_AP_mean": mean_ap[name],
            "CV_AP_std": float(result["test_ap"].std(ddof=1)),
            "CV_ROC_AUC_mean": float(result["test_roc_auc"].mean()),
        })

    # 예제의 모델 선택 기준은 AP입니다. 최소 비용 모델을 직접 찾는 절차는 아닙니다.
    best_name = max(mean_ap, key=mean_ap.get)
    selected = clone(candidates[best_name]).fit(X_train, y_train)
    validation_scores = selected.predict_proba(X_val)[:, 1]
    threshold = select_cost_threshold(y_val, validation_scores)
    print("Selected model:", best_name)
    print("Validation decision:", report(y_val, validation_scores, threshold))

    # 모델과 임계값을 모두 고정한 뒤에만 테스트를 엽니다. 여기서 재학습하지 않습니다.
    test_scores = selected.predict_proba(X_test)[:, 1]
    print("Test summary:", {
        "n": len(y_test), "positives": int(y_test.sum()),
        "positive_rate": float(y_test.mean()),
        "AP": float(average_precision_score(y_test, test_scores)),
        "ROC_AUC": float(roc_auc_score(y_test, test_scores)),
    })
    # 미리 정해 둔 비교: 기본값과 검증에서 고른 임계값. 테스트로 둘 중 하나를 고르지 않습니다.
    print("Test / default 0.5:", report(y_test, test_scores, 0.5))
    print("Test / validation threshold:", report(y_test, test_scores, threshold))


if __name__ == "__main__":
    main()
```

### 8.2 실행 결과를 읽는 방법

위 환경에서 실행한 **한 번의 합성 데이터 실습 결과**는 다음과 같습니다. 이는 실제 사기 탐지 벤치마크의 결과가 아닙니다.

| 후보 모델 | 훈련 내부 CV 평균 AP | 폴드 간 AP 표준편차 |
|---|---:|---:|
| 기본 로지스틱 회귀 | 0.0535 | 0.0089 |
| 가중 로지스틱 회귀 | 0.0530 | 0.0065 |
| SMOTE + 로지스틱 회귀 | 0.0512 | 0.0074 |
| 히스토그램 기반 그래디언트 부스팅 | 0.6203 | 0.0376 |

이 설정에서는 마지막 모델이 선택됐습니다. **이 표만으로 부스팅이 항상 최고라거나 SMOTE가 항상 해롭다고 결론 내리면 안 됩니다.** 데이터 생성 방식과 모델 설정에 의존한 교육용 관찰입니다. 폴드 간 표준편차도 평균 성능의 95% 신뢰구간은 아닙니다.

최종 테스트에는 3,000건 중 양성이 61건 있었고, 선택한 모델의 AP는 약 0.6008, ROC-AUC는 약 0.9488이었습니다.

| 미리 정한 비교 | 기본 임계값 0.5 | 검증에서 선택한 임계값 약 0.01992 |
|---|---:|---:|
| 정밀도 | 94.4% | 37.4% |
| 재현율 | 27.9% | 65.6% |
| TP / FP / FN / TN | 17 / 1 / 44 / 2,938 | 40 / 67 / 21 / 2,872 |
| 경보 건수 | 18 | 107 |
| 1,000건당 비용 | 147.00 | 92.33 |

낮은 임계값은 사기를 더 찾았지만 오경보도 늘렸습니다. 이 예제의 비용 가정에서는 유리했으나, 검토 인력의 여력이 부족하면 경보 107건이 감당 가능한지 별도로 판단해야 합니다. **무엇을 얻었고 무엇을 포기했는지 함께 설명하는 것이 결과 해석입니다.**

### 8.3 실제 데이터로 바꿀 때의 경계선

이 코드는 독립적인 숫자형 관측을 가정합니다. 고객 그룹, 시간 순서, 범주형 특성, 건별 비용이 있다면 각각의 처리를 추가해야 합니다. 테스트 결과를 보고 다시 모델을 선택하거나, 검증 임계값을 정한 뒤 모델을 재학습하고 같은 임계값이 그대로 유효하다고 가정하지 마세요. 재학습하면 점수 분포가 바뀔 수 있어 다시 검증이 필요합니다.[^cv][^threshold]

## 9. 최신 연구가 바꾸는 실무 권고

이 절은 원문에 추가한 외부 연구 검토입니다. **학회·학술지에 발표된 결과와 공개 사전논문(preprint)을 구분**합니다. 사전논문은 빠르게 공개된 연구로, 아래에서 별도 출판을 확인하지 못한 경우 동료심사를 거친 확정적 결론처럼 다루지 않습니다.

### 9.1 CLIMB, NeurIPS 2025: “균형을 맞추면 좋아진다”는 가정부터 검증

CLIMB는 **실제 표형 데이터셋 73개와 불균형 학습 방법 29개**를 비교한 벤치마크입니다. 단순한 재균형화의 한계, 앙상블의 효과, 데이터 품질의 중요성을 보고했습니다. 평가 지표에 따라 방법의 순위가 달라지는 점도 강조합니다.[^climb]

이 가이드에 반영한 실무적 해석은 **샘플링을 기본 의무 단계로 두지 않고, 재표본추출 없는 모델·가중치·샘플링·앙상블을 비교한다**는 것입니다. 다만 이 연구의 중심 비교는 트리 기반 학습과 특정 데이터 선정 조건에 의존합니다. 시계열이나 모든 딥러닝 모델에 같은 우열이 성립한다고 확장하지 않습니다.

### 9.2 NeurIPS 2024의 지표 연구: AUPRC의 무조건적 우위를 피하기

McDermott 등의 연구는 불균형이라는 이유만으로 AUPRC가 AUROC보다 항상 우월하다고 할 수 없음을 이론과 실험으로 보였습니다. 특정 하위집단의 양성 빈도 차이가 평가에 미치는 영향도 다뤘습니다.[^auroc-auprc]

따라서 이 가이드는 PR 계열 지표의 유용성을 설명하되 ROC-AUC를 버리라고 권하지 않습니다. **어떤 오류를 중요하게 평가하는지 먼저 정한다**는 방향으로 수정했습니다.

### 9.3 2026년 연구: 고정된 샘플링에서 적응적 학습과 더 넓은 비교로

**ART**는 학습 중 클래스별 성능을 확인해 훈련 데이터의 재표본추출 분포를 바꾸는 방법입니다. 쉽게 말하면 “처음에 어려웠던 클래스”뿐 아니라 “지금도 어려운 클래스”에 학습을 더 배정합니다. 공개 수정본에서는 반복 학습을 사용하는 모델에 대한 적용과 함께, 검증 비용 증가와 잡음이 많은 클래스에 과도하게 집중할 위험을 논의합니다.[^art]

이는 유망한 비교 후보이지, 입문자가 SMOTE나 기본 모델을 건너뛰어야 한다는 근거는 아닙니다. 공개 수정본의 표형 데이터 실험은 Pima·Yeast·Red Wine 세 데이터셋 중심이므로, 그 결과를 모든 표형 문제의 우승으로 일반화하지 않습니다.[^art]

또한 **2026년 8월 공개된 45개 이진 과제 비교 사전논문**은 한 사기 데이터셋에서 관찰한 임계값 조정과 SMOTE의 효과가 다른 과제에서는 달라질 수 있다고 보고했습니다. 다만 45개 과제 중 30개가 합성 데이터이고, 일부 실제 데이터셋을 여러 이진 과제로 사용했습니다. 광범위한 독립 현장 데이터 검증으로 받아들이기보다는, **단일 데이터셋의 결론을 일반화하지 말라는 보조 근거**로 읽습니다.[^threshold-study]

## 10. 딥러닝으로 확장할 때

### 10.1 배치 샘플링과 가중 손실

**배치(batch)** 는 한 번의 학습 갱신에 사용하는 작은 데이터 묶음입니다. 양성이 매우 드물면 일부 배치에는 양성이 없을 수 있습니다. 예를 들어 양성 비율 1%, 독립적으로 32개를 뽑는다고 가정하면 양성이 하나도 없을 확률은 $0.99^{32}\approx72.5\%$입니다.

양성을 더 자주 뽑거나 양성의 손실 가중치를 높이는 방법을 비교할 수 있습니다. PyTorch의 `BCEWithLogitsLoss(pos_weight=...)`는 양성 손실 항의 가중치를 조절합니다. **logit은 확률로 변환하기 전의 출력값**이며, 이 손실 함수에는 sigmoid를 미리 적용하지 않은 logit을 전달합니다.[^torch]

샘플링과 가중치를 함께 쓸 때는 소수 클래스가 얼마나 강조되는지 점검해야 합니다. 최종 평가는 균형 배치가 아니라 운영 분포를 반영하는 데이터에서 수행합니다.[^tf]

### 10.2 Focal Loss: 이미 쉬운 문제의 영향 줄이기

Focal Loss는 “소수 클래스라서”가 아니라 **현재 모델이 쉽게 맞히는 사례의 손실을 줄이는 방식**입니다. 가중치가 포함된 형태는 다음과 같습니다.[^focal]

$$
\mathrm{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t
$$

$p_t$는 정답 클래스에 부여한 확률, $\alpha_t$는 클래스 가중치, $\gamma\ge0$는 쉬운 사례를 얼마나 약하게 반영할지 정하는 값입니다. $\gamma=0$이면 가중 교차엔트로피가 됩니다.

예를 들어 $\gamma=2$이면 $p_t=0.9$인 쉬운 사례의 조절 계수는 $0.01$, $p_t=0.2$인 어려운 사례의 계수는 $0.64$입니다. 이는 학습의 상대적 관심을 어려운 사례로 옮깁니다. **어려운 사례가 잘못된 정답일 수도 있으므로 잡음 해결책과 동일하지 않습니다.**[^focal]

원래 논문의 주된 검증 대상은 객체 탐지입니다. 표형 데이터에서도 후보로 실험할 수 있지만 자동으로 최선이라고 가정하지 말고, 일반 손실·가중 손실·트리 기반 모델과 비교하세요. 이 개정본의 실행 검증은 8절의 sklearn 예제에 한정되며, 딥러닝 학습 실험까지 수행한 것은 아닙니다.

## 11. 실무 체크리스트와 이해 확인

### 11.1 실험 전후에 확인할 네 가지

- **분할과 누출:** 실제 예측 시 알 수 있는 특성만 사용했는가? 시간·고객 조건을 반영했는가? 전처리·샘플링은 각 훈련 부분에서만 학습했는가?
- **공정한 비교:** 아무 처리도 하지 않은 기준선이 있는가? 후보마다 같은 분할과 일관된 선택 기준을 사용했는가? 불균형 비율뿐 아니라 실제 양성 수와 정답 품질도 확인했는가?
- **독립적인 최종 평가:** 임계값과 모델을 테스트를 보지 않고 고정했는가? AP의 계산 방식, 양성 비율, 경보량과 오류 비용을 함께 기록했는가?
- **불확실성과 운영:** 분할별 성능 차이를 확인했는가? 소수 사례가 적을 때 성능을 단정하지 않는가? 배포 뒤 발생률·점수 분포·정밀도·재현율 변화를 다시 점검할 계획이 있는가?

앞의 세 항목은 이 가이드에서 설명한 평가·분할·임계값 원칙을 모은 것입니다. 마지막 항목은 표본 수가 적거나 운영 분포가 달라질 때 결과를 과신하지 않기 위한 점검 제안입니다.[^cv][^sk-pitfalls][^cost-example]

특히 실제 양성이 10건인 테스트에서는 한 건을 더 맞히는 것만으로 재현율이 10%포인트 바뀝니다. 80%라는 숫자만 쓰기보다 **“양성 10건 중 8건 탐지”**라고 함께 기록하세요. 가능한 경우 표본의 독립성·시간·그룹 구조에 맞는 불확실성 평가를 추가하고, 교차검증 표준편차를 신뢰구간으로 오해하지 않습니다.

### 11.2 스스로 설명할 수 있는지 확인하기

**질문 1. 사기가 1%인데 정확도가 99%면 충분할까요?**  
아닙니다. 모든 사례를 정상으로 예측했을 수도 있습니다. 혼동행렬과 재현율을 확인해야 합니다.

**질문 2. SMOTE를 전체 데이터에 먼저 적용하면 왜 안 될까요?**  
평가할 사례의 정보가 합성 과정에 들어갈 수 있고, 평가 분포도 실제 운영 분포와 달라질 수 있기 때문입니다.

**질문 3. 재현율이 올랐으면 모델이 무조건 좋아진 것일까요?**  
아닙니다. 오경보가 얼마나 늘었는지, 정밀도·경보량·전체 비용이 목표에 맞는지도 봐야 합니다.

## 12. 원문에서 수정한 핵심 사항

| 원문의 내용 또는 코드 | 개정 내용 | 이유 / 근거 |
|---|---|---|
| 샘플링과 비용 가중치의 병행을 기본 권장 | 기본 모델과 단독 방법부터 비교 | 효과는 데이터·모델·지표에 따라 다름. CLIMB 및 2026 비교 연구 |
| `SMOTE → StandardScaler` | 숫자형 예제에서 `결측값 처리 → StandardScaler → SMOTE` | SMOTE의 이웃 탐색 전에 특성 단위를 맞춤 |
| 테스트 데이터로 임계값을 정한 후 같은 데이터에서 보고 | 훈련 내부 CV / 임계값 검증 / 최종 테스트 분리 | 테스트를 이용한 선택으로 생기는 낙관적 평가 방지 |
| `PR-AUC(AP)`를 사실상 같은 뜻으로 사용 | AP와 사다리꼴 PR-AUC를 구분 | scikit-learn의 계산 정의에 맞춤 |
| 불균형이면 PR 계열 지표를 우선시하는 단순 설명 | 목적·양성 비율·ROC 관점도 함께 고려 | NeurIPS 2024의 지표 연구 반영 |
| 완전 균형 SMOTE 뒤 `class_weight="balanced"` 병행의 효과 강조 | 이 경우 자동 가중치는 둘 다 1임을 설명 | 가중치 식과 학습에 입력되는 클래스 빈도 |
| 재현율 “상한선”을 안전 제약으로 제시 | 최소 탐지율을 요구하는 경우 재현율 “하한선” | 놓치지 않아야 하는 사건의 최소 탐지 목표와 방향 일치 |
| IBA를 “지수형 균형정확도”로 표현 | Index of Balanced Accuracy로 정정 | 지수함수라는 뜻이 아님 |
| Borderline-1/2의 생성 설명이 불분명 | 변형별 이웃 범위와 DANGER 조건 구분 | 공식 API와 구현 코드 기준 |
| 중복된 출처 번호와 중첩된 수식 구분자 | 이름 있는 각주와 단일 수식 구분자로 정리 | 근거 추적과 Markdown 가독성 개선 |

각 수정의 세부 근거는 해당 본문 문단의 각주에 연결했습니다. 기존 블로그 중심 참고문헌은 논문 원문과 라이브러리 공식 문서 중심으로 재구성했습니다.

---

## 참고문헌

아래 자료는 2026년 10월 15일에 확인했습니다. `stable` 문서는 이후 바뀔 수 있으므로 코드 재현 시에는 8절의 **실행 확인 버전**도 함께 확인하세요. 설명용 계산과 8절의 합성 데이터 결과는 이 개정 작업에서 직접 작성·계산한 것이며, 논문의 보고 수치가 아닙니다.

[^climb]: Liu, Z. et al. **CLIMB: Class-imbalanced Learning Benchmark on Tabular Data.** NeurIPS 2025, Datasets and Benchmarks Track. [학회 공식 기록](https://papers.nips.cc/paper_files/paper/2025/hash/45ee52e746c65e5e60205bedf2d18386-Abstract-Datasets_and_Benchmarks_Track.html), [공개 본문 v2](https://arxiv.org/html/2505.17451v2). 비교 범위와 한계는 본문의 실험 설정·결론·부록을 함께 참고.

[^sk-pitfalls]: scikit-learn. **Common pitfalls and recommended practices**, 데이터 누출과 전처리 원칙. [공식 문서](https://scikit-learn.org/stable/common_pitfalls.html).

[^im-pitfalls]: imbalanced-learn. **Common pitfalls and recommended practices**, 전체 데이터 재표본추출의 누출 및 평가 분포 문제. [공식 문서](https://imbalanced-learn.org/stable/common_pitfalls.html).

[^metrics]: scikit-learn. **Metrics and scoring: quantifying the quality of predictions**. [공식 문서](https://scikit-learn.org/stable/modules/model_evaluation.html).

[^precision]: scikit-learn. **precision_score**, 정밀도의 정의와 분모가 0일 때의 처리. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html).

[^recall]: scikit-learn. **recall_score**, 재현율의 정의. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html).

[^f1]: scikit-learn. **f1_score**, F1의 정의와 평균 방식. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

[^roc]: scikit-learn. **roc_auc_score**. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

[^ap]: scikit-learn. **average_precision_score**, AP가 사다리꼴 PR 면적과 다름을 명시. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

[^auroc-auprc]: McDermott, M. B. et al. **A Closer Look at AUROC and AUPRC under Class Imbalance.** NeurIPS 2024. [학회 공식 기록](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4df3510ad02a86d69dc32388d91606f8-Abstract-Conference.html).

[^im-metrics]: imbalanced-learn. **Metrics**, G-mean과 Index of Balanced Accuracy. [공식 문서](https://imbalanced-learn.org/stable/metrics.html).

[^cv]: scikit-learn. **Cross-validation: evaluating estimator performance**, 계층화·그룹·시계열 분할과 평가. [공식 문서](https://scikit-learn.org/stable/modules/cross_validation.html).

[^scale]: scikit-learn. **StandardScaler**, 평균·표준편차를 이용한 변환. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

[^over]: imbalanced-learn. **Over-sampling**, 무작위 오버샘플링, 합성 방식, 범주형 처리 및 한계. [공식 문서](https://imbalanced-learn.org/stable/over_sampling.html). Borderline의 정확한 판정 조건은 아래의 구현 코드 각주를 우선 참고.

[^ensemble]: imbalanced-learn. **Ensemble of samplers**, Balanced Random Forest, Easy Ensemble 등. [공식 문서](https://imbalanced-learn.org/stable/ensemble.html).

[^smote-paper]: Chawla, N. V. et al. **SMOTE: Synthetic Minority Over-sampling Technique.** Journal of Artificial Intelligence Research, 16, 321–357, 2002. [저자 공개본 및 원출판 정보](https://arxiv.org/abs/1106.1813). 논문 출판은 2002년이며 arXiv 등록 연도와 다름.

[^smote-api]: imbalanced-learn. **SMOTE**, `sampling_strategy`와 `k_neighbors`. [공식 API](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html).

[^adasyn]: imbalanced-learn. **ADASYN**, He 등의 2008년 원논문 정보 포함. [공식 API](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html).

[^adasyn-code]: imbalanced-learn 0.14.1. **ADASYN 구현**, 난이도 정규화·반올림·소수 클래스 이웃 선택. [공식 소스 코드](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/0.14.1/imblearn/over_sampling/_adasyn.py).

[^borderline]: imbalanced-learn. **BorderlineSMOTE**, `kind`, `m_neighbors`, `k_neighbors`. [공식 API](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html).

[^smote-code]: imbalanced-learn 0.14.1. **BaseSMOTE 구현**, `_in_danger_noise`의 DANGER/NOISE 판정과 Borderline-2의 거리 조절. [공식 소스 코드](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/0.14.1/imblearn/over_sampling/_smote/base.py).

[^smotenc]: imbalanced-learn. **SMOTENC**, 연속형·범주형 혼합 데이터 처리. [공식 API](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html).

[^weights]: scikit-learn. **compute_class_weight**, `balanced` 가중치 계산식. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html).

[^threshold]: scikit-learn. **Tuning the decision threshold for class prediction**, 점수 예측과 의사결정 구분. [공식 문서](https://scikit-learn.org/stable/modules/classification_threshold.html).

[^cost-example]: scikit-learn. **Post-tuning the decision threshold for cost-sensitive learning**, 비용 기반 임계값 선택과 과적합 주의. [공식 예제](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html).

[^tuned-api]: scikit-learn. **TunedThresholdClassifierCV**, 기본 `scoring`과 교차검증 설정. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html).

[^calibration]: scikit-learn. **Probability calibration**, 신뢰도 곡선 및 `CalibratedClassifierCV`. [공식 문서](https://scikit-learn.org/stable/modules/calibration.html).

[^yang]: Yang, C. et al. **Impact of random oversampling and random undersampling on the performance of prediction models developed using observational health data.** Journal of Big Data, 11, 7, 2024. [논문 원문](https://link.springer.com/article/10.1186/s40537-023-00857-7). 무작위 재표본추출과 위험 확률 보정에 관한 연구이며, 모든 샘플링 방법·도메인을 포괄하지 않음.

[^scorer]: scikit-learn. **make_scorer**, 예측 응답 선택에는 `response_method` 사용. 이 가이드 코드는 내장 문자열 scorer를 사용. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).

[^art]: Basandrai, A., Jain, S., and Ilanthenral, K. **ART: Adaptive Resampling-based Training for Difficulty-aware Classification.** 방법과 한계에 대한 설명은 [공개 수정본 v2, 2026-02-15](https://arxiv.org/html/2509.00955v2)에 근거함. 학술지판 서지: Machine Learning with Applications, 25, 100967, 2026년 9월, DOI 10.1016/j.mlwa.2026.100967. [출판사 기록](https://www.sciencedirect.com/science/article/pii/S2666827026001325). 학술지판 전체 본문은 이 검토에서 직접 열람하지 못했으므로 공개본과 완전히 같다고 가정하지 않음.

[^threshold-study]: Musaev, D. **When Single-Dataset Conclusions Fail: A 45-Task Study of Threshold Tuning and Resampling for Imbalanced Classification.** arXiv:2608.16147v1, 2026-08-17. [공개 사전논문](https://arxiv.org/html/2608.16147v1). 이 검토에서 별도의 동료심사 출판은 확인하지 못함. 과제 구성과 한계는 3.4절 및 7.1절 참고.

[^torch]: PyTorch 2.9.0. **BCEWithLogitsLoss**, `pos_weight` 및 sigmoid와 손실의 결합. [공식 소스의 API 설명](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/modules/loss.py).

[^tf]: TensorFlow. **Classification on imbalanced data**, 가중치·오버샘플링·평가 절차. [공식 튜토리얼](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).

[^focal]: Lin, T.-Y. et al. **Focal Loss for Dense Object Detection.** ICCV 2017. [저자 공개본 v2](https://arxiv.org/html/1708.02002v2), 특히 3절의 가중 교차엔트로피와 Focal Loss 정의. v2 공개일은 2018-02-07.
