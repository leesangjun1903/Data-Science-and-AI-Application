# Imbalanced Data 가이드 : Tabular Data 중심으로

핵심은 “평가 지표부터 바꾸고, 데이터/알고리즘/의사결정 임계값을 함께 튜닝한다”입니다.[1][2][3][4]

## 핵심 요약
불균형 데이터에서는 정확도 대신 **정밀도-재현율**, **ROC/PR 곡선**, **G-mean/IBA** 같은 지표를 우선 사용하고, 샘플링(SMOTE/ADASYN), 비용가중치(class_weight), 임계값 튜닝을 함께 적용한 **파이프라인**을 구성하는 것이 효과적입니다.[2][3][5][6]

## 왜 불균형 데이터가 문제인가
- 다수 클래스만 찍어도 정확도는 높게 나와 “좋은 모델”로 착각하기 쉽습니다. 예를 들어 99:1 데이터에서 전부 0으로 예측해도 99% 정확도를 달성하지만, 소수 클래스(관심 사건)는 한 건도 못 잡습니다.[7][8][2]
- 소수 클래스 표본이 적으면 분포 대표성이 떨어져 과적합/일반화 실패가 발생하기 쉽습니다. 이때 단순 정확도는 현상을 가립니다.[8][2][7]

## 올바른 평가 지표
- 혼동행렬 기반: 정밀도 $$Precision=\frac{TP}{TP+FP}$$, 재현율 $$Recall=\frac{TP}{TP+FN}$$, F1 $$=\frac{2PR}{P+R}$$처럼 소수 클래스 성능을 직접 측정합니다.[4][2]
- 커브 기반: ROC-AUC, PR-AUC를 확인하되, 극단적 불균형일수록 PR-AUC가 더 민감하게 소수 클래스 성능을 반영합니다.[2][8]
- 불균형 특화: G-mean(민감도×특이도의 기하평균), IBA(지수형 균형정확도) 등을 종합 리포트로 확인합니다. imbalanced-learn의 classification_report_imbalanced가 편리합니다.[3][2]

## 실무 전략 개요
- 데이터 전처리: 누락값/이상치 처리 → 누적 정보 누출 방지(시간순 분할) → 피처 스케일링 순서를 유지합니다.[9][1]
- 학습 전략(병행 권장):  
  - Oversampling: **SMOTE**, Borderline-SMOTE, **ADASYN** 등으로 소수 클래스를 합성해 결정경계를 보강합니다.[5][10][11]
  - Cost-sensitive: **class_weight='balanced'** 또는 커스텀 비용으로 소수 클래스 에러를 더 크게 패널티합니다.[12][13][14]
  - 임계값 튜닝: 확률 출력 후 **컷오프를 사업 비용 최소화** 기준으로 최적화합니다.[6][2]

## 샘플링 방법 한눈에 보기
- SMOTE: 소수 클래스 이웃 간 보간으로 합성 표본 생성. 경계가 완만해질 수 있으나, 드물게 이상치도 잇는 위험이 있습니다.[15][5]
- Borderline-SMOTE: 경계(in-danger) 샘플만 증강해 결정경계 품질을 개선합니다. Borderline-1/2 변형이 존재합니다.[16][5]
- ADASYN: 지역 난이도(주변 다수 비율)에 비례해 더 어려운 영역에 표본을 많이 생성합니다. 경계 학습에 집중하지만, 노이즈 과강조 가능성이 있어 모델/규제와 같이 쓰는 것을 권장합니다.[17][5]

SMOTE와 ADASYN은 모두 **불균형 데이터 문제를 해결하기 위한 오버샘플링 기법**으로, 소수 클래스의 데이터를 인위적으로 생성하여 학습 데이터의 균형을 맞춥니다.

---

### 1. SMOTE (Synthetic Minority Over-sampling TEchnique)

- SMOTE는 소수 클래스의 각 데이터 포인트 $\( x_i \)$ 에 대해 K-최근접 이웃(K-NN)을 찾습니다.
- 그 이웃 중 하나 $\( x_{\text{near}} \)$ 를 선택한 후, 두 점 사이의 벡터 차이에 임의의 값 $\( \delta \in [0,1] \)$ 을 곱해 새로운 합성 샘플을 만듭니다.
  
수식으로 표현하면:

$$
\[x_{\text{new}} = x_i + \delta \times (x_{\text{near}} - x_i)
\]
$$

여기서 $\( \delta \)$ 는 균등분포에서 무작위로 선택한 값입니다.

- 이렇게 하면 기존 소수 클래스 샘플들의 분포를 따라가면서 새로운 예제를 만듭니다.
- 단점: 노이즈가 있는 소수 클래스 샘플까지 증폭시켜 과적합 위험이 있습니다.

---

### 2. ADASYN (Adaptive Synthetic Sampling)

- ADASYN은 SMOTE를 바탕으로, 소수 클래스 내에서도 학습이 어려운 (즉, 다수 클래스가 주변에 많은) 샘플에 더 많은 합성 샘플을 할당하는 기법입니다.
- 먼저 각 소수 클래스 샘플 $\( x_i \)$ 의 주변 K개의 최근접 이웃 중 다수 클래스 비율을 구해 어려움 정도를 수치화합니다:

$$
\[r_i = \frac{\text{Number of majority class neighbors of } x_i}{K}
\]
$$

- 이 $\( r_i \)$ 값들을 전체 소수 클래스 샘플에 대해 정규화하여 합이 1이 되도록 스케일링합니다:

$$
\[\hat{r}_i = \frac{r_i}{\sum_{j} r_j}
\]
$$

- 생성할 전체 샘플 수를 $\( G \)$ 라고 할 때, 각 소수 클래스 샘플별로 생성할 샘플 수는:

$$
\[g_i = \hat{r}_i \times G
\]
$$

- 이후 SMOTE 방식으로 각 샘플 별로 $\( g_i \)$ 개만큼 새로운 데이터를 합성합니다.
- 이로써 상대적으로 경계선 혹은 다수 클래스와 겹치는 어려운 영역에 더 많은 데이터를 생성하여 모델 학습을 돕습니다.

---

요약하자면, **SMOTE는 소수 클래스 균등 생성을 통해 데이터를 늘리고**, **ADASYN은 소수 클래스 내에서 어려운 샘플에 더 집중해 데이터 생성을 차등적으로 수행**하는 방식입니다[1][3][5].

출처 :
[1] https://datascience-hyemin.tistory.com/50
[2] https://woongsonvi.github.io/ai/AI6/
[3] https://sonstory.tistory.com/94
[4] https://dining-developer.tistory.com/27
[5] https://wikinist.tistory.com/196

### Borderline-SMOTE

**Borderline-SMOTE**는 SMOTE의 변형 기법으로, 소수 클래스 중에서 다수 클래스 경계(borderline) 근처에 위치한 어려운 샘플에 집중해 합성 데이터를 생성하는 방법입니다. 경계 근처 샘플은 분류기가 잘못 분류할 가능성이 높은 샘플이므로, 이들을 중심으로 증강하여 모델의 일반화 성능을 향상시키려는 목적입니다[1][2][3].

---

### Borderline-SMOTE의 기본 원리 및 수식

1. **경계 샘플 탐지 (Borderline Sample Identification):**  
   소수 클래스 샘플 $\( x_i \)$ 각각에 대해 전체 데이터(소수 + 다수 클래스)를 대상으로 K-최근접 이웃(KNN) 검색을 수행합니다.  
   - $\( N_{\text{maj}}(x_i) \): \( x_i \)$ 주변 K개 이웃 중 다수 클래스 개수  
   - $\( N_{\text{min}}(x_i) \): \( x_i \)$ 주변 K개 이웃 중 소수 클래스 개수

2. **샘플 군 분류:**  
   소수 클래스 샘플은 주변 다수 클래스 비율에 따라 세 모드로 분류합니다.  
   - **SAFE:** 대부분 소수 클래스 이웃  
   - **DANGER (경계 영역):** 다수 클래스 비율이 높음 (즉, 분류하기 어려움)  
   - **NOISE:** 완전히 다수 클래스에 둘러싸인 경우 (잡음)

   Borderline-SMOTE는 이 중 **DANGER** 군에 속하는 샘플에 대해서만 합성 샘플을 생성합니다.

3. **합성 샘플 생성:**  
   DANGER 샘플 $\( x_i \)$에 대해, 소수 클래스 내 K-최근접 이웃 $\( x_{\text{near}} \)$ 를 선택하고 다음 수식으로 새로운 합성 샘플을 만듭니다:  

$$
   \[   x_{\text{new}} = x_i + \delta \times (x_{\text{near}} - x_i), \quad \delta \sim U(0,1)
   \]
   $$
   
   여기서 $\(\delta\)$ 는 0과 1 사이의 균등분포에서 임의 추출한 값입니다.  

   이렇게 생성된 샘플들은 소수 클래스와 다수 클래스 경계 근처에서 더 밀집된 분포를 형성하여, 모델이 경계 근처를 더 명확히 학습하게 만듭니다[1][3][4].

---

### Borderline-SMOTE 알고리즘 요약

1. 각 소수 클래스 샘플 $\( x_i \)$ 에 대해 주변 $\( K \)$ 최근접 이웃을 찾음.  
2. 주변 이웃 중 다수 클래스 비율이 높으면 $\( x_i \)$ 를 DANGER(경계) 샘플로 분류.  
3. 오직 DANGER 샘플에 대해, 소수 클래스 내 이웃과 선형 보간해 합성 샘플 생성.  
4. 합성 샘플은 기존 소수 클래스의 경계에 집중되어 생성됨.

---

### 참고 사항

- Borderline-SMOTE는 두 가지 변형(종종 Borderline-1과 Borderline-2)가 있으며, 각각 합성 샘플 생성과정과 이웃 선정에 차이가 있습니다[2][3].  
- 다수 클래스와 경계 근처 샘플에 집중하므로, 일반 SMOTE보다 경계 분류 성능이 향상될 수 있습니다.

---

정리하면, **Borderline-SMOTE는 다수 클래스와 소수 클래스 경계에 위치한 소수 클래스 샘플을 대상으로 합성 샘플을 생성하는 기법**이며, 이를 위해 주변 다수 클래스 비율에 따라 소수 샘플을 분류하고, 아래 수식으로 새로운 샘플을 만듭니다.  

$$
\[\boxed{
x_{\text{new}} = x_i + \delta (x_{\text{near}} - x_i), \quad \text{for } x_i \text{ in DANGER group}, \quad \delta \sim U(0,1)
}
\]
$$

이를 통해 분류기가 경계 근처를 더 잘 학습하도록 데이터 분포를 조정합니다[1][3][4].

출처 :
[1] https://www.geeksforgeeks.org/machine-learning/smote-for-imbalanced-classification-with-python/
[2] https://www.youtube.com/watch?v=vQDy6EnhyL8
[3] https://cran.r-project.org/web/packages/smotefamily/smotefamily.pdf
[4] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html
[5] https://www.blog.trainindata.com/overcoming-class-imbalance-with-smote/

## 비용/가중치 조정의 포인트
- 대부분의 분류기는 class_weight를 지원하며, ‘balanced’는 자동으로 클래스 불균형 비율에 따라 가중치를 설정합니다.[14][12]
- 도메인 비용(예: False Negative가 훨씬 비쌀 때)을 반영해 커스텀 가중치/코스트 매트릭스를 설정하고, 이후 임계값까지 조정하면 사업지표 최적화에 유리합니다.[12][6]

## 임계값 튜닝
- 확률 예측 후 ROC/PR 곡선 상에서 조직의 비용 함수에 맞는 컷오프를 선택합니다. Scikit-learn의 TunedThresholdClassifierCV는 비용함수 기반 자동 튜닝을 제공합니다.[6][2]
- 재현율 상한선(규제/안전)과 정밀도 하한선(운영비/후속조치 가능 인력)을 함께 제약으로 두고 탐색하는 것이 실무적입니다.[2][6]

## 엔드투엔드 파이프라인 예시(sklearn + imbalanced-learn)
- 목적: Tabular 이진 분류(예: 사기 탐지)에서 “SMOTE → 표준화 → 비용가중 로지스틱 → 임계값 튜닝”을 교차검증으로 묶습니다.[1][5][12]
- 핵심: 데이터 누출 방지(분할→오버샘플→스케일 순), class_weight 병행, PR-AUC·G-mean 모니터링, 최종 임계값 도출입니다.[3][1][6]

```python
# Python 3.10+
# pip install scikit-learn imbalanced-learn

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import classification_report_imbalanced
import numpy as np

# 1) 데이터 생성(불균형)
X, y = make_classification(n_samples=20000, n_features=30, n_informative=10, n_redundant=5,
                           weights=[0.98, 0.02], flip_y=0.001, class_sep=1.0, random_state=42)

# 2) 파이프라인: (CV 안에서) SMOTE -> 스케일 -> 비용가중 로지스틱
pipe = ImbPipeline(steps=[
    ('smote', SMOTE(k_neighbors=5, random_state=42)),
    ('scaler', StandardScaler(with_mean=False)),  # 희소 가능 시 with_mean=False
    ('clf', LogisticRegression(class_weight='balanced', max_iter=200, n_jobs=-1, solver='saga'))
])

# 3) 스코어: PR-AUC(AP), ROC-AUC, 재현율-정밀도 trade-off용 커스텀도 가능
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'pr_auc': make_scorer(average_precision_score, needs_proba=True),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
}

cvres = cross_validate(pipe, X, y, scoring=scoring, cv=cv, return_estimator=True, n_jobs=-1)

print({k: np.mean(v) for k, v in cvres.items() if 'test_' in k})

# 4) 홀드아웃에서 임계값 튜닝(간단 버전): PR-커브 기반 또는 비용 최소화 기준
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

pipe.fit(Xtr, ytr)
proba = pipe.predict_proba(Xte)[:, 1]

prec, rec, th = precision_recall_curve(yte, proba)
# 예시 비용함수: C_FN=10, C_FP=1 → 비용 최소 임계값 선택
C_FN, C_FP = 10, 1
# 추정 FN, FP 계산을 위해 각 임계값에서 예측라벨을 만들어 근사
def cost_at_threshold(t):
    yhat = (proba >= t).astype(int)
    FN = np.sum((yte == 1) & (yhat == 0))
    FP = np.sum((yte == 0) & (yhat == 1))
    return C_FN*FN + C_FP*FP

ths = th if th.size > 0 else np.array([0.5])
best_t = min(ths, key=cost_at_threshold)

# 최종 리포트(임계값 적용)
yhat_final = (proba >= best_t).astype(int)
print(f"Chosen threshold: {best_t:.3f}")
print(classification_report_imbalanced(yte, yhat_final))
```
이 파이프라인은 CV 안에서만 오버샘플링이 수행되도록 하여 누출을 막고, class_weight와 함께 로지스틱 회귀의 결정경계를 조정합니다. 이후 홀드아웃에서 비용함수 기반 임계값을 고르고, G-mean/IBA 포함 리포트를 출력합니다.[5][1][3][12][6]

## 딥러닝으로 확장할 때
- 배치 샘플링: 소수 클래스를 더 자주 뽑는 **스트래티파이드 배치**를 사용하거나, 학습 중 가중 손실(예: BCE with logits + pos_weight)을 적용합니다.[9][1]
- Focal Loss: 어려운 샘플에 더 큰 가중치를 부여해 클래스 불균형과 hard example을 동시에 다룹니다. 마지막에 임계값 튜닝을 잊지 않습니다.[1][2]

## 실무 체크리스트
- 데이터 분할 → 샘플링/스케일링은 반드시 분할 내부에서 수행합니다(파이프라인/전처리기 사용).[9][1]
- 지표 세트: PR-AUC, 재현율, 정밀도, G-mean/IBA, 운영 KPI(경보량, 케이스 처리비용)를 함께 봅니다.[3][6][2]
- 경계-노이즈 주의: SMOTE/ADASYN은 경계를 강화하지만 아웃라이어 연결/노이즈 증폭 리스크가 있어 모델 규제·특징 선택을 병행합니다.[18][5]
- 비용 기반 운영: class_weight와 임계값을 사업 비용에 맞춰 튜닝하고, 주기적으로 데이터 드리프트를 점검합니다.[14][12][6]

## 추가로 보면 좋은 자료
- TensorFlow 튜토리얼: 불균형 분류 워크플로우 전반 예시.[1]
- 평가 지표 가이드: 정확도 한계와 PR/ROC·G-mean 소개.[2]
- imbalanced-learn: oversampling/리포트 API 문서.[5][3]
- 비용민감 학습: class_weight와 결정경계 변화 이해.[12][6]

이 글의 목적은 “올바른 지표-샘플링-가중치-임계값”을 하나의 파이프라인으로 엮어, 재현 가능한 실무 성능 향상을 만드는 것입니다. 위 코드와 체크리스트를 베이스라인으로 삼아 데이터 특성에 맞는 변형(SMOTE 변종, 다른 분류기, 비용함수)을 실험해 보시길 권합니다.[6][5][12][1]

[1](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
[2](https://www.machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)
[3](https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.classification_report_imbalanced.html)
[4](https://isi-web.org/sites/default/files/2024-02/Handling-Data-Imbalance-in-Machine-Learning.pdf)
[5](https://imbalanced-learn.org/stable/over_sampling.html)
[6](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html)
[7](https://www.sciencedirect.com/science/article/pii/S0031320319300950)
[8](https://pmc.ncbi.nlm.nih.gov/articles/PMC10688675/)
[9](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets)
[10](https://www.kaggle.com/code/residentmario/oversampling-with-smote-and-adasyn)
[11](https://www.geeksforgeeks.org/machine-learning/smote-for-imbalanced-classification-with-python/)
[12](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/CostSensitive.html)
[13](https://www.geeksforgeeks.org/python/how-to-implement-cost-sensitive-learning-in-decision-trees/)
[14](https://www.blog.trainindata.com/cost-sensitive-learning-for-imbalanced-data/)
[15](https://towardsdatascience.com/class-imbalance-smote-borderline-smote-adasyn-6e36c78d804/)
[16](https://www.sciencedirect.com/science/article/pii/S2666827024000732)
[17](https://housekdk.gitbook.io/ml/ml/tabular/imbalanced-learning/oversampling-basic-smote-variants)
[18](http://www.diva-portal.org/smash/get/diva2:1519153/FULLTEXT01.pdf)
[19](https://rfriend.tistory.com/773)
[20](https://www.kaggle.com/code/para24/survival-prediction-using-cost-sensitive-learning)
[21](https://www.linkedin.com/pulse/class-weights-cost-sensitive-learning-enhancing-model-debasish-deb-tesrf)

# Reference
https://rfriend.tistory.com/773
