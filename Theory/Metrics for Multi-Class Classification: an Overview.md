# Metrics for Multi-Class Classification: an Overview 

### 1. 논문의 핵심 주장 및 기여

"Metrics for Multi-Class Classification: an Overview"는 다중 클래스 분류 문제에서 모델의 성능을 평가하기 위한 다양한 평가 지표들을 체계적으로 검토하는 백서이다. 본 논문의 핵심 주장은 단일 지표로는 분류 모델의 전면적인 성능 평가가 불가능하며, 데이터셋의 특성과 문제의 맥락에 따라 적절한 메트릭을 선택해야 한다는 것이다.[1]

**주요 기여:**

- **혼동 행렬(Confusion Matrix) 기반 메트릭의 체계적 분류**: Accuracy, Precision, Recall, F1-Score, Balanced Accuracy 등의 전통적 메트릭을 상세히 설명[1]
- **다중 클래스 확장의 명확화**: Macro-Average, Micro-Average, 가중 평균 방식의 차이점을 명확히 제시[1]
- **포괄적 메트릭 비교**: Matthews Correlation Coefficient (MCC)와 Cohen's Kappa 같은 통계적 지표까지 포함[1]
- **실무적 가이던스 제공**: 각 메트릭의 장단점과 사용 상황을 구체적으로 제시[1]

***

### 2. 논문이 해결하는 문제와 제안 방법

#### 2.1 핵심 문제

다중 클래스 분류 문제에서는 다음과 같은 근본적 문제들이 존재한다:[1]

$$Y \in \{1, 2, ..., K\}$$ 를 K개의 클래스로 정의할 때, 단순히 정확도(accuracy)만으로는 클래스별 성능 차이를 파악할 수 없다. 특히 불균형 데이터셋에서는 다수 클래스의 높은 정확도가 소수 클래스의 저조한 성능을 은폐할 수 있다.[1]

#### 2.2 제안된 방법들

**1) 기본 메트릭 (Confusion Matrix 기반):**

$$
\text{Accuracy} = \frac{\sum_{k=1}^{K} \text{TP}_k}{\sum_{i,j} C_{ij}}
$$

여기서 $$C_{ij}$$ 는 혼동 행렬의 원소이다.[1]

**2) 클래스별 성능 평가:**

각 클래스 k에 대해:

$$
\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}
$$

$$
\text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}
$$

[1]

**3) 다중 클래스 F1-Score:**

Macro F1-Score:

$$
\text{Macro F1} = 2 \times \frac{\text{Macro Precision} \times \text{Macro Recall}}{\text{Macro Precision} + \text{Macro Recall}}
$$

여기서 Macro Precision과 Macro Recall은 각 클래스의 값들의 산술 평균이다.[1]

Micro F1-Score:

$$
\text{Micro F1} = \frac{\sum_{k=1}^{K} \text{TP}_k}{\text{Grand Total}} = \text{Accuracy}
$$

[1]

**4) Balanced Accuracy (불균형 데이터 대응):**

$$
\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \text{Recall}_k
$$

[1]

**5) Matthews Correlation Coefficient (다중 클래스):**

$$
\text{MCC} = \frac{c \times s - \sum_{k=1}^{K} p_k \times t_k}{\sqrt{(s^2 - \sum_{k=1}^{K} p_k^2)(s^2 - \sum_{k=1}^{K} t_k^2)}}
$$

여기서 c는 정확히 분류된 원소의 총 개수, s는 전체 원소 개수, $$p_k$$는 클래스 k로 예측된 횟수, $$t_k$$는 실제 클래스 k의 횟수이다.[1]

**6) Cohen's Kappa (다중 클래스):**

$$
\kappa = \frac{c \times s - \sum_{k=1}^{K} p_k \times t_k}{s^2 - \sum_{k=1}^{K} p_k \times t_k}
$$

[1]

#### 2.3 모델 구조

논문의 메트릭 분류 구조:

- **혼동 행렬 기반**: Accuracy, Balanced Accuracy, Precision, Recall
- **조화 평균 기반**: F1-Score (Macro, Micro)
- **상관 관계 기반**: MCC, Cohen's Kappa
- **확률 분포 기반**: Cross-Entropy[1]

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상 방법

**1) 메트릭 선택의 최적화:**

- **불균형 데이터셋**: Balanced Accuracy, Macro F1-Score, MCC 선택 권장[1]
- **클래스별 모니터링**: Macro 방식으로 각 클래스의 개별 성능 파악 가능[1]
- **가중 평균 적용**: 클래스의 중요도를 반영한 Balanced Accuracy Weighted 활용[1]

$$
\text{Balanced Accuracy Weighted} = \frac{\sum_{k=1}^{K} \text{Recall}_k \times w_k}{K \times W}
$$

여기서 $$w_k$$ 는 클래스 k의 가중치이다.[1]

**2) ROC-AUC와 Precision-Recall 곡선의 활용:**

최신 연구에 따르면, 단일 임계값 대신 다양한 임계값에서의 성능을 시각화하는 것이 중요하다. 특히 불균형 데이터에서는 ROC 곡선이 Accuracy보다 신뢰할 수 있는 평가를 제공한다.[2]

**3) 강건한 메트릭 개발:**

최근 연구에서는 클래스 불균형에 견딜 수 있는 메트릭 수정안들이 제시되었다. 예를 들어, F-score와 MCC의 강건한 변형은 소수 클래스의 탐지율(True Positive Rate)을 0이 아닌 값으로 유지한다.[2]

#### 3.2 핵심 한계

**1) Accuracy의 한계:**

불균형 데이터셋에서 다수 클래스 예측에만 최적화된 모델도 높은 정확도를 달성할 수 있다. 예를 들어, 소수 클래스가 1%인 데이터셋에서 모든 샘플을 다수 클래스로 분류하면 99% 정확도를 얻는다.[1]

**2) Cross-Entropy의 한계:**

Cross-Entropy는 예측 확률 분포의 전체 형태를 고려하지 않고, 참 클래스의 확률만 사용한다. 따라서 서로 다른 확률 분포이지만 같은 true class 확률을 가진 예측들을 구별하지 못한다.[1]

**3) MCC의 변동성:**

MCC는 불균형이 심한 설정에서 훈련 중 매우 큰 변동을 보인다. 이는 모델 훈련의 수렴성 평가를 어렵게 만든다.[1]

**4) Macro F1의 소수 클래스 편향:**

Macro F1-Score는 모든 클래스에 동일한 가중치를 부여하므로, 작은 클래스의 오류가 전체 점수에 과도하게 영향을 미칠 수 있다.[1]

***

### 4. 일반화 성능 향상 가능성

#### 4.1 클래스 불균형과 일반화 성능의 관계

최신 연구에 따르면, **클래스 분포**는 모델의 일반화 성능에 직접적인 영향을 미친다. 특히:[3][2]

- **문제점**: 전통적 메트릭(F-score, Jaccard, MCC)은 소수 클래스 비율이 0으로 접근할 때 Bayes 분류기의 참 양성률(TPR)도 0으로 수렴한다.[2]
- **결과**: 불균형 문제에서 이들 메트릭은 소수 클래스를 무시하는 분류기를 우호적으로 평가한다.[2]

#### 4.2 강건한 메트릭을 통한 일반화 성능 개선

**강건한 F-score 수정안:**

$$
\text{MCC}_{\text{rb}} = \sqrt{\frac{d + \pi(1-\pi)}{1}} \times \frac{\pi_{1|1}\pi_{0|0} - (1-\pi_{1|1})(1-\pi_{0|0})}{\sqrt{d + [\pi\pi_{1|1} + (1-\pi)(1-\pi_{0|0})][\pi(1-\pi_{1|1}) + (1-\pi)\pi_{0|0}]}}
$$

이 수정안은 매개변수 d를 통해 클래스 불균형에 대한 강건성을 제어하며, 임계값 δ*이 π가 0으로 접근해도 유계를 유지하도록 보장한다.[2]

#### 4.3 다중 클래스 불균형 평가의 새로운 접근

**IMCP (Imbalanced Multiclass Classification Performance) 곡선:**

최근 연구에서는 클래스 분포에 무관한 평가 메트릭이 제시되었다. 이 방법은:[3]

- Hellinger 거리를 이용한 예측 확률 분포와 실제 분포 간의 거리 측정[3]
- 각 샘플의 불확실성을 정량화[3]
- 클래스별 개별 성능 평가 가능[3]

**IMCP의 이점:**

$$
\text{AU(IMCP)} = \int_0^1 \text{(sorted accuracy)} \, dx
$$

이 메트릭은 종양 분류 같은 실제 문제에서 정확도 92.4%인 시스템이 실제로는 AU(IMCP) 57.8%만을 달성할 수 있음을 보여준다. 이는 높은 정확도가 실제 신뢰성을 반영하지 못함을 시사한다.[3]

#### 4.4 일반화 성능 향상을 위한 실전 권장사항

**1) 메트릭 조합 사용:**

단일 메트릭 대신 여러 메트릭을 함께 사용해야 한다:[1]
- 정확도는 전체 성능의 기초선으로 제시
- Balanced Accuracy로 클래스별 성능 평가
- F1-Score (특히 Macro)로 클래스 간 성능 편차 확인
- MCC로 전반적 상관성 검증

**2) 임계값 최적화:**

분류 규칙의 임계값을 메트릭별로 최적화할 수 있다. 예를 들어:

$$
\delta^* = \arg\max_{\delta} M(C_\delta)
$$

여기서 다른 메트릭은 다른 최적 임계값을 산출한다.[2]

**3) 교차 검증 전략:**

클래스 불균형이 있는 경우 계층화된 K-겹 교차 검증(stratified K-fold cross-validation)을 사용하여 각 fold에서 클래스 분포를 유지해야 한다.[1]

***

### 5. 논문이 앞으로의 연구에 미치는 영향

#### 5.1 학문적 영향

**1) 메트릭 선택의 중요성 강조:**

본 논문은 메트릭이 단순한 수치가 아니라 모델 개발 과정의 핵심 결정 도구임을 보여준다. 이는 AutoML 시스템과 hyperparameter tuning에서 목적 함수 선택의 중요성으로 이어진다.[1]

**2) 통계적 기초의 중요성:**

MCC와 Cohen's Kappa의 상세 분석은 분류 메트릭이 단순한 산술이 아니라 통계적 독립성과 상관성의 개념과 깊게 연결되어 있음을 보여준다.[1]

#### 5.2 최신 연구 동향 (2023-2025)

최신 연구는 논문의 핵심 메시지를 발전시키고 있다:

**1) 불균형 문제의 심화:**

최근 연구는 표준 메트릭의 문제를 더욱 명확히 했다. 소수 클래스 비율이 매우 작을 때 (π → 0), F-score와 MCC는 다음과 같은 문제를 보인다:[2]

- 최적 임계값 δ*가 무한대로 발산 (F-score의 경우) 또는 매우 큼 (MCC의 경우)[2]
- 결과적으로 소수 클래스의 탐지 불가능[2]

**2) 새로운 메트릭 개발:**

- **IMCP 곡선**: 클래스 분포 무관한 다중 클래스 평가[3]
- **강건 F-score/MCC**: 불균형에 견딜 수 있는 매개변수화된 버전[2]
- **메타러닝 기반 메트릭**: 데이터셋 특성에 적응하는 메트릭[4]

**3) 의료 진단 응용:**

최신 연구는 종양 분류, COVID-19 심각도 평가, 신경퇴행성 질환 진단 등에서 불균형 다중 클래스 평가의 중요성을 강조한다.[3]

#### 5.3 앞으로의 연구 시 고려할 점

**1) 메트릭과 비즈니스 목표의 정렬:**

- **의료 진단**: 거짓 음성을 최소화하기 위해 Recall 우선
- **신용 평가**: 거짓 양성을 최소화하기 위해 Precision 우선
- **균형 잡힌 시스템**: Balanced Accuracy 또는 F1-Score

**2) 도메인 특화 메트릭 개발:**

각 분야의 비용 불균형을 반영한 비용 가중 메트릭(cost-weighted metrics) 개발이 필요하다.[2]

**3) 설명 가능성(Explainability)과의 통합:**

IMCP 같은 새로운 메트릭은 클래스별 성능뿐 아니라 불확실성 영역을 시각화하여 모델 이해도를 향상시킨다.[3]

**4) 강건성 검증:**

분류 메트릭의 강건성을 이론적으로 증명하기 위해, 클래스 불균형 외에도 다음을 고려해야 한다:[5][2]
- 라벨 노이즈
- 도메인 시프트
- 분포 변화(concept drift)

**5) 실시간 모니터링:**

배포된 모델의 성능 모니터링에는 단일 메트릭 대신 메트릭 조합을 사용하여 성능 저하를 조기에 감지해야 한다.[5][1]

***

### 결론

"Metrics for Multi-Class Classification: an Overview"는 다중 클래스 분류 문제에서 모델 평가의 기초를 제공한다. 논문의 핵심 통찰은 **메트릭 선택이 모델 개발의 중심**이라는 것이다. 최근 연구는 클래스 불균형에 견딜 수 있는 새로운 메트릭을 개발하고 있으며, 이는 의료 진단, 신용 평가, 이상 탐지 등 실무 응용에서 특히 중요해진다.

**연구자로서의 실천 권고:**
1. 단일 메트릭이 아닌 **메트릭 조합** 사용
2. 불균형 데이터에서는 **Balanced Accuracy 또는 Macro F1-Score** 우선 고려
3. **IMCP 곡선**과 같은 최신 방법 검토
4. 메트릭과 **비즈니스 목표의 명시적 정렬**
5. **클래스별 성능의 개별 모니터링**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d98b17e7-e789-4150-9b30-f03c7b7d1b94/2008.05756v1.pdf)
[2](https://arxiv.org/abs/2404.07661)
[3](https://pmc.ncbi.nlm.nih.gov/articles/PMC11087593/)
[4](https://www.ijcai.org/proceedings/2024/0524.pdf)
[5](https://www.sciencedirect.com/topics/computer-science/generalization-performance)
[6](http://arxiv.org/pdf/2412.14489.pdf)
[7](http://arxiv.org/pdf/1310.1949.pdf)
[8](https://arxiv.org/html/2501.14460v1)
[9](https://arxiv.org/pdf/2205.15860.pdf)
[10](https://www.mdpi.com/1424-8220/23/1/9/pdf?version=1672202200)
[11](https://arxiv.org/pdf/2401.05069.pdf)
[12](https://arxiv.org/pdf/2112.09727.pdf)
[13](http://arxiv.org/pdf/2305.13349.pdf)
[14](https://www.machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)
[15](https://arxiv.org/html/2409.01498v1)
[16](https://openreview.net/forum?id=p9k5MS0JAL&noteId=JSsjmMmfKP)
[17](https://community.deeplearning.ai/t/evaluation-metrics-for-imbalanced-image-classification/320325)
[18](https://causalwizard.app/inference/article/metrics)
[19](https://www.sciencedirect.com/science/article/pii/S131915782400051X)
