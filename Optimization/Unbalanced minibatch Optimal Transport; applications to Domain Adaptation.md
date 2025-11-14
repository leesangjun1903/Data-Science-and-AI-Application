# Unbalanced minibatch Optimal Transport; applications to Domain Adaptation

### 핵심 주장 및 주요 기여 요약

본 논문은 **균형 잡힌 미니배치 최적 수송(balanced minibatch optimal transport, MBOT)의 근본적 한계**를 지적하고, **불균형 최적 수송(Unbalanced Optimal Transport, UOT)을 미니배치 수준에서 결합**한 새로운 접근법을 제시합니다.[1]

핵심 기여는 다음과 같습니다:

**1. 미니배치 OT의 문제점 규명**: 미니배치 전략은 계산 효율성을 제공하지만, 표본 추출로 인해 **불바람직한 부드러움 효과(smoothing effects)**를 야기하며, 전체 데이터의 클러스터 구조를 반영하지 못해 **부정확한 표본 쌍(mismatched pairs)**을 생성합니다.[1]

**2. 불균형 미니배치 최적 수송 제안**: 미니배치 수준에서 불균형 OT를 적용함으로써 부분 도메인 적응(partial domain adaptation)을 자연스럽게 지원하면서도 계산 효율성을 유지합니다.[1]

**3. 이론적 보장**: 무편향 추정량(unbiased estimators), 집중 한계(concentration bounds), Clarke 미분을 통한 구배 존재성을 증명하여 확률적 경사 하강법(SGD) 수렴성을 보장합니다.[1]

---

### 문제 설정 및 해결 방법

#### 문제의 정의

전통적 최적 수송은 한 확률 분포 $$\alpha$$를 다른 확률 분포 $$\beta$$로 이송할 때의 최소 비용을 계산합니다. 하지만 다음의 두 가지 도전 과제가 있습니다:[1]

1. **계산 복잡도**: 표준 OT의 계산 복잡도는 $$O(n^3 \log n)$$로, 대규모 데이터셋에서 불가능합니다.
2. **미니배치 효과**: 미니배치를 사용하면 계산 복잡도가 감소하지만, 작은 배치 내에서 분포 통계가 왜곡되어 클러스터 간 잘못된 매칭이 발생합니다.[1]

#### 불균형 최적 수송 정식화

불균형 OT는 엄격한 주변(marginal) 제약을 완화합니다:[1]

$$UOT_{\tau,\epsilon}(\alpha, \beta, c) = \min_{\pi \in \mathcal{M}^+(\mathcal{X} \times \mathcal{Y})} \int c \, d\pi + \epsilon KL(\pi | \alpha \otimes \beta) + \tau(KL(\pi_1 \| \alpha) + KL(\pi_2 \| \beta))$$

여기서:
- $$\tau \geq 0$$: 주변 페널티화 계수
- $$\epsilon \geq 0$$: 엔트로피 정규화 계수
- $$\pi_1, \pi_2$$: 수송 계획의 주변분
- $$KL$$: 쿨백-라이블러 발산

이 정식화의 핵심은 **주변 제약을 완전히 만족할 필요가 없다**는 것으로, 이를 통해 **이상치(outliers)와 부분 매칭에 강인성을 제공**합니다.[1]

#### 미니배치 불균형 OT 정의

미니배치 UOT는 다음과 같이 정의됩니다:[1]

$$\hat{h}^m(X, Y) = \binom{n}{m}^{-2} \sum_{I,J \in \mathcal{P}_m} h(u_m, u_m, C_{I,J})$$

**불완전 추정량(incomplete estimator)**:
$$\hat{h}^m_k(X, Y) = k^{-1} \sum_{(I,J) \in D_k} h(u_m, u_m, C_{I,J})$$

여기서 $$D_k$$는 $$k$$개의 무작위로 선택된 미니배치 쌍들의 집합입니다.[1]

---

### 이론적 성질

#### 1. 강인성 증명 (Lemma 1)

**이상치에 대한 UOT의 강인성**:[1]

$$z$$가 $$\alpha$$의 지지집합 밖에 있는 이상치이고, $$\tilde{\alpha} = \zeta\alpha + (1-\zeta)\delta_z$$일 때:

$$UOT_{\tau,0}(\tilde{\alpha}, \beta, c) \lesssim \zeta \cdot UOT_{\tau,0}(\alpha, \beta, c) + 2\tau(1-\zeta)(1-e^{-m(z)/2\tau})$$

이에 비해 균형 OT는 $$z$$가 멀어질수록 손실이 무한정 증가합니다. 반면 UOT의 손실은 $$e^{-m(z)/2\tau}$$ 항으로 인해 **포화(saturation)**되므로 더 강인합니다.[1]

#### 2. 집중 한계 (Theorem 1)

**편차 한계**:[1]

$$\mathbb{P}\left(\hat{h}^m_k - \mathbb{E}h \leq \frac{M_h}{\sqrt{n}}\sqrt{\frac{1+\log 2}{\delta}} + \frac{M_h}{\sqrt{2k}}\sqrt{\frac{1+\log 2}{\delta}}\right) \geq 1-\delta$$

이 한계의 핵심:
- 샘플 수 $$n$$이 증가하고 배치 수 $$k$$가 증가할 때 수렴
- 수렴률 $$\sim m^2/n$$은 거의 최적(nearly optimal)
- **차원 의존성이 없음** (기존 UOT의 저주를 회피)

#### 3. 편향 없는 구배 (Theorem 2)

Wasserstein 거리의 악명 높은 문제인 **편향된 구배(biased gradients)**를 극복합니다.[1]

**Clarke 정규성**을 통해 증명:
$$\mathbb{E}[\nabla h(u_m, u_m, C_m|X,Y)] = \nabla \mathbb{E}[h(u_m, u_m, C_m|X,Y)]$$

이는 **SGD가 거의 확실하게 Clarke 일반화 미분의 임계점으로 수렴**함을 의미합니다.[1]

---

### 일반화 성능 향상 메커니즘

#### 1. 미니배치 평활화 효과 제거

**Figure 1**에서 보이듯이, 균형 MBOT는 표본 크기가 작을 때 다른 클러스터의 샘플들을 오매칭시킵니다. 반면 UMBOT는 이러한 "불가능한" 쌍들에 페널티를 부여하여 **클러스터 구조를 보존**합니다.[1]

#### 2. 부분 도메인 적응 (PDA)

전통적 OT에서는 **클래스 비율 불일치(class imbalance)**가 부정적 이전(negative transfer)을 유발합니다. UOT의 주변 완화는 자연스럽게 목표 도메인에서 존재하지 않는 클래스를 처리할 수 있습니다.[1]

#### 3. 표본 효율성 개선

**Figure 6**의 배치 크기 민감도 분석에서:
- JUMBOT: 배치 크기 변화에 대해 안정적 ($$\pm 0.1\%$$ 변동)
- DEEPJDOT: 작은 배치에서 성능 급락 (4-6% 감소)

이는 UMBOT가 **작은 배치에서도 견고한 일반화 능력**을 제공함을 시사합니다.[1]

***

### 실험 결과 분석

#### 도메인 적응 벤치마크

| 데이터셋 | DEEPJDOT | JUMBOT | 향상도 |
|---------|----------|---------|---------|
| Office-Home (평균) | 66.6% | 70.1% | +3.5pp |
| VisDA-2017 | 68.0% | 72.5% | +4.5pp |
| 숫자 데이터셋 | 94.5% | 98.0% | +3.5pp |

특히 **Office-Home**은 실제 도메인 적응의 가장 어려운 벤치마크로, JUMBOT의 우월성은 견고합니다.[1]

#### 부분 도메인 적응

목표 도메인이 소스의 25개 클래스만 포함할 때:[1]
- JUMBOT: 75.5% (12개 중 9개 작업에서 최고)
- 이전 최고: BA3US 73.6%
- **+1.9pp 향상**

#### 오버피팅 분석

**Figure 7**에서 DEEPJDOT는 에포크 30부터 모든 클래스에서 오버피팅을 보이지만, JUMBOT은 안정적입니다. 이는 **UOT의 정규화 효과**를 시사합니다.[1]

#### 매칭 품질 분석

MNIST→M-MNIST 작업에서 다른 레이블 간 매칭 비율:[1]
- DEEPJDOT: ~7% (잘못된 매칭)
- JUMBOT: ~0.7% (강인한 매칭)

***

### 한계점

#### 1. 하이퍼파라미터 민감도
$$\tau$$ (주변 페널티)와 $$\epsilon$$ (엔트로피 정규화)의 선택이 중요하며, 최적값이 데이터셋마다 상이합니다.[1]

#### 2. 계산 복잡도
미니배치 UOT는 여전히 일반화 Sinkhorn 알고리즘으로 $$O(n^2)$$ 복잡도를 가지며, 매우 큰 배치에서는 GPU 메모리 문제가 남아있습니다.[1]

#### 3. 이론-실제 간극
집중 한계는 $$\delta$$ 값에 민감하며, 현실적 설정에서 이론적 보장의 실용성 제한이 있습니다.[1]

***

### 앞으로의 연구에 미치는 영향 및 고려사항

#### 1. 최근 후속 연구 동향

**부분 최적 수송 활용** (2022): Nguyen et al.의 m-POT는 JUMBOT보다 더 선택적으로 전체 질량을 이송하여 추가적 유연성을 제공합니다.[2]

**불균형 CO-OT** (2023): Unbalanced COOT는 표본뿐 아니라 **특성 정렬(feature alignment)**도 동시에 학습하여 이질적 도메인 적응에 더 강인합니다.[3]

**적응적 정규화** (2023): OT with Adaptive Regularisation(OTARI)는 점별 질량 제약으로 **더 희소한 수송 계획**을 생성합니다.[4]

#### 2. 대규모 모델 시대의 함의

**Vision Transformer와의 통합**: 최근 논문들은 SAM 등 대규모 사전학습 모델을 도메인 적응에 활용하고 있으며, UMBOT의 구배 편향 제거가 이들 모델의 미세조정(fine-tuning)에 유리합니다.[5][6]

**자기지도 학습(SSL)과의 조합**: SSL과 OT 기반 도메인 적응의 결합이 새로운 방향으로 부상하고 있습니다.[6]

#### 3. 향후 연구 시 고려사항

**1. 하이퍼파라미터 자동 선택**
- 현재 $$\tau$$, $$\epsilon$$은 수동 튜닝 필요
- **교차 검증 기반 자동화** 또는 **메타 학습** 적용 고려

**2. 미분 가능 Sinkhorn 구현 개선**
- KeOps 패키지 활용으로 GPU 메모리 효율성 향상
- 저랭크 분해 기반 근사 탐색

**3. 동적 도메인 시나리오**
- 지속적 학습(continual learning) 설정에서의 UMBOT 활용
- 시간 변화 분포에 대한 강인성 분석

**4. 멀티소스 도메인 적응**
- 다중 소스에서 최적의 가중치 조합 학습
- UMBOT의 부분 이송 특성을 활용한 신뢰할 수 없는 소스 필터링

**5. 인과적 도메인 적응**
- 인과 그래프 기반 도메인 불변 특성 학습
- UMBOT로 인과 구조 정렬

#### 4. 이론적 확장 방향

**1. 비유클리드 기하학**
- 리만 다양체 상에서의 UMBOT
- 그래프 기반 도메인 적응에의 응용

**2. 개인 정보 보호 측면**
- Differential privacy를 UMBOT에 적용
- 페더레이션 학습에서의 개인정보 보호 도메인 적응

**3. 보증 있는 부분 도메인 적응**
- 대역폭 선택 최적화
- 부분 매칭의 최적성 인증서 개발

***

### 결론

본 논문 "Unbalanced Minibatch Optimal Transport; applications to Domain Adaptation"는 **미니배치 전략의 본질적 평활화 문제를 식별하고**, 이를 불균형 OT로 우아하게 해결하는 **이론적으로 견고하면서 실무적으로 효과적인** 접근법을 제시합니다.[1]

이론적 기여(편향 없는 구배, 집중 한계)와 실증적 성능(SOTA 달성) 간의 균형, 그리고 자연스러운 부분 도메인 적응 지원은 이 연구를 **도메인 적응 분야의 이정표**로 만듭니다.[1]

앞으로 **대규모 기초 모델과의 통합**, **동적 환경 적응**, **인과 구조 학습**이 핵심 연구 방향이 될 것으로 예상되며, UMBOT의 강인성 메커니즘은 이러한 신흥 과제들에서 중요한 역할을 할 것입니다.[4][3][5][6]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/620f8b8d-ee9d-4e75-a608-8b5b3b1cdec5/2103.03606v1.pdf)
[2](https://proceedings.mlr.press/v162/nguyen22e/nguyen22e.pdf)
[3](https://arxiv.org/abs/2205.14923)
[4](http://arxiv.org/pdf/2310.02925.pdf)
[5](https://openreview.net/forum?id=xsF3ngGYPv)
[6](https://arxiv.org/html/2510.15615v1)
[7](https://arxiv.org/pdf/2209.04594.pdf)
[8](https://arxiv.org/abs/2205.15424)
[9](http://arxiv.org/pdf/2404.10261.pdf)
[10](https://arxiv.org/pdf/2401.15952.pdf)
[11](http://arxiv.org/pdf/2112.02073.pdf)
[12](http://arxiv.org/pdf/2503.08155.pdf)
[13](https://icml.cc/media/icml-2021/Slides/8555.pdf)
[14](https://arxiv.org/abs/2103.03606)
[15](https://openreview.net/pdf/c287715aed992250c354d29d2a6bab5728f2101e.pdf)
[16](https://project.inria.fr/maclean/files/2021/07/20210705_slides_Nicolas_Courty.pdf)
[17](https://www.scribd.com/document/813805061/Improving-and-Generalizing-Flow-Based-Generative-Models-with-Minibatch-Optimal-Transport)
[18](https://pure.ewha.ac.kr/en/publications/deep-unsupervised-domain-adaptation-a-review-of-recent-advances-a)
[19](https://github.com/kilianFatras/JUMBOT)
