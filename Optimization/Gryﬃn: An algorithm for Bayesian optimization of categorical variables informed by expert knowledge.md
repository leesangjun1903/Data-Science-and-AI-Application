# Gryﬃn: An algorithm for Bayesian optimization of categorical variables informed by expert knowledge

### 1. 핵심 주장과 주요 기여

**Gryffin**은 화학 및 재료과학에서 범주형 변수(categorical variables)의 효율적인 선택을 위한 베이지안 최적화 프레임워크입니다. 이 논문의 핵심 주장은 다음과 같습니다.[1]

**주요 기여:**

1. **범주형 최적화의 직접 탐색**: 기존 방식(예: 변분 오토인코더를 이용한 연속공간 매핑)과 달리, Gryffin은 범주형 공간을 직접 탐색합니다.[1]

2. **도메인 지식 활용**: 물리화학 기술자(physicochemical descriptors)를 통해 범주형 선택지 간의 유사성을 정량화하여 검색을 가속화합니다.[1]

3. **동적 기술자 개선**: 최적화 과정 중 제공된 기술자를 변환하여 더 정보력 있는 새로운 기술자를 자동으로 구성합니다.[1]

4. **과학적 통찰력 제공**: 알고리즘이 학습한 기술자의 중요도를 해석하여 물리적, 화학적 직관을 제공합니다.[1]

***

### 2. 문제 정의 및 제안 방법

#### 2.1 해결하는 문제

화학 및 재료과학에서의 자율 실험 플랫폼 개발에는 두 가지 주요 과제가 있습니다:[1]

- **혼합 변수 공간**: 온도나 유량 같은 연속 변수와 촉매나 용매 같은 범주형 변수를 동시에 최적화해야 합니다.
- **범주형 변수 최적화의 어려움**: 범주형 변수 간에 자연적인 순서가 없어 기존 연속 최적화 방법을 직접 적용할 수 없습니다.

기존 접근 방식의 한계:[1]

- One-hot 인코딩은 모든 범주형 선택지를 동등하게 유사한 것으로 취급하여, 실제 화학적 유사성을 반영하지 못합니다.
- 핸드크래프트된 휴리스틱이나 인간의 개입이 필요하여 처리량이 낮습니다.

#### 2.2 제안 방법과 수식

**Naive Gryffin**은 커널 밀도 추정(kernel density estimation)을 기반으로 합니다. 범주형 변수를 심플렉스(simplex) $$\Delta^{n-1} = \{z \in \mathbb{R}^n | z_i \in [0,1], \sum_i z_i = 1\}$$에서 표현하고, **Concrete 분포**를 사용하여 부드러운 근사를 수행합니다[1]:

$$\text{Concrete 분포의 온도 파라미터: } \tau \sim n^{-1}$$

여기서 $$n$$은 수집된 관측값의 개수이고, 온도 파라미터는 균등 분포와 범주형 분포 간의 보간을 제어합니다.[1]

획득 함수(acquisition function)는 Phoenics 프레임워크에서 구성되며, 구성된 커널 밀도를 균등 분포와 비교하여 탐색-활용 트레이드오프를 조절합니다.[1]

**Static Gryffin**은 기술자 기반 메트릭 변환을 도입합니다. 심플렉스 위의 무한소 선 요소의 길이를 다음과 같이 재정의합니다:[1]

```math
\Delta s^2 = \sum_{m=1}^{\#\text{descs}} \sum_{i,j=1}^{\#\text{opts}} \left(x_i^m - \sum_{k=1}^{\#\text{opts}} z_k x_k^m\right)^2
```

여기서 $$x^m$$은 $$m$$번째 기술자, $$z_k$$는 범주형 선택지의 좌표입니다.[1]

**Dynamic Gryffin**은 기술자 변환을 학습합니다:[1]

$$x' = \text{softsign}(W \cdot x + b), \quad \text{softsign}(x) = \frac{x}{1 + |x|}$$

여기서 $$W$$와 $$b$$는 최적화 피드백으로부터 학습되는 파라미터입니다. 손실 함수는 세 가지 목표를 균형 있게 유지합니다:[1]

1. 새로운 기술자와 측정값 간의 상관관계 증가
2. 새로운 기술자 간 상관관계 감소
3. 중요도가 낮은 기술자 제거

***

### 3. 모델 구조

#### 3.1 아키텍처 구성요소

**핵심 구성요소**:[1]

| 구성요소 | 설명 |
|---------|------|
| **서로게이트 모델** | Bayesian Neural Network(BNN) 기반 커널 밀도 추정 |
| **획득 함수** | 가중치가 적용된 균등 분포 추가 (λ 파라미터로 조절) |
| **메트릭 변환** | 물리화학 기술자 기반 심플렉스 메트릭 재정의 |
| **기술자 개선** | 소프트사인 활성화 함수를 가진 선형 변환 |

#### 3.2 계산 복잡도

Gryffin은 선형 시간 복잡도로 확장됩니다:[1]

- 관측값 개수에 대해: $$O(n)$$
- 검색 공간 차원에 대해: 선형
- 병렬 최적화를 자연스럽게 지원

저밀도 영역의 커널 밀도 추정을 근사하여 계산 비용을 추가로 감소시킵니다.[1]

***

### 4. 성능 향상 및 일반화

#### 4.1 합성 벤치마크 성과

**테스트 표면에서의 성능**:[1]

- **Naive Gryffin**: 최첨단 방법과 경쟁 가능한 수준
- **Static Gryffin**: 비볼록 표면에서 다른 방법 대비 몇 배 탐색 공간 축소
- **Dynamic Gryffin**: 노이즈가 있는 표면에서 가장 우수한 성능

기술자 상관관계에 따른 성능:[1]

- Pearson 상관계수 0.8 이상: Static Gryffin이 naive 방식 대비 현저한 개선
- 상관계수 0.1 이상: Dynamic Gryffin이 여전히 naive 대비 우수

#### 4.2 차원에 따른 확장성

다항식 감소: 옵션 수 증가에 따라 $$y = \alpha x^{-\gamma}$$ (단, $$\gamma \in [1.0, 1.25]$$)[1]

지수 감소: 파라미터 개수 증가에 따라 $$\gamma \in [1.6, 2.0]$$[1]

이는 상대적으로 큰 차원에서만 차원의 저주가 나타남을 시사합니다.[1]

#### 4.3 실제 응용 성과

**비풀러렌 수용체 발견 (4,216개 후보)**:[1]
- Naive Gryffin: 11% 탐색
- Static Gryffin: 8.7% 탐색 (22% 감소)
- Dynamic Gryffin: 6.9% 탐색 (38% 감소)

**하이브리드 유기-무기 페로브스카이트 (192개 후보)**:[1]
- Naive Gryffin: <9% 탐색
- Static/Dynamic Gryffin: <8% 탐색

**스즈키-미야우라 반응 (연속-범주형 혼합)**:[1]
- Gryffin 변종들이 모든 벤치마크 대비 우수한 성능
- Dynamic Gryffin이 원하는 반응 수율 달성까지 7-8번의 평가만 필요

#### 4.4 일반화 성능

**일반화 향상 메커니즘**:[1]

1. **도메인 지식 활용**: 물리화학 기술자를 통해 귀납 편향(inductive bias) 제공
2. **적응형 메트릭**: Dynamic Gryffin의 메트릭 변환이 새로운 데이터에 적응
3. **로버스트성**: 정보가 부족한 기술자에도 견디며, 성능이 naive 수준 이하로 떨어지지 않음

**오버피팅 방지**:[1]

- 소프트사인 함수의 약간의 비선형성만 사용하여 복잡도 제한
- 낮은 데이터 시나리오에 견디도록 설계
- 정규화 효과를 통해 노이즈가 있는 환경에서도 안정적

***

### 5. 한계

#### 5.1 주요 한계점[1]

1. **기술자 선택**: 알고리즘의 성능이 제공된 기술자의 품질에 크게 의존합니다. 최적의 기술자 집합을 사전에 선택하기 어렵습니다.

2. **단일 기술자 중요도 학습**: Static Gryffin은 모든 기술자에 동등한 중요도를 부여하며, Dynamic Gryffin이 이를 학습해야 합니다.

3. **획득 함수 최적화**: 비볼록 획득 함수의 최적화가 계산 비용의 상당 부분을 차지합니다.

4. **이질 잡음**: 이질 잡음(heteroscedastic noise)에 대한 처리가 개선 필요

#### 5.2 현재 미해결 문제[1]

- 작은 라이브러리에서 기술자의 중요도 지표가 일반화되지 않을 수 있음
- Suzuki-Miyaura 반응의 경우, 7개 리간드만으로는 학습된 특성이 큰 라이브러리로 일반화되지 않음[1]

***

### 6. 영향 및 향후 연구 고려사항

#### 6.1 학문적 영향

**최근 발전 (2020년 이후)**:

베이지안 최적화 분야에서 다음과 같은 발전이 이루어졌습니다:[2][3][4]

1. **혼합 범주형-연속 최적화**: CoCaBO, BOUNCE, Think Global and Act Local 등 고차원 혼합 공간을 다루는 방법들이 개발되었습니다.[3][4][5]

2. **확장성 개선**: CatCMA와 같은 방법들이 고차원 문제로의 확장성을 개선했습니다.[6]

3. **대규모 언어 모델과 통합**: Reasoning BO는 LLM의 추론 능력과 베이지안 최적화를 결합하여 탐색-활용 트레이드오프를 동적으로 조절합니다.[7]

#### 6.2 일반화 성능 개선 방향[8][9][10]

**머신러닝 커뮤니티의 최신 인사이트**:

1. **정규화 재검토**: 최근 연구에서 대규모 신경망은 명시적 정규화 없이도 일반화되며, 이는 기존 이론과 상충합니다.[11][12]

2. **데이터 증강**: 훈련 데이터의 다양성 증가를 통한 일반화 성능 향상.[13]

3. **메타학습**: 도메인 시프트를 모의하여 미지 도메인으로의 일반화를 개선.[8]

4. **전이학습 개선**: Co-Tuning 같은 프레임워크가 범주 간 관계를 명시적으로 모델링하여 전이 성능을 향상시킵니다.[14]

#### 6.3 Gryffin 연구 시 고려사항

**기술자 선택 및 구성**:
- 수백 개의 기술자 계산이 가능하지만, 동적 Gryffin이 불필요한 기술자를 식별하므로 포함 여부에 신중할 필요 없음[1]
- 도메인 지식 기반 선택이 성능을 크게 향상시킴[1]

**배치 최적화 확장**:
- 여러 λ 값으로 여러 후보를 동시에 제안 가능[1]
- 효율적인 분산 학습을 위한 구성 솔버(compositional solvers) 활용 가능[1]

**이질 잡음 처리**:
- Power transform이나 robust surrogate model 구현 가능[1]
- 실험 불확실성 모델링 개선

**자동 기술자 발견**:
- 변분 오토인코더나 unsupervised learning을 보완 기술로 활용[1]
- 사용자 정의 기술자와 자동 발견 기술자의 결합

#### 6.4 향후 응용 분야

**확대 가능 분야**:[15][1]

1. **약물 발견 및 단백질 공학**: 분자 구조 최적화
2. **배치 프로세스 엔지니어링**: 생물공정 조건 최적화
3. **촉매 설계**: 반응 조건 및 촉매 선택의 동시 최적화
4. **신재료 탐색**: 태양전지, 배터리, 자기 재료 등

***

### 결론

Gryffin은 화학 및 재료과학에서 **범주형 변수의 효율적인 베이지안 최적화**를 위한 실용적이고 해석 가능한 프레임워크를 제공합니다. 도메인 지식을 활용한 메트릭 변환과 동적 기술자 개선을 통해 기존 방법 대비 수십 퍼센트의 탐색 공간 감소를 달성하며, 동시에 과학적 통찰력을 제공합니다.[1]

최근의 베이지안 최적화 발전과 신경망 일반화에 관한 새로운 이해는 Gryffin의 원리를 고차원 문제와 혼합 도메인으로 확장할 기회를 제공합니다. 특히 LLM과의 통합이나 meta-learning 기반 도메인 적응이 향후 중요한 연구 방향이 될 것으로 예상됩니다.[4][10][2][3][7][11]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/eaf0b46d-9aee-4302-9a29-5727c95fe4c0/2003.12127v2.pdf)
[2](https://arxiv.org/pdf/2206.03301.pdf)
[3](https://arxiv.org/abs/2102.07188)
[4](https://arxiv.org/pdf/1906.08878.pdf)
[5](https://arxiv.org/html/2307.00618v2)
[6](http://arxiv.org/pdf/2405.09962.pdf)
[7](https://arxiv.org/html/2505.12833v2)
[8](https://www.sciencedirect.com/topics/computer-science/generalization-performance)
[9](https://arxiv.org/pdf/2502.09193.pdf)
[10](http://arxiv.org/pdf/2209.01610.pdf)
[11](https://arxiv.org/html/2409.15156)
[12](https://arxiv.org/pdf/1611.03530.pdf)
[13](https://arxiv.org/html/2209.01610v3)
[14](https://proceedings.neurips.cc/paper/2020/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf)
[15](https://pmc.ncbi.nlm.nih.gov/articles/PMC12067035/)
[16](https://arxiv.org/pdf/1910.06403.pdf)
[17](http://arxiv.org/pdf/2502.06044.pdf)
[18](https://arxiv.org/pdf/2109.09264.pdf)
[19](https://arxiv.org/html/2402.19427v1)
[20](https://arxiv.org/abs/1805.07072)
[21](https://pmc.ncbi.nlm.nih.gov/articles/PMC7441167/)
[22](https://www.sciencedirect.com/science/article/abs/pii/S0952197624002768)
[23](https://glanceyes.com/entry/Deep-Learning-%EC%B5%9C%EC%A0%81%ED%99%94Optimization)
[24](https://www.sciencedirect.com/science/article/pii/S2152265025001399)
[25](https://ieeexplore.ieee.org/iel8/6287639/10820123/10815962.pdf)
[26](http://arxiv.org/pdf/1809.01465.pdf)
[27](http://arxiv.org/pdf/2205.08836.pdf)
[28](https://arxiv.org/pdf/2308.03236.pdf)
[29](http://arxiv.org/pdf/2409.14123.pdf)
[30](https://arxiv.org/html/2502.06178v2)
[31](https://codefinity.com/blog/Avoiding-Overfitting-in-Neural-Networks)
[32](https://arxiv.org/pdf/2502.06178.pdf)
[33](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_General_Incremental_Learning_With_Domain-Aware_Categorical_Representations_CVPR_2022_paper.pdf)
[34](https://papers.nips.cc/paper_files/paper/2024/file/10e9204f14c4daa08041343455435308-Paper-Conference.pdf)
[35](https://www.sciencedirect.com/science/article/abs/pii/S0141029625010478)
[36](https://pmc.ncbi.nlm.nih.gov/articles/PMC12312616/)
[37](https://www.nature.com/articles/s42005-024-01837-w)
