# Vision Transformer for Small-Size Datasets

### 1. 핵심 주장 및 주요 기여도 요약

이 논문의 핵심 주장은 **Vision Transformer(ViT)가 소규모 데이터셋에서 학습하지 못하는 근본적인 이유는 국소성 귀납적 편향(locality inductive bias)의 부재**라는 것입니다.[1]

논문의 주요 기여도는 다음과 같습니다:[1]

첫째, **Shifted Patch Tokenization(SPT)**을 통해 인접한 픽셀들 간의 공간적 관계를 더 효과적으로 임베딩합니다. 입력 이미지를 대각선 4개 방향으로 이동시킨 후 원본 이미지와 연결하여 패치 분할을 수행함으로써 토큰화 과정의 수용 영역을 확대합니다.

둘째, **Locality Self-Attention(LSA)**은 대각선 마스킹과 학습 가능한 온도 스케일링을 통해 주의 집중을 국소적으로 만듭니다. 이는 표준 ViT에서 발생하는 주의 점수 분포의 평탄화 문제를 해결합니다.

실험 결과, Tiny-ImageNet에서 평균 **2.96% 성능 향상**을 달성했으며, 특히 Swin Transformer는 **4.08%의 획기적인 성능 개선**을 기록했습니다.[1]

---

### 2. 해결하고자 하는 문제

#### 2.1 문제의 본질

표준 ViT는 ImageNet에서 ResNet-50을 능가하는 성능을 보였으나, 이는 JFT-300M과 같은 대규모 데이터셋에서의 사전학습에 의존합니다. CIFAR-100이나 Tiny-ImageNet 같은 소규모 데이터셋에서 처음부터 학습할 때 성능이 급격히 저하됩니다.[1]

이 문제는 ViT의 구조적 특성에서 비롯됩니다. 합성곱 신경망(CNN)이 비선형 활성화 함수와 함께 일반화 능력이 뛰어난 반면, ViT는 다음 두 가지 문제를 갖습니다:[1]

**문제 1: 불충분한 토큰화**
- 표준 ViT는 이미지를 겹치지 않는(non-overlapping) 패치로 분할합니다
- 각 패치의 수용 영역이 패치 크기와 동일하여 매우 작습니다
- 예: 224×224 이미지에서 패치 크기 16인 ViT의 수용 영역은 16이지만, ResNet50은 483입니다 - **약 30배 차이**[1]
- 이로 인해 인접 픽셀들 간의 공간적 관계가 불충분하게 임베딩됩니다

**문제 2: 평탄한 주의 분포**
- 이미지 특성상 토큰 수가 자연어나 오디오보다 훨씬 많습니다
- Query와 Key가 같은 입력에서 선형 투영되므로 자기-토큰 관계(self-token relation)가 상호-토큰 관계보다 큽니다
- 표준 softmax의 온도 $$\sqrt{d_k}$$로 인한 온도 스케일링이 주의 점수를 평탄화시킵니다[1]
- 결과적으로 배경에 주의가 산포되어 대상 객체의 형태를 제대로 포착하지 못합니다

***

### 3. 제안 방법 및 수식

#### 3.1 Shifted Patch Tokenization (SPT)

**기본 개념**: Temporal Shift Module(TSM)에서 영감을 받아 시간 차원을 공간 차원으로 확장한 개념입니다.[1]

**수식 표현**:

표준 ViT 토큰화:
$$P(x) = [x_1^p; x_2^p; ...; x_N^p]$$

여기서 $$x_i^p \in \mathbb{R}^{P^2 \cdot C}$$는 i번째 평탄화된 벡터, P는 패치 크기, N = HW/P²는 패치 수입니다.[1]

표준 선형 투영:
$$T(x) = P(x)E_t$$

여기서 $$E_t \in \mathbb{R}^{(P^2 \cdot C) \times d}$$는 학습 가능한 선형 투영, d는 숨겨진 차원입니다.[1]

**SPT 공식**:
$$S(x) = \text{LN}(P([x \; s_1 \; s_2 \; ... \; s_{N_S}])) E_S$$

여기서:[1]
- $$s_i \in \mathbb{R}^{H \times W \times C}$$: 전략 S에 따른 i번째 이동된 이미지
- $$E_S \in \mathbb{R}^{(P^2 \cdot C \cdot (N_S+1)) \times d_S}$$: 학습 가능한 선형 투영
- $$N_S$$: 이동 개수(기본값: 4개 대각선 방향)

**패치 임베딩 레이어 적용**:

$$S_{pe}(x) = \begin{cases} [x_{cls}; S(x)] + E_{pos} & \text{if } x_{cls} \text{ exists} \\ S(x) + E_{pos} & \text{otherwise} \end{cases}$$

[1]

여기서 $$x_{cls}$$는 클래스 토큰, $$E_{pos}$$는 위치 임베딩입니다.

**이미지 이동 전략**: 패치 크기의 절반만큼 대각선 4개 방향(좌상향, 우상향, 좌하향, 우하향)으로 이동시킵니다. 이동된 이미지들은 원본과 동일 크기로 자릅니다.

**수용 영역 계산**:

$$r_{token} = r_{trans} \cdot j + (k - j)$$

표준 ViT에서 $$r_{trans} = 1$$이므로 $$r_{token} = k$$(패치 크기)입니다. SPT는 이를 확대하여 더 풍부한 공간 정보를 임베딩합니다.[1]

#### 3.2 Locality Self-Attention (LSA)

LSA는 두 가지 기술로 구성됩니다:[1]

**기술 1: 대각선 마스킹(Diagonal Masking)**

유사도 행렬에서 자기-토큰 관계를 제거합니다:

$$R^M_{i,j}(x) = \begin{cases} R_{i,j}(x) & \text{if } i \neq j \\ -\infty & \text{if } i = j \end{cases}$$

여기서 $$R_{i,j}(x) = xE_q(xE_k)^{\top}$$는 Query와 Key의 내적으로 계산된 유사도입니다.[1]

**기술 2: 학습 가능한 온도 스케일링(Learnable Temperature Scaling)**

표준 softmax의 고정된 온도 $$\sqrt{d_k}$$ 대신 학습 가능한 매개변수 τ를 사용합니다:

$$L(x) = \text{softmax}(R^M(x)/\tau) x E_v$$

여기서 $$E_v \in \mathbb{R}^{d \times d_v}$$는 Value의 선형 투영입니다.[1]

**작동 원리**: 
- 대각선 마스킹은 자기-토큰 관계를 제거하여 상호-토큰 관계에 상대적으로 높은 주의 점수를 부여합니다
- 학습 가능한 온도 스케일링은 표준 ViT보다 낮은 온도를 학습하여 주의 분포를 날카롭게 만듭니다[1]
- 실험 결과, 최적의 온도는 $$\sqrt{d_k}$$보다 낮게 학습됩니다 (Figure 3 참조)

**Kullback-Leibler 발산 개선**: LSA는 주의 점수 분포의 Kullback-Leibler 발산을 약 0.5 증가시켜 평탄화 문제를 해결합니다.[1]

---

### 4. 모델 구조

#### 4.1 전체 아키텍처

논문은 SPT와 LSA를 기존 ViT 아키텍처에 **모듈로 추가**하는 제너릭 방식을 제안합니다. 이는 다양한 ViT 변형(ViT, T2T, PiT, Swin Transformer, CaiT)에 쉽게 적용됩니다.[1]

**기본 ViT 구조**:
1. 입력 이미지 → SPT (패치 토큰화)
2. 클래스 토큰 추가 + 위치 임베딩
3. Transformer 인코더 (다중 블록)
   - LSA가 각 블록의 자기-주의 메커니즘을 대체
   - 피드포워드 네트워크
4. 분류 헤드

#### 4.2 풀링 레이어에 SPT 적용

계층적 ViT(Swin, PiT)의 경우, SPT를 풀링 레이어로도 사용합니다:[1]

$$S_{pool}(y) = \begin{cases} [x_{cls}E_{cls}; S(R(y))] & \text{if } x_{cls} \text{ exists} \\ S(R(y)) & \text{otherwise} \end{cases}$$

여기서 $$R(y)$$: $$y \in \mathbb{R}^{N \times d} \rightarrow \mathbb{R}^{(H/P) \times (W/P) \times d}$$는 2D-행렬을 3D-텐서로 변환하는 reshape 연산입니다.

#### 4.3 실험 설정의 구성

소규모 데이터셋 실험:[1]
- ViT: 깊이 9, 숨겨진 차원 192, 헤드 수 12, 패치 크기 8
- PiT-XS, T2T-14, Swin-T, CaiT-XXS24: 각 논문의 기본 설정
- 중간 시각적 토큰 개수: 64로 설정하여 계산-정확도 트레이드오프 고려

***

### 5. 성능 향상 및 한계

#### 5.1 정량적 성능 향상

**소규모 데이터셋(표 2)**:[1]

| 모델 | CIFAR-10 | CIFAR-100 | SVHN | Tiny-ImageNet |
|------|----------|-----------|------|--------------|
| ViT | 93.58% | 73.81% | 97.82% | 57.07% |
| SL-ViT | 94.53% | 76.92% | 97.79% | 61.07% |
| **개선율** | **+0.95%** | **+3.11%** | **-0.03%** | **+4.00%** |
| T2T | 95.30% | 77.00% | 97.90% | 60.57% |
| SL-T2T | 95.57% | 77.36% | 97.91% | 61.83% |
| **개선율** | **+0.27%** | **+0.36%** | **+0.01%** | **+1.26%** |
| Swin | 94.46% | 76.87% | 97.72% | 60.87% |
| SL-Swin w/ Spool | 95.93% | 79.99% | 97.92% | 64.95% |
| **개선율** | **+1.47%** | **+3.12%** | **+0.20%** | **+4.08%** |
| CaiT | 94.91% | 76.89% | 98.13% | 64.37% |
| SL-CaiT | 95.81% | 80.32% | 98.28% | 67.18% |
| **개선율** | **+0.90%** | **+3.43%** | **+0.15%** | **+2.81%** |

**중규모 데이터셋 (ImageNet):**[1]

| 모델 | 기본 성능 | SPT+LSA 성능 | 개선율 |
|------|----------|------------|--------|
| ViT-Tiny | 69.95% | 71.55% | **+1.60%** |
| PiT-XS | 75.58% | 77.02% | **+1.44%** |
| Swin-T | 79.95% | 81.01% | **+1.06%** |

#### 5.2 절제 연구(Ablation Study)

**LSA 구성 요소별 기여도 (표 4):**[1]

| 모델 | CIFAR-100 | Tiny-ImageNet |
|------|-----------|--------------|
| ViT (기본) | 73.81% | 57.07% |
| T-ViT (온도만) | 74.35% | 57.95% |
| M-ViT (대각선만) | 74.34% | 58.29% |
| L-ViT (온도+대각선) | 74.87% | 58.50% |

- 학습 가능한 온도: +0.88% (Tiny-ImageNet)
- 대각선 마스킹: +1.22% (Tiny-ImageNet)
- 두 기술의 상승 효과 존재

**SPT와 LSA 분리 기여도 (표 5):**[1]

| 모델 | CIFAR-100 | Tiny-ImageNet |
|------|-----------|--------------|
| ViT (기본) | 73.81% | 57.07% |
| L-ViT | 74.87% | 58.50% |
| S-ViT | 76.29% | 60.67% |
| SL-ViT | 76.92% | 61.07% |

- SPT 단독: +3.60% (Tiny-ImageNet)
- LSA 단독: +1.43% (Tiny-ImageNet)
- 함께 적용: +4.00% (Tiny-ImageNet)

#### 5.3 계산 효율성

**처리량 오버헤드 최소화**:[1]
- ViT: 1.12% 레이턴시 증가
- T2T: 1.15% 레이턴시 증가
- CaiT: 1.06% 레이턴시 증가

CNN 모델(ResNet, EfficientNet)과의 비교에서도 경쟁력 있는 처리량 유지

#### 5.4 한계점

**데이터셋 의존성:**
- 최적 이동 방향이 데이터셋마다 다름 (보충 자료)[1]
- CIFAR-10/100: 4개 대각선 또는 기본 방향
- Tiny-ImageNet: 8개 전방향
- 이동 비율도 데이터셋마다 최적값이 다름 (0.25~0.5)

**계산 비용:**
- SPT 적용 시 토큰 수 증가로 메모리 사용량 증가
- 표 2에서 일부 모델(PiT, Swin with Spool)의 FLOPs 증가

**일반화 성능의 한계:**
- ImageNet에서의 개선율이 소규모 데이터셋보다 작음 (+1.06~1.60%)
- 대규모 데이터셋에서는 추가 개선 효과 감소

***

### 6. 일반화 성능 향상 가능성 (중점)

#### 6.1 일반화 메커니즘

이 논문의 방법이 일반화 성능을 향상시키는 메커니즘은 **귀납적 편향 강화**입니다:[1]

**국소성 귀납적 편향 추가:**
- SPT: 이미지의 공간적 인접성 정보를 토큰 단계에서 명시적으로 임베딩
- LSA: 모델이 역함수나 필터 매커니즘 없이도 국소적 주의를 학습
- 결과: 모델이 국소 패턴을 더 빨리 인식하여 적은 데이터로 일반화 가능

**주의 분포 개선:**
- Figure 5의 시각화에서 배경 노이즈가 감소하고 객체 형태가 명확히 됨[1]
- 이는 모델이 핵심 특성에 집중하여 변동에 덜 민감해짐을 의미

**데이터 효율성 증대:**
- 표 2에서 SL-CaiT와 SL-Swin이 CNN(ResNet, EfficientNet)을 능가
- 동일 데이터량에서 ViT가 더 나은 성능 달성 가능

#### 6.2 분포 외 일반화 (Out-of-Distribution Generalization)

최신 연구 기반:[2]

ViT는 CNN에 비해 **형태(shape) 편향**이 강하고 **질감(texture) 편향이 약함**을 보입니다. 이는 다음을 의미합니다:[2]
- 픽셀 수준 분석보다 의미적 구조에 집중
- 도메인 변화에 더 강건
- 분포 외 일반화 시 CNN보다 우수한 성능

이 논문의 SPT+LSA는 이러한 ViT의 장점을 유지하면서 국소 정보를 추가하여 소규모 데이터셋에서도 이점을 활용합니다.

#### 6.3 개선 가능성

**긍정적 신호:**
- 중규모 데이터셋(ImageNet)에서도 일관된 개선 (+1.06~1.60%)
- 오버피팅 감소 효과 (Tiny-ImageNet에서 최대 +4.08%)
- 다양한 ViT 아키텍처에 범용 적용 가능

**한계:**
- 대규모 데이터셋에서는 개선 효과 미미
- 데이터셋 특성에 따라 최적 하이퍼파라미터 조정 필요
- 모든 데이터셋에 최적인 단일 설정 부재

---

### 7. 앞으로의 연구 영향 및 고려사항

#### 7.1 연구 영향

**기존 연구 확장:**

이 논문이 발표된 이후(2021년 12월) 후속 연구들이 유사한 접근을 취하고 있습니다:[3][4][5][6][7]

1. **GvT (Graph-based Vision Transformer, 2024):**[3]
   - 스파스 그래프 기반 주의 메커니즘으로 국소 편향 강화
   - 소규모 데이터셋에서 처음부터 학습 가능
   - 최신 성능 달성

2. **Pre-training with Masked Auto-Encoder (2024):**[4]
   - 경량 ViT를 자기-감독 사전학습으로 개선
   - 소규모 데이터셋 성능 향상

3. **Self-supervised Inductive Biases (2022):**[5]
   - 자기-감독 학습으로 귀납적 편향 학습
   - 대규모 사전학습 불필요

4. **Spatial Entropy Regularization (2023):**[6]
   - 공간 엔트로피를 국소 편향으로 사용
   - 정규화 기법으로 아키텍처 변경 최소화

5. **LIFE Module (2024):**[7]
   - Local Information Enrichment 모듈
   - 확대된 수용 영역으로 국소 문맥 학습
   - 검출, 분할 등 다양한 하위 작업에 적용 가능

#### 7.2 최신 연구 기반 고려사항

**데이터 증강 강화:**[8]
- GenFormer: 생성 이미지를 데이터 증강으로 사용하여 ViT 성능 향상
- SPT+LSA와 고급 증강 기법 결합 시너지 예상

**도메인 적응 및 일반화:**[2]
- ViT의 형태 편향 특성과 국소 편향 결합
- 분포 변화에 더 강건한 모델 개발 가능
- 의료 이미징, 원격 센싱 등 특수 도메인에 응용

**계산 효율성:**[9]
- big.LITTLE ViT: 토큰 프루닝으로 효율성 개선
- SPT의 토큰 수 증가를 동적 프루닝으로 완화 가능

**하이브리드 접근:**[10][11]
- CNN과 Transformer 결합 (CMT, Shunted Self-Attention)
- SPT+LSA는 순수 Transformer 접근 강화

#### 7.3 실무 적용 시 고려사항

**의료 영상 처리 분야 (당신의 관심사):**
- 뼈 억제 문제는 소규모, 특수 데이터셋 기반 → SPT+LSA 직접 적용 가능
- 국소 구조 정보(뼈 경계, 폐 윤곽선)에 SPT 특히 효과적
- LSA의 국소 주의 메커니즘이 의료 영상의 미세한 디테일 포착에 유리

**데이터셋 특성별 최적화:**
- CIFAR-like 자연 이미지: 기본 설정 (4개 대각선, 이동비율 0.5)
- 의료 영상: 이동 방향과 비율 재조정 필요
- 보충 자료의 실험 설계 참고하여 하이퍼파라미터 튜닝

**모듈식 설계의 장점:**
- 기존 ViT 구현에 SPT와 LSA 추가 용이
- 점진적 개선 가능 (SPT만 먼저, 후에 LSA 추가 등)
- 다양한 백본(Swin, T2T 등)에 쉽게 적용

**향후 연구 방향:**
1. 특정 도메인(의료, 위성 영상 등)에 최적화된 이동 전략 개발
2. 적응형 이동 비율을 동적으로 결정하는 메커니즘
3. SPT의 토큰 수 증가 문제를 해결하는 효율적 압축 기법
4. 강화 학습을 통한 자동 하이퍼파라미터 최적화

---

### 결론

**Vision Transformer for Small-Size Datasets**는 ViT가 소규모 데이터셋에서 실패하는 근본 원인을 명확히 규명하고, 이를 해결하기 위한 **간단하지만 효과적인 두 가지 모듈(SPT, LSA)**을 제안합니다.[1]

논문의 강점은:
- 문제 진단의 명확성 (국소성 귀납적 편향 부재)
- 제너릭 모듈식 설계로 다양한 ViT 아키텍처에 적용 가능
- 최소한의 계산 오버헤드로 최대 4% 이상의 성능 개선
- 명확한 시각화와 절제 연구로 각 컴포넌트의 기여도 입증

최신 연구 동향을 보면, 이 논문의 핵심 통찰(국소 편향 강화의 중요성)은 후속 연구에 광범위하게 영향을 미쳤으며, 자기-감독 학습, 그래프 기반 주의, 및 다양한 정규화 기법 등으로 발전했습니다. 특히 의료 영상 처리와 같은 **제한된 데이터 환경에서 ViT 활용이 증가하는 추세**를 고려할 때, 이 논문의 방법론은 실질적 중요성이 높습니다.[4][5][6][7][3]

---

### 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0b44cb5c-d488-4885-9e4a-df640dc06744/2112.13492v1.pdf)
[2](https://arxiv.org/pdf/2404.04452.pdf)
[3](http://arxiv.org/pdf/2404.04924.pdf)
[4](https://arxiv.org/pdf/2402.03752.pdf)
[5](https://arxiv.org/pdf/2210.07240.pdf)
[6](http://arxiv.org/pdf/2206.04636.pdf)
[7](https://www.sciencedirect.com/science/article/abs/pii/S0031320324002619)
[8](https://arxiv.org/html/2408.14131v2)
[9](https://arxiv.org/html/2410.10267v1)
[10](https://arxiv.org/pdf/2107.06263.pdf)
[11](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_Shunted_Self-Attention_via_Multi-Scale_Token_Aggregation_CVPR_2022_paper.pdf)
[12](https://www.frontiersin.org/articles/10.3389/fbuil.2024.1321634/full)
[13](https://www.semanticscholar.org/paper/7a29f47f6509011fe5b19462abf6607867b68373)
[14](https://www.jmir.org/2025/1/e57723)
[15](https://arxiv.org/abs/2509.20580)
[16](https://arxiv.org/abs/2112.13492)
[17](http://arxiv.org/pdf/2106.03746.pdf)
[18](https://www.sciencedirect.com/science/article/pii/S0031320324004631)
[19](https://viso.ai/deep-learning/vision-transformer-vit/)
[20](https://www.semanticscholar.org/paper/Vision-Transformer-for-Small-Size-Datasets-Lee-Lee/164e41a60120917d13fb69e183ee3c996b6c9414)
[21](https://arxiv.org/pdf/2112.13492.pdf)
[22](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Delving_Deep_Into_the_Generalization_of_Vision_Transformers_Under_Distribution_CVPR_2022_paper.pdf)
[23](https://www.nature.com/articles/s41598-025-10408-0)
