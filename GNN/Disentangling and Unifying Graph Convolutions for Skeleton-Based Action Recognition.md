# Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition

### 1. 핵심 주장과 주요 기여도 요약

본 논문의 핵심은 **스켈레톤 기반 행동 인식에서 그래프 신경망을 활용할 때 발생하는 두 가지 근본적인 문제를 해결**하는 것입니다.[1]

**첫 번째 문제: 편향된 가중치 문제(Biased Weighting Problem)**[1]
기존 방법들은 인접 행렬의 고차 다항식(adjacency matrix polynomials)을 사용하여 다중 스케일 정보를 캡처합니다. 그러나 무방향 그래프에서는 순환 경로(cyclic walks)가 존재하기 때문에, 원거리 노드보다 근처 노드에 대한 엣지 가중치가 편향됩니다. 스켈레톤 그래프에서는 이로 인해 집계된 특징이 로컬 신체 부위 신호에 지배되어 장거리 관절 의존성을 효과적으로 포착하지 못합니다.[1]

**두 번째 문제: 간접적인 정보 흐름**[1]
대부분의 기존 방법은 공간 모듈(GCN)과 시간 모듈(RNN/TCN)을 분해하여 교차 배치하므로, 복잡한 공간-시간 관절 의존성을 직접 포착하기 어렵습니다. 예를 들어, "일어서기" 행동은 상체가 앞으로 기울어지는 순간과 하체가 일어서는 미래의 움직임 사이에 강한 상관관계가 있지만, 분해된 모델링으로는 이를 효과적으로 캡처하지 못합니다.

**제안된 두 가지 주요 기여:**[1]
1. **분해된 다중 스케일 집계 방식**: k-인접 행렬(k-adjacency matrix) $$\hat{A}_k$$를 제안하여 근거리 이웃과 원거리 이웃 간의 중복 의존성을 제거하고, 각 이웃에서의 특징을 분해하여 거리와 무관하게 관절 관계를 모델링합니다.

2. **G3D 모듈**: 공간-시간 도메인 전체에서 직접 그래프 엣지를 도입하여 스킵 연결을 만들고, 공간-시간 그래프 상에서 방해받지 않는 정보 흐름을 가능하게 합니다.

이 두 접근법을 결합한 **MS-G3D**는 NTU RGB+D 60, NTU RGB+D 120, Kinetics Skeleton 400에서 최첨단 성능을 달성합니다.[1]

***

### 2. 해결하는 문제와 제안 방법(수식 포함)

#### 2.1 기존 다중 스케일 집계의 문제점

기본 그래프 합성곱 신경망(GCN)은 다음과 같이 정의됩니다:[1]

$$
\mathbf{X}^{(l+1)}_t = \sigma\left(\mathbf{D}^{-1/2}\mathbf{\tilde{A}}\mathbf{D}^{-1/2}\mathbf{X}^{(l)}_t \mathbf{W}^{(l)}\right)
$$

여기서 $$\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}$$는 자기 루프가 추가된 인접 행렬이고, $$\mathbf{D}$$는 차수 행렬입니다.[1]

기존 다중 스케일 접근법은 다음과 같이 고차 인접 행렬의 합을 사용합니다:[1]

$$
\mathbf{X}^{(l+1)}_t = \sum_{k=0}^{K} \bar{\mathbf{A}}^k \mathbf{X}^{(l)}_t \mathbf{W}^{(l)}_k
$$

여기서 $$\bar{\mathbf{A}}$$ 는 정규화된 인접 행렬 형태입니다. 그러나 이 방식에서 $$\bar{\mathbf{A}}^k_{i,j}$$는 노드 $$v_i$$와 $$v_j$$ 사이의 길이 k인 경로의 개수를 나타내므로, **순환 경로의 존재로 인해 근처 노드에 대한 가중치가 크게 편향됩니다.**[1]

#### 2.2 제안된 분해된 다중 스케일 집계(Disentangled Multi-Scale Aggregation)

이 문제를 해결하기 위해 k-인접 행렬 $$\hat{\mathbf{A}}_k$$를 정의합니다:[1]

$$
\hat{\mathbf{A}}^k_{i,j} = \begin{cases} 
1 & \text{if } d(v_i, v_j) = k \\
1 & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}
$$

여기서 $$d(v_i, v_j)$$ is는 두 노드 사이의 최단 경로 거리입니다. 자기 루프를 포함하는 것은 현재 관절과 k-홉 이웃 간의 관계를 학습하고, k-홉 이웃이 없을 때 각 관절의 정체성 정보를 유지하기 위해 중요합니다.[1]

이를 통해 다음과 같은 분해된 집계 수식을 얻습니다:[1]

$$
\mathbf{X}^{(l+1)}_t = \sum_{k=0}^{K} \mathbf{D}_k^{-1/2}\hat{\mathbf{A}}_k\mathbf{D}_k^{-1/2}\mathbf{X}^{(l)}_t \mathbf{W}^{(l)}_k
$$

여기서 $$\mathbf{D}_k^{-1/2}\hat{\mathbf{A}}_k\mathbf{D}_k^{-1/2}$$는 정규화된 k-인접 행렬입니다. 이 공식은 **근거리 이웃과 원거리 이웃 간의 중복 의존성을 제거**하여 편향된 가중치 문제를 해결합니다.[1]

#### 2.3 G3D: 통합 공간-시간 그래프 합성곱(Unified Spatial-Temporal Graph Convolution)

대부분의 기존 방법이 공간과 시간을 분해하여 처리하므로, 본 논문은 직접적인 정보 흐름을 위해 공간-시간 엣지를 도입하는 **G3D 모듈**을 제안합니다.[1]

슬라이딩 윈도우 크기 $$\tau$$를 고려하여, 공간-시간 부분 그래프 $$G_\tau = (\mathcal{V}, \mathcal{E})$$를 구성합니다:[1]

$$
\mathbf{\tilde{A}}_\tau = \begin{pmatrix}
\mathbf{A} & \mathbf{A} & \cdots & \mathbf{A} \\
\mathbf{A} & \mathbf{A} & \cdots & \mathbf{A} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{A} & \mathbf{A} & \cdots & \mathbf{A}
\end{pmatrix} \in \mathbb{R}^{N\tau \times N\tau}
$$

여기서 각 부분행렬 $$\mathbf{A}_{i,j} = \mathbf{A}$$는 프레임 i의 모든 노드가 프레임 j의 그들의 1-홉 공간 이웃과 연결되어 있음을 의미합니다. 이는 각 노드가 윈도우 내의 모든 프레임에서 자신과 그 1-홉 공간 이웃에 조밀하게 연결되도록 합니다.[1]

G3D의 통합 공간-시간 합성곱은:[1]

$$
\mathbf{X}^{(l+1)}_t = \sigma\left(\mathbf{D}_\tau^{-1/2}\mathbf{\tilde{A}}_\tau\mathbf{D}_\tau^{-1/2}\mathbf{X}^{(l)}_t \mathbf{W}^{(l)}\right)
$$

또한 **팽창된 윈도우(dilated window)**를 사용하여 더 큰 시간 수용 영역을 얻을 수 있습니다. 팽창 율 d를 가진 윈도우는 매 d 프레임마다 프레임을 선택합니다.[1]

#### 2.4 MS-G3D: 분해된 다중 스케일 G3D

이 두 방법을 결합하면:[1]

$$
\mathbf{X}^{(l+1)}_t = \sum_{k=0}^{K} \mathbf{D}_{\tau,k}^{-1/2}\hat{\mathbf{A}}_{\tau,k}\mathbf{D}_{\tau,k}^{-1/2}\mathbf{X}^{(l)}_t \mathbf{W}^{(l)}_k
$$

여기서 $$\hat{\mathbf{A}}\_{\tau,k}$$와 $$\mathbf{D}_{\tau,k}$$ 는 공간-시간 도메인에서 정의된 k-인접 및 차수 행렬입니다. 흥미롭게도, G3D의 증가된 노드 차수로 인해 편향된 가중치 문제가 더 심화될 수 있지만, **제안된 분해된 집계 방식이 이를 보완**합니다.[1]

***

### 3. 모델 구조

#### 3.1 전체 아키텍처

MS-G3D 모델은 다음과 같은 구조를 가집니다:[1]

**STGC 블록(Spatial-Temporal Graph Convolutional Block)**: 전체 모델은 r개의 STGC 블록이 쌓여 있으며, 각 블록은 **두 가지 경로**를 가집니다:[1]

1. **G3D 경로(G3D Pathway)**: 
   - 공간-시간 윈도우를 구성
   - 분해된 다중 스케일 그래프 합성곱 수행
   - 완전 연결층으로 윈도우 특징을 읽어냄
   - 여러 $$\tau$$ 및 $$d$$ 값을 가진 여러 경로 사용 가능

2. **인수분해된 경로(Factorized Pathway)**:
   - 다중 스케일 그래프 합성곱(MS-GCN): 전체 스켈레톤 그래프에서 최대 K = 12 스케일로 모델링
   - 다중 스케일 시간 합성곱(MS-TCN): 확장된 시간 맥락 포착
   - 각 STGC 블록의 출력은 96, 192, 384 채널로 지정됨

**다중 스케일 시간 모델링**: 기존 고정 커널 크기 대신, 다양한 팽창 비율(1, 2, 3, 4)을 가진 1D 합성곱 층을 사용하여 더 큰 수용 영역을 얻습니다. 병목 설계를 통해 계산 비용을 낮춥니다.[1]

**적응형 그래프(Adaptive Graphs)**: 각 $$\hat{\mathbf{A}}\_k$$ 및 $$\hat{\mathbf{A}}\_{\tau,k}$$에 학습 가능한 그래프 잔여 마스크 $$\mathbf{A}_{res}$$를 추가하여 엣지를 동적으로 강화, 약화, 추가 또는 제거할 수 있습니다:[1]

$$
\mathbf{X}^{(l+1)}_t = \sum_{k=0}^{K} \mathbf{D}_k^{-1/2}(\hat{\mathbf{A}}_k + \mathbf{A}_{res,k})\mathbf{D}_k^{-1/2}\mathbf{X}^{(l)}_t \mathbf{W}^{(l)}_k
$$

#### 3.2 관절-뼈 이중 스트림 융합

논문의 시각화 직관에 기반하여, 별도의 모델을 **뼈 특징**(인접한 관절의 벡터 차이로 초기화)을 사용하여 학습합니다. 최종 예측은 관절 및 뼈 모델의 소프트맥스 점수를 합산합니다.[1]

---

### 4. 성능 향상

#### 4.1 정량적 성능 개선

| 데이터셋 | 평가 설정 | MS-G3D | SOTA 기존 방법 | 향상도 |
|---------|---------|--------|-------------|--------|
| **NTU RGB+D 60** | X-Sub | **91.5%** | 89.9% (DGNN) | +1.6% |
| | X-View | **96.2%** | 96.1% (2s-AGCN) | +0.1% |
| **NTU RGB+D 120** | X-Sub | **86.9%** | 82.9% (2s-AGCN) | +4.0% |
| | X-Set | **88.4%** | 84.9% (2s-AGCN) | +3.5% |
| **Kinetics Skeleton 400** | Top-1 | **38.0%** | 36.9% (DGNN) | +1.1% |
| | Top-5 | **60.9%** | 59.6% (DGNN) | +1.3% |

[1]

#### 4.2 분해된 다중 스케일 집계의 효과

다중 스케일 수를 다양하게 하여 분해된 집계의 효과를 검증했습니다:[1]

- **GCN 경로에서**: K=4일 때 분해된 방식이 인접 행렬 멱승 대비 1.4% 성능 향상 달성
- **G3D 경로에서**: 더 큰 그래프 밀도로 인해 편향 가중치 문제가 더 심화되어, K=12에서 0.8%의 더 큰 성능 갭 발생
- **적응형 마스크 추가**: 편향 가중치 불균형을 부분적으로 완화하지만, 분해된 방식이 근본적인 해결책

#### 4.3 G3D 모듈의 효과

경량 모델에서 G3D를 추가했을 때:[1]

- 기본 모델(2s-AGCN): 86.0%
- 인수분해된 경로만: 87.8%
- G3D 경로 추가(τ=5, d=1): 89.2%
- 성능 향상은 동일한 매개변수 수에서 달성되어, G3D의 효율성 증명

#### 4.4 하이퍼파라미터 분석

- **윈도우 크기 $$\tau$$**: τ=5에서 최적 성능, τ=7에서는 이웃이 너무 커서 일반화 특징이 됨
- **팽창 비율 $$d$$**: d=1, 2의 조합이 최적으로, 더 큰 팽창은 시간 해상도 감소로 이점 감소
- **공간-시간 연결**: 교차-공간시간 엣지(Eq. 5)가 그리드형 또는 자기 엣지 대비 0.4~0.5% 우수

***

### 5. 모델 일반화 성능 향상 가능성

#### 5.1 현재 논문에서의 일반화 관점

**관절-뼈 이중 스트림 융합의 일반화**:[1]
논문은 NTU RGB+D 60 데이터셋에서 관절-뼈 융합의 일반화 성능을 검증했습니다:
- 관절 정보만: 89.4% (X-Sub)
- 뼈 정보만: 90.1% (X-Sub)
- 융합: 91.5% (X-Sub)

이는 제안된 아키텍처가 **다양한 입력 모드에 적응 가능**함을 시사합니다.

**G3D 하이퍼파라미터의 안정성**:[1]
모든 G3D 구성이 기본 모델 대비 일관되게 성능 향상을 보여주어, 모델의 **견고한 안정성**을 입증합니다.

#### 5.2 최신 연구에서의 일반화 개선 방향

최근(2024-2025) 스켈레톤 기반 행동 인식의 일반화 연구는 다음과 같은 방향으로 진화하고 있습니다:[2][3][4][5]

**1. 크로스 도메인 일반화**[3][2]
- **"Recovering Complete Actions for Cross-dataset Skeleton Action Recognition"**(NeurIPS 2024): 시간적 불일치를 다루기 위해 완전한 행동 복원 및 리샘플링 프레임워크를 제안. 다양한 데이터셋 간 성능 격차 20% 이상 개선[3]
- **"Cross-Graph Domain Adaptation"**(CVPR 2024): 그래프 구조를 적응적으로 학습하여 크로스 도메인 전이 개선

**2. 제로샷 및 소수샷 학습**[4][5][6][7]
- **"Bridging the Skeleton-Text Modality Gap"**(2024): 확산 모델을 사용한 공간-텍스트 정렬로 보이지 않는 행동 예측 성능 2.36%-13.05%포인트 향상
- **"CrossGLG: LLM Guides One-shot Skeleton-based 3D Action Recognition"**(2024): 대규모 언어 모델(LLM)의 텍스트 설명을 활용하여 소수샷 학습 성능 대폭 개선

**3. 자기지도 학습 및 비지도 학습**[8][9]
- **"SkeletonMAE: Spatial-Temporal Masked Autoencoders"**(2023): 마스킹 자기인코더를 통한 자기지도 학습으로 라벨 없이 일반화 가능한 골격 특징 학습
- **"Unsupervised Cross-Attention Encoder-Decoder"**(2024): 비지도 학습으로 마스킹 및 재구성을 통해 NTU 및 NW-UCLA 데이터셋에서 인상적인 일반화 능력 달성

**4. 전이 학습 및 메타 학습**[10][11]
- **"Skeleton-Based Action Recognition Using Graph Convolution and Cross-Domain Transfer Learning"**(2024): ST-GCTN 아키텍처에서 매개변수 기반 전이 학습으로 훈련 데이터 볼륨 감소 및 표현력 확장
- **계층적 상호 작용 그래프 학습**(Nature, 2025): 위상 특정 그래프 구조를 시간에 따라 동적으로 조정하여 행동별 최적 연결성 모델링

**5. 다중 모달 정렬 및 비전-언어 모델**[12]
- **"Vision-Language Meets the Skeleton"**(2024): 비전-언어 모델과의 지식 증류로 더 나은 상호 연결 공간 학습
- **"Dual Visual-Text Alignment for Zero-shot"**(2024): 직접 정렬 및 강화된 정렬 모듈로 모달리티 간 격차 효과적으로 축소

***

### 6. 모델의 한계

#### 6.1 논문에서 명시된 한계

**1. 적응형 마스크의 제한**:[1]
- 적응형 그래프 잔여 마스크 $$\mathbf{A}_{res,k}$$는 모든 가능한 행동에 대해 최적화되므로 미세한 엣지 조정만 가능
- 그래프 구조에 주요 결함이 있을 때는 불충분

**2. G3D와 인수분해된 경로 간의 균형 필요**:[1]
- G3D는 **지역적 공간-시간 의존성** 포착에 최적
- **장거리 의존성**은 인수분해된 경로가 더 효율적
- 따라서 최적 성능을 위해 두 경로의 조합 필수

**3. 시간 윈도우 하이퍼파라미터의 민감성**:[1]
- 윈도우 크기 τ와 팽창 비율 d 사이의 신중한 균형 필요
- 큰 τ는 수용 영역 증대하지만 일반 특징으로 인해 판별력 감소
- 큰 d는 시간 해상도 손실로 세부 스켈레톤 동작 손실

#### 6.2 최신 연구에서 드러난 한계

**1. 크로스 데이터셋 일반화 부족**:[3]
- 같은 데이터셋 내 설정(Cross-Subject, Cross-View)에서는 우수하나, 서로 다른 데이터셋 간(예: NTU → Kinetics) 성능 저하 20%+ 발생
- 이는 **시간적 분포 불일치** 및 **카메라 각도 변화**에 대한 견고성 부족을 의미

**2. 포즈 추정 오류에 대한 취약성**:[9]
- 스켈레톤 데이터가 자동 포즈 추정(OpenPose)에서 파생될 때 신뢰도 저하
- 낮은 신뢰도 포즈 필터링 후에도 노이즈가 존재하면 성능 악화

**3. 불균형 데이터셋에 대한 한계**[13]
- 최근 연구(2025)는 장꼬리 분포 문제 주목: 일부 행동 클래스는 훨씬 적은 샘플 보유
- MS-G3D는 이러한 불균형을 다루기 위한 특화 메커니즘 부재

**4. 계산 복잡도**[1]
- 다중 G3D 경로와 여러 τ, d 조합 사용 시 계산 비용 증가
- 실시간 응용(감시, 로봇)에는 부담스러울 수 있음

***

### 7. 후속 연구에 미치는 영향 및 고려할 점

#### 7.1 논문의 연구 커뮤니티에 미친 영향[14][15][16][17][1]

**높은 인용도**: arXiv 기준 1,400+ 인용(2024년 기준), 스켈레톤 기반 행동 인식 분야의 **기본 참고 문헌**으로 정립

**다중 경로 설계 패러다임**: MS-G3D의 G3D 경로와 인수분해된 경로를 분리하는 아이디어가 이후 많은 연구에서 채용됨[15][16]

**공간-시간 통합 그래프 합성곱의 재평가**: 기존의 순차적 공간-시간 처리 방식을 비판하고 직접 공간-시간 상호작용의 중요성 강조

#### 7.2 앞으로의 연구 시 고려할 점

**1. 도메인 일반화 강화**[2][3]
- **완전한 행동 복원 사전 활용**: 시간 불일치를 다루기 위해 경계 포즈 및 선형 시간 변환 학습
- **메타 학습 적용**: 새로운 도메인에 빠르게 적응하기 위한 메타 학습 프레임워크 통합
- **실제 응용 고려**: 포즈 추정 오류, 부분 가시성(occlusion) 등 현실 조건 대비

**2. 멀티모달 정렬 기법 적용**[5][6][4]
- **LLM 기반 텍스트 정보**: CrossGLG처럼 대규모 언어 모델에서 생성된 고수준 의미 정보 활용
- **비전-언어 표현학습**: 스켈레톤과 텍스트 간 공유 표현 공간 학습으로 제로샷 일반화 개선
- **확산 모델 활용**: 공간-텍스트 정렬 모달리티 격차 해소

**3. 자기지도 및 비지도 학습 확대**[8][9]
- **마스킹 기반 사전학습**: SkeletonMAE처럼 라벨 불필요한 사전학습으로 초기화 개선
- **대조학습**: 긍정 및 부정 쌍 간 대조로 판별력 있는 표현 학습
- **생성 모델**: VAE 또는 확산 모델로 적대적 데이터 생성 및 증강

**4. 적응형 그래프 학습 고도화**[16]
- **동적 그래프 구조**: 각 프레임 또는 행동 단계마다 그래프 위상 변경
- **생물학적 제약**: 인간 해부학 및 운동학적 제약 통합
- **주의 메커니즘**: 행동별 중요 관절에 선택적 주의

**5. 계산 효율성**
- **경량 아키텍처**: 모바일/임베디드 기기 배포를 위한 효율적 모델
- **양자화 및 프루닝**: 스켈레톤 데이터 특성에 맞춘 압축 기법
- **동적 계산**: 행동 복잡도에 따른 적응적 계산 비용 조절

**6. 특수 도메인 적용**[11][13]
- **신경학적 장애 진단**: SkelMamba(2024)처럼 임상 응용 고려
- **장꼬리 분포**: 클래스 불균형 문제 처리 메커니즘
- **다중 사람 장면**: 군중 내 개별 행동 인식

**7. 이론적 심화**
- **그래프 신경망의 표현력 분석**: 왜 k-인접 분해가 근본적으로 더 나은가에 대한 이론적 증명
- **정보 흐름 분석**: 공간-시간 직접 연결의 정보이론적 이점 정량화
- **편향 가중치 문제의 일반화**: 다른 그래프 신경망 구조에 적용 가능성

***

## 결론

**"Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition"**은 스켈레톤 기반 행동 인식에서 **두 가지 근본적 문제(편향된 가중치, 간접적 정보 흐름)를 우아하게 해결**한 influential 논문입니다. 제안된 **분해된 다중 스케일 집계**와 **G3D 모듈**은 이론적 깊이와 실증적 효과성을 모두 갖추고 있으며, 세 대규모 데이터셋에서 최첨단 성능을 달성했습니다.[1]

다만 **크로스 도메인 일반화, 포즈 추정 오류 견고성, 불균형 데이터 처리** 측면에서는 개선 여지가 있으며, 최신 연구는 **멀티모달 정렬(텍스트, 비전-언어), 자기지도 학습, 동적 그래프 구조** 등으로 이러한 한계를 보완하고 있습니다. 향후 연구자들은 이러한 방향들을 참고하여 더욱 강건하고 일반화 가능한 스켈레톤 기반 행동 인식 모델을 개발할 수 있을 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f01df306-d8a2-414e-a05d-fb19799875f1/2003.14111v2.pdf)
[2](https://ieeexplore.ieee.org/document/10485873/)
[3](https://crv.pubpub.org/pub/pb2qi4uo)
[4](https://arxiv.org/abs/2403.10082)
[5](https://ieeexplore.ieee.org/document/10495700/)
[6](https://ieeexplore.ieee.org/document/10379525/)
[7](https://www.semanticscholar.org/paper/5ec8485886ceeedc5ead51f11f1301122f6de5c5)
[8](https://ieeexplore.ieee.org/document/10484500/)
[9](https://arxiv.org/abs/2410.23641)
[10](https://arxiv.org/abs/2409.14336)
[11](https://link.springer.com/10.1007/s00371-024-03548-3)
[12](http://arxiv.org/pdf/2411.11288.pdf)
[13](http://arxiv.org/pdf/2411.19544.pdf)
[14](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Disentangling_and_Unifying_Graph_Convolutions_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.pdf)
[15](https://www.themoonlight.io/en/review/recovering-complete-actions-for-cross-dataset-skeleton-action-recognition)
[16](https://www.nature.com/articles/s41598-025-19399-4)
[17](https://arxiv.org/pdf/2112.09413.pdf)
[18](http://arxiv.org/pdf/2403.10082.pdf)
[19](https://arxiv.org/pdf/2308.14024.pdf)
[20](https://arxiv.org/pdf/2309.11445.pdf)
[21](https://arxiv.org/pdf/2209.02399.pdf)
[22](http://arxiv.org/pdf/2405.20606.pdf)
[23](http://arxiv.org/pdf/2404.15719.pdf)
[24](https://proceedings.neurips.cc/paper_files/paper/2024/file/a78f142aec481e68c75276756e0a0d91-Paper-Conference.pdf)
[25](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Skeleton-Based_Action_Recognition_With_Directed_Graph_Neural_Networks_CVPR_2019_paper.pdf)
[26](https://madison-proceedings.com/index.php/aetr/article/view/2116)
[27](https://ieeexplore.ieee.org/document/9695721/)
[28](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Lerch_Unsupervised_3D_Skeleton-Based_Action_Recognition_Using_Cross-Attention_With_Conditioned_Generation_WACVW_2024_paper.html)
