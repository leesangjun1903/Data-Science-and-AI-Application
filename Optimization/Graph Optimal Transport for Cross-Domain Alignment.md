# Graph Optimal Transport for Cross-Domain Alignment

### 1. 핵심 주장과 주요 기여

**Graph Optimal Transport (GOT)**는 최적 수송(Optimal Transport, OT) 이론을 활용하여 **교차 도메인 정렬(Cross-Domain Alignment)**을 해결하는 원칙적인 프레임워크입니다.[1]

기존의 주의 메커니즘(attention mechanism) 기반 방법들의 한계를 극복합니다:[1]

- **명시적 정렬 신호 부재**: 기존 방법은 과제 특화 손실함수만 사용하여 정렬을 직접적으로 장려하지 않음
- **해석성 부족**: 학습된 주의 행렬이 밀도가 높아 모델의 결정 근거 파악이 어려움

**주요 기여:**[1]
1. 교차 도메인 정렬을 그래프 매칭 문제로 공식화한 GOT 프레임워크 제안
2. Wasserstein Distance(WD)와 Gromov-Wasserstein Distance(GWD)를 결합하여 노드 및 간선(edge) 정보 동시 활용
3. 기존 신경망 모델에 **드롭인 정규화기(drop-in regularizer)**로 통합 가능
4. 희소하고 자기정규화된 수송 계획으로 **높은 해석성** 달성
5. 5가지 다양한 과제에서 일관된 성능 개선 입증

***

### 2. 문제 정의 및 해결 방법

#### 2.1 문제 설정

교차 도메인 정렬은 이미지의 객체나 문장의 단어 등 **두 도메인 간의 연관된 엔티티를 연결**하는 작업입니다.[1]

약한 감독(weakly supervised) 환경에서 작동합니다:[1]
- 도메인별 엔티티 쌍은 제공됨 (예: 이미지와 캡션)
- 하지만 구체적인 정렬 지표는 없음 (예: 이미지의 "개" 영역과 텍스트의 "개" 단어의 대응)

#### 2.2 GOT 프레임워크의 방법론

**동적 그래프 구성(Dynamic Graph Construction):**[1]

각 도메인의 엔티티들을 그래프로 표현합니다. 노드 $$i$$는 특성 벡터 $$x_i$$로 표현되고, 유사도 행렬로부터 간선을 구성합니다:

$$
C_x = \max(C_x - \tau, 0), \quad C_x = \{\cos(x_i, x_j)\}_{i,j}
$$

여기서 $$\tau = 0.1$$은 임계값입니다.[1]

**손실함수 설계:**[1]

원래의 감독 손실에 정렬 정규화항을 추가합니다:

$$
L(\theta) = L_{\text{sup}}(X, Y, l) + \alpha \cdot L_{\text{CDA}}(X, Y)
$$

#### 2.3 최적 수송 거리

**Wasserstein Distance (WD) - 노드 매칭:**[1]

두 확률 분포 $$\mu, \nu$$ 간의 최적 수송 비용을 정의합니다:

$$
D_w(\mu, \nu) = \min_{T \in \Pi(u,v)} \sum_{i=1}^{n} \sum_{j=1}^{m} T_{ij} \cdot c(x_i, y_j)
$$

여기서 $$T$$는 수송 계획으로 $$T_{ij}$$는 도메인 X의 $$u_i$$에서 도메인 Y의 $$v_j$$로 이동되는 질량을 나타냅니다.[1]

**Gromov-Wasserstein Distance (GWD) - 간선 매칭:**[1]

그래프의 구조적 유사성을 비교합니다:

$$
D_{gw}(\mu, \nu) = \min_{\hat{T} \in \Pi(u,v)} \sum_{i,i',j,j'} \hat{T}_{ij} \hat{T}_{i'j'} L(x_i, y_j, x'_i, y'_j)
$$

비용함수 $$L$$은 그래프 내 두 노드 쌍 간의 거리 차이를 측정합니다:[1]

$$
L(x_i, y_j, x'_i, y'_j) = \|c_1(x_i, x'_i) - c_2(y_i, y'_i)\|
$$

**GOT 통합 거리:**[1]

WD와 GWD를 하나의 공유 수송 계획으로 통합합니다:

$$
D_{\text{got}}(\mu, \nu) = \min_{T \in \Pi(u,v)} \sum_{i,i',j,j'} T_{ij} \left[\lambda c(x_i, y_j) + (1-\lambda)T_{i'j'} L(x_i, y_j, x'_i, y'_j)\right]
$$

여기서 $$\lambda$$는 두 거리의 상대적 중요도를 제어합니다.[1]

**계산 효율성:**[1]

Sinkhorn 알고리즘을 사용하여 효율적으로 해결합니다:

$$
\min_{T \in \Pi(u,v)} \sum_{i,j} T_{ij} c(x_i, y_j) + \beta H(T)
$$

여기서 $$H(T) = \sum_{i,j} T_{ij} \log T_{ij}$$는 엔트로피 정규화항입니다.[1]

#### 2.4 GOT의 이점

**희소성(Sparsity):**[1]
정확한 최적 수송은 최대 $$(2r-1)$$개의 0이 아닌 원소만을 포함하여 더 해석 가능한 정렬 생성

**자기정규화(Self-normalization):**[1]
수송 계획 $$T$$의 모든 원소의 합이 1로 정규화되어 확률 분포로 해석 가능

**효율성:**[1]
행렬-벡터 곱만 필요하여 대규모 딥러닝 모델에 쉽게 통합 가능

***

### 3. 모델 구조 및 아키텍처

#### 3.1 전체 계산 구조

[figure 2를 참고하면] GOT의 계산 그래프는 다음과 같이 구성됩니다:[1]

1. **입력 처리**: 원본 데이터 $$\tilde{X}, \tilde{Y}$$
2. **신경망 처리**: 신경망 $$f_\theta(\cdot)$$를 통한 문맥화된 표현 생성 → X, Y
3. **그래프 구성**: 
   - 도메인 X에 대한 비용 행렬 $$C_x$$ 생성
   - 도메인 Y에 대한 비용 행렬 $$C_y$$ 생성
4. **OT 거리 계산**:
   - WD와 GWD 알고리즘 적용
   - 교차 도메인 비용 행렬 $$C_{xy}$$ 계산
5. **수송 계획 도출**: 최종 수송 계획 T 생성

#### 3.2 알고리즘 상세

**Algorithm 1 - Wasserstein Distance:**[1]
```
입력: {xi}, {yj}, 정규화 파라미터 β
초기화: σ = (1/n)1_n, T^(1) = 11^T
비용 행렬 생성: C_ij = c(x_i, y_j), A_ij = exp(-C_ij/β)

반복 (t=1,2,3,...):
  Q = A ⊙ T^(t)  (Hadamard 곱)
  K번 반복:
    δ = (1/n)Qσ
    σ = (1/n)Q^T δ
  T^(t+1) = diag(δ)Q diag(σ)

출력: 수송 계획 T, Wasserstein 거리 D_wd = ⟨C^T, T⟩
```

**Algorithm 2 - Gromov-Wasserstein Distance:**[1]
```
입력: {xi}, {yj}, 확률 벡터 p, q
도메인 내 유사도 계산:
  [C_x]_ij = cos(x_i, x_j)
  [C_y]_ij = cos(y_i, y_j)

교차 도메인 비용 행렬:
  C_xy = C_x^2 p1_m^T + C_y q(C_y^2)^T

반복: Algorithm 1을 이용해 수송 계획 T 해결

출력: T, Gromov-Wasserstein 거리 D_gw = ⟨L^T, T⟩
```

**Algorithm 3 - GOT 거리:**[1]
```
입력: {xi}, {yj}, 하이퍼파라미터 λ

도메인 내 유사도:
  [C_x]_ij = cos(x_i, x_j)
  [C_y]_ij = cos(y_i, y_j)

변환된 표현 계산:
  x'_i = g_1(x_i)   (MLP 변환)
  y'_j = g_2(y_j)

교차 도메인 유사도:
  C_ij = cos(x'_i, y'_j)

공유 수송 계획인 경우:
  L_unified = λC + (1-λ)L
  Algorithm 2의 Line 8에 L_unified 대입
  새로운 T 해결

출력: GOT 거리 D_got
```

#### 3.3 핵심 설계 선택

**공유 수송 계획의 이점:**[1]
- 비공유 방식 대비 성능 향상: EN-VI 데이터셋에서 29.92 vs 29.77 BLEU
- 계산 효율성: Sinkhorn 알고리즘을 한 번만 실행
- 직관: WD와 GWD가 노드 및 간선 정보를 통해 서로 강화

***

### 4. 성능 향상 및 실험 결과

#### 4.1 이미지-텍스트 검색 (Vision-and-Language Tasks)

**Flickr30K 데이터셋:**[1]

| 메서드 | R@1 | R@5 | R@10 | Rsum |
|--------|-----|-----|------|------|
| SCAN (기준) | 67.7 | 88.9 | 94.0 | 452.2 |
| SCAN + WD | 70.9 | 92.3 | 95.2 | 472.3 |
| SCAN + GWD | 69.5 | 91.2 | 95.2 | 468.6 |
| SCAN + GOT | **70.9** | **92.8** | **95.5** | **474.8** |

**COCO 데이터셋:**[1]

| 메서드 | R@1 | R@5 | R@10 | Rsum |
|--------|-----|-----|------|------|
| SCAN (기준) | 46.4 | 77.4 | 87.2 | 384.8 |
| SCAN + GOT | **50.5** | **80.2** | **89.8** | **403.9** |

**성능 분석:**[1]
- Flickr30K: 최대 +5.1% (R@1 지표)
- COCO: 최대 +8.8% (R@1 지표)
- GWD 단독보다 WD 단독이 더 효과적: 간선 정보만으로는 충분하지 않음을 시사

#### 4.2 시각 질의응답 (VQA 2.0)

**BAN 모델에서의 성능:**[1]

| 메서드 | 정확도 |
|--------|-------|
| BAN | 66.00 |
| BAN + GWD | 66.21 |
| BAN + WD | 66.26 |
| BAN + GOT | **66.44** |

**아키텍처별 효과:**[1]

| 모델 | BUTD | BAN-1 | BAN-2 | BAN-4 | BAN-8 |
|------|------|-------|-------|-------|-------|
| 기준 | 63.37 | 65.37 | 65.61 | 65.81 | 66.00 |
| +GOT | 65.01 | 65.68 | 65.88 | 66.10 | **66.44** |
| 향상도 | +2.6% | +0.5% | +0.4% | +0.4% | +0.7% |

**중요한 발견:**[1]
- 더 간단한 모델(BUTD)에서 더 큰 개선: 2.6%
- 복잡한 모델(BAN-8)도 개선 가능: 0.7%
- BAN-4 + GOT가 BAN-8 단독 성능 초과

#### 4.3 이미지 캡셔닝

**COCO 데이터셋:**[1]

| 메서드 | CIDEr | BLEU-4 | BLUE-3 | BLEU-2 | BLEU-1 | ROUGE | METEOR |
|--------|-------|--------|--------|--------|---------|-------|---------|
| MLE | 106.3 | 34.3 | 45.3 | 59.3 | 75.6 | 55.2 | 26.2 |
| MLE + WD | 107.9 | 34.8 | 46.1 | 60.1 | 76.2 | 55.6 | 26.5 |
| MLE + GWD | 106.6 | 33.3 | 45.2 | 59.1 | 75.7 | 55.0 | 25.9 |
| MLE + GOT | **109.2** | **35.1** | **46.5** | **60.3** | **77.0** | **56.2** | **26.7** |

**상대적 개선도:**[1]
- WD 대비 GOT의 상대 개선율: $$\frac{109.2-107.9}{107.9-106.3} = 81.25\%$$
- GWD의 추가적 구조 정보가 약 80% 이상의 추가 이득 제공

#### 4.4 기계 번역 (Machine Translation)

**EN-VI 데이터셋 (IWSLT):**[1]

| 모델 | EN-VI uncased | EN-VI cased |
|------|---------------|-------------|
| Transformer | 29.25 ± 0.18 | 28.46 ± 0.17 |
| + WD | 29.49 ± 0.10 | 28.68 ± 0.14 |
| + GWD | 28.65 ± 0.14 | 28.34 ± 0.16 |
| + GOT | **29.92 ± 0.11** | **29.09 ± 0.18** |

**EN-DE 데이터셋 (WMT):**[1]

| 모델 | EN-DE uncased | EN-DE cased |
|------|---------------|-------------|
| Transformer | 25.60 ± 0.07 | 25.12 ± 0.12 |
| + GOT | **26.05 ± 0.17** | **25.54 ± 0.15** |

**성능 분석:**[1]
- EN-VI: +0.67 BLEU (uncased), +0.63 BLEU (cased)
- EN-DE: +0.45 BLEU (uncased), +0.42 BLEU (cased)
- GWD만 사용하면 성능 저하: 구조 정보만으로는 부족

#### 4.5 추상적 요약 (Abstractive Summarization)

**English Gigawords 데이터셋:**[1]

| 메서드 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|---------|---------|---------|
| LSTM | 36.11 | 16.39 | 32.32 |
| + GWD | 36.31 | 17.32 | 33.15 |
| + WD | 36.81 | 17.34 | 33.34 |
| + GOT | **37.10** | **17.61** | **33.70** |

**성능 개선:**[1]
- ROUGE-1: +0.99
- ROUGE-2: +1.22
- ROUGE-L: +1.38

***

### 5. 일반화 성능 향상 가능성

#### 5.1 다중 과제 일반화

GOT의 주목할 만한 특징은 **다양한 도메인과 과제에서의 일관된 성능 향상**입니다:[1]

1. **비전-언어 과제**: 이미지-텍스트 검색, VQA, 이미지 캡셔닝
2. **언어 과제**: 기계 번역, 텍스트 요약

모든 과제에서 성능 개선을 달성했습니다.

#### 5.2 모델 아키텍처 무관성

GOT는 **드롭인 정규화기**로서 기존 모델에 쉽게 적용 가능합니다:[1]

- SCAN (이미지-텍스트 검색)
- BAN, BUTD (VQA)
- LSTM, Transformer (시퀀스 모델)

다양한 아키텍처에 적용 가능한 범용성을 입증합니다.

#### 5.3 계산 오버헤드 최소화

**실제 학습 시간:**[1]
- SCAN: 6시간 34분
- SCAN + GOT: 6시간 57분
- **추가 시간: 약 7% (23분)**

Envelope Theorem으로 인해 GOT는 순전파(forward pass) 중에만 계산되므로 역전파(backward pass)에 미치는 영향 최소화.

#### 5.4 해석성 개선

**시각화 비교 (Figure 3):**[1]

SCAN의 주의 행렬 vs GOT의 수송 계획:
- SCAN: 밀도가 높고 노이즈가 많음
- GOT: 희소하고 자명한(clear) 정렬
- 예시: "sidewalk"와 "skateboard"가 정확히 대응되는 이미지 영역과 매칭

**희소성의 이점:**
- 모델 예측의 근거를 명확하게 제시
- 오류 분석 및 모델 디버깅 용이
- 신뢰성 있는 의사결정 가능

#### 5.5 도메인 간 관계 모델링

GOT의 두 가지 거리 측정:

1. **Wasserstein Distance**: 
   - 개별 엔티티 간의 직접적 매칭
   - 노드 표현의 유사성 포착

2. **Gromov-Wasserstein Distance**:
   - 구조적 관계 보존
   - 중복된 엔티티의 문맥적 차이 구분
   - "red book on blue desk" 예시: GWD가 여러 책과 책상 중 정확한 매칭 가능

#### 5.6 약한 감독 환경에서의 효과

**명시적 정렬 신호 추가:**[1]

기존 방법: $$L(\theta) = L_{\text{sup}}(X, Y, l)$$ (과제 손실만)

GOT: $$L(\theta) = L_{\text{sup}}(X, Y, l) + \alpha \cdot L_{\text{CDA}}(X, Y)$$

명시적 정렬 정규화가 모델 학습을 강화하여 더 나은 일반화 성능 달성.

***

### 6. 모델의 한계

#### 6.1 방법론적 한계

1. **그래프 구성의 제약:**[1]
   - 임계값 $$\tau = 0.1$$ 선택의 효과에 대한 상세 분석 부족
   - 다양한 데이터 유형에 대한 최적 임계값 설정 방법 미제시

2. **하이퍼파라미터 민감도:**[1]
   - 가중치 $$\lambda$$ 선택: EN-VI에서 최적값 0.8, 다른 과제에서는 다를 수 있음
   - 정규화 파라미터 $$\alpha$$ 조정 필요

3. **계산 복잡도:**[1]
   - Sinkhorn 알고리즘의 반복 횟수 K 설정 기준 미명시
   - 대규모 시스템(큰 n, m)에서의 성능 보장 미확인

#### 6.2 성능상 한계

1. **개선의 크기 변동:**
   - VQA: 0.4~0.7% 개선 (상대적으로 작음)
   - 복잡한 모델에서 개선 폭 감소 경향

2. **GWD 단독 사용의 성능 저하:**[1]
   - 기계 번역에서 GWD만 사용시 성능 악화
   - 구조 정보만으로는 충분하지 않음을 시사

3. **도메인 특성에 따른 효과:**
   - 간단한 모델에서 더 효과적 (BUTD: 2.6%)
   - 이미 복잡한 모델에서는 개선 여지 제한

#### 6.3 실험적 한계

1. **일반화 범위:**
   - 주로 이미지-텍스트 및 텍스트-텍스트 과제에 집중
   - 다른 모달리티(비디오, 오디오 등)에 대한 검증 부족

2. **비교 기준 부족:**
   - 동시대 다른 OT 기반 정렬 방법과의 상세 비교 미흡
   - 그래프 기반 주의 메커니즘(GAT, GMN)과의 직접 비교 제한적

3. **통계적 유의성:**
   - 일부 개선값의 표준편차 범위가 겹치는 경우 존재
   - 반복 실험 횟수 상이 (EN-DE: 1회 vs 기계 번역 ablation: 5회)

***

### 7. 논문의 앞으로의 연구 영향 및 고려사항

#### 7.1 최신 연구에서의 GOT 영향 (2023-2025)

**1. Optimal Transport 기반 정렬의 확산:**[2][3][4]

최근 연구들이 GOT의 아이디어를 다양한 분야에 확장하고 있습니다:

- **교차 도메인 요약 (2023)**: SCCS 모델이 GOT의 OT 정렬 원리를 활용하여 동영상과 기사의 교차 도메인 요약 성능 향상 (텍스트 8%, 동영상 6.6% 개선)[2]

- **감정 분석 (2025)**: OTESGN 모델이 Optimal Transport를 그래프 신경망과 결합하여 관점 기반 감정 분석에서 최첨단 성능 달성[5]

- **이미지 검색 (2024)**: ProtoOT가 Optimal Transport를 활용하여 비지도 교차 도메인 이미지 검색에서 K-means 클러스터링과 통합[6]

**2. 다중 도메인 적응의 확대:**[3]

- **협업적 다중 도메인 적응 (2024)**: CMDA-OT가 여러 소스 도메인과 대상 도메인 간 OT 기반 적응 수행, 개인정보 보호 및 모델 적응 동시 해결[3]

- **활동 인식 (2024)**: TROT가 시간적 관계 OT를 활용하여 사용자 간 활동 인식에서 기존 방법 능가[4]

**3. 오프라인 강화학습에의 적용 (2025):**[7]

- **OTDF**: Optimal Transport를 사용한 데이터 필터링으로 소스 도메인 데이터와 목표 도메인 데이터 정렬, 오프라인 RL에서 정규화 성능 60% 향상[7]

#### 7.2 현재 연구 트렌드

**A. 멀티모달 정렬의 강조:**[8][9][10]

최신 문헌에서 **멀티모달 도메인 적응 (MMDA)**이 중요한 주제로 부상하고 있습니다:[8]

- GOT의 그래프 기반 정렬 접근은 비전-언어 모델에서 텍스트와 시각 특성의 **의미적 갭** 해결에 직접 적용 가능
- MMA (Multi-Modal Adapter)는 텍스트와 비전 브랜치 간 교차 모달 상호작용을 GOT의 원칙과 유사하게 구조화[9]

**B. 기초 모델과의 통합:**[11]

Vision-Language Models (CLIP 등)의 일반화 성능 향상이 집중 연구 분야입니다:[11]

- GOT의 **명시적 정렬 신호** 개념이 CLIP 기반 미세조정(fine-tuning)에 적용될 수 있음
- 영점 샷 전이(zero-shot transfer) 상황에서 도메인 시프트 극복에 유용

**C. 희소 주의 메커니즘의 복부:**[12][13]

해석 가능성 강화 트렌드에서 GOT의 희소 수송 계획이 주목받고 있습니다:[13][12]

- SparseMAP, Stream 등의 희소 주의 방법들이 구조 귀납과 해석성에서 GOT와 유사한 철학 공유
- 긴 문맥에서의 주의 패턴 분석(Sparse Tracing)이 GOT의 희소성 개념과 일맥상통[13]

#### 7.3 미래 연구 시 고려할 점

**1. 대규모 모델과의 호환성:**

- **현재 상황**: 2024-2025 주요 VLM (CLIP-ViT, LLaVA 등)의 파라미터 수 급증
- **고려사항**: 억 단위 파라미터 모델에 대한 GOT의 스케일링 성능 검증 필요
- **추천**: Sinkhorn 알고리즘의 근사(approximation) 및 분산(distributed) 계산 연구

**2. 도메인 이질성(Domain Heterogeneity) 처리:**

- **현재 문제**: GOT는 주로 이미지-텍스트 쌍 또는 문자-번역 쌍에만 집중
- **확장 방향**: 비디오-음성, 3D 모델-스케치 등 더 이질적인 모달리티 간 정렬 연구
- **기술적 어려움**: 상이한 모달리티 간 거리 함수 $$c(·,·)$$ 설계의 어려움

**3. 자기감독(Self-supervised) 학습과의 결합:**

- **최신 트렌드**: 논문의 향후 계획(Section 5, Conclusions)에서 언급된 바와 같이, 자기감독 표현 학습(self-supervised representation learning)과의 결합
- **기회**: GOT의 명시적 정렬이 대조 학습(contrastive learning) 손실에 통합되어 더 견고한 표현 학습 가능
- **사례**: OTESGN에서 대조 손실을 GOT와 결합하여 성능 향상[5]

**4. 계산 효율성 개선:**

- **현재 오버헤드**: 기계 번역 ablation에서 공유 vs 비공유 수송 계획의 효과 검증 시 상당한 반복 계산 필요
- **개선 방향**:
  - Entropic OT의 대안으로 **Unbalanced OT** 또는 **Sliced Wasserstein Distance** 고려
  - 동적 프로그래밍 또는 근사 알고리즘으로 계산 시간 단축

**5. 도메인 시프트 강건성 평가:**

- **미충족 갭**: 논문은 다양한 과제에서의 성능만 보이고, 실제 도메인 시프트 상황(out-of-distribution) 평가 부족
- **필요한 연구**: 
  - ImageNet → Sketch, 자연 이미지 → 의료 이미지 등의 극단적 도메인 시프트에서의 GOT 성능
  - DOMAINNET, PACS 같은 표준 도메인 적응 벤치마크에서의 평가

**6. 이론적 보장:**

- **현재 한계**: GOT의 수렴성(convergence) 및 근사 품질(approximation quality)에 대한 이론적 분석 부족
- **학술적 가치**: 
  - Sinkhorn 알고리즘의 수렴 속도 분석
  - 엔트로피 정규화 파라미터 $$\beta$$와 근사 정확도 간의 관계 규명

**7. 추론 단계(Inference) 최적화:**

- **현재 상황**: GOT는 학습 중에만 사용됨
- **잠재력**: 
  - 추론 시 희소 수송 계획을 활용한 선택적 계산(selective computation)
  - 실시간 시스템(자율주행, 라이브 번역)에서의 적용 가능성 탐색

**8. 다문화 및 언어 다양성:**

- **글로벌화**: 기계 번역 실험이 EN-VI, EN-DE에만 국한
- **확장 필요**: 
  - 언어 선택지가 제한적인 언어 쌍(rare language pairs)에서의 성능
  - 비인도유럽 언어군(동아시아, 아프리카 언어) 포함

***

### 8. 종합 평가

**Graph Optimal Transport**는 교차 도메인 정렬 문제에 대한 원칙적이고 효과적인 해결책을 제시합니다. Optimal Transport 이론의 수학적 엄밀성과 희소성, 해석성 같은 실질적 이점을 결합하여 기존 주의 메커니즘의 한계를 극복합니다.

**주요 강점:**
- 5가지 다양한 과제에서 일관된 성능 개선
- 드롭인 정규화기로서의 범용성
- 희소하고 해석 가능한 정렬 결과
- 최소한의 계산 오버헤드

**제약 사항:**
- 대규모 모델에서 개선 폭 감소
- 하이퍼파라미터 민감도
- 극단적 도메인 시프트 평가 부족

**미래 방향:**
최신 연구(2023-2025)에서 보듯이, GOT의 원리는 다중 도메인 적응, 자기감독 학습, 기초 모델 적응 등으로 활발히 확장되고 있습니다. 계산 효율성, 대규모 모델 호환성, 극단적 도메인 시프트 강건성에 대한 추가 연구가 필요하며, 이러한 개선이 이루어질 경우 더욱 광범위한 응용이 가능할 것으로 예상됩니다.

***

### 참고 문헌 인덱스

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b9bd74c4-0c06-44de-b180-b4e54c4823aa/2006.14744v3.pdf)
[2](https://aclanthology.org/2023.findings-acl.101.pdf)
[3](https://arxiv.org/html/2404.06599)
[4](http://arxiv.org/pdf/2403.15423.pdf)
[5](https://www.themoonlight.io/ko/review/otesgnoptimal-transport-enhanced-syntactic-semantic-graph-networks-for-aspect-based-sentiment-analysis)
[6](https://arxiv.org/html/2402.18411)
[7](https://openreview.net/pdf?id=LRrbD8EZJl)
[8](https://arxiv.org/html/2501.18592v3)
[9](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.html)
[10](https://www.sciencedirect.com/science/article/abs/pii/S1566253524006407)
[11](https://arxiv.org/html/2506.18504v1)
[12](https://aclanthology.org/W18-5450.pdf)
[13](https://arxiv.org/html/2510.19875v1)
[14](https://arxiv.org/html/2503.15779)
[15](http://arxiv.org/pdf/2405.09400.pdf)
[16](https://arxiv.org/pdf/1507.00504.pdf)
[17](https://arxiv.org/abs/2406.03319)
[18](https://arxiv.org/abs/2006.14744)
[19](https://www.nature.com/articles/s41598-024-53311-w)
[20](https://www.themoonlight.io/ko/review/fusion-of-graph-neural-networks-via-optimal-transport)
[21](https://proceedings.neurips.cc/paper_files/paper/2023/file/1e5f58d98523298cba093f658cfdf2d6-Paper-Conference.pdf)
[22](https://www.semanticscholar.org/paper/Graph-Optimal-Transport-for-Cross-Domain-Alignment-Chen-Gan/2a81f6bf76bcb70244aa40217ff316025971bd0f)
[23](https://koreascience.or.kr/article/JAKO200111920777562.pdf)
[24](https://web.math.princeton.edu/~amits/publications/2305.12310.pdf)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC12520601/)
[26](https://arxiv.org/html/2504.00691v1)
[27](https://arxiv.org/pdf/2307.14061.pdf)
[28](https://aclanthology.org/2022.emnlp-main.488.pdf)
[29](http://arxiv.org/pdf/2311.12327.pdf)
[30](http://arxiv.org/pdf/2405.06217.pdf)
[31](http://arxiv.org/pdf/2311.15569.pdf)
[32](http://arxiv.org/pdf/2108.10904v3.pdf)
[33](http://arxiv.org/pdf/2311.17091.pdf)
[34](https://aclanthology.org/2024.emnlp-main.124.pdf)
[35](https://openaccess.thecvf.com/content_CVPR_2020/papers/Munro_Multi-Modal_Domain_Adaptation_for_Fine-Grained_Action_Recognition_CVPR_2020_paper.pdf)
[36](https://proceedings.mlr.press/v189/pandey23a/pandey23a.pdf)
[37](https://arxiv.org/pdf/2506.03189.pdf)
