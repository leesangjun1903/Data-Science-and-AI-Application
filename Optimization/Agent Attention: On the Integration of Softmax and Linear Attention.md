# Agent Attention: On the Integration of Softmax and Linear Attention
https://github.com/LeapLabTHU/Agent-Attention

## 1. 핵심 주장과 주요 기여 요약

Agent Attention은 Softmax attention의 높은 표현력과 linear attention의 효율성을 통합한 혁신적인 어텐션 패러다임을 제시합니다. 논문의 핵심 기여는 다음과 같습니다:[1]

**새로운 어텐션 패러다임**: 기존 (Q, K, V) 구조에 agent 토큰 A를 추가하여 (Q, A, K, V) 4원체 구조를 제안합니다. Agent 토큰은 쿼리 토큰 Q의 "대리인" 역할을 하여 K와 V로부터 정보를 집계하고 이를 다시 Q에게 전파합니다.[1]

**계산 복잡도 획기적 개선**: Agent attention은 기존 Softmax attention의 O(N²) 복잡도를 O(N)으로 감소시키면서도 전역적 맥락 모델링 능력을 유지합니다. Agent 토큰의 개수가 쿼리 토큰보다 훨씬 적게 설계될 수 있기 때문입니다.[1]

**이론적 통합**: Agent attention이 일반화된 linear attention과 수학적으로 동등함을 증명하여, Softmax attention과 linear attention의 우아한 통합임을 보였습니다.[1]

## 2. 문제 정의, 제안 방법, 모델 구조 및 성능

### 해결하고자 하는 문제

현대 Vision Transformer 모델에서 사용되는 Softmax attention은 모든 쿼리-키 쌍 간의 유사도를 계산하여 O(N²)의 계산 복잡도를 갖습니다. 이로 인해 고해상도 영상이나 긴 시퀀스 처리 시 계산 부담이 급격히 증가합니다. 기존 해결책들(Swin Transformer의 윈도우 어텐션, PVT의 희소 어텐션 등)은 장거리 의존성 모델링 능력을 희생합니다.[1]

### 제안 방법 (수식 포함)

Agent attention은 두 단계의 Softmax attention 연산으로 구성됩니다:[1]

**1단계 - Agent Aggregation**:

$$V_A = \text{Attn}^S(A, K, V) = \sigma(AK^T)V$$

**2단계 - Agent Broadcast**:

$$O^A = \text{Attn}^S(Q, A, V_A) = \sigma(QA^T)V_A$$

전체 공식은:

$$O^A = \sigma(QA^T)\sigma(AK^T)V$$

이는 일반화된 linear attention과 동등함이 증명됩니다:

$$O^A = \phi_q(Q)\phi_k(K)^TV$$

여기서 $$\phi_q(Q) = \sigma(QA^T)$$, $$\phi_k(K) = (\sigma(AK^T))^T$$입니다.[1]

### 향상된 Agent Attention 모듈

**Agent Bias**: 위치 정보를 효과적으로 활용하기 위해 agent bias를 도입합니다:

$$O^A = \sigma(QA^T + B_2)\sigma(AK^T + B_1)V$$

**Diversity restoration**: Feature diversity 부족 문제를 해결하기 위해 depthwise convolution(DWC) 모듈을 추가합니다:

$$O = \sigma(QA^T + B_2)\sigma(AK^T + B_1)V + \text{DWC}(V)$$

**전체 복잡도**:

$$\Omega = 4NC^2 + NC + 4nNC + k^2NC$$

여기서 n은 agent 토큰 수, k는 DWC 커널 크기입니다.[1]

### 성능 향상 결과

**ImageNet-1K 분류**: Agent-DeiT-T는 DeiT-T 대비 2.7% 향상(72.2% → 74.9%), Agent-PVT-S는 PVT-L보다 30% 적은 파라미터와 40% 적은 FLOPs로 더 높은 성능 달성.[1]

**COCO 객체 탐지**: Agent-PVT 모델들은 기존 PVT 대비 3.9-4.7 box AP 향상, Agent-Swin 모델들은 최대 1.5 box AP 향상.[1]

**ADE20K 의미 분할**: Agent-PVT-T는 기존 PVT-T 대비 3.61 mIoU 향상, Agent-Swin-T는 2.17 mIoU 향상.[1]

**추론 속도**: CPU에서 1.7-2.1배, GPU에서 1.4-1.7배 빠른 추론 속도 달성.[1]

## 3. 일반화 성능 향상 가능성

Agent attention의 일반화 성능 향상은 여러 측면에서 확인됩니다:

**대용량 수용 필드**: 선형 복잡도 덕분에 전역 수용 필드를 유지하면서도 동일한 계산량을 보장합니다. 실험 결과 윈도우 크기가 증가할수록 성능이 지속적으로 향상됨을 보였습니다.[1]

**고해상도 시나리오 우수성**: 고해상도 영상 처리에서 특히 뛰어난 성능을 보입니다. Agent-DeiT-S는 448×448 해상도에서 83.1% 정확도를 달성하여, 표준 해상도의 DeiT-B보다 높은 성능을 보였습니다.[1]

**다양한 아키텍처 적응성**: DeiT, PVT, Swin, CSwin 등 다양한 Vision Transformer 아키텍처에 플러그인 방식으로 적용 가능하며, 모든 경우에서 일관된 성능 향상을 보였습니다.[1]

**제로샷 적용 가능**: Stable Diffusion에 추가 훈련 없이 직접 적용하여 1.84배 빠른 생성 속도와 0.9 낮은 FID 점수를 달성했습니다.[1]

## 4. 한계점

**Agent 토큰 수 설계**: Agent 토큰의 최적 개수는 작업과 모델 깊이에 따라 달라지며, 이에 대한 체계적인 설계 원칙이 부족합니다.[1]

**Feature Diversity 문제**: 일반화된 linear attention의 특성상 feature diversity 부족 문제가 있어 DWC 모듈이 필수적입니다.[1]

**훈련 없는 적용 제한**: Stable Diffusion 적용 시 agent bias와 DWC를 사용할 수 없어 성능 제약이 있습니다.[1]

## 5. 연구 영향 및 향후 고려사항

### 연구에 미치는 영향

**Attention 메커니즘 패러다임 전환**: Softmax와 linear attention을 대립적 관계로 보던 기존 관점을 통합적 관점으로 전환시켰습니다.[1]

**효율적 Vision Transformer 설계**: 고해상도 영상 처리와 장시퀀스 모델링을 위한 새로운 방향을 제시했습니다.[1]

**확장성 입증**: 비디오 모델링, 멀티모달 파운데이션 모델 등 초장시퀀스 작업으로의 확장 가능성을 보였습니다.[1]

### 향후 연구 고려사항

**Agent 토큰 최적화**: 동적 agent 토큰 선택, 적응적 개수 조정, 계층적 agent 구조 등의 연구가 필요합니다.

**이론적 분석 심화**: Agent attention의 표현력, 수렴성, 일반화 능력에 대한 더 깊은 이론적 분석이 요구됩니다.

**도메인 특화 적용**: 의료 영상, 자율주행, 자연어 처리 등 특정 도메인에 최적화된 agent attention 설계 연구가 중요합니다.

**하이브리드 아키텍처**: Agent attention과 다른 효율적 attention 메커니즘의 결합을 통한 더 나은 성능-효율성 균형점 탐색이 필요합니다.

이 연구는 Transformer 아키텍처의 근본적 한계를 해결하는 혁신적 접근법으로, 향후 대규모 모델과 고해상도 처리에서 핵심적 역할을 할 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/cac55c27-8e5c-4fbe-bc73-cb0d5f54a6e1/2312.08874v3.pdf)
