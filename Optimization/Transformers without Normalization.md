
# Transformers without Normalization

## 1. 핵심 주장 및 주요 기여 요약

본 논문의 핵심 주장은 **정규화 계층(Normalization Layer)이 Transformer 모델에서 필수불가결하지 않다**는 것입니다. 연구진은 단순한 **Dynamic Tanh (DyT)** 함수로 정규화 계층을 완전히 대체할 수 있으며, 동일하거나 더 나은 성능을 달성할 수 있음을 입증했습니다.[1]

주요 기여는 다음과 같습니다:

- **DyT 함수 제안**: 요소별 연산 $$DyT(x) = \tanh(\alpha x)$$를 통해 정규화 계층을 대체하는 간단한 방법 제시
- **광범위한 실증 검증**: 컴퓨터 비전(ViT, DiT), 자연어 처리(LLaMA), 음성 처리(wav2vec 2.0), DNA 서열 모델링 등 다양한 도메인에서 효과성 입증
- **이론적 통찰**: 정규화 계층이 S자 모양의 tanh 함수와 유사한 입-출력 매핑을 수행함을 발견하고 이를 활용

***

## 2. 논문이 해결하는 문제 및 제안 방법

### 2.1 문제 정의

정규화 계층은 배치 정규화(Batch Normalization, BN) 도입 이후 약 10년간 신경망의 필수 요소로 간주되어 왔습니다. 그러나 정규화 계층은:[1]

- **계산 오버헤드**: 통계량(평균, 분산) 계산 필요
- **구조적 제약**: 정규화 계층을 교체하려는 시도가 거의 없음
- **일반화 메커니즘 이해 부족**: 정규화가 어떻게 작동하는지 충분히 이해되지 않음

### 2.2 핵심 발견: Layer Normalization의 Tanh 유사성

논문은 실제로 학습된 네트워크의 정규화 계층 입-출력 관계를 분석했습니다. Vision Transformer (ViT-B), wav2vec 2.0, Diffusion Transformer (DiT-XL)에서:

- **초기 계층**: 주로 선형 변환
- **깊은 계층**: **S자 형태의 tanh 함수 곡선**을 보임[1]

이는 정규화 계층이 다음 두 가지 효과를 동시에 수행함을 의미합니다:

1. **선형 변환**: 중앙값 부근에서 거의 선형적으로 작동
2. **비선형 극값 억압**: 극단값을 비선형적으로 억압 (squashing)

### 2.3 제안 방법: Dynamic Tanh (DyT)

#### 기본 수식

$$DyT(x) = \gamma \cdot \tanh(\alpha x) + \beta$$

여기서:
- **$$\alpha$$**: 학습 가능한 스칼라 매개변수 (입력 범위 조절)
- **$$\gamma$$**: 학습 가능한 채널별 벡터 (스케일링, 형태: (C,))
- **$$\beta$$**: 학습 가능한 채널별 벡터 (시프팅, 형태: (C,))

#### 핵심 특징

| 특징 | 설명 |
|------|------|
| **드롭인 교체** | 기존 정규화 계층을 그대로 대체 가능 |
| **초기화** | $$\alpha_0 = 0.5$$ (LLM 제외), $$\gamma = 1$$, $$\beta = 0$$ |
| **통계량 불필요** | 배치 통계 계산 불필요 (요소별 연산만 수행) |
| **정규화가 아님** | 함수 자체는 정규화가 아니라 활성화 함수처럼 작동 |

### 2.4 모델 구조 통합

DyT는 다음 위치에 배치됩니다:

- Attention 블록 내 정규화 계층
- FFN 블록 내 정규화 계층  
- 최종 선형 투영 전 정규화 계층

***

## 3. 성능 향상 분석

### 3.1 컴퓨터 비전 작업

**표 1: ImageNet-1K 지도 분류 정확도**

| 모델 | LN | DyT | 변화 |
|------|-----|-----|------|
| ViT-B | 82.3% | 82.5% | ↑0.2% |
| ViT-L | 83.1% | 83.6% | ↑0.5% |
| ConvNeXt-B | 83.7% | 83.7% | - |
| ConvNeXt-L | 84.3% | 84.4% | ↑0.1% |

**자가 지도 학습**:[1]

- MAE ViT-B: 83.2% (동일)
- DINO ViT-B (patch 8): 84.1% → 84.5% (↑0.4%)

### 3.2 생성 모델 (Diffusion Transformer)

**표 2: ImageNet FID 점수 (낮을수록 좋음)**

| 모델 | LN | DyT | 변화 |
|------|-----|-----|------|
| DiT-B | 64.9 | 63.9 | ↓1.0 |
| DiT-L | 45.9 | 45.7 | ↓0.2 |
| DiT-XL | 19.9 | 20.8 | ↑0.9 |

### 3.3 대규모 언어 모델 (LLaMA)

**표 3: LLaMA 사전학습 손실 및 제로샷 성능**

| 모델 | RMSNorm 점수 | DyT 점수 | 변화 |
|------|-------------|---------|------|
| LLaMA 7B | 0.513 / 1.59 | 0.513 / 1.60 | - / ↑0.01 |
| LLaMA 13B | 0.529 / 1.53 | 0.529 / 1.54 | - / ↑0.01 |
| LLaMA 34B | 0.536 / 1.50 | 0.536 / 1.50 | - / - |
| LLaMA 70B | 0.549 / 1.45 | 0.549 / 1.45 | - / - |

LLaMA의 경우 RMSNorm을 DyT로 대체했습니다.

### 3.4 음성 및 DNA 서열 모델링

- **wav2vec 2.0 Base**: 1.95 (동일)
- **wav2vec 2.0 Large**: 1.92 → 1.91 (↓0.01)
- **HyenaDNA**: 85.2% (동일)
- **Caduceus**: 86.9% (동일)

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 손실 함수 지형(Loss Landscape) 관점

DyT는 정규화 계층의 핵심 메커니즘 중 하나인 **손실 함수의 예각성(Sharpness) 감소**를 통해 일반화 성능을 개선합니다.[2]

정규화 계층과 가중치 감소의 결합은 그래디언트 하강법이 **Edge of Stability (EoS) 체제**에 진입하도록 유도하고, 이 과정에서 손실 함수의 곡률을 지속적으로 감소시킵니다. DyT의 tanh 함수의 비선형적 극값 억압 특성도 유사한 효과를 제공합니다.[2]

$$\frac{d}{dt}\text{Sharpness}(t) \propto -\text{Gradient Contribution}$$

### 4.2 활성화값 범위 제어

DyT의 learnable $$\alpha$$ 매개변수는 학습 중 입력 활성화값의 표준편차의 역수($$1/\text{std}$$)와 추적 관계를 보입니다.[1]

**그림 8 분석**:
- **훈련 중**: $$\alpha$$와 $$1/\text{std}$$이 함께 변동
- **훈련 후**: 강한 양의 상관관계 ($$r > 0.8$$)
- **깊은 계층**: 더 큰 표준편차와 작은 $$\alpha$$ 값

이는 극값 억압이 깊은 계층에서 더 활발함을 시사합니다.

### 4.3 수렴 안정성

학습 손실 곡선 분석 (그림 5, 6):
- LN과 DyT의 수렴 궤적이 매우 유사
- ViT-B, ConvNeXt-B, LLaMA (모든 크기)에서 안정적 수렴
- 하이퍼파라미터 튜닝 거의 불필요

### 4.4 극값 처리 메커니즘

정규화 계층과 달리 DyT는:

1. **토큰별 정규화 불필요**: 전체 활성화에 대해 스칼라 $$\alpha$$로 작동
2. **비선형 극값 억압**: tanh 함수로 인한 자동 경계 제한
3. **채널별 미세 조정**: $$\gamma, \beta$$를 통한 보정

극값을 squashing하지 않으면 (identity 함수 사용 시) 훈련이 발산하며, 이는 극값 억압의 중요성을 입증합니다.[1]

***

## 5. 논문의 한계

### 5.1 기술적 한계

1. **배치 정규화와의 호환성**: 기존 ResNet 같은 모델에서는 성능 저하
   - ResNet-50: 76.2% (BN) → 68.9% (DyT)
   - VGG19: 72.7% (BN) → 71.0% (DyT)

2. **계산 효율성**: 최적화된 컴파일 후에는 정규화 계층과 성능 동등
   - 컴파일 전: DyT 52.4% 빠른 추론
   - torch.compile 후: 동등한 성능

3. **하이퍼파라미터 민감도**: LLM에서만 $$\alpha_0$$ 초기화에 민감
   - 비LLM: $$\alpha_0 = 0.5$$ 일반적 (안정적)
   - LLM: 모델 크기에 따라 다른 값 필요 (표 10)

### 5.2 응용 범위의 제한

- Layer Normalization/RMSNorm이 있는 Transformer에만 적용
- Batch Normalization이 주된 네트워크에서는 미검증
- 대규모 프로덕션 환경에서의 장기 영향 미지수

***

## 6. 다른 방법과의 비교

**표 9: ImageNet 분류 정확도 (비교 방법)**

| 방법 | ViT-B | ViT-L | MAE ViT-B | MAE ViT-L |
|------|-------|-------|-----------|-----------|
| LN | 82.3% | 83.1% | 83.2% | 85.5% |
| Fixup | 77.2% | 78.1% | 73.7% | 74.1% |
| SkipInit | 74.1% | 75.6% | 73.1% | 74.0% |
| σReparam | 82.5% | 83.0% | 83.2% | 85.4% |
| **DyT** | **82.8%** | **83.6%** | **83.7%** | **85.8%** |

DyT는 초기화 기반(Fixup, SkipInit) 및 가중치 정규화 기반(σReparam) 방법을 모두 능가합니다.[1]

***

## 7. 앞으로의 연구에 미치는 영향 및 고려 사항

### 7.1 이론적 영향

**정규화 계층의 역할 재정의**: 최근 연구(Peri-LN, 2025)에서 정규화 계층의 배치 위치가 훈련 동역학에 미치는 영향을 체계적으로 분석하고 있습니다. DyT는 이러한 이론적 이해를 보완하며, 정규화가 **선택지(option)**이지 **필수**가 아님을 시사합니다.[3]

**일반화 이론 강화**: 손실 함수의 예각성 감소(Sharpness Reduction) 관점의 일반화 이론이 강화될 것으로 예상됩니다. DyT의 비선형 극값 억압도 유사한 효과를 제공하므로, 정규화가 아닌 다른 메커니즘으로도 일반화를 개선할 수 있음을 입증합니다.[4][2]

### 7.2 실무적 응용 고려 사항

#### (1) 도메인 적응 및 특화

**최신 사례**: 자율주행 궤적 예측(DyTTP, 2025)에서 DyT를 적용하여 계산 오버헤드를 줄이면서 안정성을 유지하고 있습니다. 이는 리소스 제약이 있는 실시간 시스템에 DyT가 유용함을 보여줍니다.[5]

**시계열 Transformers**: 최근 논문(UnitNorm, 2024)에서 시계열 데이터에 최적화된 정규화 전략을 제안하고 있습니다. DyT도 유사하게 시계열 특화 버전으로 개발될 수 있습니다.[6]

#### (2) 하이퍼파라미터 튜닝 전략

**비LLM 모델**: 기본값 $$\alpha_0 = 0.5$$로 충분하며, 추가 튜닝은 선택사항입니다.[1]

**LLM 모델**: 모델 너비(width)가 결정 요소이며, 깊이(depth)는 영향 미미합니다.[1]
- 1024 width: $$\alpha_0 = 1.0$$
- 4096 width: $$\alpha_0 = 0.8/0.2$$ (Attention/Other)
- 8192 width: $$\alpha_0 = 0.2/0.05$$

#### (3) 레이어별 차등 초기화

LLaMA에서 Attention 블록과 FFN 블록에 다른 $$\alpha_0$$을 사용할 때 성능 향상:[1]

$$\alpha_0^{\text{attention}} > \alpha_0^{\text{FFN}}$$

이는 Attention의 activation range가 FFN과 다르기 때문으로 추측됩니다.

### 7.3 앞으로의 연구 방향

#### (1) 이론적 심화

- DyT가 정규화와 동등하지 않은 이유에 대한 수학적 분석
- 손실 함수 지형에 미치는 정확한 영향 정량화
- Edge of Stability와 DyT의 관계 규명

#### (2) 아키텍처 확장

- Batch Normalization 대체 DyT 개발
- 다른 활성화 함수(e.g., GELU, ReLU와의 조합) 연구
- 혼합 정확도(Mixed Precision) 훈련에서의 효과

#### (3) 효율성 최적화

- GPU/TPU 특화 컴파일 최적화
- 모바일/엣지 디바이스 배포 전략
- 동적 $$\alpha$$ 조정(training → inference)

#### (4) 도메인 특화

- 멀티모달 모델(Vision-Language)에서의 효과
- 시계열 예측 태스크 최적화
- 생물정보학(DNA, 단백질) 시퀀스 모델링

### 7.4 산업계 영향

**훈련 효율성**: DyT 도입으로 대규모 언어 모델의 메모리 효율성과 훈련 시간이 개선될 수 있습니다.

**추론 최적화**: 최적화 전 DyT가 정규화 계층 대비 52.4% 추론 속도 향상을 보이므로, 엣지 배포에 유리합니다.[1]

**아키텍처 단순화**: 정규화 계층이 선택지가 됨으로써 모델 아키텍처 설계의 유연성이 증가합니다.

***

## 8. 결론

**"Transformers without Normalization"** 논문은 정규화 계층이 현대 신경망의 필수 요소라는 오랜 통념에 도전합니다. Dynamic Tanh는 정규화 계층의 핵심 역할(극값 억압 + 활성화 범위 제어)을 수행하면서도, 훨씬 단순하고 통계량 계산이 불필요한 대안을 제시합니다.

특히 **일반화 성능 측면**에서 DyT는:
1. 손실 함수의 예각성 감소 메커니즘을 부분적으로 보존
2. 깊은 계층에서의 극값 자동 억압
3. 하이퍼파라미터 튜닝 최소화

를 통해 기존 정규화 계층과 동등하거나 우수한 성능을 달성합니다.[1]

앞으로의 연구는 **이론적 이해 심화**, **도메인 특화 개선**, **효율성 최적화**로 나뉠 것으로 예상되며, 이는 더욱 단순하고 효율적인 미래 신경망 아키텍처로의 전환을 촉발할 수 있습니다.[5][3]

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/91696d92-c571-43bb-afec-8240295223b9/2503.10622v2.pdf)
[2](https://openreview.net/forum?id=xp5VOBxTxZ)
[3](https://proceedings.mlr.press/v267/kim25u.html)
[4](http://arxiv.org/pdf/2412.13573.pdf)
[5](http://arxiv.org/pdf/2504.05356.pdf)
[6](https://arxiv.org/pdf/2405.15903.pdf)
[7](https://arxiv.org/html/2503.10622)
[8](https://arxiv.org/pdf/2305.02790.pdf)
[9](https://arxiv.org/pdf/2112.02624.pdf)
[10](https://arxiv.org/pdf/2502.00585.pdf)
[11](http://arxiv.org/pdf/2502.09503.pdf)
[12](https://arxiv.org/html/2502.02732v1)
[13](https://arxiv.org/abs/2503.10622)
[14](https://www.sciencedirect.com/science/article/abs/pii/S0926580524004515)
[15](https://cvpr.thecvf.com/virtual/2025/poster/32739)
[16](https://arxiv.org/abs/2206.07085)
[17](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aidi.202400035)
[18](https://arxiv.org/html/2503.10622v1)
[19](https://www.machinelearningmastery.com/using-normalization-layers-to-improve-deep-learning-models/)
[20](https://journaleit.org/wp-content/uploads/8_July_2025.pdf)
[21](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Transformers_without_Normalization_CVPR_2025_paper.pdf)
[22](https://arxiv.org/pdf/2209.08473.pdf)
[23](http://arxiv.org/pdf/2312.13555.pdf)
[24](https://arxiv.org/pdf/1712.09913.pdf)
[25](https://arxiv.org/pdf/2412.10146.pdf)
[26](https://arxiv.org/html/2403.00567)
[27](https://arxiv.org/pdf/2207.01847.pdf)
[28](https://arxiv.org/html/2412.13321v1)
[29](https://arxiv.org/html/2412.10146v1)
[30](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
[31](https://arxiv.org/html/2509.01842v3)
[32](https://dl.acm.org/doi/10.5555/3327345.3327535)
[33](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)
[34](https://arxiv.org/abs/2412.10146)
[35](https://arxiv.org/html/2501.19399v1)
[36](https://clova.ai/en/tech-blog/stable-training-preventing-divergence-with-peri-ln)
[37](https://proceedings.neurips.cc/paper_files/paper/2024/file/52f050499cf82fa8efb588e263f6f3a7-Paper-Conference.pdf)
