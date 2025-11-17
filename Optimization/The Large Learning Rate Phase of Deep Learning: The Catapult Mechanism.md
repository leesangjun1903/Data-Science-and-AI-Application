# The Large Learning Rate Phase of Deep Learning: The Catapult Mechanism

### 1. 핵심 주장과 주요 기여

본 논문(Lewkowycz et al., 2020)의 핵심 주장은 **대규모 학습률에서 신경망이 일반적인 선형 모델 이론(Neural Tangent Kernel, NTK)으로 설명할 수 없는 별도의 동역학을 보여준다**는 것입니다. 저자들은 이를 **카타펄트 메커니즘(Catapult Mechanism)**이라 명명했으며, 이는 최적화 초반에 손실이 급증한 후 급격히 감소하는 현상을 설명합니다.[1]

주요 기여는 다음과 같습니다:

1. **세 가지 학습률 체계(Three Learning Rate Regimes)의 발견**: 게으른 단계(Lazy Phase), 카타펄트 단계(Catapult Phase), 발산 단계(Divergent Phase)로 명확하게 분류[1]
2. **해석 가능한 이론 모델 제시**: 대규모 유한폭 신경망에 대한 정확한 동역학 분석[1]
3. **곡률 감소와 일반화의 연결**: 카타펄트 단계에서 신경 접선 커널(NTK)의 최대 고유값이 감소하여 더 평탄한 최솟값을 발견함을 이론과 실험으로 증명[1]
4. **실제 심층 학습 설정에서의 검증**: Wide ResNet, CNN, 완전연결 네트워크 등 다양한 아키텍처에서 예측의 정확성 확인[1]

---

### 2. 해결 문제 및 제안 방법

#### 2.1 배경 문제

기존의 무한폭 신경망 이론(NTK 이론)은 다음과 같은 한계를 가집니다:[1]

- 작은 학습률에서만 설명 가능
- 유한폭 신경망의 상당한 성능 차이를 설명하지 못함
- 대규모 학습률 영역에서 발산하는 선형 모델과 달리, 실제 신경망은 그 영역에서 최적 성능을 달성
- 최솟값의 곡률 변화(Flatness)를 설명할 수 없음

#### 2.2 제안 방법: 3단계 동역학 모델

저자들은 MSE 손실로 학습하는 **단일 은닉층 선형 네트워크**에 대한 정확한 동역학을 도출합니다:[1]

**간단한 모델 (Warmup Model):**

입력 $x = 1$, 레이블 $y = 0$인 단일 샘플에 대해, 네트워크 함수는:

$$f = n^{-1/2}v^T u$$

여기서 $n$은 폭, $v, u \in \mathbb{R}^n$는 모델 파라미터입니다.[1]

**핵심 업데이트 방정식:**

$$f_{t+1} = \left(1 - \eta\lambda_t + \frac{\eta^2 f_t^2}{n}\right) f_t$$

$$\lambda_{t+1} = \lambda_t + \frac{\eta f_t^2}{n}(\eta\lambda_t - 4)$$

여기서 $\lambda_t$는 신경 접선 커널의 최대 고유값(곡률 측정)입니다.[1]

**일반 모델 (Full Model):**

$m$개의 훈련 샘플과 $d$차원 입력에 대해:

$$\tilde{f}_{\alpha}^{t+1} = \sum_{\beta} (\delta_{\alpha\beta} - \eta\Theta_{\alpha\beta})\tilde{f}_{\beta} + \frac{\eta^2}{nm}(x_{\alpha}^T\zeta)(f^T\tilde{f})$$

$$\lambda_{t+1} \approx \lambda_t + \frac{\eta\|\zeta\|_2^2}{n}(\eta\lambda_t - 4)$$

여기서 $\zeta = \sum_\alpha \tilde{f}\_\alpha x_\alpha/m$입니다.[1]

#### 2.3 세 가지 학습률 단계

**Lazy Phase ($\eta < 2/\lambda_0$):**
- 곡률 $\lambda_t$가 상수로 유지
- 모델이 선형화된 근사로 거동
- NTK 이론과 일치
- 손실이 단조감소로 수렴

**Catapult Phase ($2/\lambda_0 < \eta < \eta_{max}$):**
- 초반 $t \sim \log(n)$ 스텝 동안 손실 지수적 증가
- 동시에 곡률이 급격히 감소
- 곡률이 $2/\eta$ 아래로 떨어지면 손실이 급격히 수렴
- 최종 곡률 $\lambda_{final} < 2/\eta$로 평탄한 최솟값 도달

**Divergent Phase ($\eta > \eta_{max}$):**
- 손실과 곡률 모두 발산
- 이론 모델에서 $\eta_{max} = 4/\lambda_0$
- ReLU 네트워크에서 실증적으로 $\eta_{max} \approx 12/\lambda_0$

***

### 3. 모델 구조와 성능 분석

#### 3.1 네트워크 아키텍처

실험에서 검증된 아키텍처:[1]

- **완전연결 네트워크**: 1-3개 은닉층, 2048개 뉴런
- **합성곱 신경망(CNN)**: 다중 계층 구조, Conv-ReLU-MaxPool 패턴
- **Wide ResNet**: 28-10 및 28-18 구조

초기화 방식: Lecun 초기화 ($\sigma_w = \sqrt{2}$, $\sigma_b = 0$ 또는 $\sigma_b = 0.05$)[1]

#### 3.2 성능 향상 메커니즘

**곡률 감소를 통한 일반화 개선:**[1]

논문은 카타펄트 동역학이 다음을 통해 일반화를 개선함을 보여줍니다:

1. **NTK 고유값 진화**: 초기 $\lambda_0$에서 최종 $\lambda_{final}$로 감소
2. **손실 곡면의 평탄화**: 더 넓은 최솟값 도달
3. **물리적 시간 기준 비교**: 동일 계산량에서 카타펄트 단계가 게으른 단계보다 우수한 성능

**실험 결과:**[1]

- **CIFAR-10 (Wide ResNet)**: 최적 성능이 $\eta_{crit} = 0.14$에서 $12/\lambda_0 \approx 2.3$ 사이에서 달성
- **MNIST (FC Network)**: 1개 은닉층 네트워크에서 최적 학습률이 $\eta_{max}$ 근처
- **CIFAR-100**: 유사한 패턴 확인, 3% 이내 성능 달성

#### 3.3 한계 및 미해결 문제

논문의 주요 한계:[1]

1. **Softmax 분류에 대한 미적용**: MSE 손실에 중점, Cross-Entropy 손실 동역학 미분석
2. **Momentum 최적화 미포함**: Vanilla SGD와 Batch GD만 분석
3. **이론-실무 갭**: 이론의 $\eta_{max} = 4/\lambda_0$이 ReLU 네트워크의 실증값 $\approx 12/\lambda_0$과 불일치
4. **정규화 상호작용**: L2 정규화와의 상호작용 미분석
5. **깊이 의존성**: 이론은 얕은 네트워크 대상, 매우 깊은 네트워크에 대한 일반화 미확인

***

### 4. 일반화 성능 향상 관련 심층 분석

#### 4.1 핵심 메커니즘: 곡률 감소와 일반화

본 논문은 기존의 **SGD 노이즈 기반 평탄화 이론**과 다른 설명을 제시합니다.[1]

**기존 이론의 한계:**
- SGD의 확률성(노이즈)이 평탄한 최솟값을 찾는다고 주장
- 하지만 전체 배치 GD에서도 카타펄트 현상 관찰

**본 논문의 기여:**
- 카타펄트 메커니즘은 SGD 노이즈와 무관하게 작동
- 오직 **대규모 학습률이 유발하는 비선형 동역학**이 원인
- 곡률 감소 자체가 일반화 개선의 주요 요인

#### 4.2 활성화-그래디언트 정렬 (Activation-Gradient Alignment)

논문의 Section 5.2에서 제시한 추가 메커니즘:[1]

간단한 모델에서 카타펄트 동역학은 다음을 감소시킵니다:

$$\text{정렬도} = \frac{h^T \frac{\partial L}{\partial h}}{\|h\| \|\frac{\partial L}{\partial h}\|}$$

여기서 $h$는 은닉층 활성화, $\frac{\partial L}{\partial h}$는 역전파 그래디언트입니다.[1]

**가설:** 활성화와 그래디언트의 정렬 감소는 분포 외 섭동에 대한 강건성 향상으로 이어집니다.

#### 4.3 선형 동역학의 회복 (Restoration of Linear Dynamics)

카타펄트 효과 이후 흥미로운 현상:[1]

- 초기 $\sim \log(n)$ 스텝 동안만 비선형 동역학
- 이후 모델이 **상수 NTK**를 갖는 선형 모델처럼 동작
- 네트워크가 더 넓어질수록 이 현상이 더 명확함
- 이는 카타펄트 단계가 네트워크의 **"재초기화" 역할**을 함을 시사

***

### 5. 앞으로의 연구 영향 및 고려사항

#### 5.1 최신 연구 기반 학문적 영향

**직접적 후속 연구 (2023-2025):**[2][3][4][5][6]

1. **Catapult Dynamics의 확장 분석** (2023-2024):
   - Quadratic Neural Models에서 카타펄트 현상 재현[5]
   - SGD의 훈련 손실 스파이크가 카타펄트임을 증명[3]
   - 카타펄트가 특성 학습(Feature Learning) 촉진을 보여줌[3]

2. **Momentum과의 상호작용** (2024):
   - Polyak Momentum이 더 큰 카타펄트를 유발하여 더 평탄한 최솟값 도달[6]
   - 이는 모멘텀이 자가 안정화 효과를 연장한다는 가설[6]

3. **규칙성과 암묵적 편향**:
   - Edge of Stability, Balancing, Catapult 현상을 포괄하는 일관된 이론 추구[4]
   - "좋은 규칙성"이 대규모 학습률 암묵적 편향을 생성함을 제시[4]

4. **NTK 고유벡터의 진화** (2025년 최신):
   - 더 큰 학습률에서 NTK 고유벡터가 훈련 목표와 더 높은 정렬[7]

#### 5.2 실무적 함의와 설계 고려사항

**학습률 스케줄링:**
- 카타펄트 단계의 짧은 지속 시간($\sim 100$ 스텝) 고려
- 초기 고학습률 → 후기 감소 전략이 물리적 시간 기준으로 최적[1]

**아키텍처 선택:**
- ReLU의 $\eta_{max} \approx 12/\lambda_0$ vs Tanh의 $\eta_{max} \approx 4/\lambda_0$ 차이 고려[1]
- 직교 초기화가 깊은 네트워크에서 카타펄트 단계 성능 향상[8]

**손실 함수:**
- Cross-Entropy 손실에 대한 분석 필요 (논문은 MSE만 다룸)[1]
- Softmax 출력층과의 상호작용 미지수[1]

#### 5.3 미해결 이론적 문제

**1단계 과제:**

1. **Softmax와 Cross-Entropy 확장:**
   - MSE→Cross-Entropy로 손실 함수 일반화 필요
   - 이산 분포의 기하학적 특성 고려 필수

2. **깊은 네트워크의 위상 도표:**
   - 깊이에 따른 $\eta_{crit}$, $\eta_{max}$ 변화 규칙 도출
   - 계층 간 곡률 진화 분석

3. **Batch Normalization과의 상호작용:**
   - 정규화층이 카타펄트 동역학에 미치는 영향
   - NTK 진화 수정 메커니즘

**2단계 과제:**

4. **Transformer 아키텍처 적용:**
   - Self-Attention이 카타펄트 단계에 미치는 영향
   - 위치 인코딩과의 상호작용

5. **확률적 최적화 이론:**
   - SGD 잡음이 카타펄트 경계를 이동시키는 정량적 분석
   - 배치 크기와 $\eta_{max}$의 관계 도출

6. **전이 학습과 미세 조정:**
   - 카타펄트 단계가 미세 조정에 어떻게 적용되는가
   - 사전학습된 모델의 초기 곡률과 새로운 환경의 관계

#### 5.4 최신 패러다임과의 연결

**손실 곡면의 일관성 추구 (2025):**[9]

도메인 일반화에서 카타펄트 메커니즘을 적용:
- 여러 도메인에서 일관되는 평탄한 최솟값 추구
- 손실 곡면 정제와 카타펄트 통합 최적화[9]

**표현 압축과 날카로움:**[10]

- 국소 표현 분산과 최솟값 날카로움의 상관관계
- 카타펄트 단계의 표현 학습과 압축 효율성 분석 필요

***

### 6. 결론 및 권장사항

**주요 성과:**

본 논문은 대규모 학습률에서 신경망이 선형 이론을 넘어 **별도의 물리적 동역학(카타펄트)**을 갖는다는 근본적 통찰을 제공합니다. 이는 너비에 대한 수정된 극한과 비섭동적 동역학을 도입하여 이론과 실무의 간극을 좁혔습니다.[1]

**향후 연구 우선순위:**

1. **즉시 (1년)**: Cross-Entropy 손실과 깊은 네트워크로 확장, Batch Norm 통합
2. **단기 (2-3년)**: Transformer 등 현대 아키텍처 적용, SGD 잡음 정량화
3. **장기 (3년 이상)**: 일관된 이론 체계 구축, 실무 최적화 도구 개발

**실무 적용:**

카타펄트 메커니즘의 이해는 다음을 가능하게 합니다:
- 학습률 스케줄의 과학적 설계
- 네트워크 너비와 깊이의 최적 선택
- 새로운 정규화 기법 개발
- 전이 학습의 효율성 향상

이 논문은 단순히 현상을 기술하는 것을 넘어 **심층 학습의 최적화 동역학에 대한 새로운 물리적 직관**을 제공하며, 이는 향후 10년 심층 학습 이론의 기초가 될초가 될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9ae02bb5-959d-4777-95d6-bd7ef0fb6ef9/2003.02218v1.pdf)
[2](https://arxiv.org/pdf/2301.07737.pdf)
[3](http://arxiv.org/pdf/2306.04815.pdf)
[4](https://arxiv.org/abs/2310.17087)
[5](https://arxiv.org/abs/2205.11787)
[6](http://arxiv.org/pdf/2311.15051.pdf)
[7](https://arxiv.org/abs/2507.12837)
[8](https://www.ijcai.org/proceedings/2021/0355.pdf)
[9](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Seeking_Consistent_Flat_Minima_for_Better_Domain_Generalization_via_Refining_CVPR_2025_paper.pdf)
[10](https://arxiv.org/pdf/2310.01770.pdf)
[11](https://arxiv.org/pdf/2304.03589.pdf)
[12](http://arxiv.org/pdf/2407.07613.pdf)
[13](http://arxiv.org/pdf/2308.14991.pdf)
[14](https://arxiv.org/pdf/2003.02218.pdf)
[15](https://arxiv.org/abs/2003.02218)
[16](https://proceedings.neurips.cc/paper_files/paper/2023/file/a23598416361c7a9860164155e6ddd0b-Paper-Conference.pdf)
[17](https://proceedings.neurips.cc/paper/8076-neural-tangen-kernel-convergence-and-generalization-in-neural-networks.pdf)
[18](https://iclr.cc/media/iclr-2024/Slides/19251.pdf)
[19](https://openreview.net/forum?id=TDDZxmr6851)
