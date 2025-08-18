# A Survey of Quantization Methods for Efficient Neural Network Inference 

## 1. 핵심 주장 및 주요 기여  
본 논문은 **신경망 추론의 효율성을 높이기 위한 양자화(quantization) 기법**을 폭넓게 정리한 종설 논문이다.  
- 양자화의 역사적 배경 및 기본 개념(균등·비균등, 대칭·비대칭, 정적·동적, 채널·그룹·레이어 단위, “fake” vs. integer-only 등)을 체계적으로 소개  
- 양자화 이후 손실된 정확도를 보완하기 위한 학습 절차(Quantization-Aware Training, Post-Training Quantization, Zero-Shot Quantization) 및 **STE**(Straight-Through Estimator) 활용 방법을 상세히 설명  
- 8비트 이하의 저비트 양자화(4비트, 2비트, 바이너리) 기법과 하드웨어 가속기(Edge TPU, GPU INT8/INT4 유닛, ARM Cortex-M, RISC-V 등)와의 관계를 분석  
- 혼합 정밀도(mixed-precision) 양자화, 하드웨어 인식(HAWQ) 자동 탐색, 지식 증류 기반 성능 보강 등 **최신 연구 동향**을 정리  
- **엣지 디바이스** 환경에서 전력·메모리·대역폭 제약을 만족하며 추론 효율을 극대화하는 양자화 설계를 조망  

## 2. 문제 정의, 제안 기법, 모델 구조, 성능 및 한계  
### 2.1 해결하고자 하는 문제  
- 과도한 메모리·연산량을 요구하는 **과매개화(over-parameterized)** 신경망 모델을  
- 엣지/모바일·데이터센터의 전력·대기시간(SLA)·메모리 제약 하에서도  
- **정확도 손실을 최소화**하면서 더 낮은 비트 정밀도로 표현하고 연산을 수행하기  

### 2.2 제안하는 주요 방법  
1. **Uniform Quantization**  

$$ Q(r) = \mathrm{Int}\bigl(\tfrac{r}{S}\bigr) - Z, \quad S=\tfrac{\beta-\alpha}{2^b-1} $$  
   
   - S: 스케일 팩터, Z: 제로포인트, $$[α,β]$$는 클리핑 범위  
   - *Symmetric*: $$Z=0,\;S=(2\max|r|)/(2^b{-}1)$$  
   - *Asymmetric*: $$S=(β-α)/(2^b{-}1),\;Z=\mathrm{round}(-α/S)$$  

2. **Non-uniform Quantization**  

$$ Q(r)=X_i,\text{if }r\in[Δ_i,Δ_{i+1}),\quad\min_{Q}\|Q(r)-r\|^2 $$  
   
   - 로그 스케일, 벡터/이진 근사, 군집화(k-means) 기반 학습형(learnable) 양자화  
3. **Quantization-Aware Training (QAT)**  
   - 순전파/역전파에서 양자화 연산 삽입  
   - 비분화점(round) 연산의 기울기를 **STE**로 근사하여 전파  
4. **Post-Training Quantization (PTQ)**  
   - 재학습 없이 가중치·활성화 클리핑 범위만 소량의 교정 데이터로 결정  
   - 채널별 편향 보정, OCS(outlier channel splitting), AdaRound 적응형 반올림 기법  
5. **Zero-Shot Quantization**  
   - **원본 데이터 접근 불필요**: BatchNorm 통계나 GAN/역전파로 생성한 합성 데이터로 캘리브레이션·미세튜닝  
6. **Integer-Only Quantization**  
   - 파라미터·연산 모두 정수 연산으로만 수행  
   - Dyadic 스케일(분모 2^k) 기반 비트 시프트만으로 스케일링  
7. **Mixed-Precision Quantization & Hardware-Aware**  
   - 레이어별 민감도(Hessian trace) 측정→비트폭 자동 배정(HAWQ)  
   - 실제 GPU(T4)·ASIC 지연 측정 기반 ILP 최적화  
8. **Extreme Low-Bit (≤4bit, 2bit, Binary/Ternary)**  
   - $$W\approx\sum_{i=1}^mα_i b_i,b_i∈\{±1\}^n$$ (ABC-Net, HORQ)  
   - 손실기반 최적화(Loss-Aware Binarization), 지식 증류, 게이트/잔차 구조 개선  

### 2.3 모델 구조 및 성능 향상  
- **INT8**: ResNet-50 3.9×, Inception-V3 5.0× 속도 향상(1080Ti)  
- **INT4**: ResNet-50 추가 50–60% 속도 개선(T4)  
- **Mixed INT4/8**: ResNet-50 23% 추가 가속(T4) 무정밀도 손실  
- **Binary Networks**: BNN, XNOR-Net 등 최대 8× 추론량 절감, 게이트형/스케일 결합으로 정확도 회복  
- **Zero-Shot PTQ**: ZeroQ 1% 이내 정확도 손실로 8비트 양자화  

### 2.4 한계  
- **초저비트(≤4bit) 양자화**: 과도한 정확도 손실 회복 위해 복잡한 미세튜닝·지식 증류·군집화 필요  
- **재학습 비용**: QAT 수백 에폭 요구, PTQ는 정확도 한계  
- **소프트웨어 지원 미비**: INT4 이하 추론 라이브러리·전용 가속기 부족  
- **데이터·하드웨어 종속성**: 캘리브레이션·성능 향상 기법이 모델·데이터셋·아키텍처별로 튜닝 요구  

## 3. 모델 일반화 성능 향상 관점  
- **과매개화의 이점**: 실수 가중치→양자화된 가중치 간 큰 편차에도 **일반화 성능**(분류 정확도, 언어 모델 perplexity)은 거의 유지 가능  
- **Mixed-Precision**: 민감한 레이어(Sharp Hessian)에는 고정밀, 둔감한 레이어(Flat Hessian)에는 저정밀 부여→일반화·성능 균형 최적화  
- **Regularization-like 효과**: STE 및 stochastic rounding 도입 시 가중치/활성화의 불확실성 주입이 **과적합 억제**에 기여  
- **Knowledge Distillation**: 양자화로 인한 정보 손실을 “soft label”로 보완하여 일반화 성능 강화  

## 4. 향후 연구 영향 및 고려 사항  
### 4.1 기대 효과  
- **엣지·IoT AI 보급 가속**: 전력·메모리 제약 환경에서 복잡 모델 실시간 추론 실현  
- **저비트 연산기 설계**: ASIC·FPGA·GPU TensorCore 차세대 유닛의 설계 방향 제시  
- **자동화 도구 발전**: Mixed-precision·하드웨어 인식 자동 탐색 및 소프트웨어 스택 정비  

### 4.2 연구 시 고려할 점  
- **재현성·표준화**: 다양한 데이터·하드웨어·아키텍처에서 일관된 벤치마크·라이브러리 필요  
- **초저비트 한계 돌파**: 재학습·증류 없이도 2–4비트 수준에서 안정적 일반화 성능 확보  
- **통합 압축·코-디자인**: 양자화·가지치기·지식증류·아키텍처 검색·하드웨어 설계를 **공동 최적화**하는 프레임워크  
- **양자화 학습**: 8비트 이하 훈련 안정화 기법(수치 불안정성 해소)  

> 위 구조화된 개요를 통해, 본 종설 논문이 제시하는 양자화의 전반적 개념·기법과 이들의 실제 하드웨어·모델 일반화 성능에 대한 함의를 폭넓게 이해할 수 있다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b5edda4d-6105-4d1a-a89d-8820023fe830/2103.13630v3.pdf

# III. BASIC CONCEPTS OF QUANTIZATION

신경망 양자화(Quantization)는 연산 효율 및 메모리 절감을 위해 **실수 값을 저정밀 정수 값**으로 변환하는 과정입니다. 이 절에서는 양자화의 기초 개념과 주요 기법을 목차별로 자세히 살펴봅니다.

***

## III-A. 문제 설정 및 표기법

- 학습된 신경망 파라미터는 θ = {W₁, …, W_L}로 표현되며, N개 샘플 (xᵢ, yᵢ)에 대한 손실  

$$L(θ)=\frac1N\sum_{i=1}^N\ell(xᵢ,yᵢ;θ)$$를 최소화합니다.  

- 양자화 목표:  
  1. **파라미터 θ**와  
  2. **중간 활성화 hᵢ, aᵢ**  
  를 b비트 이하 정수로 표현해 **추론 정확도**(일반화 성능) 손실을 최소화.

***

## III-B. 균등 양자화 (Uniform Quantization)

실수 r을 일정 간격 S로 나누고, 정수 오프셋 Z를 더해 b비트 범위의 정수 Q(r)로 매핑:

$$
Q(r) = \mathrm{round}\bigl(\tfrac{r}{S}\bigr)-Z
$$

- $S =(β−α)/(2ᵇ−1)$ : [α, β] 구간을 2ᵇ−1개로 균일 분할  
- Z : zero-point (비대칭 클리핑 시 오프셋)  
- **복원** $$\tilde r=S\bigl(Q(r)+Z\bigr)$$ (반올림 오차 존재)

***

## III-C. 대칭 vs. 비대칭 양자화

### 비대칭 양자화
- $α=r_\min, β=r_max → S=(β−α)/(2ᵇ−1), Z=round(−α/S)$ 
- 데이터 분포가 한쪽으로 치우칠 때 해상도 유지에 유리  

### 대칭 양자화
- α=−β → Z=0 → 구현 단순화  
- $S=2·max|r|/(2ᵇ−1)$ (full range) 또는 max|r|/(2ᵇ−1−1) (restricted range)  

> **요약**:  
> - *대칭* = 제로 포인트 0, 구현 용이  
> - *비대칭* = 데이터 편향 대응, 해상도 우수

***

## III-D. 캘리브레이션: 정적 vs. 동적 양자화

- **정적(Static) 양자화**  
  - 추론 전 미리 샘플 데이터로 [α, β] 고정  
  - 연산 오버헤드 無, 정확도 소폭 저하  
- **동적(Dynamic) 양자화**  
  - 입력별 실시간 [min, max] 계산  
  - 입력 변화 즉시 반영, 구현·연산 부하 높음

***

## III-E. 양자화 그레인(Granularity)

파라미터 블록별 별도 스케일링으로 해상도 확보:

1. **Layerwise**: 레이어 전체를 하나의 [α, β]로 양자화  
2. **Groupwise**: 채널 그룹 단위로 분할  
3. **Channelwise**: 채널(필터)별 개별 [α, β]  
4. **Sub-channelwise**: 커널 내부 서브그룹  

> **표준**: 채널별 양자화(channelwise)가 구현 오버헤드 최소·정확도 최대.

***

## III-F. 비균등 양자화 (Non-Uniform Quantization)

- 구간 경계 Δᵢ 및 레벨 Xᵢ 비균일 분포  
- 룰 기반: 로그 스케일, 벡터/이진 분해  
- 최적화 기반: $$\min_Q\|Q(r)-r\|^2$$  
- 학습형(learnable): Δᵢ, Xᵢ를 역전파로 조정  
- 군집화: k-means로 텐서 클러스터링 후 센트로이드 사용  

> **장점**: 분포 밀집 구간 해상도↑  
> **단점**: 하드웨어 매핑·속도 저하

***

## III-G. 미세조정(Fine-tuning)

### 1. Quantization-Aware Training (QAT)
- 순·역전파에 양자화 삽입  
- **STE**(Straight-Through Estimator)로 비분화점 연산 기울기 근사  
- 학습 과정 중 b비트 가중치로 주기적 투영  

### 2. Post-Training Quantization (PTQ)
- 재학습 없이 캘리브레이션만 수행  
- *Bias correction*, *range equalization*, *AdaRound* 등으로 정확도 보완  

### 3. Zero-Shot Quantization (ZSQ)
- **원본 데이터 無**: BatchNorm 통계나 GAN·역전파 기반 합성 데이터로 캘리브레이션 및 소량 미세튜닝

***

## III-H. 확률적(Stochastic) 양자화

- 실수→정수 반올림을 확률적 방식으로 수행:  
  $$\mathrm{round}(x)$$ 대신  
  $$\lfloor x\rfloor$$ 또는 $$\lceil x\rceil$$ 선택 확률 ∝ 거리  
- **QuantNoise**: 매 순전파마다 가중치 랜덤 부분양자화  
- *탐색성 향상* 효과, 추가 난수 생성 오버헤드 존재

***

이상으로 양자화의 범위 설정에서 그레인별 세분화, 학습·비학습 미세조정, 그리고 균등·비균등·확률적 기법까지 전반을 정리했습니다. 이러한 기초 위에서, 이후 장에서 다루는 **저비트·혼합정밀도·하드웨어 연계** 기법이 구체화됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b5edda4d-6105-4d1a-a89d-8820023fe830/2103.13630v3.pdf

# IV. ADVANCED CONCEPTS: QUANTIZATION BELOW 8 BITS

이 장에서는 **8비트 이하(b < 8)**로 더 낮은 정밀도를 쓰기 위한 양자화의 고급 기법을 단계별로 정리합니다.

***

## IV-A. 가짜 양자화(Fake/Simulated) vs. 정수 전용(Integer-Only) 양자화

### 1. 가짜(Fake) 양자화

- 모델 파라미터·활성화값을 정수로 저장하지만
- 연산(곱셈·덧셈)은 **FP32** 부동소수점으로 수행
- 실행 중 `dequantize → FP32 연산 → requantize` 단계 필요
- 메모리 절감 효과 있으나, 저정밀 하드웨어 가속기 활용 불가


### 2. 정수 전용(Integer-Only) 양자화

- 모든 연산을 **INT8/INT4/…** 정수로만 수행
- `int32` 누산(accumulation)과 시프트(스케일링)만 사용 → **FP 연산 완전 제거**
- GPU Tensor Core·ASIC·엣지 프로세서의 정수 유닛 활용 가능
- 다만 양자화 수식 내 나눗셈 제거를 위해 **dyadic scale**(분모 2ᵏ) 적용

***

## IV-B. 혼합 정밀도(Mixed-Precision) 양자화

- 모델의 각 레이어마다 최적 비트폭 b_l을 다르게 설정
- **정밀도-성능 민감도**(Hessian 기반 trace, 지연 측정 등) 및 **하드웨어 효율**(지연·전력) 고려
- 탐색 방법

1. **RL/DNAS**(Differentiable NAS) 기반 정책 탐색
2. **HAWQ**: 레이어별 Hessian trace로 민감도 계량 → ILP(정수 선형계획) 최적화
3. 주기 함수 정규화로 비트폭 학습

> **효과**: INT8 대비 INT4/8 혼합 시 최대 50% 추가 가속(정확도 손실 無)

***

## IV-C. 하드웨어 인식(Hardware-Aware) 양자화

- 동일 양자화가 GPU·CPU·ASIC마다 이득이 다름
- HW-Agnostic 방식: 레이어별 추정 지연표 LUT로 탐색(RL)
- HW-Aware 방식: 실측 지연·전력 측정 결과로 혼합 정밀도 비트폭 최적화
- FPGAs·TPU·GPU별 캘리브레이션 필요

***

## IV-D. 지식 증류(Distillation) 활용 양자화

- **Teacher–Student** 구조: 대형 고정밀 모델(teacher)의 소프트 타깃(soft logits)을 student에 전이
- 학습 손실:

```math
L = \alpha\,H\bigl(y,\sigma(z_s)\bigr)\;+\;\beta\,H\bigl(\sigma(z_t/T),\,\sigma(z_s/T)\bigr)
```

  - y: 실제 레이블, z_t/z_s: teacher/student 로짓
  - T: softmax 온도, α/β: 가중치
- 중간 특성 매칭, self-distillation 기법도 병용 가능

***

## IV-E. 극단 저비트(Extreme Low-Bit) 양자화

### 1. 바이너리(1-bit) 네트워크

- BinaryConnect, BNN, XNOR-Net: +1/−1만 사용
- 곱셈→XNOR + 비트카운트로 변환 → **8× 연산량 절감**
- SCALE 인자 α를 곱해 근사 오차 ↓


### 2. 토너리(2-bit) 네트워크

- 값 범위 \{−1, 0, +1\}
- Zero 처리로 희소성↑, 연산 간소화


### 3. 정확도 회복 기법

- **Error-Minimization**: ABC-Net, HORQ 등 다수 이진 근사 조합
- **Loss-Aware**: 양자화 직접 손실 최소화
- **개선된 학습**: STE 대체 연속 근사, 스무딩 함수, 잔차 \& 게이트 구조 추가

> **주의**: 극단 저비트는 대개 상당한 미세조정·튜닝 필요

***

## IV-F. 벡터 양자화(Vector Quantization)

- DSP 벡터 양자화 기법 차용
- 텐서 내 가중치를 k-means 클러스터링 → 각 클러스터 중심(c_j) 코드북 사용
- W ≈ c₁[n₁] + c₂[n₂] + … 형태로 근사
- **Product Quantization**: 서브벡터별 재귀 클러스터링
- 모델 사이즈 8× 축소, 허프만 코딩 병행시 추가 압축 가능

***

이상으로 **8비트 이하** 저정밀 양자화의 주요 개념과 기법을 정리했습니다. 다음 장에서는 이들을 엣지/클라우드 하드웨어에 적용하는 실무 노하우 및 최신 연구 동향을 살펴봅니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b5edda4d-6105-4d1a-a89d-8820023fe830/2103.13630v3.pdf
