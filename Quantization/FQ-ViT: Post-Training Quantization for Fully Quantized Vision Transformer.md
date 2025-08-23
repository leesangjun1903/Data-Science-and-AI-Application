# FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer

## 1. 핵심 주장 및 주요 기여  
FQ-ViT는 **사후(PTQ) 완전 정수 양자화된 비전 트랜스포머**를 구현하여, LayerNorm과 Softmax를 포함한 모든 모듈을 8-bit(가중치·활성화) 및 4-bit(어텐션 맵)로 양자화하면서도, 원본 FP32 대비 **±1% 이내의 손실 없는 정확도**를 달성한다.  
1. LayerNorm 입력의 심각한 채널 간 분포 편차를 해결하는 **Power-of-Two Factor (PTF)** 제안  
2. 어텐션 맵의 극도로 비균일한 분포를 효과적으로 유지하는 **Log-Int-Softmax (LIS)** 제안  
3. 두 기법 결합으로 완전 정수 연산만으로 비전 트랜스포머 추론 가능  

## 2. 문제 정의  
- 비전 트랜스포머는 CNN보다 파라미터·FLOPs가 많아 경량 하드웨어로의 배포가 어려움.  
- 사후 양자화 시, 기존 MinMax나 그룹·채널별 스케일링만으로는 LayerNorm과 Softmax 양자화 시 **심각한 성능 저하** 발생.  
- LayerNorm 입력의 채널별 값 범위는 최대 600× 이상 차이, Softmax 어텐션 맵 값은 대부분 0∼0.01에 몰려 있음.  

## 3. 제안 방법  
### 3.1 Power-of-Two Factor (PTF) for LayerNorm  
- 입력 $$X\in\mathbb{R}^{B\times L\times C}$$에 대해, 전체 레이어 스케일 $$s$$, 제로포인트 $$z_p$$와 각 채널별 PTF $$\alpha_c\in\{0,\dots,K\}$$를 도입:  

$$
    X_Q = \mathrm{clip}\bigl(\lfloor X / (2^{-\alpha_c} s)\rceil + z_p,\;0,\;2^b-1\bigr)
  $$  

- $$\alpha_c$$는 각 채널별로 양자화 오류를 최소화하도록 선택  
- 추론 시 비트시프트 연산만으로 $$\alpha_c$$ 반영, 정수 도메인에서 평균·분산 계산 가능  
- 하드웨어 오버헤드 없이도 채널별 스케일링 효과 달성  

### 3.2 Log-Int-Softmax (LIS) for Softmax  
- Softmax 어텐션 맵 $$\mathrm{Attn}\in(0,1)$$를 4-bit log2 양자화:  

$$
    \mathrm{Attn}_Q = \mathrm{clip}\bigl(\lfloor -\log_2(\mathrm{Attn})\rceil,\;0,\;2^b-1\bigr)
  $$  

- 양자화된 $$\mathrm{Attn}_Q$$와 values $$V_Q$$의 곱을 비트 시프트로 대체:  

$$
    \mathrm{Attn}\cdot V_Q = V_Q \gg \mathrm{Attn}_Q
  $$  

- Softmax 자체도 정수 전용 다항 근사(i-exp) 및 정수 로그 연산으로 연산  

## 4. 모델 구조 및 실험 결과  
- ViT, DeiT, Swin Transformer 다양한 백본에 적용  
- **ImageNet Top-1 정확도** (FP32 vs PTQ):  
  - ViT-L: 85.81% → 85.03% (8/8/8) → 84.89% (8/8/4)  
  - DeiT-B: 81.85% → 81.20% → 80.85%  
- **COCO 객체 탐지 mAP** (Cascade Mask R-CNN w/ Swin-S):  
  - FP32 52.0 → 51.4 (8/8/8) → 50.8 (8/8/4)  

## 5. 성능 향상 및 한계  
- 성능 향상: 완전 양자화(FQ-ViT) 시에도 기존 PTQ 대비 **거의 손실 없는 정확도**  
- 연산 효율: 비트시프트 기반 정수 전용 연산으로 하드웨어 리소스·전력 사용 절감  
- 한계:  
  - PTF 하이퍼파라미터 $$K$$ 튜닝 필요 (기본값 3)  
  - i-exp 근사 오차 누적 가능성  
  - 대규모 데이터셋 외 특수 도메인 일반화 검증 부족  

## 6. 일반화 성능 향상 가능성  
- PTF는 입력 분포 편차에 따라 동적으로 스케일을 맞추므로, 도메인 변화 시에도 **채널별 양자화 오류 최소화** 기대  
- LIS의 순서 보존(order-aware) 특성은 과소표집된 어텐션 패턴도 잘 유지해 **다양한 비전 태스크로 확장 가능**  

## 7. 향후 연구 영향 및 고려 사항  
- **더 낮은 비트 폭(2-4bit)** 양자화 적용 연구  
- PTF 자동 튜닝 기법 개발로 **하이퍼파라미터 의존성 감소**  
- i-exp 다항 근사 개선을 통한 **정수 전용 Softmax 정확도 보강**  
- **비교 연구**: QAT와 결합한 양자화–훈련 혼합 기법  
- **다양한 도메인**(의료영상, 위성영상)으로 일반화 검증  

> FQ-ViT는 “완전 양자화”라는 과제를 풀어내며, 향후 비전 트랜스포머의 AI 칩 배치 및 경량화 연구에 중요한 기준을 제공할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/81cd3d6e-356d-4c30-9bb7-557ae1c2541c/2111.13824v4.pdf)
