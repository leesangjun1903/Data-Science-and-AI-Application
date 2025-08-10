# Decoupled Kullback–Leibler Divergence Loss

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
Decoupled Kullback–Leibler (DKL) Divergence loss는 기존 KL 손실이 사실상  
1) 가중치 MSE(wMSE) 손실과  
2) 소프트 레이블 교차 엔트로피(CE) 손실의 합으로 **동일한 그래디언트**를 만들어냄을 이론적으로 증명한다.  

**주요 기여**  
- KL 손실의 그래디언트 관점 해석 및 DKL 등가성 증명  
- 비대칭 최적화(asymmetric optimization) 제거 기법 제안  
- 클래스 단위 전역 정보(class-wise global information) 도입을 통한 샘플 편향 완화  
- 위 두 개선을 종합한 Improved KL(IKL) 손실 제안  
- CIFAR-10/100 및 ImageNet에서 **최신 수준의** 적대적 강인성 및 지식 증류 성능 달성[1]

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제  
1. **비대칭 최적화**: 지식 증류 시 Teacher 쪽 wMSE 그래디언트가 차단되어 활용되지 않음  
2. **샘플 편향(sample-wise bias)**: wMSE 가중치가 개별 예측에 의존해 hard example에서 잘못된 학습 유도  

### 2.2 DKL 손실 등가성  
기존 KL 손실:  

$$
L_{KL}(x_m,x_n)=\sum_{j=1}^C s^m_j\log\frac{s^m_j}{s^n_j}
$$  

→ 그래디언트 전개 후  

$$
\frac{\partial L_{KL}}{\partial o^m_j}
=\sum_k(\Delta^m_{j,k}-\Delta^n_{j,k})\,w^m_{j,k},\quad
\frac{\partial L_{KL}}{\partial o^n_j}
=s^m_j(s^n_j-1)+s^n_j(1-s^m_j).
$$  

여기서 $$\Delta^m_{j,k}=o^m_j-o^m_k,w^m_{j,k}=s^m_j\,s^m_k$$.  

Theorem 1: KL 손실은 다음 DKL 손실과 **동일한 그래디언트**를 갖는다:[1]

$$
L_{DKL}=\frac{\alpha}{4}\bigl\|W^m(∆^m-∆^n)\bigr\|^2
-\beta s^{m\top}\log s^n
$$  

– 첫항: wMSE  
– 둘째항: 소프트 레이블 CE  

### 2.3 Improved KL (IKL) 손실  
두 가지 개선 반영:  

1. **비대칭 최적화 제거**  

$$
   \underbrace{\frac{\alpha}{4}\|W^m(\Delta^m - \Delta^n)\|^2}_{\text{wMSE}}
   \longrightarrow
   \frac{\alpha}{4}\bigl\|W^m(\Delta^m - \mathrm{stopgrad}(\Delta^n))\bigr\|^2
   $$  
  
→ Teacher→Student wMSE 그래디언트 활성화  

2. **클래스 단위 전역 정보 삽입**  

$$
   W^m_{j,k}=s^m_j\,s^m_k
   \longrightarrow
   \bar W^y_{j,k}=\bar s^y_j\,\bar s^y_k,\quad
   \bar s^y=\frac1{|X_y|}\sum_{x\in X_y}s(x)
   $$  
   
→ 소속 클래스 전역 평균 확률로 가중치 정규화  

최종 IKL 손실:  

$$
L_{IKL}
=\frac{\alpha}{4}\bigl\|\bar W^y(\Delta^m - \Delta^n)\bigr\|^2
-\beta\,s^{m\top}\log s^n.
$$

### 2.4 모델 구조 및 학습 설정  
- **Adversarial Training**: TRADES+AWP 기반, WideResNet-34-10/28-10 사용  
- **Knowledge Distillation**: KD/DKD 대비, ResNet 및 MobileNet 계열  

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
- **적대적 강인성**: CIFAR-100 ℓ∞, ϵ=8/255에서 Auto-Attack AA 31.91%로 기존 대비 +0.71% (기본 aug) 및 39.18%로 +0.35% (50M 합성 데이터)[1]
- **지식 증류**: ImageNet ResNet34→ResNet18 71.91% (+0.85%), ResNet50→MobileNet 72.84%[1]
- **일반화**: 클래스 경계 마진(margin) 증가로 특징 공간 분리도 개선[1]

### 3.2 한계  
- **하이퍼파라미터 민감도**: α, β, τ 조정 필요 (α₄=5, β=5, τ=4 권장)  
- **클래스 수 증가 시 메모리**: $$C×C$$ 가중치 행렬 유지  
- **계산 오버헤드**: 소폭 증가하나 전체 학습 비용 대비 미미함  

## 4. 일반화 성능 관점 강조  
- **클래스 단위 전역 정보**는 intra-class 일관성 강화 → hard example에서도 안정적 학습  
- **비대칭 최적화 제거**로 Teacher 지식 전이 완전 활용  
- 이로 인해 **Out-of-Distribution** 및 **Semi-supervised** 학습에서도 유망(예비 실험: FixMatch, Mean-Teacher에서도 성능 향상)  

## 5. 향후 연구 영향 및 고려사항  
- **다른 도메인**(연속학습, 반지도·비지도, OOD robustness)에 IKL 확장 가능성  
- **가중치 행렬 효율화**(저메모리·저계산) 기법 연구 필요  
- **하이퍼파라미터 자동 최적화**로 실험 부담 경감  
- **이론적 일반화 보장** 연구(경계 마진과 일반화 관계 정량화)  

***

 Jiequan Cui et al., “Decoupled Kullback–Leibler Divergence Loss,” *arXiv:2305.13948v3*, Oct 27, 2024.[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/1f1d0461-0a03-47ec-b3bd-fb6a4a135d53/2305.13948v3.pdf
