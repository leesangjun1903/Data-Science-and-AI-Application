# IntraQ: Learning Synthetic Images with Intra-Class Heterogeneity for Zero-Shot Network Quantization

**핵심 주장**  
IntraQ는 제로샷 네트워크 양자화(Zero-Shot Quantization, ZSQ)에서 실제 데이터의 **클래스 내 이질성(intra-class heterogeneity)** 을 보존하는 새로운 합성 이미지 생성 기법을 제안함으로써, 합성 데이터만으로도 양자화 후 모델의 일반화 성능을 크게 향상시킬 수 있음을 보인다.

**주요 기여**  
1. **로컬 객체 강화(Local Object Reinforcement)**: 합성 이미지의 다양한 위치와 크기에서 객체를 학습하도록 무작위 크롭-리사이즈 과정을 도입.  
2. **여유 거리 제약(Marginal Distance Constraint)**: 클래스 내 피처 간 거리가 너무 좁아지지 않도록 상·하한 마진(λ_l, λ_u)을 둔 코사인 거리 제약식 적용 (식 6).  
3. **소프트 인셉션 손실(Soft Inception Loss)**: 단일 원-핫 레이블 대신 확률 분포를 U(ε, 1)에서 샘플링한 소프트 레이블과 MSE로 매칭(식 8)하여 복합 장면도 학습.  
4. **실험적 검증**: 5,120장 합성 이미지만으로 MobileNetV1 4-bit 양자화 시 ImageNet Top-1 51.36% 달성(종전 대비 +9.17%) 등 SOTA 대비 유의미한 성능 향상.

***

## 1. 해결하고자 하는 문제

기존 제로샷 양자화 기법들은 배치 정규화 통계(batch normalization statistics) 정합(BNS alignment)만으로 합성 이미지를 생성하거나, 원-핫 인셉션 손실로 클래스 경계만 학습하여 합성 데이터가 **클래스 내에서 지나치게 균질(homogeneous)**하게 분포된다. 이로 인해 양자화 후 모델이 실제 테스트 데이터의 **다양한 표현**을 일반화하지 못해 성능 저하가 발생한다.

***

## 2. 제안 방법

### 2.1 로컬 객체 강화 (Local Object Reinforcement)  
합성 이미지 $$\tilde I$$를 확률 $$p$$로 무작위 크롭 후 원 크기로 리사이즈하여  

$$
\tilde I_{\mathrm{LOR}} = 
\begin{cases}
\mathrm{resize}(\mathrm{crop}_\eta(\tilde I)), & \text{확률 } p,\\
\tilde I, & \text{확률 } 1-p,
\end{cases}
$$

여기서 $$\mathrm{crop}_\eta$$는 축소율을 $$\mathcal U(\eta,1)$$에서 샘플링한 영역을 크롭한다.

### 2.2 여유 거리 제약 (Marginal Distance Constraint)  
합성 이미지 $$\tilde I_{\mathrm{LOR}}$$의 피처 $$V_F(\tilde I_{\mathrm{LOR}})$$와 같은 클래스 평균 피처 $$C(\tilde I_{\mathrm{LOR}})$$ 간 코사인 거리를  

$$
L_{\mathrm{MDC}} = \max\bigl[\lambda_l - \cos(V_F,\;C),\,0\bigr] \;+\;\max\bigl[\cos(V_F,\;C)-\lambda_u,\,0\bigr]
$$

로 제약하여  
- $$\cos\ge\lambda_l$$로 **분산 확보**,  
- $$\cos\le\lambda_u$$로 **클래스 응집**을 동시에 달성한다.

### 2.3 소프트 인셉션 손실 (Soft Inception Loss)  
원-핫 인셉션 손실 대신, 사전 학습된 모델 출력 $$F(\tilde I_{\mathrm{LOR}})_c$$가  

$$\mathcal U(\epsilon,1)$$에서 샘플된 스칼라 값과 일치하도록  

$$
L_{\mathrm{SIL}} = \mathrm{MSE}\bigl(F(\tilde I_{\mathrm{LOR}})_c,\; U(\epsilon,1)\bigr)
$$

로 학습하여 단일 객체 편향을 완화하고 복합 장면을 생성한다.

### 2.4 전체 합성 손실 및 양자화 후 파인튜닝  
합성 이미지 최적화 손실:  

$$
L(\tilde I_{\mathrm{LOR}})=L_{\mathrm{BNS}}+L_{\mathrm{MDC}}+L_{\mathrm{SIL}}
$$  

양자화 네트워크 파인튜닝 손실:  

$$
L_Q = \mathrm{CE}(Q(\tilde I),y)\;+\;\alpha\,\mathrm{KL}(Q(\tilde I)\,\|\,F(\tilde I))
$$

***

## 3. 모델 구조

전체 워크플로우는 (1) 무작위 가우시안 노이즈 → (2) Local Object Reinforcement → (3) 사전학습 모델 통과 → (4) BNS 정합, MDC, SIL 손실로 합성 이미지 학습 → (5) 합성 이미지로 양자화 모델 파인튜닝 으로 구성된다.

***

## 4. 성능 향상

- **CIFAR-10/100**: ResNet-20 4-bit에서 91.49%/64.98% 달성, 종전 최고 대비 +0.19%/+0.61% 개선.  
- **ImageNet**:  
  - ResNet-18 W4A4에서 66.47% 달성(+1.97% vs. GZNQ),  
  - MobileNetV1 W4A4에서 51.36% 달성(+9.17% vs. DSG+IL).  
- **이질성 유지**: 합성 이미지 클래스 내 평균 코사인 거리 0.42로 실제 데이터(0.44)와 근접.

***

## 5. 한계

- **실제 데이터 대비 성능 여전히 격차** 존재(예: ResNet-18 W4A4 실제 파인튜닝 67.89% vs. IntraQ 66.47%).  
- **다른 비전 과제(예: 객체 검출)**에 대한 적용성 검증 부족.  
- **하드웨어 제약**으로 다양한 네트워크·데이터셋 확장 실험 제한.

***

## 6. 모델 일반화 성능 향상 관점

IntraQ는 클래스 내 피처 분포를 확장하고 복합 장면을 학습함으로써, **합성 데이터만으로도 실제 데이터의 다양한 표현을 더 잘 포착**하여 양자화 모델의 일반화 능력을 크게 높인다. 로컬 크롭으로 다양한 시각 배치를 학습하고, MDC로 피처 스프레드를 장려하며, 소프트 레이블로 오버피팅을 억제하는 삼중 전략이 시너지 효과를 발휘한다.

***

## 7. 향후 연구에 미치는 영향 및 고려사항

- **이질성 기반 합성 데이터**: ZSQ뿐 아니라 데이터 증강·합성 학습 전반에 클래스 내 다양성 보존 전략으로 활용 가능.  
- **하이퍼파라미터 민감도**: η, λ_l, λ_u, ε, α 등의 최적화가 모델·데이터셋별로 달라질 수 있으므로 자동 탐색 기법 연구 필요.  
- **다양한 태스크 확장성**: 객체 검출, 세분화 등 복합 비전 과제에 IntraQ의 이질성 보존이 일반화 성능에 미치는 영향 검증이 요구됨.  
- **경량화 및 효율성**: 합성 이미지 생성 비용·메모리 효율을 높이기 위한 경량 모델 설계 및 학습 가속화 연구가 필요하다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0498066e-eb3b-4dbf-8589-323ac0680582/2111.09136v5.pdf)
