# Learning the Stein Discrepancy for Training and Evaluating Energy-Based Models without Sampling

## 핵심 주장 및 주요 기여  
**주장:** Stein 불일치(Stein Discrepancy)를 신경망으로 학습하면, 샘플링 없이도 고차원 에너지 기반 모델(EBM)의 학습과 평가가 가능하다.  
**기여:**  
- **학습된 Stein 불일치(Learned Stein Discrepancy, LSD)** 정의: 비정규화 밀도 모델의 로그 밀도 기울기($$\nabla_x \log q(x)$$)만으로 모델과 데이터 분포 간 불일치를 추정할 수 있는 비대칭 거리 지표를 신경망 평론가(critic) $$f_\phi$$로 학습.[1]
- **효율적 추정:** Hutchinson 추정기를 활용해 $$\mathrm{Tr}[\nabla_x f_\phi(x)]$$ 항을 한 번의 벡터–자코비안 곱으로 계산하도록 개선.[1]
- **모델 평가 → 가설검정·모델 비교:** LSD를 이용해 두 모델 중 더 데이터에 근접한 모델을 선택하거나, goodness-of-fit 검정을 고차원에서도 효율적으로 수행.[1]
- **모델 학습:** Minimax 최적화(critic–모델 번갈아 업데이트)로 EBM 파라미터를 직접 학습하여 MCMC 기반 방법보다 안정적이고 확장 가능함을 입증.[1]

## 문제 정의  
에너지 기반 모델(EBM)은 정규화 상수 $$Z$$ 계산이 불가능해 최대우도 학습과 샘플링에 MCMC가 필수적이다. 그러나 유한 스텝 MCMC는 편향 샘플을 생성하며, 샘플러 파라미터에 민감해 학습/평가 결과가 왜곡된다.  

## 제안 방법  
1. **Stein 항목 기반 불일치 정의**  
   Stein 항등식:  

$$
   \mathbb{E}_{p}\bigl[\nabla_x \log p(x)^\top f(x) + \mathrm{Tr}[\nabla_x f(x)]\bigr]=0.
   $$  
   
   이를 데이터 분포 $$p$$ 대신 모델 분포 $$q$$로 대체하고, critic 함수 $$f$$를 최대로 조정하면 Stein 불일치  

$$
   \mathrm{SD}(p,q)=\sup_{f\in\mathcal{F}}\mathbb{E}_{p}[\nabla_x \log q(x)^\top f(x)+\mathrm{Tr}[\nabla_x f(x)]]
   $$  
   
   가 된다.[1]

2. **신경망 critic 학습 (LSD)**  

$$
   \max_{\phi}\;\mathbb{E}_{x\sim p}\bigl[\nabla_x\log q(x)^\top f_\phi(x)\;+\;\mathrm{Tr}[\nabla_x f_\phi(x)]\bigr]\;-\;\lambda\;\mathbb{E}_{p}\bigl[\|f_\phi(x)\|^2\bigr].
   $$  
  
   - $$\mathrm{Tr}[\nabla_x f]$$는 Hutchinson 추정기로 효율 계산.[1]
   - $$\lambda$$로 critic 출력 제약.  

3. **모델 평가 기법**  
   - **모델 비교:** LSD 값이 작은 모델을 선택하며, 분산-aware 검증 지표(mean–std)로 과적합 방지.[1]
   - **Goodness-of-Fit Test:** test statistic  

$$
     t=\frac{\bar{s}}{\hat{\sigma}/\sqrt{n}},\quad s(x)=\nabla_x\log q(x)^\top f_\phi(x)+\mathrm{Tr}[\nabla_x f_\phi(x)],
     $$  
     
  에 따른 Z-검정으로 $$p=q$$ 여부 판단.[1]

4. **모델 학습 알고리즘**  
   - Alternating updates: critic 최대화 ↔ 모델 파라미터 최소화 (minimax).[1]
   - Score Matching 대비 2차 도함수 불안정성 회피, MCMC 제거로 안정적 확장 가능.

## 모델 구조  
- critic 및 energy 함수에 각각 2층 MLP(은닉 300 유닛, Swish) 사용.  
- 학습 시 Adam 옵티마이저, critic 업데이트 반복 횟수 및 $$\lambda$$ 하이퍼파라미터 조정.

## 성능 향상  
- **가설검정:** 기존 선형 커널 기반 KSD, MMD 대비 고차원(200차원)에서도 검정력 유지 및 속도 우수.[1]
- **모델 비교:** Gaussian-Bernoulli RBM, Normalizing Flow에서 perturbation 구분, log-likelihood 추적 정확도 우수.[1]
- **모델 학습:** Linear ICA(50차원)에서 최대우도 성능 근접, 기존 unnormalized 방법(NCE, Score Matching) 대비 성능·안정성 우월.[1]
- **이미지 모델링:** MNIST/FashionMNIST EBM 학습 후 SGLD 샘플링으로 다양한 샘플 획득, 모드 캡처 확인.[1]

## 한계  
- critic 학습 위한 데이터·모델 선택 시 과적합 주의 필요.  
- 샘플링 제거에도 최종 시각화·응용에는 여전히 MCMC 의존.  
- 대규모 자연영상에는 추가 아키텍처·최적화 기술 필요.

## 일반화 성능 향상 관점  
critic이 데이터 분포 특성을 직접 학습하므로, 고차원·복잡도 높은 분포에서도 모델 간 미세 차이를 효과적으로 구분할 수 있어, 일반화 평가 지표로 강력하다. 또한 Hessian 불필요성 및 MCMC 오차 제거로 추정 편향이 줄어들어 실제 테스트 분포에서의 성능 일치도가 높아진다.[1]

## 향후 영향 및 고려 사항  
- **후속 연구:** 더 깊은 critic/GAN 스타일 minimax 기법, convolutional 아키텍처 통합으로 자연영상 EBM 학습 확대.  
- **샘플러 의존성 제거:** implicit sampler 학습(Hu et al. 2018)과 결합해 완전 샘플링-프리 generative 모델 개발.  
- **변형 불일치:** 다양한 함수 클래스·regularization을 통해 disco­very of robust critic architectures.  

앞으로 EBM 분야에서 **샘플링 의존성 제거**, **고차원 확장성**, **정교한 모델 비교**를 위한 기본 도구로 LSD가 자리매김할 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1364d136-6ac7-4529-9221-e0d97bade43a/2002.05616v4.pdf)
