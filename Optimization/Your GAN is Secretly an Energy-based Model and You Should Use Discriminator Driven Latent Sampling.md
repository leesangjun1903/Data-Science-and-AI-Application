# Your GAN is Secretly an Energy-based Model and You Should Use Discriminator Driven Latent Sampling | 2020 · 151회 인용

# 핵심 요약

**Your GAN is Secretly an Energy-based Model and You Should Use Discriminator Driven Latent Sampling** 논문은 기존 GAN 샘플링의 한계를 극복하기 위해 **GAN의 생성기와 판별기를 합친 에너지 기반 모델(Energy-Based Model, EBM)** 관점에서 재해석하고, 잠재 공간에서 효율적인 MCMC 샘플링 기법인 **Discriminator Driven Latent Sampling (DDLS)**를 제안한다. 주요 기여는 다음과 같다:[1]

- GAN의 암묵적 생성 분포와 판별기의 logit 점수를 합쳐 구성한 에너지 함수가 진짜 데이터 분포에 수렴함을 이론적으로 증명  
- 픽셀 공간이 아닌 잠재 공간에서 에너지 기반 MCMC를 수행함으로써 샘플링 효율성을 획기적으로 개선  
- DDLS 적용만으로 기존에 학습된 여러 유형의 GAN(SNGAN, WGAN, DCGAN 등)의 샘플 품질 및 다양도를 정성·정량적으로 크게 향상  
- 클래스 조건 정보나 추가 파라미터 없이 무조건 생성(Unconditional) 설정에서 BigGAN 수준 성능(예: CIFAR-10 Inception Score 8.22 → 9.09, FID 21.7 → 15.42) 달성  

# 문제 정의

기존 GAN은 생성기 $$G(z)$$만을 이용해 잠재 변수 $$z\sim p(z)$$에서 샘플을 그리지만,  
1) GAN 학습이 완전 수렴하지 않아 $$p_g\neq p_{d}$$ 이고  
2) 픽셀 공간에서 직접 MCMC 샘플링은 고차원·다중모달 분포에서 비효율적이라는 두 가지 문제가 존재한다.[1]

따라서 판별기가 제공하는 정보로 생성기 편향을 보정하면서 효율적으로 샘플링할 방법이 필요하다.

# 제안 방법

1. **GAN을 EBM으로 재해석**  
   - 최적 판별기 $$D^*(x)$$가 성립할 때, 판별기 logit $$d(x)$$와 생성기 암묵적 밀도 $$\log p_g(x)$$의 합으로 정의한 분포  

$$
       p^*(x)\propto \exp\bigl(\log p_g(x)+d(x)\bigr)
     $$
     
  는 실제 데이터 분포 $$p_d(x)$$에 일치함을 보임.[1]

2. **잠재 공간 EBM 유도**  
   - 위 분포를 직접 샘플링하기 위해 픽셀 공간이 아닌 잠재 공간 $$z$$에 대응하는 에너지 함수로  

$$
       E(z) = -\log p_0(z) \;-\; d\bigl(G(z)\bigr)
     $$
     
  를 정의하면, 이로부터 생성된 분포

$$
       p_t(z)\propto \exp\bigl(-E(z)\bigr)
     $$
     
  를 MCMC(예: Langevin dynamics)로 샘플링하고 생성기 $$x=G(z)$$를 적용해 고품질 이미지를 생성할 수 있음.[1]

3. **Discriminator Driven Latent Sampling (DDLS)**  
   - **알고리즘**  
     ```
     z_0 ∼ p_0(z)
     for i in 0…N–1:
       z_{i+1} = z_i – (ϵ/2)∇_zE(z_i) + √ϵ·n_i,    n_i∼N(0,I)
     return G(z_N)
     ```
   - **효율성**: 픽셀 공간 MCMC 대비 훨씬 짧은 mixing time을 보이며, 판별기가 유도하는 그래디언트로 빠르게 고품질 영역으로 이동.

# 모델 구조

- **생성기 $$G$$**: 기존 GAN 구조 그대로(DCGAN, WGAN, SNGAN 등) 활용  
- **판별기 $$D$$**: 학습된 판별기를 고정하거나 추가로 로지스틱 출력층을 붙여 재교정(calibration)  
- **잠재 공간 샘플러**: Langevin dynamics 기반 MCMC로 100–1000 스텝 수행  

# 성능 향상

- CIFAR-10 무조건 생성에서 **Inception Score** 8.22 → **9.09**, **FID** 21.7 → **15.42** 개선  
- CelebA, ImageNet 등 다양한 데이터셋에서도 일관된 품질 및 다양도 향상  
- 훈련된 GAN에 추가 학습 없이, 오직 샘플링 단계만 개선하여 BigGAN 수준 성능에 근사[1]

# 한계 및 일반화 성능

- **모드 드롭핑**: 기본적으로 생성기 지원 영역(support)에 의존하므로 생성기가 놓친 모드는 복구 어려움.  
  → 출력에 소량의 가우시안 노이즈를 추가한 확장 잠재 변수 $$(z,z')$$로 DDLS 수행 시 어느 정도 완화 가능  
- **계산 비용**: MCMC 스텝 수가 증가할수록 샘플링 비용 상승  
- **Hyperparameter 민감도**: 스텝 크기 ϵ, 노이즈 크기, 스텝 수 등에 따라 품질 및 mixing 속도가 달라짐  
- **일반화 가능성**: 잠재 공간 샘플링 아이디어는 다양한 GAN 변형(WGAN, f-GAN 등)에 적용 가능하며, VAE나 Flow 기반 모델에도 확장 여지 있음  

# 향후 연구 및 고려 사항

- **레이어별 잠재 변수 확장**: Generator 내부 각 계층에 노이즈를 추가해 다단계 DDLS 적용  
- **VAE와의 융합**: VAE prior에 판별기 기반 에너지 보정 적용  
- **하이퍼파라미터 자동화**: MCMC 스텝 크기 및 스텝 수 최적화를 위한 메타러닝  
- **실제 응용**: 의료 영상·합성 데이터 생성 등 민감 분야에서 모드 커버리지와 공정성(fairness) 보장 연구  

이처럼 DDLS는 **GAN 샘플 품질과 다양도를 판별기 신호로 보정**하면서, **잠재 공간 MCMC**를 통해 효율적인 고품질 생성이라는 새로운 방향을 제시한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c04e2ca1-c495-4f8d-bebe-d0a2e48b567d/2003.06060v3.pdf)
