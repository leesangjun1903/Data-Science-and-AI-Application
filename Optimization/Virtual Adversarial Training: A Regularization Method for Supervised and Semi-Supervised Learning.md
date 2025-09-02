# Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning

**핵심 주장:**  
Virtual Adversarial Training(VAT)은 입력 주변에서 모델의 조건부 출력 분포 $$p(y|x)$$의 국소적 민감도를 측정하는 새로운 정규화 항인 *virtual adversarial loss*를 도입하여, 지도학습과 반지도학습 모두에서 일반화 성능을 획기적으로 향상시킨다.

**주요 기여:**  
1. **Virtual Adversarial Direction 정의:** 레이블 정보 없이도 입력 $$x$$ 주변에서 출력 분포를 최대한 변화시키는 방향 $$r_{\mathrm{vadv}}$$를 제안.  
2. **Local Distributional Smoothness(LDS) 측정:**  

$$
     \mathrm{LDS}(x,\theta)
     = D\bigl[p(y|x,\hat\theta),\,p(y|x + r_{\mathrm{vadv}},\theta)\bigr],
   $$  
   
   여기서 $$D$$는 KL divergence.  
3. **효율적 근사 계산:** 파워 방법과 유한차분으로 $$r_{\mathrm{vadv}}\approx \epsilon\,\frac{g}{\|g\|\_2}$$ (여기서 $$g=\nabla_r D|_{r=\xi d}$$)를 한 번의 역전파로 계산.  
4. **적용 범위:** 네트워크 아키텍처 불문, 레이블 없는 데이터에도 적용 가능 → 반지도학습 활용.  
5. **최소 하이퍼파라미터:** 오직 노름 구속 크기 $$\epsilon$$와 정규화 계수 $$\alpha$$ 두 개만 조정.

***

# 문제 정의

기존의 랜덤 노이즈 기반 정규화(데이터 증강, 드롭아웃 등)는 **특정 방향**의 작은 교란(adversarial perturbation)에 취약하여, 사람이 인식하기 어려운 미세한 교란에도 모델 예측이 크게 흔들릴 수 있다. 이를 해결하고자:

- **목표:** 입력 주변에서 모델 예측이 가장 민감한 방향을 찾아내어, 그 방향으로의 변화에 **강인**하도록 학습한다.

***

# 제안 방법

## 1. Virtual Adversarial Perturbation  
레이블 $$y$$ 정보를 쓰지 않고, 현재 추정된 분포 $$p(y|x,\hat\theta)$$를 “가상 레이블”로 삼아 다음 최적화로 정의:  

$$
  r_{\mathrm{vadv}} = \underset{\|r\|\le\epsilon}{\arg\max}\;D\bigl[p(y|x,\hat\theta),\,p(y|x+r,\hat\theta)\bigr].
$$

## 2. Local Distributional Smoothness (LDS)  

$$
  \mathrm{LDS}(x,\theta)
  = D\bigl[p(y|x,\hat\theta),\,p(y|x+r_{\mathrm{vadv}},\theta)\bigr].
$$

## 3. 전체 목적 함수  
지도학습 음의 로그우도 $$\ell(D_l,\theta)$$에 LDS 정규화 항을 더함:  

$$
  \min_\theta\;\ell(D_l,\theta)\;+\;\alpha\,\frac{1}{|D_l|+|D_{ul}|}
    \sum_{x\in D_l\cup D_{ul}}\mathrm{LDS}(x,\theta).
$$

## 4. 효율적 근사  
1) 해시안 행렬 고유벡터 대신 파워 방법 한 단계로  

```math
  r_{\mathrm{vadv}}\approx\epsilon\,\frac{g}{\|g\|_2},\quad
  g = \nabla_r\,D\bigl[p(y|x,\hat\theta),p(y|x+r,\hat\theta)\bigr]\Big|_{r=\xi d}.  
```

2) 이로써 **추가 역전파 2회**만으로 정규화 그라디언트를 계산.

## 5. 모델 구조  
- 완전연결 네트워크(MLP) 및 CNN 모두에 적용 가능.  
- 실험에서는 ReLU + Batch Normalization 기반 MLP(4층) 및 두 가지 규모의 CNN(Conv-Small, Conv-Large)을 사용.

***

# 성능 향상 및 한계

## 성능 향상  
- **MNIST(지도학습):** 기존 기법 대비 테스트 오류율 0.64% 달성  
- **CIFAR-10(지도학습):** 5.81% 오류율  
- **SVHN(반지도):** 레이블 1,000개만으로 6.83% → 추가 엔트로피 최소화 시 4.28%  
- **CIFAR-10(반지도):** 13.15%(엔트로피 최소화 포함)로 동종 최상위 성능  

특히 **일관된 일반화 개선**을 보이며, 작은 $$\epsilon$$만 조정해도 튜닝이 비교적 단순.

## 한계  
1. **큰 $$\epsilon$$** 사용 시 과도한 스무딩으로 입력이 비자연적 영역까지 확장되어 과소적합 유발 가능.  
2. **하이퍼파라미터 상호작용:** $$\alpha$$와 $$\epsilon$$이 커지면 서로 간 독립 최적화 어려움.  
3. **고차원 입력:** 파워 방법 수렴 속도는 스펙트럼 분포에 의존. 필요시 반복 횟수 $$K>1$$ 필요.

***

# 일반화 성능 강화 기전

1. **적응적 스무딩:** 모델이 **가장 취약한 방향**을 선택해 로컬 평활화 → 불필요한 방향에 자원 낭비 방지.  
2. **스펙트럼 노름 제어:** Hessian의 최댓값(eigenvalue)을 페널티 → 모델 민감도 최댓값 감소.  
3. **그래디언트 안정화:** 파워 방법 적용으로 정규화 그라디언트 분산 축소 → 학습 안정성↑.

***

# 향후 영향 및 연구 고려사항

- **생성 모델 결합:** VAT의 분포 스무딩과 GAN/VAE의 입력 분포 모델링을 결합하면 더욱 강인한 반지도 학습 가능.  
- **스케일 적응형 $$\epsilon$$:** 입력별로 최적 $$\epsilon$$을 학습하거나 예측하는 메커니즘 연구.  
- **고차원 영역 확장:** 자연 이미지 외 텍스트, 시계열에도 VAT 적용성 및 근사 개선 검토.  
- **스펙트럼 제어 심화:** Hessian 스펙트럼 전반을 활용한 다중 고유값 페널티 및 스무딩 방향 다변화.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9307424b-aa12-4ade-b87b-972f332c6a46/1704.03976v2.pdf)
