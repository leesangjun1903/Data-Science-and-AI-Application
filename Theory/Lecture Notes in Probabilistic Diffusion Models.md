# Lecture Notes in Probabilistic Diffusion Models

## 1. 문제 정의  
확률적 확산 모델(diffusion models)은 복잡한 데이터 분포 $$p_{\mathrm{complex}}(x)$$를 단순한 분포 $$p_{\mathrm{prior}}(x)$$로 점진적으로 변환하고, 그 반대(역확산) 과정을 학습하여 새로운 데이터를 생성하는 방식입니다.  
- **목표**: 자연 이미지나 기타 고차원 데이터 샘플을 생성하기 위해,  
  1) 데이터 분포에서 순수 노이즈 분포로 가는 **순방향 확산 과정**을 정의  
  2) 노이즈 분포에서 데이터 분포로 가는 **역방향 확산 과정**을 모델링 및 학습  
를 통해 복잡한 분포를 효과적으로 모델링하고 샘플링하는 것.

## 2. 순방향 확산 과정 (Forward Diffusion)  
1. 원본 샘플 $$x_0\sim p_{\mathrm{complex}}(x)$$에 작은 가우시안 노이즈를 점진적으로 추가  
2. $$t=1,\dots,T$$ 시점마다 분산 스케줄 $$\{\beta_t\}$$에 따라  

$$
     q(x_t\mid x_{t-1})
     = \mathcal{N}\bigl(x_t;\,\sqrt{1-\beta_t}\,x_{t-1},\;\beta_t I\bigr).
   $$  

3. 닫힌 형태로 직접 표현 가능:  

$$
     q(x_t\mid x_0)
     = \mathcal{N}\bigl(x_t;\,\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\bigr),
     \quad \bar\alpha_t=\prod_{i=1}^t(1-\beta_i).
   $$  

4. 충분히 큰 $$T$$에서 $$\bar\alpha_T\to0$$이 되어 $$x_T\sim \mathcal{N}(0,I)=p_{\mathrm{prior}}$$.

## 3. 역방향 확산 과정 (Reverse Diffusion)  
- **진짜 역확산 분포** $$q(x_{t-1}\mid x_t)$$는 베이즈 정리를 써야 하지만, 데이터 분포 $$p_{\mathrm{complex}}$$를 통합할 수 없어 직접 계산 불가.  
- 대신 **가우시안 형태**로 근사하여 학습:

$$
    p_\theta(x_{t-1}\mid x_t)
    = \mathcal{N}\bigl(x_{t-1};\,\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\bigr).
  $$

### 3.1 최적화 목표: 변분 하한 (ELBO)  
역확산 모델의 로그우도 $$\log p_\theta(x_0)$$를 최대화하기 위해, 순방향·역방향 합동 분포를 이용한 Evidence Lower Bound(ELBO)를 유도하여 학습한다.  
결과적으로 얻은 손실 함수(음의 ELBO) 중 핵심 일관성(consistency) 항은  

$$
  L_{t-1}
  = \mathbb{E}\_{x_t\sim q(x_t\mid x_0)}
    \bigl[
      \mathrm{KL}(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t))
    \bigr].
$$  

가우시안 간 KL을 계산하면, 이상적인 평균은  

$$
  \tilde\mu_t(x_t,x_0)
  = \frac{\sqrt{\bar\alpha_{t-1}}\;\beta_t}{1-\bar\alpha_t}\,x_0
    + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,
  \quad
  \alpha_t=1-\beta_t,
$$  

분산 $$\tilde\beta_tI$$ 임을 알 수 있다.

### 3.2 네트워크가 학습하는 것  
- $$x_0$$는 알려지지 않으므로, 노이즈 예측 네트워크 $$\hat\epsilon_\theta(x_t,t)$$를 도입.  
- 식 (11)에서  

$$
    x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_t,
    \quad \epsilon_t\sim\mathcal{N}(0,I)
  $$  
  
  를 쓰면, 이상적 평균 $$\tilde\mu_t$$도 $$\epsilon_t$$로 재표현 가능.  

- 따라서 네트워크는 $$\epsilon_t$$를 예측하도록 학습:  

$$
    L_{t-1}
    \propto \mathbb{E}\_{x_0,\epsilon}
    \bigl\|
      \hat\epsilon_\theta(x_t,t) - \epsilon
    \bigr\|^2.
  $$

## 4. 샘플링 알고리즘  
### 4.1 DDPM (Denoising Diffusion Probabilistic Model)  
학습된 $$\hat\epsilon_\theta$$를 이용한 역확산 스텝 하나:  

$$
  x_{t-1}
  = \frac{1}{\sqrt{\alpha_t}}
    \Bigl(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,
      \hat\epsilon_\theta(x_t,t)\Bigr)
    + \sqrt{\tilde\beta_t}\,z,\quad z\sim\mathcal{N}(0,I).
$$  

이를 $$t=T,\dots,1$$ 순으로 반복하여 $$x_0$$를 얻는다.

### 4.2 DDIM (Denoising Diffusion Implicit Models)  
- 순방향 과정 재정의 시그마 $$\sigma_t$$ 만 바꾸어 비마르코프(non-Markov) 생성 가능  
- 특히 $$\sigma_t=0$$인 경우, **결정적(deterministic)** 역확산으로 잠재 공간이 1대1 대응  
- 샘플링:  

$$
    x_{t-1}
    = \tfrac{1}{\sqrt{\alpha_t}}
      \bigl(x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta(x_t,t)\bigr)
    + \sqrt{1-\bar\alpha_{t-1}}\,\hat\epsilon_\theta(x_t,t).
  $$

## 5. 텍스트 조건부 확산(Text-Conditioning)  
### 5.1 분류기 안내(Classifier Guidance)  
- 역확산 분포를 클래스 레이블 $$y$$에 조건부로 바꾸는 아이디어:  

$$
    p_{\theta,\xi}(x_{t-1}\mid x_t,y)
    \propto p_\theta(x_{t-1}\mid x_t)\;p_\xi(y\mid x_{t-1}),
  $$  

- 로지스틱 선형 근사 및 1차 테일러 전개로, 평균에 $$\nabla_x\log p_\xi(y\mid x)$$ 항을 추가한 가우시안으로 근사.

### 5.2 분류기-프리 안내(Classifier-Free Guidance)  
- 외부 분류기 없이 텍스트 임베딩 $$y$$를 직접 네트워크 입력으로 사용  
- 일부 학습 시 텍스트를 마스킹하여, $$\hat\epsilon_\theta(x_t,t)$$와 $$\hat\epsilon_\theta(x_t,y,t)$$ 차이를 이용해 안내 강도 조절  
- 약칭: guidance scale $$s$$로 조절.

***

**핵심 요약**  
1. **순방향**: 데이터에 단계적 가우시안 노이즈 추가 → 단순 분포  
2. **역방향**: 역확산 분포를 가우시안으로 근사, 노이즈 예측 네트워크 학습  
3. **손실**: 변분 하한(ELBO) 유도로 노이즈 예측 오차 최소화  
4. **샘플링**: DDPM(확률적)과 DDIM(결정적) 방식 제공  
5. **조건부 생성**: 외부 분류기나 텍스트 임베딩을 활용한 안내 기법으로 텍스트-이미지 생성 지원.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/1f5a1231-d67a-42d4-a183-c7d185d4aec0/2312.10393v1.pdf
