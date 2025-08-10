# Variational Inference: A Review for Statisticians
# 핵심 주장 및 주요 기여 요약

**핵심 주장**  
이 논문은 베이지안 통계에서 후방분포(posterior) 계산이 어려운 문제를 해결하기 위해, 최적화 관점의 근사 추론 기법인 **변분 추론(Variational Inference, VI)** 을 체계적으로 정리하고 통계학적 관점에서 조망한다. VI는 표본추출 기반 방법(MCMC)에 비해 계산 속도가 빠르고 대규모 데이터에 잘 확장되지만, 통계적 성질에 대한 이해는 아직 부족하다는 점을 강조한다.[1]

**주요 기여**  
1. VI의 **기본 원리**(증거하한(Evidence Lower BOund, ELBO) 도출과 KL 발산 최소화)를 명확히 설명.  
2. **평균장(mean-field) 변분 근사**를 이용한 좌표 상승 알고리즘(CAVI)을 정리하고, 지수족 모델(exponential family) 및 결합 가우시안 혼합 모델 예시를 통해 구체적 업데이트식을 제시.  
3. **지수족 및 조건부 공액(conjugacy) 모델**에 대한 일반화된 VI 기법을 설명하고, 전역·국소 변수 구조에서의 CAVI 유도 방식 제시.  
4. 대규모 데이터에 적용 가능한 **확률적 변분 추론(Stochastic Variational Inference, SVI)** 알고리즘을 소개하여, 자연 그래디언트 및 확률적 최적화를 활용한 확장성을 보임.  
5. 다양한 분야(영상, 자연어, 생물정보학 등)에서의 실제 응용 사례를 폭넓게 검토하고, VI의 이론적 이해와 확장 가능한 연구 방향을 제시.[1]

# 문제 설정 및 제안 방법 상세 설명

## 1. 근사 추론의 문제  
베이지안 모델에서 관심 있는 후방분포는  

$$ p(z \mid x) = \frac{p(z,x)}{p(x)}, \quad p(x)=\int p(z,x)\,dz $$  

이지만, 분모 $$p(x)$$ 계산이 고차원 통합을 요구하여 불가능하거나 계산비용이 지수적으로 증가한다.[1]

## 2. 변분 추론 공식화  
1) **변분 패밀리** $$\mathcal{Q}$$ 정의: 근사 분포 $$q(z)$$들이 속하는 사전 분포족.  
2) **KL 발산** 최소화  

$$
q^*(z)=\arg\min_{q\in\mathcal{Q}}\mathrm{KL}(q(z)\parallel p(z\mid x))
$$  

직접 최적화가 불가능하므로, 다음 **ELBO** 를 도입:[1]

$$
\mathrm{ELBO}(q)=\mathbb{E}_q[\log p(z,x)] - \mathbb{E}_q[\log q(z)]
$$  

이는 $$\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}(q\,\|\,p)$$ 관계로부터  

$$\mathrm{KL}\ge0$$ 임을 이용해  

$$\log p(x)\ge\mathrm{ELBO}(q)$$ 이므로 ELBO를 최대화하면 KL이 최소화된다.

## 3. 평균장 변분 패밀리 및 좌표 상승 알고리즘 (CAVI, coordinate ascent variational inference)  
- **평균장 가정**: $$q(z)=\prod_{j=1}^m q_j(z_j)$$ (잠재변수 간 독립 가정)  
- **좌표 상승 업데이트**: 각 $$q_j$$ 를 고정한 상태에서  

$$
q_j^*(z_j)\propto \exp\Bigl(\mathbb{E}\_{-j}[\log p(z_j,z_{-j},x)]\Bigr)
$$  

를 반복 적용하여 ELBO의 국소 최대화 수행.[1]

### 예: 가우시안 혼합 모델  
- 관측 $$x_i$$, 군집 평균 $$\mu_k$$, 할당 변수 $$c_i$$  
- 변분 인자는 $$\phi_{ik}=q(c_i=k)$$ 와 $$\mathcal{N}(\mu_k\mid\hat\mu_k,\hat\sigma_k^2)$$  
- 업데이트식  

$$
  \phi_{ik}\propto\exp\Bigl(\mathbb{E}[\log p(x_i\mid\mu_k)]\Bigr),\quad
  \hat\mu_k=\frac{\sum_i\phi_{ik}x_i}{1/\sigma^2+\sum_i\phi_{ik}},\;
  \hat\sigma_k^2=\frac{1}{1/\sigma^2+\sum_i\phi_{ik}}
  $$  
  
를 좌표 상승 반복.[1]

## 4. 지수족 모델 및 조건부 공액 구조  
- **지수족 완전조건부**:  

$$
  p(z_j\mid z_{-j},x)=h(z_j)\exp\{\eta_j(z_{-j},x)^\top t(z_j)-a(\eta_j)\}
  $$  
  
이면, $$q_j$$ 도 동일 지수족이며 자연모수는  
  
$$\nu_j=\mathbb{E}\_{-j}[\eta_j(z_{-j},x)]$$ 로 업데이트.[1]

- **전역·국소 변수 모델**: 전역변수 $$\beta$$, 국소변수 $$z_i$$ 및 데이터 $$x_i$$  
  - 전역 prior: 공액 형태 $$p(\beta)$$  
  - 변분 업데이트  

$$
    \phi_i = \mathbb{E}\_\lambda\bigl[\eta(\beta,x_i)\bigr],\quad
    \lambda = \alpha + \sum_i \mathbb{E}_{\phi_i}[t(z_i,x_i)]
    $$  
  
를 좌표 상승 방식으로 반복.[1]

## 5. 확률적 변분 추론 (SVI, stochastic variational inference)  
- 대규모 데이터에서 전체 집합 순회가 비효율적이므로, **자연그래디언트**  
 
$$g(\lambda)=\widehat\alpha - \lambda$$ 을 도입.  

- 확률적 최적화: 임의 샘플 $$t$$ 에 대해  

$$
  \hat g(\lambda)=\alpha + n\mathbb{E}\_{\phi_t}[t(z_t,x_t)] - \lambda,
  \quad
  \lambda\leftarrow (1-\epsilon_t)\lambda + \epsilon_t\bigl(\alpha + n\,\mathbb{E}\_{\phi_t}[t]\bigr)
  $$  
  
을 반복하며 수렴.[1]
- 문서 180만 건 토픽 모델링 사례에서 대규모 확장성 입증.[1]

***

각 방법에 대한 수식 유도 및 응용 예시는 본 논문에서 자세히 다루며, VI의 이론적·실용적 이해를 확장하는 데 중요한 토대를 제시한다.[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/1effb1d1-c1ca-4378-91c6-f794587a3418/1601.00670v1.pdf
