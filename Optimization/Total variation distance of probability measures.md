Probability distributions - torch.distributions : https://pytorch.org/docs/stable/distributions.html

# Total variation distance of probability measures
총변동거리(Total Variation Distance, TV Distance)는 두 확률분포 간의 통계적 차이를 측정하는 거리입니다. 확률이 할당된 같은 사건에 대해 두 분포가 부여하는 확률의 최대 절대 차이로 정의됩니다.

정확한 정의는 다음과 같습니다:

측정공간 (($$\Omega, \mathcal{F}$$))에서 확률측도 (P)와 (Q)가 있을 때,

$$
[
\delta(P, Q) = \sup_{A \in \mathcal{F}} |P(A) - Q(A)|
]
$$

여기서 (A)는 가능한 모든 사건을 뜻하며, 이 값은 두 분포가 어떤 사건에 대해 부여하는 확률 차이 중 최대값입니다.

이산확률분포(확률질량함수)인 경우, 총변동거리는 다음과 같이 표현됩니다:

$$
[
\frac{1}{2}\sum_x |P(x) - Q(x)|
]
$$

이 식은 두 분포의 확률 값 차이의 절대값을 모두 더한 후 절반을 취한 값입니다.

총변동거리는 0에서 1 사이의 값을 가지며, 0이면 두 분포가 동일함을, 1이면 완전히 다른 분포임을 의미합니다. 따라서 총변동거리는 확률분포 간 차이를 직관적으로 이해하고 비교하는 데 매우 유용한 지표입니다.

요약하면, 총변동거리는 두 확률분포가 얼마나 다른지를 최대 사건 확률 차이로 측정하는 대표적인 거리 척도입니다.
https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures

# Variational Bayesian methods

**핵심 개념 요약**  
변분 베이지안 방법은 복잡한 후방분포 $$p(\mathbf{Z} \mid \mathbf{X})$$를 직접 계산하기 어려울 때, 단순한 근사분포 $$q(\mathbf{Z})$$를 도입하여 최적의 근사분포를 찾는 기법이다. 이때 최적화 목표는 변분 원리(ELBO: Evidence Lower Bound)를 최대화하거나 KL 발산을 최소화하는 것이다.

## 1. 문제 설정

데이터 $$\mathbf{X}=\{x_n\}\_{n=1}^N$$와 잠재 변수 $$\mathbf{Z}=\{z_n\}_{n=1}^N$$에 대해, 베이지안 모형의 사전분포와 우도는:

$$
p(\mathbf{X}, \mathbf{Z}) = p(\mathbf{X}\mid \mathbf{Z})\,p(\mathbf{Z}).
$$

후방분포는

$$
p(\mathbf{Z}\mid \mathbf{X}) = \frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{X})}
\quad\text{단, }p(\mathbf{X})=\int p(\mathbf{X},\mathbf{Z})\,d\mathbf{Z}.
$$

직접 계산 불가능할 때, 근사분포 $$q(\mathbf{Z})$$를 도입한다.

## 2. 변분 원리(ELBO) 도출

우리는 로그 주변 우도 $$\ln p(\mathbf{X})$$를 다음과 같이 변형한다:

$$
\ln_p(\mathbf{X})
= \int q(\mathbf{Z}) \ln \frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{Z}\mid \mathbf{X})}\,d\mathbf{Z}
= \underbrace{\int q(\mathbf{Z}) \ln \frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}\,d\mathbf{Z}}_{\mathcal{L}(q)} 
\;+\;\underbrace{\int q(\mathbf{Z}) \ln \frac{q(\mathbf{Z})}{p(\mathbf{Z}\mid \mathbf{X})}\,d\mathbf{Z}}\_{\mathrm{KL}[\,q\,\|\,p]\,\ge0}.
$$

여기서  
- $$\mathcal{L}(q)$$는 **Evidence Lower Bound** (ELBO),  
- $$\mathrm{KL}[q\|p]$$는 근사분포와 참 후방분포 간의 KL 발산이다.

따라서

$$
\ln p(\mathbf{X}) = \mathcal{L}(q) + \mathrm{KL}[\,q(\mathbf{Z})\,\|\,p(\mathbf{Z}\mid \mathbf{X})],
$$

$$\mathrm{KL}\ge0$$ 이므로 $$\mathcal{L}(q) \le \ln p(\mathbf{X})$$.  
$$\mathcal{L}(q)$$를 최대화하면 $$\mathrm{KL}$$을 최소화하여 $$q(\mathbf{Z})\approx p(\mathbf{Z}\mid \mathbf{X})$$를 얻는다.

## 3. ELBO 전개

$$
\mathcal{L}(q)
= \int_q(\mathbf{Z}) \ln\_p(\mathbf{X},\mathbf{Z})\,d\mathbf{Z}
$$

$$
\int_q(\mathbf{Z}) \ln\_q(\mathbf{Z})\,d\mathbf{Z}.
$$

이 식은 기대 우도(term1)와 엔트로피(term2)의 합으로 해석된다:
- 기대 우도: $$\mathbb{E}\_{q}[\ln p(\mathbf{X},\mathbf{Z})]$$,
- 엔트로피: $$-\mathbb{E}\_{q}[\ln q(\mathbf{Z})]$$.

## 4. 인수 분해 가정 (Mean-Field Approximation)

대부분의 응용에서 $$q$$를 계산 가능하도록 다음과 같이 팩터화한다:

$$
q(\mathbf{Z}) = \prod_{i=1}^M q_i(z_i).
$$

이때 각 팩터 $$q_i(z_i)$$를 순차적으로 최적화한다.  
변분 최적화 결과는 다음 갱신식 형태를 갖는다:

$$
\ln q_i^*(z_i)
= \mathbb{E}_{j\neq i}\bigl[\ln p(\mathbf{X},\mathbf{Z})\bigr] + \mathrm{const}.
$$

여기서 $$\mathbb{E}_{j\neq i}[\cdot]$$는 $$q_j$$들에 대한 기댓값이다.

## 5. 구체적 예: 가우시안 혼합 모델

가우시안 혼합 모델의 잠재 변수는 군집 할당 $$\mathbf{Z}$$와 군집별 모수 $$\{\mu_k,\Lambda_k\}_{k=1}^K$$이다.  

변분 근사분포를

$$
q(\mathbf{Z}, \{\mu_k,\Lambda_k\})
= \prod_{n=1}^N q(z_n)\;\prod_{k=1}^K q(\mu_k,\Lambda_k)
$$

로 설정하고, 각 팩터를 다음과 같이 갱신한다.

1. 군집 할당 $$q(z_n)$$ 갱신:

$$
\ln q(z_n=k)
\propto \mathbb{E}_{\mu_k,\Lambda_k}\bigl[\ln \pi_k\,\mathcal{N}(x_n\mid \mu_k,\Lambda_k^{-1})\bigr].
$$

2. 모수 $$q(\mu_k,\Lambda_k)$$ 갱신: 공액 사전(conjugate prior) 덕분에 정상 분포-위샤트 분포로 갱신.

$$
q(\mu_k, \Lambda_k)
= \mathcal{N}\bigl(\mu_k\mid m_k,(\beta_k\Lambda_k)^{-1}\bigr)\;
\mathcal{W}\bigl(\Lambda_k\mid W_k,\nu_k\bigr).
$$

각 하이퍼파라미터 $$(m_k,\beta_k,W_k,\nu_k)$$는 이전 식에서 기대값을 계산하여 닫힌형 갱신식을 얻는다.

## 6. 수렴 기준 및 구현

- **ELBO 값 모니터링**: 각 반복마다 ELBO $$\mathcal{L}(q)$$를 계산하여 증가 여부 확인  
- **KL 발산**: $$\mathrm{KL}[q\|p]$$이 충분히 작아지면 종료  
- **파라미터 변화**: $$\|\theta^{(t+1)}-\theta^{(t)}\|$$이 작은 경우 종료  

## 7. 요약 및 장단점

**장점**  
- 계산 효율적: MCMC보다 속도 우수  
- 확률적 해석: 근사 후방분포 활용 가능

**단점**  
- 근사 품질: Mean-field 가정으로 과도한 팩터화 시 부정확  
- 수렴: 전역 최적이 보장되지 않음

---  

위와 같은 수식 전개를 통해 **Variational Bayesian methods**의 원리와 구체적 갱신식을 체계적으로 이해할 수 있다.

https://en.wikipedia.org/wiki/Variational_Bayesian_methods

https://mpatacchiola.github.io/blog/2021/01/25/intro-variational-inference.html  
https://alpopkes.com/posts/machine_learning/kl_divergence/  
https://alpopkes.com/posts/machine_learning/variational_inference/  
https://hyunw.kim/blog/2017/10/27/KL_divergence.html  
https://statproofbook.github.io/P/norm-kl.html  
https://shakirm.com/papers/VITutorial.pdf

https://glanceyes.com/entry/VAE-%EA%B3%84%EC%97%B4-%EB%AA%A8%EB%8D%B8%EC%9D%98-ELBOEvidence-Lower-Bound-%EB%B6%84%EC%84%9D    
https://hugrypiggykim.com/2018/09/07/variational-autoencoder%EC%99%80-elboevidence-lower-bound/  


# KL divergence Loss
https://hwiyong.tistory.com/408

# VAE
https://devs0n.tistory.com/191   
https://www.kaggle.com/code/sushovansaha9/vae-pytorch-lightning-elbo-loss  
https://velog.io/@hong_journey/VAEVariational-AutoEncoder-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0  

# ELBO
DDPM ELBO : https://xoft.tistory.com/33   
https://glanceyes.com/entry/VAE-%EA%B3%84%EC%97%B4-%EB%AA%A8%EB%8D%B8%EC%9D%98-ELBOEvidence-Lower-Bound-%EB%B6%84%EC%84%9D    

# Wasserstein Distance
https://www.slideshare.net/slideshow/wasserstein-gan-i/75554346

# marginalize 의미
https://ploradoaa.tistory.com/107  
Marginal Likelihood 란? (빈도주의VS. 베이지안 관점 비교) : https://m.blog.naver.com/sw4r/221380395720

