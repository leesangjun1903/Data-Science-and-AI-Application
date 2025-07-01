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

