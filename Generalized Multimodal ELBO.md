# Generalized Multimodal ELBO

**주요 요약 및 기여**  
본 논문은 **MoPoE-VAE (Mixture-of-Products-of-Experts VAE)**라는 새로운 증거하한(ELBO) 공식을 제안한다. 이는 기존의 MVAE(Product-of-Experts)와 MMVAE(Mixture-of-Experts)를 일반화하여, 모든 모달리티 부분집합에 대한 후방분포 근사를 **PoE**와 **MoE**로 계층적으로 결합함으로써 두 접근법의 장점을 타협 없이 통합한다. 이로써 다양한 모달리티 조합에서 뛰어난 **잠재표현 일반화**와 **생성 샘플 일관성**, **로그우도** 간의 최적 트레이드오프를 달성한다.[1]

***

## 1. 문제 정의  
실세계 데이터는 서로 다른 유형(modalities)의 정보가 공존하며, 이들 간의 **공유 정보(self-supervision)**를 활용해 잠재 표현을 학습하고 데이터를 생성하는 것이 목표다.  
기존 멀티모달 VAE들은 다음 두 가지 문제에 직면한다:  
- **Posterior approximation trade-off:**  
  - MVAE(PoE)는 결합 분포를 잘 근사하나, 개별 모달 후방분포 최적화에 취약  
  - MMVAE(MoE)는 개별 모달 후방분포 최적화에는 강하나, 조합 정보 통합 성능 저하  
- **모달리티 결측 처리의 비확장성:**  
  - 𝑀개 모달이 있으면 부분집합 갯수가 2^M으로 증가하여 별도 네트워크 필요  

***

## 2. 제안 방법 및 모델 구조  
### 2.1 일반화된 ELBO 공식  
모든 부분집합 X_k에 대하여  

$$
\tilde q_{\phi}(z \mid X_k) =\mathrm{PoE}\bigl(\{q_{\phi_j}(z\mid x_j)\}\_{x_j\in X_k}\bigr)
=\frac{\prod_{x_j\in X_k}q_{\phi_j}(z\mid x_j)}{\displaystyle\int\prod_{x_j\in X_k}q_{\phi_j}(z\mid x_j)\,dz}\,,
$$

전체 결합 후방분포는 이들을 균등 평균한 Mixture 형태로 정의한다:[2]

$$
q_{\phi}(z\mid X) \;=\;\frac{1}{2^M}\sum_{X_k\subseteq X}\tilde q_{\phi}(z\mid X_k)\,.
$$  

이에 따른 **MoPoE-ELBO**는  

$$
\mathcal{L}\_\mathrm{MoPoE}=\mathbb{E}\_{q_{\phi}(z\mid X)}[\log p_{\theta}(X\mid z)]-
\mathrm{KL}\Bigl(\tfrac{1}{2^M}\sum_{X_k}\tilde q_{\phi}(z\mid X_k)\Big\|p_{\theta}(z)\Bigr)
$$  

로 주어지며, 이는 각 부분집합 ELBO의 볼록 결합을 단일 유효 하한으로 통합한다.[2]

### 2.2 모델 아키텍처  
- **Unimodal encoders** $$q_{\phi_j}(z\mid x_j)$$: 각 모달별 잠재인코더  
- **Products-of-Experts (PoE)**: 부분집합별 인코더 출력 결합  
- **Mixture-of-Experts (MoE)**: 부분집합 PoE들의 균등 평균  
- **공동 디코더** $$p_{\theta}(X\mid z)$$: 잠재 변수 $$z$$로부터 모든 모달을 재구성  

이 계층적 구조로 2^M 부분집합을 직접 학습하지 않고도 **Missing-modality** 상황을 효율적으로 처리한다.

***

## 3. 성능 향상 및 한계  
### 3.1 성능 향상  
| 모델    | Joint Log-Likelihood | Joint Coherence | Latent Classification | Missing-Modality Robustness |
|:-------:|:--------------------:|:--------------:|:---------------------:|:----------------------------:|
| MVAE    | 매우 우수            | 낮음           | 보통                  | 부분집합 PoE만 최적화        |
| MMVAE   | 낮음                 | 매우 우수      | 우수                  | 단일 모달 최적화에 특화      |
| **MoPoE-VAE** | **우수**             | **우수**        | **우수**               | **모든 부분집합 동등 처리**    |

- MNIST-SVHN-Text, PolyMNIST, CelebA 실험에서 MoPoE-VAE가 **일관성(Coherence)**과 **로그우도** 간의 최적 트레이드오프를 달성.[2]
- 다양한 개수의 입력 모달에서 **잠재표현 분류 정확도** 및 **조건부 생성 일관성**에서 최고 성능.

### 3.2 한계  
- **계산 복잡도**: 부분집합 평균(MoE) 연산에 따른 추가 비용  
- **추상 평균 함수** 선택 제약: Gaussian PoE에 의존하며, 이외 평균 함수를 적용시 이론적 타이트니스 재검증 필요  
- **고차원 모달리티**(예: 고해상도 영상) 확장 시 효율성 저하 가능성  

***

## 4. 일반화 성능 향상 관점  
1. **Missing-modality 일반화**: 모든 부분집합을 동등 처리하여, 일부 모달이 결측되어도 **일관된 잠재표현** 및 **생성 품질** 유지.[2]
2. **자기지도적 학습**: 공유 정보뿐 아니라 모달별 정보도 활용해 **잠재 분포**가 더욱 풍부하고 견고하게 학습됨.  
3. **트레이드오프 최적화**: β-하이퍼파라미터를 조절하며 **다양한 목표**(로그우도 vs. 생성 일관성) 간 균형 조정 가능.

***

## 5. 향후 연구 영향 및 고려사항  
- **다양한 추상 평균 함수 탐색**: Arithmetic/Geometric 외 다른 평균 함수를 도입해 **타이트니스(ELBO bound tightness)**와 **표현력** 개선  
- **Diffusion decoder 결합**: 고품질 이미지 생성에는 최신 **확산모델(decoder)** 통합 연구  
- **계산 효율화**: MoE 단계의 **부분집합 수 줄이기** 또는 **샘플링 기반 근사**로 대규모 모달리티 확장  
- **모달별 특화 잠재공간**: CelebA에서 시도된 대로 **모달별 잠재 분리**가 일반화 성능에 미치는 영향 평가  

위와 같이 MoPoE-VAE는 멀티모달 대표모델 연구에 **이론적 통합**과 **실험적 우수성**을 기여하며, 차세대 멀티모달 생성 모델 개발의 기반이 될 것으로 기대된다.

[1] https://www.semanticscholar.org/paper/8de5f12826bd6726f541c8155191a3127eec3710
[2] http://arxiv.org/pdf/2105.02470.pdf
[3] https://ccforum.biomedcentral.com/articles/10.1186/s13054-015-0809-9
[4] http://ieeexplore.ieee.org/document/6619144/
[5] https://ccforum.biomedcentral.com/articles/10.1186/cc12714
[6] https://www.semanticscholar.org/paper/5c12d9ca6393c8a3717b79dfa7f6a3952e948c71
[7] https://arxiv.org/abs/2403.09027
[8] https://arxiv.org/abs/2410.15475
[9] https://arxiv.org/abs/2403.09530
[10] https://arxiv.org/abs/2307.07093
[11] https://arxiv.org/abs/2311.10125
[12] https://arxiv.org/pdf/2105.02470.pdf
[13] https://arxiv.org/pdf/2201.06718.pdf
[14] https://arxiv.org/html/2408.16883
[15] http://arxiv.org/pdf/2410.19315.pdf
[16] http://arxiv.org/pdf/2303.15963.pdf
[17] https://arxiv.org/pdf/2202.03390.pdf
[18] http://arxiv.org/pdf/2307.05222.pdf
[19] http://arxiv.org/pdf/2307.05435.pdf
[20] http://arxiv.org/pdf/2407.09705.pdf
[21] http://arxiv.org/pdf/2502.15336.pdf
[22] https://iclr.cc/media/iclr-2021/Slides/2632.pdf
[23] https://openreview.net/forum?id=5Y21V0RDBV
[24] https://www.semanticscholar.org/paper/Generalized-Multimodal-ELBO-Sutter-Daunhawer/8de5f12826bd6726f541c8155191a3127eec3710
[25] https://arxiv.org/abs/2105.02470
[26] https://thomassutter.github.io/publication/2021-04-26-Generalized%20Multimodal%20ELBO
[27] https://github.com/thomassutter/MoPoE
[28] https://papertalk.org/papertalks/29288
