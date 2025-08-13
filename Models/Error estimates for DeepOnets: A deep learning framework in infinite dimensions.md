# Error estimates for DeepOnets: A deep learning framework in infinite dimensions

Deep Operator Network(DeepONet)은 함수(입력)에서 함수(출력)로 가는 **비선형 연산자**를 학습하는 딥러닝 구조입니다. 복잡한 고차원 문제—예: 편미분방정식의 해 연산자—를 다룰 때 **차원 저주(curse of dimensionality)** 없이 효율적으로 근사할 수 있는 이론적 근거를 제공합니다.

아래에서는 DeepONet이 어떤 구조를 갖고, 학습 시 발생하는 오차를 어떻게 분해·제어하며, 이론적으로 왜 “차원 저주를 극복”하는지 쉽고 자세히 설명하겠습니다.

## 1. DeepONet 구조

DeepONet는 크게 세 부분으로 이루어집니다.  
– **인코더** $$E$$: 무한차원 입력 함수 $$u$$를 유한차원 벡터로 압축  
– **브랜치 네트워크** $$A$$: 압축된 입력 벡터를 다시 다른 유한차원 벡터로 근사  
– **트렁크 네트워크** $$\tau$$와 **재구성기** $$R$$: 브랜치 출력과 공간 좌표 $$y$$를 결합해 최종 출력 함수 생성  

수식으로 쓰면,

$$
N(u)(y) = \bigl(R\circ A\circ E\bigr)(u)\;(y)
$$

여기서,

$$
E(u)=\bigl(u(x_1),\dots,u(x_m)\bigr)\in\mathbb R^m,
\quad
A\bigl(E(u)\bigr)\in\mathbb R^p,
$$

$$
\tau(y)=(\tau_1(y),\dots,\tau_p(y)),\quad
R(\alpha)(y)=\sum_{k=1}^p \alpha_k\,\tau_k(y).
$$

– $$m$$은 센서 개수(입력 차원), $$p$$는 트렁크 개수(출력 차원)  
– $$\{x_j\}$$는 입력 함수 포인트 평가 지점  
– $$\{\tau_k\}$$는 출력 함수의 기저 함수(네트워크가 학습)

## 2. 오차 분해

학습된 DeepONet $$N$$과 진짜 연산자 $$G$$ 사이 **전체 오차**는

```math
\mathcal E
=
\Bigl\|G(u)-N(u)\Bigr\|_{L^2(\mu;L^2_y)}
=
\Bigl(\int\!\!\int |G(u)(y)-N(u)(y)|^2\,dy\,d\mu(u)\Bigr)^{1/2}
```

로 측정합니다. 여기서 $$\mu$$는 입력 함수 $$u$$의 분포입니다.

이 오차를 세 가지로 나눕니다[1]:
1. **인코딩 오차** $$\mathcal E_E$$: 입력 함수→점값 벡터 변환 손실  
2. **근사 오차** $$\mathcal E_A$$: 브랜치 네트워크의 유한차원 함수 근사 손실  
3. **재구성 오차** $$\mathcal E_R$$: 기저 함수로 출력 함수 복원시 손실  

결과적으로

```math
\boxed{\mathcal E \le L_{\!G}\,L_{R\!P}\,\bigl(\mathcal E_E\bigr)^\alpha + L_{R}\,\mathcal E_A + \mathcal E_R}
```

– $$L_G$$, $$L_R$$, $$L_{R P}$$: 해당 단계의 Lipschitz 상수  
– $$\alpha\in(0,1]$$: $$G$$가 $$\alpha$$-Hölder일 때 (특히 Lipschitz일 땐 $$\alpha=1$$)

## 3. 인코딩 오차 $$\mathcal E_E$$

– 입력 함수 $$u$$를 점값 $$u(x_j)$$으로만 대표 → 정보 손실  
– 공분산 연산자 $$\Gamma_\mu$$의 고유함수 $$\{\phi_k\}$$ 스펙트럼이 빠르게 떨어지면 적은 센서로도 정확히 재구성 가능  
– 이론적으로 무작위로 점을 고르는 것만으로도 거의 최적[3.9]  

$$
\mathcal E_E
\lesssim
\sqrt{\sum_{k>m/C\log m}\lambda_k(\mu)},
$$

$$\lambda_k(\mu)$$: $$\Gamma_\mu$$ 고유값

**의미**: 측도 기반 상위 공분산 성분이 작아지면, 입력  분해차원을 높여도 손실이 급감

## 4. 재구성 오차 $$\mathcal E_R$$

– 출력 함수 분포 $\( G_{\\#}\mu \)$ 의 공분산 $\Gamma_{G{\\#}\mu}$ 고유함수로 최적 기저 구성[3.14]  
– 유한개 기저 $$\{\tau_k\}$$로 근사 못하는 부분 → 고유값 꼬리  

$$
\mathcal E_R
\gtrsim
\sqrt{\sum_{k>p}\lambda_k(G{\\#}\mu)}
$$

– 만약 $$G(u)\in H^s$$, 표준 푸리에/다항 기저로

$$\mathcal E_R\lesssim p^{-s/n}$$[3.7]

**의미**: 출력 함수가 매끄러울수록 적은 기저로도 정확히 재구성

## 5. 근사 오차 $$\mathcal E_A$$

– 유한차원 매핑 $$R\circ G\circ D: \mathbb R^m\to\mathbb R^p$$을 뉴럴넷으로 근사  
– 일반 Lipschitz라면 $$\sim \varepsilon^{-m/2}$$ (차원 저주!), 하지만  
– 해당 매핑이 **해석적(holomorphic)** 특성 보이면 파라메트릭 해법 수준으로 효율적 근사[3.35]

## 6. 차원 저주 극복

연산자 근사에 차원 저주가 없다는 건, 원하는 정확도 $$\varepsilon$$를 얻기 위해 필요한 네트워크 크기(size)가  

$$
\mathrm{size}\lesssim\mathrm{poly}(1/\varepsilon),
$$

즉 지수적($$\exp(1/\varepsilon)$$)이 아니고 다항적($$(1/\varepsilon)^c$$) 증가.

위 세 오차를 조합하여 각 예제(ODE, 타원 PDE, 반포 PDE, 보존 법칙 PDE)에서  
– 인코딩 오차는 고유값 스펙트럼으로  
– 재구성 오차는 매끄러움(=Sobolev 정규성)으로  
– 근사 오차는 해석성(holomorphy)·수치해법 모사로  

각각 algebraic rate로 줄일 수 있음을 보였습니다. 따라서 DeepONet은 **무한차원-유한차원 연산자 근사**에서 효율성을 보장합니다.

## 7. 일반화 오차

학습할 때 실제 손실(모집단 오차)  

$$\mathcal L(N)=\mathcal E^2$$를  

표본 손실(경험적 오차) $$\mathcal L_N$$ 으로 대체하므로  

$$
\underbrace{\mathcal L(N_N)-\mathcal L(N^*)}\_{\text{일반화 오차}}
\;\lesssim
\sqrt{\frac{\log({\\#}\text{parameters})}{N_{\text{samples}}}}
$$

– 표본 개수 $$N$$이 늘면 $$1/\sqrt N$$ 감소  
– 파라미터 개수 의존은 $$\log$$ 스케일이므로 고차원 문제에도 문제없음[5.3]

# 결론

DeepONet은
-  인코딩→근사→재구성 단계로 오차를 분해해 이론적 제어 가능  
-  공분산 스펙트럼, 매끄러움, 해석성 등 **함수공간** 특성을 활용해 **차원 저주 극복**  
-  무한차원 입력공간에서도 표본 수 $$\sqrt{N}$$ 스케일로 일반화오차 감소

이를 통해 **PDE 해 연산자**, **물리 기반 시스템** 등 복잡한 연산자를 신뢰성 있게 근사·예측할 수 있는 강력한 이론적·실용적 도구임을 보였습니다.

[1] https://journals.sagepub.com/doi/10.1177/26320843231190024
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/14a12a30-31ba-49de-b525-c96b7efbc963/2102.09618v3.pdf
[3] http://link.springer.com/10.1007/PL00005457
[4] https://osf.io/5vbyr
[5] http://projecteuclid.org/euclid.aoms/1177699172
[6] https://www.earthdoc.org/content/papers/10.3997/2214-4609.202035022
[7] https://arxiv.org/abs/2212.12474
[8] https://ieeexplore.ieee.org/document/8365826/
[9] https://onlinelibrary.wiley.com/doi/book/10.1002/9780470192573
[10] https://academic.oup.com/imatrm/article/6/1/tnac001/6542709
[11] https://arxiv.org/html/2408.04157v1
[12] https://openreview.net/forum?id=21kO0u6LN0
[13] https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2021/2021-07_fp.pdf
[14] https://doc.global-sci.org/uploads/admin/article_pdf/20250212/a89eaba6c13d9dd6990e3e7d3e6205a2.pdf
[15] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4763746
[16] https://arxiv.org/pdf/2102.09618.pdf
[17] https://academic.oup.com/imatrm/article-pdf/6/1/tnac001/42785544/tnac001.pdf
[18] https://scispace.com/pdf/on-the-training-and-generalization-of-deep-operator-networks-2233v2bw6l.pdf
[19] https://arxiv.org/abs/2102.09618
[20] https://www.sciencedirect.com/science/article/pii/S0952197624003142
[21] https://arxiv.org/abs/2205.11359
[22] https://www.osti.gov/servlets/purl/1977482
[23] https://proceedings.mlr.press/v84/suzuki18a.html
[24] https://www.sciencedirect.com/science/article/abs/pii/S0893608022002349
[25] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5160138
[26] https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X
[27] http://www.tandfonline.com/doi/abs/10.1198/tech.2004.s813
[28] https://arxiv.org/html/2503.03178v1
[29] https://arxiv.org/pdf/2309.01020.pdf
[30] https://arxiv.org/pdf/2111.05512.pdf
[31] https://arxiv.org/pdf/2410.04344.pdf
[32] https://arxiv.org/pdf/1804.09060.pdf
[33] https://arxiv.org/pdf/2306.08218.pdf
[34] http://arxiv.org/pdf/2409.00994v1.pdf
[35] https://arxiv.org/html/2405.11907
[36] http://arxiv.org/pdf/2302.03271.pdf
[37] https://arxiv.org/pdf/2311.00060.pdf
[38] https://openreview.net/forum?id=nnTKcGNrbV
[39] https://www.sciencedirect.com/science/article/pii/S0893608020302033
[40] https://www.jmlr.org/papers/volume22/21-0806/21-0806.pdf
[41] https://proceedings.neurips.cc/paper_files/paper/2022/hash/949b3011c50300a2b4e60377466f52a8-Abstract-Conference.html

## 1. 핵심 주장 및 주요 기여  
이 논문은 무한 차원 바나흐 공간 사이의 비선형 연산자(Operator)를 근사하기 위해 제안된 DeepONet(Deep Operator Network)의 오차 구조를 체계적으로 해석·분석하고, 다음을 보인다.  
- **전역 근사 정리**: 입력 공간이 비연속이고 비콤팩트일지라도, 측도 기반 L²(μ)-오차에서 DeepONet이 임의 정확도로 근사 가능함을 보이는 일반화된 보편 근사 정리(Theorem 3.1).  
- **오차 분해**: DeepONet 전체 오차를 인코딩(encoding), 근사(approximation), 재구성(reconstruction) 세 항으로 분해하고 각 항에 대한 상·하한을 유도(Theorem 3.3).  
- **재구성 오차**: 출력 측의 공분산 연산자 스펙트럼에 따른 재구성 오차의 하한·상한(Theorems 3.6, 3.7).  
- **인코딩 오차**: 무작위 센서 배치로도 공분산 고유함수 기반 최적 오차에 근사 가능함을 보이는 인코딩 오차 해석(Theorem 3.9).  
- **근사 항 오차**: 유계 선형 및 특정 비선형(PDE 해 연산자) 연산자에서 DeepONet이 차원 저주(curse of dimensionality)를 깨고 정확도 ε 달성 시 파라미터 수가 다항적 성장함을 네 가지 PDE 예제(강제 진자, 타원형 PDE, Allen–Cahn, 고속 보존 법칙)로 보임(Section 4).

## 2. 해결 문제 및 제안 방법  
### 2.1 문제 정의  
- 대상: 입력 함수 u∈X(주로 C(D) 혹은 L²(D))에서 출력 함수 G(u)∈Y(주로 L²(U))로 매핑하는 비선형 연산자 G.  
- 목표: 연산자 근사 N(u)=R∘A∘E(u)로 오차  

$$
\|G - N\|_{L²(μ)} = \Bigl(\int_X\int_U|G(u)(y)-N(u)(y)|²\,dy\,dμ(u)\Bigr)^{1/2}
$$  

를 ε 이하로 보장하면서 모델 파라미터(센서 수 m, 브랜치·트렁크 크기 p)가 ε⁻^Θ 가 아닌 다항적(ε⁻^const) 증� 성장하도록 함.

### 2.2 DeepONet 구조  
1. **인코더** E: u↦(u(x₁),…,u(x_m)) ∈ℝᵐ (센서 점 x_j).  
2. **브랜치 네트워크** A:ℝᵐ→ℝᵖ로 유한 차원 매핑.  
3. **트렁크 네트워크** τ: U→ℝᵖ, τ(u)=(τ₀(y),…,τ_p(y)).  
4. **재구성** R(α)=τ₀+∑_{k=1}^p α_k τ_k(y).  

### 2.3 오차 분해  

$$
\|G-N\| \le \underbrace{Lip(R∘P)\,Lip(G)\|D∘E-Id\|^α}\_{\text{인코딩 오차}} + \underbrace{Lip(R)\,\|A-P∘G∘D\|}\_{\text{근사 오차}} + \underbrace{\|R∘P-Id\|}_{\text{재구성 오차}}.
$$

### 2.4 성능 향상: 차원 저주 극복  
- **인코딩**: μ 공분산 고유함수 스펙트럼 급감시 m센서로 재구성 오차 ε 달성. 무작위 센서도 최적 수준 오차 보장.  
- **재구성**: G(u)∈H^s(U)일 때 p≈ε^{–n/s} 트렁크로 오차 ε.  
- **근사**: 선형 연산자는 당연히 선형 근사로 차원 저주 없음. 비선형→해석적(holomorphic)이면 p,m≈poly(ε⁻1)로 ε 달성.

### 2.5 한계  
- 상수·로그항 숨김.  
- 트렁크·브랜치 구조 최적화·학습 필요.  
- 심층 최적화·학습 알고리즘 수렴 보장은 미해결.

## 3. 일반화 성능 향상  
- **경험적 위험 vs. 진짜 위험**: 학습 손실 L_Nᵤ approximates 실손실 L.  
- **커버링 수 기반 일반화 오차** $ε_{gen} ≲O(√{(d_θ·log(B√N))/N})$ , where d_θ=파라미터 수, B=가중치 범위.  
- 무한차원 입력 공간에도 샘플 수 N에 따라 1/√N 감소 → 차원 저주 극복.  
- 깊이·넓이 증가(d_θ↑) 시 샘플 수도 충분히 커야 함(N≫d_θ).

## 4. 향후 연구 영향 및 고려 사항  
- **확장**: Navier–Stokes, 다중 연산자 학습, 시계열 예측(RNN 구조) 등.  
- **최적성**: 상수·로그 항 최적화, 다양한 네트워크 구조(활성화·스킵·인덕티브 바이어스) 비교.  
- **학습**: SGD 수렴 이론, 불확실성·노이즈 민감도, 오염된 데이터 적용.  
- **실험 검증**: 여러 실세계 PDE 사례에 적용해 경험적 이론 확인.  

이 논문은 무한차원 연산자 추정 분야에 체계적 이론틀을 제공하며, DeepONet이 차원 저주 없이 효율적이고 일반화 가능함을 보였다. 후속 연구 시, 네트워크 구조·훈련 안정성·응용 범위 확장의 관점에서 본 논문 결과를 기반으로 실용적 알고리즘 개발이 가능하다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/14a12a30-31ba-49de-b525-c96b7efbc963/2102.09618v3.pdf
