# Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations

# 핵심 요약

**Physics‐Informed Neural Networks (PINNs)**는 - 신경망(neural networks)을 일반 비선형 편미분방정식(PDE)을 만족하도록 자동미분(auto‐differentiation)으로 제약을 가한 새로운 학습 프레임워크이다[1].  
- **핵심 주장**: 작은 데이터 환경에서도 물리 법칙을 손실 함수에 직접 통합함으로써 일반 심층 학습이 갖는 과적합 현상을 억제하고, PDE의 순방향(forward) 해석 및 역문제(inverse problem)를 모두 효과적으로 해결할 수 있다[1].

# 해결 문제 및 제안 방법

## 1. 문제 설정  
PDE로 기술되는 물리계에 대하여  
- 순방향 문제(forward problem): 계수 λ가 주어졌을 때 해 u(t,x)를 추정  
- 역문제(inverse problem): 해 측정 데이터로부터 미지 계수 λ를 식별  

수식(일반형):  

$$
u_t + \mathcal{N}[u;\lambda] = 0,\quad x\in\Omega,\ t\in[0,T].
$$

## 2. Continuous‐Time 솔버  
- 해 u(t,x)를 신경망 $$u(t,x;\theta)$$으로 근사  
- PDE 잔차 $$f := u_t + \mathcal{N}[u;\lambda]$$ 역시 자동미분으로 계산  
- 손실함수:  

$$
\mathrm{MSE} = \frac{1}{N_u}\sum|u(t_u,x_u)-u_{\mathrm{data}}|^2 + \frac{1}{N_f}\sum|f(t_f,x_f)|^2.
$$ 

– 첫 항은 경계·초기조건 데이터 오차, 둘째 항은 PDE 잔차 페널티[1].

## 3. Discrete‐Time 솔버 (Runge–Kutta)  
- q단계 Runge–Kutta(time stepping)와 연동  

$$
u^{n+c_i} = u^n - \Delta t\sum_j a_{ij}\,\mathcal{N}(u^{n+c_j}),\quad
u^{n+1} = u^n - \Delta t\sum_j b_j\,\mathcal{N}(u^{n+c_j}).
$$  

- 단일 시점 또는 두 시점 데이터만으로 대형 Δt를 사용한 예측 가능  
- 손실함수는 시점별 SSE 합:  

$$
\mathrm{SSE} = \sum_j\sum_i|u^n_j(x_i^n)-u_i^n|^2 + \sum_j\sum_i|u^{n+1}_j(x_i^{n+1})-u_i^{n+1}|^2.
$$ 

–높은 q로 θ 수 소폭 증가시키고 Δt²q 오더 정확도 달성[1].

## 4. 모델 구조  
- 심층 피드포워드 신경망  
　–활성화: tanh  
　–레이어: 4–9개, 은닉유닛 20–200  
- 자동미분(체인룰)으로 입력(t,x)에 대한 미분 연산 수행  
- L-BFGS(전체 배치) 및 Adam(미니배치) 최적화  

# 성능 향상 및 한계

## 성능  
- 소량(수백 점) 데이터만으로 Burgers’, Schrödinger, Allen–Cahn, Navier–Stokes 등 해 정확도 10⁻³–10⁻⁴ 수준 달성[1].  
- 역문제: Navier–Stokes 점유 흐름에서 압력장과 점성 계수 ν, 비선형 항 계수 식별 오차 ≤1% 재현[1].  
- Discrete‐Time: Δt=0.8 단일 스텝으로 Allen–Cahn 상대 L2 오차 ≈7×10⁻³ 달성, 기존 방법보다 10⁶배 시뮬레이션 단계 절감[1].

## 일반화 성능 향상  
- 물리 제약(f PDE 잔차)이 강력한 *정칙화*(regularization) 작용  
- 소수 데이터와 잡음 환경에서도 과적합 억제  
- 물리 법칙에 의한 전역 스무딩 효과로 unseen 영역 예측력 확보[1].

## 한계  
1. **고차원 확장성**: collocation 점 N_f가 차원수 지수적으로 증가  
2. **학습 안정성**: 다항항 손실 조합 시 weight balancing 필요  
3. **로컬 최적해**: PDE 복잡도에 따른 손실풍경(loss landscape) 난해  
4. **불확실성 정량화 미비**: 예측 신뢰구간 산출 체계 부재  
5. **하이퍼파라미터 민감도**: 네트워크 깊이·너비 및 collocation 수에 민감  

# 향후 연구 제언

1. **불확실성 정량화**: Bayesian PINNs, 딥프리어(Deep Prior) 통합  
2. **적응형 샘플링**: 잔차 기반 adaptive collocation으로 고차원 효율화  
3. **하이브리드 메타러닝**: PINNs + MAML/Transfer learning 융합하여 빠른 파라미터 적응  
4. **네트워크 구조 최적화**: 신경진화(NAS), SVD‐PINNs로 경량화·가속화  
5. **경계·초기조건 불확실성**: 강건 경계조건 핸들링 및 로버스트 최적화  

PINNs는 물리 법칙을 직접 학습 프로세스에 통합함으로써 “작은 데이터, 불완전 데이터, 역추정(역문제)” 환경을 모두 아우를 수 있는 범용 툴로 자리매김하였다. 향후 고차원·불확실성 문제와의 결합 및 적응형 학습 전략이 개발된다면, 과학·공학 분야 전반에 걸친 PDE 기반 모델링 패러다임을 획기적으로 혁신할 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f226c0c4-c183-4a3f-b08c-0950e5930db1/1-s2.0-S0021999118307125-am.pdf
[2] https://arxiv.org/abs/2505.03806
[3] https://www.mdpi.com/2075-1680/14/5/385
[4] https://www.semanticscholar.org/paper/15939888f20649f34e8a728b892a034593ee4e5f
[5] https://www.tandfonline.com/doi/full/10.1080/17445302.2024.2344929
[6] https://linkinghub.elsevier.com/retrieve/pii/S0021999118307125
[7] https://pubs.aip.org/pof/article/36/1/013615/3023015/Advancing-fluid-dynamics-simulations-A
[8] https://www.cambridge.org/core/product/identifier/S0962492923000089/type/journal_article
[9] https://pubs.aip.org/pof/article/36/3/036129/3278829/Physics-informed-neural-networks-for-transonic
[10] https://arxiv.org/html/2503.18181v1
[11] https://github.com/maziarraissi/PINNs
[12] https://arxiv.org/html/2501.18879v1
[13] https://neuralfields.cs.brown.edu/paper_4.html
[14] https://openreview.net/forum?id=dY44CURN4v
[15] https://pubmed.ncbi.nlm.nih.gov/37293975/
[16] https://www.sciencedirect.com/science/article/pii/S0021999118307125
[17] https://arxiv.org/html/2501.06572v3
[18] https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125
[19] https://maziarraissi.github.io/PINNs/
[20] https://ojs.aaai.org/index.php/AAAI/article/view/29204/30272
[21] https://www.mdpi.com/2673-2688/5/3/74
[22] https://arxiv.org/html/2408.16806v1
[23] https://nagroup.ewha.ac.kr/physics-informed-machine-learning-for-chemical-reactor/
[24] https://www.sciencedirect.com/science/article/abs/pii/S0925231225008392
[25] https://repository.kisti.re.kr/bitstream/10580/19182/1/KISTI%20%EC%9D%B4%EC%8A%88%EB%B8%8C%EB%A6%AC%ED%94%84%20%EC%A0%9C74%ED%98%B8.pdf
[26] https://arxiv.org/abs/2405.15603
[27] https://arxiv.org/abs/2402.07251
[28] http://arxiv.org/pdf/2309.14722.pdf
[29] https://arxiv.org/pdf/2105.01838.pdf
[30] https://arxiv.org/pdf/2303.07127.pdf
[31] https://arxiv.org/html/2504.00910v1
[32] https://arxiv.org/abs/2109.12754
[33] http://arxiv.org/pdf/1811.04026.pdf
[34] https://pmc.ncbi.nlm.nih.gov/articles/PMC10511457/
[35] https://arxiv.org/html/2410.13228
[36] https://drpress.org/ojs/index.php/ajst/article/download/16292/15810
[37] https://arxiv.org/pdf/2104.02556.pdf
[38] https://towardsdatascience.com/essential-review-papers-on-physics-informed-neural-networks-a-curated-guide-for-practitioners/
[39] https://arxiv.org/abs/1711.10566
[40] https://osf.io/z8pu5/download
