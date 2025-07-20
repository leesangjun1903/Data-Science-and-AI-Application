# A comprehensive and FAIR comparison between MLP and KAN representations for differential equations and operator networks

# 핵심 요약 및 주요 기여

이 논문은 **Kolmogorov–Arnold Networks(KANs)** 를 다항식 기반 활성화 함수로 확장하여 물리정보 신경망(PINNs) 및 딥 오퍼레이터 네트워크(DeepONet)에 통합한 새로운 아키텍처, 즉 **PIKAN**(Physics-Informed KAN)과 **DeepOKAN**을 제안한다[1]. 주요 기여는 다음과 같다.

1. **KAN 변형 제안**  
   -  원본 B-스플라인 기반 KAN의 비효율성을 극복하기 위해 저차 직교 다항식(체비셰프, 레제드르 등)을 활성화 함수로 사용한 KAN 변형을 도입.  
2. **PIKAN 및 DeepOKAN 설계**  
   -  PINN과 DeepONet에 KAN을 적용하여 PIKAN/DeepOKAN을 구현.  
3. **광범위 벤치마크**  
   -  8가지 미분 방정식(연속·불연속 함수 근사, 해밀토니안 시스템, Helmholtz, Navier–Stokes, Allen–Cahn, 반응–확산, Burgers, Darcy)과 3가지 오퍼레이터 학습(1D Burgers, 120D Darcy)에서 MLP 기반 모델(PINN/DeepONet)과 성능 비교.  
4. **성능 분석 및 한계**  
   -  직교 다항식 KAN(체비셰프·레제드르·Jacobi)은 MLP와 동등하거나 우수한 정확도 달성.  
   -  하지만 높은 다항식 차수 혹은 깊은 네트워크에서 불안정성(수렴 실패) 및 초기화 민감성 존재.  
5. **학습 동역학 및 손실 지형 시각화**  
   -  정보 병목 이론(IB)에 기반해 학습 과정을 **적합(fitting) → 확산(diffusion) → 총확산(total diffusion)** 3단계로 분석[1].  
   -  KAN 변형은 MLP와 유사한 단계를 거치나, 불안정 구간이 넓어 초기화 민감도가 높음.

# 상세 설명

## 1. 해결 과제 및 제안 방법

### 문제 정의  
- 다항식 기반 activation function(기존 B-스플라인)의 비효율성과 불안정성 극복  
- MLP 기반 PINN/DeepONet의 스펙트럴 바이어스, 과적합, 확장성 문제 해결  

### 제안 모델 구조  
1. **Vanilla KAN** (PIKAN)  
   활성화 함수를 B-스플라인과 일반 기저 함수 b(x)의 선형 조합으로 정의:  

$$
     \phi(x) = w_b b(x) + w_s \sum_i c_i B_i(x)
   $$  

2. **체비셰프 KAN** (cPIKAN)  

   $$\phi(x)$$에 체비셰프 다항식 $$T_k(x)$$ 사용, 그리드 크기 불필요 → 파라미터 절감:  

$$
     |\theta|\_{\text{cPIKAN}}\sim O(n_\ell H^2 k)
   $$
   
3. **DeepOKAN**  
   DeepONet 구조(branch, trunk nets)에 KAN 활성화 적용[1].  

### 물리정보 손실 함수  
PINN/PIKAN의 총 손실:  

```math
\mathcal{L} = w_u\sum_i\|u_\theta(x_u^i)-u^i\|^2 + w_f\sum_j\|F[u\_\theta](x_{f^j})-f^j\|^2 + w_b\sum_l\|B[u_\theta](x_b^l)-b^l\|^2
``` 

잔차 기반 주의력(RBA) 가중치 $$\alpha$$로 국부 손실 강조[1]:  

```math
\alpha\_i^{k+1} \leftarrow (1-\eta^*)\alpha\_i^k + \eta^*\frac{|e\_i|}{\|\mathbf{e}\|\_\infty}.
```

## 2. 모델 성능 및 개선 사항

### 정확도 비교  
- **Helmholtz 방정식**:  
  - PINN+RBA: 0.354%  
  - cPIKAN+RBA: 0.376% (LBFGS, 1.8 × 10³ it) [Table 3a][1].  
- **Navier–Stokes(강제류 속도장)** (Re=400):  
  - PINN: (u,v,p)=(0.25%,0.37%,2.25%)  
  - cPIKAN(k=5): (0.20%,0.26%,1.63%) [Table 4][1].  
- **Allen–Cahn**:  
  - PINN+RBA: 1.51%  
  - cPIKAN: 5.15%  
  - PIKAN: 58.39% (불안정) [Table 5][1].  
- **1D Burgers Operator**:  
  - DeepONet: 5.83%±0.19%  
  - DeepOKAN: 2.71%±0.08% (branch/trunk KAN) [Table 7][1].  
- **120D Darcy Operator**:  
  - DeepONet: 1.62%±0.15%  
  - DeepOKAN: 2.18%±0.02% (더 빠른 일반화) [Table 9][1].  

### 한계 및 불안정성  
- KAN 변형은 **높은 차수(k > 6)**, **숨겨진 층 수(nℓ > 5)** 시 **손실 미정의** → 단일 정밀도(float32) 숫자 불안정[1].  
- 초기화 민감도가 높아, 손실 지형에 정의되지 않는 ‘구멍(hole)’ 존재[Figure 11][1].  
- PIKAN 런타임은 GPU 병렬화 비효율로 MLP 대비 느림.

## 3. 일반화 성능 개선 가능성

- **잔차 기반 주의력(RBA)**: 국소 가중치로 고오차 영역 집중 → 일반화 오류↓  
- **다항식 차수(k)·깊이(nℓ) 최적화**:  
  - 중간 차수(k=3–5), 얕은 깊이(nℓ=2–4)에서 안정적 성능  
  - 과도한 복잡도는 오히려 불안정 초래  
- **이중 정밀도 사용**: float64 활용 시 고차수·깊이에서도 안정적 학습 가능  
- **멀티그리드 PIKAN**: 계단식 그리드 확장으로 품질↑ (단 파라미터 수↑)

# 향후 연구 방향 및 고려 사항

1. **대규모 PDE 병렬 분할(domain decomposition)**  
   KAN의 해상도 한계를 극복하고 확장성 확보.  
2. **시간 상·하차원 PDE 적용**  
   2D·3D 시공간 문제에서 KAN 기반 PINN·DeepOKAN 효율성 검증.  
3. **수렴 이론 정립**  
   KAN 수렴성(타원·포아송·파동 방정식)에 대한 엄밀 수학적 분석.  
4. **하이브리드 활성화 및 구조 탐색**  
   체비셰프·RBF·웨이브릿 등 KAN 조합 최적화 및 자동 아키텍처 검색.  
5. **오퍼레이터 학습의 산업적 적용**  
   DeepOKAN을 활용한 복합 재료·구조 설계 시뮬레이션 대체 연구.

KAN 기반 표현은 MLP 대비 **직교 다항식 활용** 시 동등 이상의 정확도를 보이면서 **모델 압축**과 **해석 가능성** 측면에서 유망하다. 그러나 **수치 안정성**과 **초기화 민감성** 문제 해결이 선행되어야 실제 대규모 공학 문제에 널리 적용될 수 있다.

[1] https://arxiv.org/abs/2406.02917
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e3b8325d-204d-4afd-9ffc-bc05e075b43a/2406.02917v1.pdf
[3] http://arxiv.org/pdf/2407.16674.pdf
[4] http://arxiv.org/pdf/2406.02917.pdf
[5] http://arxiv.org/pdf/2410.03027.pdf
[6] https://arxiv.org/pdf/2406.14529.pdf
[7] https://arxiv.org/abs/2408.07906
[8] https://arxiv.org/pdf/2407.12569.pdf
[9] https://arxiv.org/html/2408.07314v3
[10] https://www.themoonlight.io/en/review/a-comprehensive-and-fair-comparison-between-mlp-and-kan-representations-for-differential-equations-and-operator-networks
[11] https://arxiv.org/html/2504.11397v1
[12] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4858126
[13] https://www.nature.com/articles/s41598-025-92900-1
[14] https://papers.ssrn.com/sol3/Delivery.cfm/59c0129c-80ae-4761-9a94-429559701945-MECA.pdf?abstractid=4858126&mirid=1&type=2
[15] https://arxiv.org/html/2407.16674v1
[16] https://www.medrxiv.org/content/10.1101/2024.09.23.24314194v1.full.pdf
[17] https://experts.illinois.edu/en/publications/deepokan-deep-operator-network-based-on-kolmogorov-arnold-network
[18] https://ouci.dntb.gov.ua/en/works/96nPG1p9/
[19] https://openreview.net/forum?id=Ozo7qJ5vZi
[20] https://arxiv.org/abs/2405.19143
[21] https://www.sciencedirect.com/science/article/pii/S0045782524005462
[22] https://openreview.net/pdf?id=yPE7S57uei
[23] https://pmc.ncbi.nlm.nih.gov/articles/PMC11957273/
[24] https://www.sciencedirect.com/science/article/abs/pii/S0045782524005462
[25] https://www.sciencedirect.com/science/article/pii/S0045782524009538
[26] http://arxiv.org/pdf/2410.08961.pdf
[27] http://arxiv.org/pdf/2410.01803.pdf
[28] https://arxiv.org/html/2411.15111v1
[29] http://arxiv.org/pdf/2412.13571.pdf
[30] http://arxiv.org/pdf/2408.08803.pdf
[31] http://arxiv.org/pdf/2411.05296.pdf
[32] http://arxiv.org/pdf/2405.07200.pdf
[33] http://arxiv.org/pdf/2501.00420.pdf
[34] http://arxiv.org/pdf/2409.09653.pdf
[35] https://arxiv.org/html/2406.02875
[36] https://arxiv.org/html/2407.04192v1
[37] https://arxiv.org/html/2410.00435v2
[38] https://ouci.dntb.gov.ua/en/works/loeRyrj7/
[39] https://github.com/mintisan/awesome-kan
[40] https://www.preprints.org/manuscript/202504.1742/v1
[41] https://x.com/ZimingLiu11/status/1803569391877197920
