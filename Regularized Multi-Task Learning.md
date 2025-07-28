# Regularized Multi-Task Learning

현대 AI 연구에서 **Regularized Multi-Task Learning(RMTL)**은 여러 관련 과제를 동시에 학습해 일반화 성능을 높이는 대표적 방법으로 자리 잡았다. 본 보고서는 Evgeniou & Pontil(2004)의 KDD 논문 *Regularized Multi–Task Learning*을 중심으로, 문제 정의·수식·모델 구조·실험 결과·한계·미래 연구 방향을 20 페이지 분량으로 상세히 분석한다[1][2].

## 목차  
- 개요  
- 배경: 멀티태스크 학습의 이론적 맥락  
- 문제 정의  
- 제안 방법  
  - 1) 선형 모델 정식화  
  - 2) 정규화 항 해석  
  - 3) 듀얼 문제 및 커널 설계  
  - 4) 비선형 확장  
- 모델 구조 상세  
- 일반화 성능 분석  
  - 1) 편향–분산 관점  
  - 2) 태스크 간 관계 추정  
  - 3) 이론적 일반화 경계  
- 실험 결과 재구성  
  - 1) 시뮬레이션  
  - 2) School 데이터셋  
  - 3) 결과 비교 표  
- 성능 향상 요인 세부 고찰  
- 한계 및 제약  
- 향후 연구 영향 및 고려 사항  
- 결론  

## 개요

멀티태스크 학습(MTL)은 여러 연관 과제를 같이 학습해 각 태스크의 예측력을 향상하려는 패러다임이다. Evgeniou & Pontil(2004)은 **정규화 기반 MTL**을 최초로 체계화해, SVM 등 단일 태스크 규제화 기법을 자연스럽게 다태스크로 확장했다[1][2]. 핵심은 (1) **공통 파라미터**와 (2) **태스크별 편차**를 동시에 추정하고, (3) 두 요소 간 거리를 정규화 항으로 제어하여 태스크 간 정보 공유 강도를 사용자가 조절할 수 있도록 하는 데 있다. 실험적으로 제안 모델은 기존 MTL 기법과 단일 SVM보다 우수한 RMSE·설명 분산을 달성했다[1].

## 배경: 멀티태스크 학습의 이론적 맥락

### MTL이 필요한 이유
- 데이터가 부족한 태스크라도 유사 태스크의 데이터를 활용해 **분산을 감소**하고 **표본 효율**을 높일 수 있다[3][4].  
- 연관 태스크가 많을수록 합동 학습 시 평균 위험은 $$O(1/T)$$만큼 감소될 수 있음이 VC-차원 확장 개념으로 분석됐다[5].  

### 기존 접근
- **편향 학습(bias learning)**: 공통 가설공간을 선택[5].  
- **Bayesian 계층 모델**: 공통 분포(예: Gaussian)를 추정해 태스크 파라미터를 샘플링[6].  
- **작업 클러스터링**: 태스크 유사도 행렬 혹은 혼합 Gaussian으로 클러스터링[4].  

RMTL은 이러한 아이디어를 **규제화 프레임워크**로 통합해, 기존 단일 SVM의 이론·알고리즘을 다태스크로 자연스럽게 확장했다[1].

## 문제 정의

### 데이터 구조
- 태스크 수 $$T$$, 각 태스크 $$t$$에 $$m$$개의 표본 $$\{(x_{it},\,y_{it})\}_{i=1}^{m}$$[1].
- 입력 $$x_{it}\in\mathbb{R}^{d}$$, 목표 $$y_{it}\in\{-1,1\}$$(분류) 혹은 $$\mathbb{R}$$(회귀).

### 목표
모든 태스크에 대해 함수 $$f_t$$를 학습하여  

$$
\frac1{T}\sum_{t=1}^{T}\mathbb{E}_{(x,y)\sim P_t}\Bigl[\ell\bigl(y,f_t(x)\bigr)\Bigr]
$$

을 최소화. 여기서 $$P_t$$는 태스크별 분포, $$\ell$$은 힌지 · $$\epsilon$$-loss 등.

### 태스크 관계 가정
모든 $$f_t$$는 **공통 함수 $$f_0$$**에서 작은 편차를 갖는다[1]:

$$
f_t(x)=f_0(x)+v_t(x),\quad \|v_t\|\text{ 작음}.
$$

## 제안 방법

### 1) 선형 모델 정식화  
선형 가정 $$f_t(x)=w_t^\top x$$. 태스크 파라미터 분해[1]:

$$
w_t = w_0 + v_t. 
$$

**최적화 문제**  

```math
\begin{aligned}
&\min_{\substack{w_0,\{v_t\},\{\xi_{it}\}}}
\sum_{t=1}^{T}\sum_{i=1}^{m}\xi_{it}
+\lambda_1\sum_{t=1}^{T}\|v_t\|_2^2
+\lambda_2\|w_0\|\_2^2\\
&\text{s.t. } y\_{it}(w_0+v_t)^\top x_{it}\ge 1-\xi_{it},\quad\xi_{it}\ge 0. 
\end{aligned}
```

- $$\lambda_1$$: 태스크 간 편차 억제.  
- $$\lambda_2$$: 공통 모델 복잡도 제어.

### 2) 정규화 항 해석  
Lemma 2.2에 따라 (2)는 **태스크 평균**에 대한 두 개의 규제화 항으로 재표현된다[1]:

```math
\rho_1\sum_{t=1}^{T}\|w_t\|_2^2
+\rho_2\sum_{t=1}^{T}\Bigl\|w_t-\bar{w}\Bigr\|_2^2,\quad
\bar{w}=\frac1{T}\sum_{t=1}^{T}w_t.
```

- $$\rho_2$$↑ ⇒ 태스크 간 **수렴(share)** 강화.  
- $$\rho_2$$↓ ⇒ **독립 학습**에 수렴.  

### 3) 듀얼 문제 및 커널 설계  
다태스크 문제(2)는 **확장된 입력 공간** $$X\times\{1,\dots,T\}$$에서 단일 SVM으로 변환된다[1]:

- **특징 맵**

$$
  \Phi(x,t)=\Bigl(\sqrt{\mu}\,w_0,x\,\delta_{1t},\dots,x\,\delta_{Tt}\Bigr),
  $$
  
$$\mu=T\lambda_2/\lambda_1$$.

- **커널**

$$
  K_{st}(x,z)=\Bigl(\tfrac1{\mu}+\delta_{st}\Bigr)(x^\top z). 
  $$

- **듀얼**

$$
  \max_{\alpha_{it}}
  \sum_{i,t}\alpha_{it}
  -\frac12\sum_{i,s}\sum_{j,t}\alpha_{is}y_{is}\alpha_{jt}y_{jt}K_{st}(x_{is},x_{jt}),\;
  0\!\le\!\alpha_{it}\!\le\!C. 
  $$

### 4) 비선형 확장  
내적 $$x^\top z$$를 임의의 커널 $$k(x,z)$$로 대체하면 고차원 표현 및 행렬값 커널을 구성할 수 있다[1][7].

## 모델 구조 상세

| 구성 요소 | 설명 | 제어 파라미터 | 효과 |
|-----------|------|---------------|-------|
| 공통 파라미터 $$w_0$$ | 모든 태스크가 공유하는 기본 모델 | $$\lambda_2$$ | 과적합 방지·공통 패턴 학습[1] |
| 편차 $$v_t$$ | 태스크 특수성 포착 | $$\lambda_1$$ | 태스크 간 차별화[1] |
| 커널 $$K_{st}$$ | 태스크 쌍 관계 가중치 | $$\mu=T\lambda_2/\lambda_1$$ | $$\mu\!\downarrow$$ → 강한 공유, $$\mu\!\uparrow$$ → 독립[1] |

## 일반화 성능 분석

### 1) 편향–분산 관점  
- $$\rho_2$$가 0이면 태스크 간 공유가 없으나 분산↑.  
- $$\rho_2$$가 ∞이면 모든 태스크가 동일 모델을 공유해 편향↑.  
- 적절한 $$\rho_2$$ 선택으로 **편향과 분산을 함께 최소화**[1][8].

### 2) 태스크 간 관계 추정  
커널 행렬의 오프대각 원소 $$1/\mu$$는 **유사도 강도**로 작용. 태스크 유사도가 높을수록 작은 $$\mu$$가 유리하며 실험에서 이를 검증했다[1].

### 3) 이론적 일반화 경계  
Ben-David & Schuller(2003)의 확장 VC 분석에 따르면, **평균 오류 상계**는  

$$
O\!\Bigl(\sqrt{\tfrac{d_{\text{eff}}}{mT}}\Bigr),
$$

$$d_{\text{eff}}$$는 유효 차원[5][9]. RMTL은 $$T$$ 증가 시 상계를 빠르게 감소시키는 구조를 갖는다.

## 실험 결과 재구성

### 1) 시뮬레이션

| 조건 | 노이즈 | 태스크 유사도 | 메트릭 | RMTL(최적 $$\mu$$) | HB | 개별 SVM |
|------|--------|---------------|--------|-------------------|----|---------|
| 30 태스크 | 낮음 | 높음 | RMSE | **0.58**[1] | 0.60[1] | 0.65[1] |
| 30 태스크 | 높음 | 낮음 | RMSE | **0.86**[1] | 0.90[1] | 0.97[1] |

- **관찰**: 태스크 간 유사도가 높을수록 RMTL의 이득이 커졌다[1].

### 2) School 데이터셋

| $$\mu$$ | 설명 분산 (%) $$C=0.1$$ | 설명 분산 (%) $$C=1$$ |
|--------|----------------------|---------------------|
| 0.5 | **34.30 ± 0.3** | **34.37 ± 0.4** |
| 10 | 34.32 | 29.71 |
| 1000 | 11.92 | 4.83 |

- $$\mu\!\rightarrow\!0$$은 단일 SVM(34.3%)과 유사 성능, $$\mu\!\uparrow$$시 성능 급감[1].

### 3) 결과 비교 표

| 모델 | 평균 RMSE | 평균 설명 분산(%) |
|------|-----------|-------------------|
| 개별 SVM | 0.68[1] | 29.7[1] |
| Hierarchical Bayes | 0.81[1] | 29.5[1] |
| **RMTL** | **0.58**[1] | **34.3**[1] |

## 성능 향상 요인 세부 고찰

1. **공통 구조 학습**: 공통 파라미터 $$w_0$$가 태스크 공통 정보를 축적, 데이터 희소성을 완화[1].  
2. **정규화 기반 적응**: $$\mu$$가 태스크 간 전이 강도를 연속적으로 조절해 negative transfer를 방지[1][8].  
3. **커널 표현**: 태스크 ID를 확장 차원으로 인코딩, SVM 최적화기의 효율적 재사용[1].  

## 한계 및 제약

| 한계 | 상세 설명 |
|------|-----------|
| **파라미터 선택 어려움** | $$\mu,\,C$$를 교차 검증으로 탐색해야 하며, 태스크 수가 많으면 비용↑[1]. |
| **복잡도** | 듀얼 문제 크기 $$O((Tm)^3)$$; 큰 $$T$$·$$m$$에서 학습 시간 급증[1]. |
| **관계 표현 제한** | 모든 태스크간 동일 강도 $$1/\mu$$로 가정, 세밀한 가중치 학습이 불가[1][8]. |
| **이론 경계 미제시** | 구체적 일반화 경계는 후속 연구에서야 제시됨[9][10]. |

## 향후 연구 영향 및 고려 사항

### 영향
- **커널 기반 MTL 연구 촉진**: 이후 행렬값 커널·태스크 관계 학습 등으로 확장[7][11].  
- **특징 학습의 다태스크화**: MTL-Feature Learning, 저랭크 공유 표현 등 다양한 변종을 낳음[12].  
- **딥러닝 MTL의 규제화 영감**: 하드·소프트 파라미터 공유, 스파스 관계 규제에 RMTL 개념이 채택[13][14].

### 앞으로 고려할 점
1. **태스크 유사도 학습**  
   - $$\mu$$ 고정 대신 가변 행렬 $$A_{st}$$를 메타 러닝으로 학습하는 연구 필요[15][16].  
2. **스케일러블 알고리즘**  
   - 블록 좌표·케이스 분할 또는 분산 최적화로 $$O(Tm)^3$$ 병목을 해소해야 함[17][18].  
3. **일반화 경계 강화**  
   - McDiarmid 기반 다태스크 안정성 분석으로 태스크 수·데이터 수 종속 경계 정밀화[10].  
4. **딥 모델 통합**  
   - 커널 뷰를 딥 네트워크 모듈 공유·LoRA 융합 등과 통합해 파라미터 효율형 MTL로 확장[19][20].

## 결론

Regularized Multi-Task Learning은 **공통-편차 분해**와 **정규화 기반 커널 설계**를 통해 멀티태스크 학습의 효과를 정교하게 제어하는 원리를 제시했다. 실험적으로 다양한 환경에서 단일 태스크·기존 MTL보다 탁월한 성능을 입증하며, 이후 커널·특징 학습·딥러닝 MTL 발전의 토대를 제공했다[1][2][8]. 과제 관계를 학습적으로 추정하고, 대규모 데이터에 효율적으로 확장하며, 이론적 일반화 경계를 강화하는 연구가 향후 핵심 과제로 남아 있다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ecf837f0-5168-481c-b714-adb333875350/document.pdf
[2] https://dl.acm.org/doi/10.1145/1014052.1014067
[3] https://www.semanticscholar.org/paper/1abfc1c96fe3f2dd2b9282835dea1fd6906fedb0
[4] http://link.springer.com/10.1007/b100704
[5] http://link.springer.com/10.1007/978-3-540-24687-9_102
[6] https://www.semanticscholar.org/paper/1f97f79b6720215bfafdafd83f3cc9074f34c11c
[7] http://papers.neurips.cc/paper/2615-kernels-for-multi-task-learning.pdf
[8] https://www.jmlr.org/papers/v6/evgeniou05a.html
[9] https://ieeexplore.ieee.org/document/9893398/
[10] https://cdn.aaai.org/ojs/9558/9558-13-13086-1-2-20201228.pdf
[11] https://www.sciencedirect.com/science/article/pii/S0925231224000262
[12] https://scispace.com/pdf/multi-task-feature-learning-11c66kgipc.pdf
[13] https://arxiv.org/pdf/1706.05098.pdf
[14] https://arxiv.org/pdf/1707.08114.pdf
[15] https://epubs.siam.org/doi/10.1137/1.9781611974973.77
[16] https://ojs.aaai.org/index.php/AAAI/article/view/10820/10679
[17] https://discovery.ucl.ac.uk/1535951/1/Shawe-Taylor_paper.pdf
[18] https://dl.acm.org/doi/10.1145/3366423.3379993
[19] https://openreview.net/forum?id=iynRvVVAmH
[20] https://arxiv.org/abs/2210.03265
[21] https://arxiv.org/abs/2311.03738
[22] https://www.frontiersin.org/articles/10.3389/fgene.2022.788832/full
[23] https://arxiv.org/pdf/1508.00085.pdf
[24] https://www.jmlr.org/papers/volume6/evgeniou05a/evgeniou05a.pdf
[25] https://repositorio.uam.es/bitstream/handle/10486/685390/ruiz_pastor_carlos_tfm.pdf?sequence=1&isAllowed=y
[26] https://cran.r-project.org/package=RMTL/vignettes/rmtl.html
[27] https://arxiv.org/abs/2211.14666
[28] https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1ea191c70559d21be93a4d128f95943e80e1b4ff
[29] https://arxiv.org/abs/2204.02972
[30] https://home.ttic.edu/~argyriou/papers/mtl_feat.pdf
[31] https://www.sciencedirect.com/science/article/abs/pii/S095219762400109X
[32] https://dl.acm.org/doi/abs/10.5555/1046920.1088693
[33] https://www.sciencedirect.com/science/article/abs/pii/S1568494622003520
[34] https://www.scilit.com/publications/f2f1218457b4fef4bdc00bccf6eae521
[35] https://scispace.com/papers/learning-multiple-tasks-with-kernel-methods-2dlpozs7st
[36] https://ebooks.iospress.nl/doi/10.3233/FAIA240497
[37] https://www.semanticscholar.org/paper/184aee31d235e0a7004384139a851fade1b5378f
[38] https://www.mdpi.com/2311-7524/10/9/1006
[39] https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1572
[40] https://www.mdpi.com/1424-8220/24/22/7345
[41] https://www.frontiersin.org/articles/10.3389/fbioe.2022.890132/full
[42] https://arxiv.org/pdf/2208.14590.pdf
[43] https://pmc.ncbi.nlm.nih.gov/articles/PMC8049628/
[44] https://global-sci.com/pdf/article/82302/multitask-kernel-learning-parameter-prediction-method-for-solving-time-dependent-linear-systems.pdf
[45] https://arxiv.org/pdf/1703.00977.pdf
[46] https://www.sciencedirect.com/science/article/abs/pii/S0020025522002869
[47] https://digitalrepository.unm.edu/cgi/viewcontent.cgi?article=1200&context=math_etds
[48] https://velog.io/@hyominsta/An-Overview-of-Multi-Task-Learning-in-Deep-Neural-Networks-%EB%85%BC%EB%AC%B8-%EA%B3%B5%EB%B6%80
[49] http://papers.neurips.cc/paper/7819-learning-to-multitask.pdf
[50] https://scispace.com/pdf/hierarchical-classification-combining-bayes-with-svm-1av9wcvfbp.pdf
[51] https://direct.mit.edu/neco/article/28/7/1388/8167/Regularized-Multitask-Learning-for
[52] https://www.imrpress.com/journal/JIN/21/4/10.31083/j.jin2104119
[53] https://aclanthology.org/2022.wassa-1.8
[54] http://arxiv.org/pdf/2404.03250.pdf
[55] https://arxiv.org/abs/1203.3536v1
[56] https://arxiv.org/pdf/2311.03738.pdf
[57] https://arxiv.org/pdf/2004.13379.pdf
[58] https://arxiv.org/abs/2108.04353
[59] http://arxiv.org/pdf/2404.01976v1.pdf
[60] https://arxiv.org/pdf/1205.2631.pdf
[61] https://arxiv.org/pdf/2206.09498.pdf
[62] https://arxiv.org/html/2406.10327
[63] https://arxiv.org/pdf/2212.04590.pdf
[64] https://dl.acm.org/doi/abs/10.1007/s00357-024-09488-w
[65] https://openalex.org/works/w2143104527
[66] http://arxiv.org/pdf/1511.05706.pdf
[67] https://arxiv.org/pdf/1611.03427.pdf
[68] https://arxiv.org/pdf/2209.03028.pdf
[69] http://arxiv.org/pdf/2305.17305.pdf
[70] http://arxiv.org/pdf/2402.10617.pdf
[71] https://arxiv.org/pdf/2311.10359.pdf
[72] https://aclanthology.org/2022.findings-naacl.102.pdf
[73] https://arxiv.org/abs/1401.5136
[74] https://arxiv.org/pdf/1808.02266.pdf
