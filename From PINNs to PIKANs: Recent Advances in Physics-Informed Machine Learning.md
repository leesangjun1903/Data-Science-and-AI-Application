# From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning

# 핵심 요약

**주장:**  
본 논문은 2017년 등장한 Physics-Informed Neural Networks(PINNs)의 최근 알고리즘적·응용적 발전을 체계적으로 정리하며, 특히 1957년 Kolmogorov의 함수 분해 정리를 활용한 Physics-Informed Kolmogorov–Arnold Networks(PIKANs)를 새로운 대표 모델로 제시한다.  

**주요 기여:**  
1. Representation Model: 입력/출력 변환, residual 기반 주의(attention), adaptive activation, feature expansion, hard/soft 경계조건 부여 기법  
2. Governing Equation Handling: 자동 미분(AD) 대안 – finite difference, stochastic gradient; variational(vPINNs), fractional(fPINNs), stochastic SDE 확장  
3. Optimization Process:  
   - Domain, time, multi-fidelity, sequential, transfer, curriculum 훈련  
   - Global/local adaptive weight 조정 (self-adaptive, adversarial, RBA)  
   - 샘플링(잔차·고오차 기반 adaptive, 증분·재샘플링)  
4. PIKANs:  
   - Kolmogorov–Arnold Network(KAN) 차원 축소 층 도입  
   - $$z^{(l)}\_i = \Phi_i\bigl(\sum_j\varphi_{ij}(z^{(l-1)}_j)\bigr)$$ 구조로, Chebyshev 다항식 기반 inner–outer 함수로 소수 파라미터로 뛰어난 표현력  
5. 이론·UQ: Neural Tangent Kernel(NTK) 관점의 수렴 분석, Information Bottleneck을 통한 학습 단계(피팅→확산→총확산) 규명, Bayesian/DRO/Ensemble 기반 불확실성 정량화  
6. 응용·SW: 유체·고체역학, 생물·의학, 지구물리·역학, 제어·자율주행, 열전달, 화공, 금융·환경 등 광범위한 사례 소개; DeepXDE, Modulus, SciANN 등 PINN 프레임워크 비교  

# 1. 해결 과제  
- **데이터 동화 한계:** FEM 등 전통적 수치 기법은 희소·잡음 측정치와 물리 모델을 통합하기 어려움  
- **PINNs 한계:** 스펙트럴 바이어스, 경계조건 미강제, 고차 미분 계산 비용, 다중 스케일·고차원 문제 난제, 최적화 불안정성  

# 2. 제안 방법  
## 2.1 Representation Model  
- **입력/출력 변환:**  
  $$\hat u(x) = \Gamma\bigl( M(\theta,\;I(x))\bigr) $$  

  - Periodic: $$I(x)=[\cos\omega x,\sin\omega x,\dots]$$  
  - Dirichlet: $$\Gamma(u)=g(x)+\phi(x)u$$, $$\phi|_{\partial\Omega}=0$$  
- **Feature Expansion:** Fourier, 다항식, Chebyshev 등으로 spectral bias 완화  
- **Residual-based Attention:** 샘플점 마다 local weight $$\lambda_i\propto\mathrm{EMA}(|r_i|)$$로 어려운 영역 집중[Anagnostopoulos et al.]  
- **Adaptive Activation/Weights:**  
  $$\sigma(z)=\sum_i a_i\,\sigma(f_i z)$$, $$W=g\,v/\|v\|$$  
- **PIKANs 구조:**  

$$
    z^{(l)}\_i
    =\Phi_i\Bigl(\sum_{j=1}^H\varphi_{ij}(z^{(l-1)}_j)\Bigr)
  $$
  
  Chebyshev 다항식 기반으로 MLP 대비 파라미터 절감 및 노이즈 강건[Shukla et al.].

## 2.2 Governing Equations  
- **미분:** AD 대안으로 finite difference, Stein’s identity, SDGD(차원별 확률적 경사)  
- **Variational(vPINNs):** 강-약 형태 통합, 다양한 시험함수 적용  
- **Fractional(fPINNs), SDE-GANs:** 비정수·확률미분 방정식 확장  

## 2.3 Optimization  
- **도메인·시간 분할:** XPINNs, cPINNs, soft 분할, time-causal PINNs[Wang & Perdikaris]  
- **Sequential/Curriculum Training:** 낮은 복잡도→높은 복잡도 단계적 학습으로 스펙트럴 바이어스·수렴 개선  
- **Multi-Fidelity/Stacked:** 저·고충실도 데이터·도메인 MPI 병렬 계산, residual 스테이지별 학습[Howard et al.]  
- **Loss Balancing:**  

$$
    L=\sum_\alpha m_\alpha\sum_i\lambda_{\alpha,i}f(r_\alpha(x_i))
  $$
  
  - Global weight $$m_\alpha$$: gradient annealing[Wang et al.], adversarial  
  - Local weight $$\lambda_i$$: self-adaptive, RBA  
- **Sampling:** residual/error 기반 adaptive p(x), 증분 N 점 추가  
- **Optimizer:** Adam⇒L-BFGS 전환; conflict-free updates; 자연그래디언트, Gauss-Newton 등 고차 방법  

# 3. 성능 향상 및 한계  
- **향상:**  
  - **스케일링:** 도메인 분할·멀티 GPU, separable PINNs 최대 60× 속도↑[Cho et al.]  
  - **정확도:** PIKANs 소형 모델로 MLP 대비 5–10% 오차↓, 노이즈 강건  
  - **수렴:** curriculum·self-adaptive weight로 수렴 속도↑, 수렴 실패 감소  
  - **일반화:** NTK·Information Bottleneck으로 학습 단계 파악, 조기 수렴 신호 활용  
- **한계:**  
  - **최적화 불안정성:** 고차·비선형 문제 residual 충돌  
  - **하이퍼파라미터 민감도:** weight·샘플링 전략 설계 필요  
  - **계산 비용:** 초대형 네트워크 &amp; 고차 미분 여전히 무거움  

# 4. 일반화 성능 향상 전략  
1. **Residual-based Attention &amp; Adaptive Sampling**  
   고오차 영역 집중 학습→잦은 overfit 방지  
2. **Curriculum &amp; Transfer Learning**  
   저난도→고난도 시퀀스, 사전학습된 필기학습으로 out-of-distribution 예측력 향상  
3. **Functional Priors (GAN/VAE)**  
   과거·시뮬레이션 모델로 사전 데이터 분포 학습→희소 데이터 일반화 개선[  Meng et al.]  
4. **PIKANs 채용**  
   내재적 차원 축소 구조로 representation bias↓, regularization 효과  

# 5. 전망 및 고려 사항  
- **영향:**  
  - **다중 물리·고차원 문제:** 전통 기법 한계 영역 PINN/PIKAN 대체 가능  
  - **불확실성 정량화:** 안전·의료·금융 등 리스크 관리에 직접 응용  
  - **신 SW 생태계:** DeepXDE→Modulus→NeuralUQ 융합, 자동화된 하이퍼파라미터 탐색 도구 필요  
- **차기 연구:**  
  1. **하이퍼파라미터 자동 튜닝**: adaptive weight·샘플링 동시 최적화  
  2. **하이브리드 FEM–PINN**: 경계 외 전통 FEM, 데이터 영역 PINN 분할  
  3. **스파이킹·에너지 천이**: neuromorphic 칩 최적화 및 자연그래디언트로 계산 비용↓  
  4. **이론 강화:** PINN error bound 고도화, 전체 학습 단계 모니터링 지표 개발  

이 논문은 PINN 및 PIKANs 연구의 핵심 이정표를 제시하며, 차세대 과학기반 AI 연구의 방향을 제시한다. 새 모델 구조·학습 전략은 향후 복합 물리, 고차원·고정밀 응용 연구에 중요한 기반이 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cd78440b-807c-4ae6-b7bc-2eda556f04b2/2410.13228v2.pdf
