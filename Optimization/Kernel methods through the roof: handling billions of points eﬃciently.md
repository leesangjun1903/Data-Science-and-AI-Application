# Kernel Methods Through the Roof: Handling Billions of Points Efficiently

---

## 1. Executive Summary (10문장 이내)

커널 방법(Kernel Methods)은 비모수적 학습의 강력한 이론적 토대를 제공하지만, 커널 행렬의 크기가 데이터 수 $n$의 제곱에 비례( $\mathcal{O}(n^2)$ )하여 대규모 데이터셋에 적용이 어려웠다.  
본 논문은 **Falkon** 및 **LogFalkon** 라이브러리를 통해 수십억 개의 데이터 포인트를 효율적으로 처리하는 GPU 최적화 커널 솔버를 제안한다.  
핵심 알고리즘은 Nyström 근사와 사전조건화 켤레기울기(Preconditioned Conjugate Gradient, PCG) 방법을 결합한 것으로, 최적 통계적 성능을 달성하는 데 $\mathcal{O}(n\sqrt{n}\log n)$의 시간과 $\mathcal{O}(n)$의 메모리만을 필요로 한다.  
GPU 하드웨어의 한계(제한된 메모리, 낮은 연산 밀도)를 극복하기 위해 out-of-core 연산, 다중 GPU 병렬화, 혼합 수치 정밀도, 메모리 전송-연산 파이프라이닝을 적용하였다.  
그 결과 기존 Falkon 기준 대비 약 **20배**의 속도 향상을 달성하였다.  
10억 개 포인트(TAXI 데이터셋)를 약 1시간 이내에 처리하며, EigenPro 대비 약 6배, GPyTorch 대비 10배 이상 빠른 수렴 속도를 보인다.  
로지스틱 손실을 위한 LogFalkon은 이진 분류에서 소폭이지만 일관된 정확도 향상을 제공한다.  
본 연구는 커널 방법이 딥러닝 경쟁 도구로서 대규모 실전 문제에서도 충분히 활용 가능함을 실험적으로 입증한다.  
코드는 PyTorch 기반의 오픈소스 라이브러리(https://github.com/FalkonML/falkon)로 공개되어 있다.

### 1-1. 연구의 목적과 필요성

**목적:** 커널 방법의 이론적 장점(볼록 최적화, 통계적 보장, 해석 가능성)을 대규모 데이터($n \sim 10^9$)에서 실용적으로 활용 가능하도록 GPU 완전 활용 솔버를 개발.

**필요성:**
- 기존 커널 방법은 $\mathcal{O}(n^2)$ 메모리와 $\mathcal{O}(n^3)$ 연산이 필요하여 $n \gtrsim 10^5$ 이상에서 실용적 적용이 불가능 (p.2)
- 딥러닝과의 이론적 연결(NTK 등)로 커널 방법의 중요성이 재조명되고 있음
- 기존 GPU 기반 솔버(EigenPro, GPyTorch, GPflow)는 메모리 부족 또는 확장성 한계로 수억~수십억 규모에서 작동 불가

> 💡 **커널 방법(Kernel Methods):** 데이터를 고차원 특징 공간에 비선형 매핑 후 그 공간에서 선형 모델을 학습하는 방법. 내적 연산을 커널 함수 $k(x,x')$로 대체하여 무한 차원도 다룰 수 있음.

> 💡 **Nyström 근사:** 전체 $n \times n$ 커널 행렬 대신 $m$개의 대표점(유도점, inducing points)으로 구성된 $m \times m$ 부분 행렬로 근사하는 방법. $m \ll n$으로 설정하여 계산 비용을 대폭 감소.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| Nyström + PCG로 최적 통계적 성능 유지 가능 | $m = \mathcal{O}(\sqrt{n})$개의 유도점으로도 $\mathcal{O}(n^{-1/2})$ 오차 보장 이론 | p.4, Eq.(3) |
| GPU 최적화로 20× 속도 향상 달성 | Float32 정밀도, GPU 전처리기, 2-GPU, KeOps 등 단계별 개선 측정 | p.10, Table 1 |
| EigenPro 대비 6×, GPyTorch 대비 10× 이상 빠름 | 동일 하드웨어에서 6개 데이터셋 비교 실험 | p.12, Table 2 |
| 10억 포인트 데이터를 수분~1시간 내 처리 | TAXI($n=10^9$) 3628초 달성 | p.12, Table 2 |
| 분산 시스템(28,000 CPU) 대비 동등 정확도를 훨씬 적은 자원으로 달성 | TAXI: 28,000 vCPU 6,000s vs. 2-GPU 3,628s | p.12, A.6 |
| LogFalkon이 이진 분류에서 일관된 정확도 향상 제공 | SUSY, HIGGS에서 소폭 오차 감소 확인 | p.12, Table 2 |
| Out-of-core 연산으로 GPU 메모리 제약 극복 | 표준 ML 라이브러리(PyTorch, TF)에 없는 OOC 연산 직접 구현 | p.8, Sec. 3.2 |

---

## 2-1. 세부 기술 설명

### 해결하고자 하는 문제

커널 방법의 두 가지 주요 병목:
1. **메모리 병목:** $n \times n$ 커널 행렬 $K_{nn}$을 저장하는 데 $\mathcal{O}(n^2)$ 메모리 필요
2. **계산 병목:** 직접 풀이에 $\mathcal{O}(n^3)$ 시간 필요

### 제안하는 방법

#### (A) 정규화 경험적 위험 최소화 (p.4, Eq.2)

$$\hat{f}_\lambda = \arg\min_{f \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \ell\bigl(f(x_i), y_i\bigr) + \lambda \|f\|^2_{\mathcal{H}}$$

- $\mathcal{H}$: 재현 커널 힐베르트 공간(RKHS)
- $\ell(\cdot, \cdot)$: 손실 함수 (제곱 손실 또는 로지스틱 손실)
- $\lambda \geq 0$: 정규화 하이퍼파라미터
- $\|f\|_{\mathcal{H}}$: RKHS 노름 (과적합 방지)

> 💡 **재현 커널 힐베르트 공간(RKHS):** 커널 함수 $k(x,x')$에 의해 정의되는 함수 공간. 이 공간에서의 함수는 $f(x) = \sum_i \alpha_i k(x, x_i)$ 형태로 표현 가능(표현자 정리).

#### (B) 통계적 수렴 보장 (p.4, Eq.3)

$$L(\hat{f}_\lambda) - \inf_{f \in \mathcal{H}} L(f) = \mathcal{O}\!\left(n^{-1/2}\right)$$

- $L(f)$: 기대 손실(expected loss)
- $\lambda = \mathcal{O}(1/\sqrt{n})$ 설정 시 높은 확률로 성립

#### (C) Nyström 근사 함수 표현 (p.4, Eq.4)

$$f(x) = \sum_{i=1}^m \alpha_i k(x, \tilde{x}_i)$$

- $\{\tilde{x}_1, \ldots, \tilde{x}_m\} \subset \{x_1, \ldots, x_n\}$: 균일 랜덤 샘플링된 **유도점(inducing points)**
- $m = \mathcal{O}(\sqrt{n})$으로 Eq.(3)의 통계적 보장 유지 가능

#### (D) Nyström 근사 적용 선형 시스템 (p.4, Eq.5)

$$\left(K_{nm}^\top K_{nm} + \lambda n K_{mm}\right)\boldsymbol{\alpha} = K_{nm}^\top \mathbf{y}$$

- $K_{nm} \in \mathbb{R}^{n \times m}$: 전체 데이터와 유도점 간의 커널 행렬, $(K_{nm})_{ij} = k(x_i, \tilde{x}_j)$
- $K_{mm} \in \mathbb{R}^{m \times m}$: 유도점들 간의 커널 행렬
- $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_m) \in \mathbb{R}^m$: 계수 벡터
- $\mathbf{y} = (y_1, \ldots, y_n)$: 타겟 벡터

#### (E) Nyström 사전조건자 (p.5, Eq.6)

$$\tilde{P}\tilde{P}^\top = \left(\frac{n}{m}K_{mm}^2 + \lambda n K_{mm}\right)^{-1}$$

- $K_{mm}^2 \approx K_{nm}^\top K_{nm}$를 이용한 근사
- Cholesky 분해를 통해 두 삼각 행렬로 분해:

$$\tilde{P} = \frac{1}{\sqrt{n}} T^{-1} A^{-1}, \quad T = \text{chol}(K_{mm}), \quad A = \text{chol}\!\left(\frac{1}{m}TT^\top + \lambda I_m\right)$$

> 💡 **Cholesky 분해:** 양정치(positive definite) 행렬 $M$을 하삼각 행렬 $L$로 $M = LL^\top$으로 분해하는 방법. 선형 시스템 풀이에 $\mathcal{O}(m^3)$ 소요.

> 💡 **사전조건화(Preconditioning):** 켤레기울기법의 수렴을 가속하기 위해 원래 시스템을 변환하는 방법. 좋은 사전조건자는 변환된 행렬의 조건수를 크게 줄여 적은 반복으로 수렴.

#### (F) 사전조건화 선형 연산자 (p.7, Eq.8-9)

$$\tilde{P}^\top H \tilde{P} \beta = (A^{-1})^\top (T^{-1})^\top (K_{nm}^\top K_{nm} + \lambda n K_{mm}) T^{-1} A^{-1} \boldsymbol{\beta}$$

$$= (A^{-1})^\top \left[(T^{-1})^\top K_{nm}^\top K_{nm} T^{-1} + \lambda n I\right] A^{-1} \boldsymbol{\beta}$$

- $K_{mm}$ 저장이 불필요: $(T^{-1})^\top K_{mm} T^{-1} = I$ ($K_{mm} = T^\top T$이므로)
- 단일 $m \times m$ 행렬만 메모리에 유지하면 됨

> 💡 **켤레기울기법(Conjugate Gradient, CG):** 양정치 대칭 선형 시스템 $Ax = b$를 반복적으로 푸는 방법. $k$번 반복 후 $k$차원 Krylov 부공간에서 최적해를 구함.

### 모델 구조 (Algorithm 1: Falkon)

```
입력: X ∈ ℝⁿˣᵈ, y ∈ ℝⁿ, λ (정규화), m (유도점 수), t (CG 반복 수)
1. Xₘ ← X에서 m개 균일 랜덤 샘플링
2. T, A ← Preconditioner(Xₘ, λ)  [Cholesky 기반]
3. LinOp 정의: Eq.(8-9)의 행렬-벡터 곱
4. R ← A⁻ᵀT⁻ᵀk(X, Xₘ)y
5. β ← ConjugateGradient(LinOp, R, t)
6. 반환: T⁻¹A⁻¹β
```

**계산 복잡도:** $\mathcal{O}(n\sqrt{n}\log n)$ 시간, $\mathcal{O}(n)$ 메모리

### GPU 최적화 핵심 구성요소

| 최적화 기법 | 설명 | 효과 |
|------------|------|------|
| **Float32 정밀도** | 32비트 부동소수점 사용 (단, $K_{mm}$ 계산 시 64비트 중간 변환) | 3× 속도 향상 (GPU 부분) |
| **GPU 전처리기** | Cholesky 분해 및 LAUUM 연산을 GPU에서 수행 | 전처리기 7.3× 향상 |
| **다중 GPU 병렬화** | 1D 블록-순환 방식으로 행렬 타일을 GPU에 분배 | CG 반복 1.9× 향상 |
| **KeOps 통합** | 저차원 데이터($d$ 작을 때) 커널 행렬-벡터 곱 특화 라이브러리 사용 | CG 반복 3× 향상 |
| **Out-of-Core(OOC) 연산** | GPU 메모리 초과 행렬을 타일 단위로 분할 처리 | 대용량 전처리기 가능 |
| **메모리 전송-연산 파이프라이닝** | CPU→GPU, 연산, GPU→CPU를 3개 스레드로 동시 실행 | $t$번 연산을 $t+2$ 시간 단위에 완료 |

### 성능 향상 및 한계

**성능 향상 (Table 1, p.11):**

| 최적화 단계 | 전처리기 향상 | CG 반복 향상 |
|------------|------------|------------|
| 기준선 [42] | 1× | 1× |
| Float32 | 1.8× | 3× |
| GPU 전처리기 | 7.3× | 1.1× |
| 2 GPUs | 1.5× | 1.9× |
| KeOps | 1× | 3× |
| **전체** | **19.7×** | **18.8×** |

**한계:**
- 단일 머신, 단일 커널(가우시안) 기준 비교 (분산 학습 미지원)
- 하이퍼파라미터(길이 척도, 정규화)를 수동 튜닝 필요 (GP 솔버는 자동 튜닝)
- Float32 사용으로 HIGGS에서 미세한 정확도 저하 관찰 (p.12)
- YELP 데이터셋은 어떤 비교 알고리즘도 처리 불가 (단독 결과)
- 멀티클래스 LogFalkon 미구현

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| 커널 행렬 $\mathcal{O}(n^2)$ 메모리 병목 | p.2, p.4 |
| Nyström $m=\mathcal{O}(\sqrt{n})$으로 통계 보장 유지 | p.4, Eq.(3), [41] |
| 전체 알고리즘 $\mathcal{O}(n\sqrt{n}\log n)$ 복잡도 | p.5 |
| 20× 전체 속도 향상 | p.10 (A.1), Table 1 |
| GPU 메모리 구조 및 메모리 할당 | Figure 2 (p.6) |
| 메모리 전송-연산 파이프라이닝 | Figure 3 (p.6) |
| 전처리기 행렬 in-place 진화 | Figure 4 (p.7) |
| 다중 GPU Cholesky 분해 | Figure 5 (p.8) |
| 다중 GPU 확장성 | Figure 6 (p.21), Table 4 (p.26) |
| 종합 벤치마크 결과 | Figure 1 (p.2), Table 2 (p.13) |
| 문헌 결과 비교 | Table 6 (p.28), Appendix A.6 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**연구 주제 (p.1, Abstract):**
> "우리는 GPU 하드웨어를 완전히 활용하는 솔버를 개발하고 테스트한다."

**방법:**
- Nyström 근사 + 사전조건화 CG (Algorithm 1)
- 총 복잡도: 시간 $\mathcal{O}(n\sqrt{n}\log n)$, 공간 $\mathcal{O}(n)$ (p.5)

**저자 직접 보고 성능 수치 (Table 2, p.13):**
- TAXI ($n=10^9$): RMSE $311.7 \pm 0.1$, 시간 $3628 \pm 2$ s
- HIGGS ($n=10^7$): $1-\text{AUC} = 0.1804 \pm 0.0003$, 시간 $443 \pm 2$ s
- 기존 Falkon [42] 대비 약 **20×** 속도 향상 (Table 1, p.11)

### 검토자 해석

1. **통계적 보장의 실용적 의미:** 저자들은 $m=\mathcal{O}(\sqrt{n})$이 이론적으로 충분하다고 주장하나, 실제 실험에서는 TAXI에서 $m=10^5$ ($\sqrt{10^9} \approx 3.16 \times 10^4$보다 약 3배 큰 값)을 사용함. 이는 이론적 하한이 실제 최적값과 다를 수 있음을 시사.

2. **비교의 공정성 한계:** GPyTorch/GPflow는 SVGP 모델(불확실성 정량화 가능)과 비교했으나, Falkon은 점 예측만 수행. 불확실성 추정 능력을 포기한 대가로 속도를 얻은 것으로 해석 가능.

3. **Float32 정밀도 트레이드오프:** HIGGS에서 Float32 사용 시 미세한 오차 증가가 관찰됨. 고정밀도가 필요한 과학 응용에서는 주의 필요.

4. **4-GPU 확장성의 준선형 성능 (Figure 6):** 이상적 4× 대비 실제 ~3× 수준으로, 통신 오버헤드가 여전히 병목임을 시사. 더 많은 GPU 추가 시 한계 존재.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 항목 | 취약점/비교 불가 이유 |
|------|---------------------|
| ⚠️ **TAXI RMSE 비교 (Table 6)** | Falkon 311.7 vs. ADVGP 309.7이지만, ADVGP는 28,000 vCPU 클러스터 사용. 하드웨어 비용과 탄소 발자국이 완전히 다름. |
| ⚠️ **EigenPro 대형 데이터셋 FAIL** | EigenPro가 실패한 데이터셋에서는 정확도 비교 불가. 실제 경쟁 상황에서의 정확도 우위를 검증할 수 없음. |
| ⚠️ **YELP 데이터셋 단독 결과** | 비교 알고리즘 모두 FAIL. Falkon의 YELP 결과 $0.810 \pm 0.001$은 경쟁 기준 없음. |
| ⚠️ **하이퍼파라미터 튜닝 불균형** | Falkon은 단일 길이 척도(single length-scale) 사용, GP 솔버는 자동 튜닝 가능. 다중 길이 척도 사용 시 Falkon 정확도가 더 향상될 수 있음(p.11 언급). |
| ⚠️ **5회 반복 통계** | 각 실험 5회 반복은 통계적 유의성 검정(t-test 등)을 수행하기에 충분하지 않을 수 있음. |
| ⚠️ **EigenPro 서브샘플링** | EigenPro는 일부 데이터셋에서 서브샘플(Table 3 각주)을 사용했으나, Falkon은 전체 데이터 사용. 정확도 비교가 불공평함. |
| ⚠️ **GPyTorch TIMIT 실패** | 소프트웨어 한계로 멀티클래스 미실행. 알고리즘 자체 한계가 아님(p.12 명시). |
| ⚠️ **Table 6 문헌 비교** | 다른 논문들의 하드웨어, 전처리, 분할 방식이 상이하여 직접 비교에 한계. |

---

## 6. 문서가 답하지 않는 질문

1. **다중 길이 척도(multiple length-scales) 커널 사용 시 성능은?** 저자들이 단일 길이 척도만 사용했음을 스스로 인정(p.11). ARD(Automatic Relevance Determination) 커널 적용 결과 미제공.

2. **이론적 최적 $m = \mathcal{O}(\sqrt{n})$과 실제 최적 $m$의 관계는?** 실험에서 사용한 $m$ 값들이 이론적 하한보다 크게 설정됨. $m$ 선택 가이드라인 부재.

3. **8개 GPUs 이상으로 확장 시 성능은?** Figure 6에서 4-GPU까지만 측정. 데이터센터 규모 확장성 미검증.

4. **로지스틱 손실 외 다른 손실 함수(힌지, Huber 등)에 대한 확장은?** GSC(Generalized Self-Concordant) 이론이 존재하나 실험 미제공.

5. **온라인/스트리밍 데이터 설정에서의 적용 가능성은?** 배치 학습 가정. 데이터가 순차적으로 도착하는 실시간 시나리오 미검토.

6. **유도점(inducing points) 선택 전략의 영향은?** 균일 랜덤 샘플링만 사용. k-means 클러스터링 등 더 나은 선택법 비교 없음.

7. **분류 작업에서 LogFalkon의 멀티클래스 확장은?** 현재 이진 분류만 지원. 144개 클래스 TIMIT에 LogFalkon 적용 결과 없음.

8. **메모리-정확도 트레이드오프에 대한 이론적 분석은?** Float32 사용 시 수치 오차가 통계적 보장에 미치는 영향 분석 부재.

9. **커널 파라미터 자동 최적화(GP처럼 marginal likelihood 최대화) 방법은?** 수동 튜닝만 제시됨.

10. **희소(sparse) 데이터셋에 대한 이론적 분석은?** YELP의 경우 실용적 구현만 제시되고 이론적 보장 미제공.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): 커널 솔버 종합 벤치마크
**내용:** 6개 대규모 데이터셋에서 Falkon/LogFalkon(빨간/노란선)과 EigenPro, GPyTorch, GPflow(파란 계열 점선)의 시간-오차 곡선.

**해석:**
- Falkon(빨간선)은 거의 모든 데이터셋에서 가장 빠른 수렴을 보임
- TAXI ($n=10^9$)에서 타 알고리즘이 수만 초 소요되는 동안 Falkon만 합리적 시간 내 완료
- SUSY, AIRLINE-CLS에서는 LogFalkon이 Falkon보다 정확도는 소폭 높으나 느림
- **의미:** GPU 최적화된 Nyström-PCG 방법이 범용 GP 근사 방법에 비해 실용적 우위를 갖는다는 핵심 결론을 시각적으로 뒷받침

### Figure 4 (p.7): 전처리기 행렬의 메모리 내 진화
**내용:** 단일 $m \times m$ 행렬이 Cholesky 분해와 행렬 곱셈을 거쳐 $K_{mm} \to T \to \frac{1}{m}TT^\top + \lambda I \to A^\top$으로 in-place 변환되는 과정.

**해석:**
- 핵심 메모리 절약 기법: 전처리기 계산에 추가 행렬 할당 없이 단일 $m \times m$ 버퍼 재사용
- $K_{mm}$ 원본이 CG 반복에서 불필요함을 수식적으로 증명(Eq.8-9)하여 덮어쓰기 가능
- **의미:** 메모리 사용량의 ~90%를 차지하는 전처리기를 단 하나의 행렬로 관리. $m=2\times10^5$일 때 약 150GB 절약 효과 (p.8)

### Figure 5 (p.8): 다중 GPU 블록 Cholesky 분해의 3단계
**내용:** G-1, G-2 두 GPU에서 삼각 행렬이 타일 단위로 분배되어 처리되는 병렬 Cholesky 분해 과정.

**해석:**
- 1D 블록-순환(block-cyclic) 방식으로 행이 GPU에 배분 (행 1,3 → G-1; 행 2,4 → G-2)
- 화살표: GPU 간 데이터 전송 통신. 첫 번째 타일 Cholesky → 삼각 시스템 풀기 → 후행 부분행렬 갱신의 3단계 순차 실행
- **의미:** GPU 메모리를 초과하는 대형 전처리기를 다중 GPU에서 효율적으로 계산 가능. 실제 2-GPU에서 전처리기 1.5× 속도 향상 달성

### Figure 6 (p.21): 다중 GPU 확장성 (TAXI 데이터셋)
**내용:** 1~4개 GPU 사용 시 단일 GPU 대비 속도 향상 비율. 이상적 선형 확장(점선)과 실제 성능(실선) 비교.

**해석:**
- 2-GPU: ~2×, 3-GPU: ~2.5×, 4-GPU: ~3.2× 달성 (이상적 4×에 비해 80% 효율)
- CG 반복은 GPU 수에 거의 선형 확장; 전처리기는 데이터 의존성과 통신 오버헤드로 완전 선형 미달
- **의미:** 단일 머신 내 다중 GPU 활용으로 합리적 확장성 달성. 다만 GPU가 늘어날수록 통신 오버헤드 증가로 효율 감소

### Figure 8 (p.23): 행렬-벡터 곱 구현 비교 (KeOps vs. 자체 구현)
**내용:** (a) 데이터 수 $n$ 증가 시, (b) 데이터 차원 $d$ 증가 시 시간 비교.

**해석:**
- **그림 (a):** $n$ 증가 시 두 구현 모두 선형 증가. KeOps가 약 10배 빠름 → 저차원 대규모 데이터에서 KeOps 사용이 유리
- **그림 (b):** $d$ 증가 시 자체 구현은 선형 증가, KeOps는 다항식적 증가 → 고차원($d \gtrsim 1000$)에서 KeOps 사용 불가
- **의미:** 데이터 차원에 따라 두 구현을 전환하는 하이브리드 전략의 필요성과 타당성을 실험적으로 입증. YELP($d \sim 10^7$)와 같은 고차원 희소 데이터에서 자체 구현 필수

---

## 8. 결론 및 후속 연구

### 8-1. 모델의 일반화 성능 향상 가능성

**저자 제시 시사점 (p.12, Section 5):**
1. 다른 손실 함수(힌지, Huber 등) 확장
2. 구조화 커널(hierarchically compositional kernels [9]) 도입으로 효율성 추가 향상
3. 최적화 방법 다양화

**일반화 성능과 관련된 핵심 분석:**

논문의 통계적 보장 (Eq.3, Eq.10):

$$L(\hat{f}_\lambda) - \inf_{f \in \mathcal{H}} L(f) = \mathcal{O}\!\left(n^{-1/2}\right)$$

이 수렴률은 **최적(minimax optimal)**이지만, 더 강한 가정 하에서는 개선 가능:

- **정규성 조건(regularity conditions):** 목표 함수가 RKHS의 더 매끄러운 부분 공간에 속할 경우 $\mathcal{O}(n^{-\beta})$, $\beta > 1/2$ 달성 가능 [6, 52]
- **유도점 선택 전략 개선:** 균일 랜덤 샘플링 대신 leverage score 기반 샘플링 [17]이나 k-DPP를 사용하면 더 적은 $m$으로 동일 통계 보장 가능
- **다중 커널 학습(MKL):** 단일 길이 척도 가우시안 커널의 한계 극복. 자동으로 적합한 커널 조합 학습 가능
- **커널 구성 자동화:** Neural Tangent Kernel (NTK)과의 연결을 통해 딥러닝과 유사한 표현력 확보 가능

**일반화 오차의 분해:**

```math
\underbrace{L(\hat{f}_\lambda) - L(f^*)}_{\text{총 오차}} = \underbrace{L(\hat{f}_\lambda) - \inf_{f \in \mathcal{H}} L(f)}_{\text{근사 오차}} + \underbrace{\inf_{f \in \mathcal{H}} L(f) - L(f^*)}_{\text{근사 편향}}
```

현재 연구는 첫 번째 항의 최소화에 집중. 두 번째 항(모델 클래스의 한계)은 더 표현력 있는 커널 선택으로 줄일 수 있음.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 비교는 본 논문(2020년 11월 arXiv)의 내용과 AI 연구 동향에 기반한 분석이며, 2021년 이후 발표된 구체적 논문들의 세부 결과에 대해서는 확인 한계가 있습니다. 정확한 비교를 위해 직접 원문 확인을 권장합니다.

**본 논문이 이후 연구에 미치는 영향:**

1. **GPU 최적화 커널 방법의 표준 확립:** Falkon 라이브러리는 대규모 커널 방법 연구의 기준 구현(baseline)으로 자리잡음. 이후 커널 방법 논문들은 Falkon과 비교하는 것이 일반화됨.

2. **커널-딥러닝 하이브리드 연구 촉진:** NTK(Neural Tangent Kernel) 연구와 결합되어, 딥러닝의 표현력과 커널 방법의 이론적 보장을 결합하려는 연구 방향 제시.

3. **Nyström 근사 재평가:** $m = \mathcal{O}(\sqrt{n})$ 유도점으로 최적 통계 성능을 달성한다는 실증적 증거 제공으로, Nyström 방법이 deep GP 등과 경쟁 가능함을 보임.

**앞으로 연구 시 고려할 점:**

| 고려 사항 | 설명 |
|----------|------|
| **커널 파라미터 자동 최적화** | 수동 튜닝의 한계 극복. 베이지안 최적화 또는 미분 가능 프로그래밍을 통한 자동화 필요 |
| **분산 학습 확장** | 단일 머신의 한계 극복. 연합 학습(federated learning) 환경에서의 커널 방법 적용 연구 필요 |
| **더 표현력 있는 커널** | Falkon의 속도 이점을 유지하면서 딥 커널(deep kernel), 스펙트럼 혼합 커널 등 적용 |
| **불확실성 정량화** | 커널 방법이 GP와 달리 예측 불확실성을 제공하지 않는 한계. Conformal prediction 등과 결합 가능성 |
| **온라인 학습 시나리오** | 스트리밍 데이터에서의 유도점 동적 업데이트 방법론 개발 |
| **구조화 데이터 적용** | 그래프, 시계열, 이미지 등 비유클리드 구조 데이터에 대한 커널 설계 |
| **에너지 효율성** | GPU 집약적 연산의 탄소 발자국 고려. 더 적은 자원으로 동일 성능 달성 연구 |
| **이론-실제 간극 해소** | 이론적 $m = \mathcal{O}(\sqrt{n})$과 실제 사용되는 $m$ 값의 차이에 대한 정밀 분석 |

---

## 참고자료

**주요 참고 논문 (논문 내 인용 기준):**

1. Rudi, A., Carratino, L., and Rosasco, L. **"FALKON: An optimal large scale kernel method."** Advances in Neural Information Processing Systems 29, 2017. [논문 내 [42]]

2. Rudi, A., Camoriano, R., and Rosasco, L. **"Less is more: Nyström computational regularization."** Advances in Neural Information Processing Systems 28, 2015. [논문 내 [41]]

3. Marteau-Ferey, U., Bach, F., and Rudi, A. **"Globally convergent Newton methods for ill-conditioned generalized self-concordant losses."** Advances in Neural Information Processing Systems 32, 2019. [논문 내 [30]]

4. Ma, S. and Belkin, M. **"Kernel machines that adapt to GPUs for effective large batch training."** Proceedings of the 2nd Conference on Machine Learning and Systems (EigenPro), 2019. [논문 내 [29]]

5. Gardner, J.R. et al. **"GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration."** Advances in Neural Information Processing Systems 31, 2018. [논문 내 [15]]

6. van der Wilk, M. et al. **"A framework for interdomain and multioutput Gaussian processes"** (GPflow), 2020. [논문 내 [57]]

7. Charlier, B. et al. **"KeOps."** 2020. https://github.com/getkeops/keops [논문 내 [8]]

8. Schölkopf, B. and Smola, A.J. **"Learning with Kernels."** MIT Press, 2001. [논문 내 [44]]

9. Rasmussen, C.E. and Williams, C.K.I. **"Gaussian Processes for Machine Learning."** MIT Press, 2006. [논문 내 [39]]

10. Ltaief, H. et al. **"A scalable high performant Cholesky factorization for multicore with GPU accelerators."** High Performance Computing for Computational Science, 2011. [논문 내 [27]]

**원문 논문:**
- Meanti, G., Carratino, L., Rosasco, L., and Rudi, A. **"Kernel methods through the roof: handling billions of points efficiently."** arXiv:2006.10350v2, 2020. https://arxiv.org/abs/2006.10350

**라이브러리:**
- FalkonML. https://github.com/FalkonML/falkon
