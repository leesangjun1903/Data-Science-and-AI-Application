# Fast Differentiable Sorting and Ranking

### 1. 핵심 주장과 주요 기여

**Fast Differentiable Sorting and Ranking** 논문은 기계학습에서 오랫동안 해결되지 않은 근본적인 문제를 다룬다. 정렬(sorting)과 순위(ranking) 연산은 컴퓨터과학에서 가장 기본적인 연산이지만, 신경망의 역전파(backpropagation)를 통해 학습할 수 없다는 치명적 제한이 있었다.[1]

이 논문의 **핵심 주장**은 다음과 같다:

1. **첫 번째 효율적인 미분 가능 정렬·순위 연산자**: 기존 방법들은 $$O(n^3)$$, $$O(n^2)$$, 또는 $$O(Tmn)$$의 시간복잡도를 가졌으나, 이 논문은 최초로 **$$O(n \log n)$$ 시간복잡도와 $$O(n)$$ 공간복잡도**를 달성한다.[1]

2. **정확한 계산과 미분**: 근사 알고리즘의 반복을 통해 미분하는 기존 방식과 달리, 이 논문은 정확한 계산과 미분을 제공한다.[1]

3. **순열체(Permutahedron) 기반 접근**: 정렬과 순위 문제를 순열체로의 사영(projection) 문제로 재정의하고, 등장계(isotonic) 최적화로 축소함으로써 효율성을 달성한다.[1]

### 2. 해결 문제와 제안 방법

#### 2.1 문제의 본질

정렬과 순위 연산의 미분 불가능성은 두 가지 원인에서 비롯된다:

- **정렬**: 구간별 선형함수(piecewise linear)이기 때문에 꺾이는 지점에서 미분 불가능하다. 이는 여러 개의 기울기 변화(kinks)를 만든다.
  
- **순위**: 구간별 상수함수(piecewise constant)이기 때문에 미분이 정의되지 않거나 0이 되어 역전파가 불가능하다.[1]

이는 강건한 통계(robust statistics), 순위 지표, NDCG(정규화 할인 누적 이득) 등 많은 중요한 응용을 신경망에 직접 통합할 수 없다는 의미이다.

#### 2.2 기존 방법의 한계

논문에서 비교한 기존 방법들:

| 방법 | 시간복잡도 | 특징 |
|------|-----------|------|
| Taylor et al. (2008) | $$O(n^3)$$ | 확률적 섭동 기반 |
| Qin et al. (2010) | $$O(n^2)$$ | 쌍별 거리 비교 |
| Grover et al. (2019) | $$O(n^2)$$ | 단봉형 확률 행렬 |
| Cuturi et al. (2019) | $$O(Tmn)$$ | 최적 운송(Sinkhorn) |
| **이 논문** | **$$O(n \log n)$$** | **순열체 사영** |

#### 2.3 제안 방법: 순열체와 정규화

**핵심 아이디어**: 선형계획법 공식화

정렬과 순위를 이산 최적화 문제로 표현:

$$
\sigma(\theta) = \arg\max_{\sigma \in \Sigma} \langle \theta_\sigma, \rho \rangle \quad (1)
$$

$$
r(\theta) = \arg\max_{\pi \in \Sigma} \langle \theta, \rho_\pi \rangle \quad (2)
$$

여기서 $$\rho = (n, n-1, \ldots, 1)$$은 역순 벡터, $$\Sigma$$는 모든 순열의 집합이다.

**연속화 및 정규화**: 순열체(permutahedron) $$P(w) = \text{conv}\{\mathbf{w}_\sigma : \sigma \in \Sigma\}$$로 변수를 변환하면:

$$
s(\theta) = \arg\max_{y \in P(\theta)} \langle y, \rho \rangle \quad (3)
$$

$$
r(\theta) = \arg\max_{y \in P(\rho)} \langle y, -\theta \rangle \quad (4)
$$

이를 통해 정렬은 **거의 모든 곳에서 미분 가능**해진다(piecewise linear이므로). 그러나 순위는 여전히 불연속이다.

**정규화를 통한 부드러운 연산자**: 강볼록 정규화 $$\Psi$$를 도입하면:

**이차 정규화**: $$Q(\mu) = \frac{1}{2}\|\mu\|^2$$

**엔트로피 정규화**: $$E(\mu) = \langle \mu, \log \mu - \mathbf{1} \rangle$$

정규화된 소프트 정렬과 순위는:

$$
s^\varepsilon_\Psi(\theta) := P_\Psi(\rho/\varepsilon, \theta) \quad (5)
$$

$$
r^\varepsilon_\Psi(\theta) := P_\Psi(-\theta/\varepsilon, \rho) \quad (6)
$$

여기서 $$P_\Psi(z, w)$$는 순열체로의 정규화 사영이다.

#### 2.4 모델 구조

논문의 핵심 구조는 **등장계 최적화로의 축소**이다:

**명제 3**: 정렬된 $$\mathbf{w}$$에 대해:

$$
P_\Psi(z, w) = z - v_\Psi(z_{\sigma(z)}, w)_{\sigma^{-1}(z)}
$$

여기서:

$$
v_Q(s, w) := \arg\min_{v_1 \geq \cdots \geq v_n} \frac{1}{2}\|v - (s - w)\|^2 \quad \text{(등장계 회귀)}
$$

$$
v_E(s, w) := \arg\min_{v_1 \geq \cdots \geq v_n} \langle e^{s-v}, \mathbf{1} \rangle + \langle e^w, v \rangle
$$

**Pool Adjacent Violators (PAV) 알고리즘**으로 이 문제들을 $$O(n)$$ 시간에 해결할 수 있다.

#### 2.5 자코비안 계산

**핵심 성질** (Lemma 2): 등장계 최적화 해의 자코비안은 블록 대각 구조를 가진다:

이차 정규화의 경우:

$$
\frac{\partial v_Q}{\partial s} = \begin{pmatrix} B_1 & 0 & \cdots \\ 0 & B_2 & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}, \quad B_j = \frac{1}{|B_j|}\mathbf{1}
$$

엔트로피 정규화의 경우:

$$
B_j = \mathbf{1} \otimes \text{softmax}(s_{B_j})
$$

이를 이용하여 **역전파는 $$O(n)$$ 시간에 수행**된다.

### 3. 성능 향상과 한계

#### 3.1 성능 향상

**실험 1: Top-k 분류 (CIFAR-10/100)**

- 제안 방법 ($$r_Q$$, $$r_E$$)은 기존 최적 운송(OT) 방법과 비교하여 **정확도는 동등하면서 수십 배 빠르다**.[1]
- 100개 클래스에서: OT 29시간 vs 제안 방법 21-23시간
- 2000+ 차원에서 기존 방법들은 메모리 부족(OOM)으로 실행 불가능

**실험 2: 레이블 순위 (21개 데이터셋)**

소프트 순위 계층을 추가하면:
- 15개 데이터셋에서 개선
- 4개 데이터셋에서 동등
- 2개 데이셋에서 악화

이는 간단한 선형 모델에서도 제안 방법이 유익함을 시사한다.[1]

**실험 3: 강건 회귀 (Soft Least Trimmed Squares)**

이상치 비율에 따른 성능:
- 기존 최소 제곱법(LS)은 이상치에 매우 취약
- 제안된 Soft LTS는 하드 LTS(hard LTS)와 LS 사이의 보간이 가능
- 적응적 정규화 강도 선택으로 미지의 이상치 비율에 대응 가능

#### 3.2 일반화 성능 향상 메커니즘

**정규화 매개변수 $$\varepsilon$$의 역할**:

**명제 2** (점근 성질):

$$
s^\varepsilon_\Psi(\theta) \xrightarrow{\varepsilon \to 0} s(\theta), \quad s^\varepsilon_\Psi(\theta) \xrightarrow{\varepsilon \to \infty} f_\Psi(\theta)\mathbf{1}
$$

여기서 $$f_Q(u) = \text{mean}(u)$$, $$f_E(u) = \log \text{mean}(u)$$이다.

**볼록화 효과**: $$\varepsilon$$가 증가하면:
- 부드러운 근사가 증가
- 목적함수가 더욱 볼록에 가까워짐
- 최적화가 용이해짐
- 일반화 성능이 향상될 가능성

**순서 보존 특성** (명제 2):

$$
s = s^\varepsilon_\Psi(\theta)는 항상 내림차순: s_1 \geq s_2 \geq \cdots \geq s_n
$$

$$
r = r^\varepsilon_\Psi(\theta)에 대해: r_{\sigma_1} \leq r_{\sigma_2} \leq \cdots \leq r_{\sigma_n}
$$

이는 모든 $$\varepsilon > 0$$에서 성립하며, **구조 보존을 보장**한다.

#### 3.3 한계

**1. 정규화 매개변수 선택**
- 상동 함수 모델에서는 자동으로 흡수 가능하나, 그렇지 않은 경우 $$\varepsilon$$ 튜닝 필요
- Top-k 분류에서는 로지스틱 맵으로 점수를 $$[0,1]^n$$에 스쿼시하고 $$\varepsilon$$ 튜닝이 중요함

**2. 정렬 연산의 미분 불완전성**
- 정렬은 $$\varepsilon \to 0$$에서도 여전히 구간별 선형이므로 기울기가 완전히 유연하지 않음
- 순위 연산이 더 유리한 미분 특성을 가짐

**3. 계산 장소 제약**
- PAV 알고리즘이 CPU 기반이므로, GPU에서의 $$O(n^2)$$ 방법들(작은 $$n$$에서)에 비해 상대적으로 느릴 수 있음
- 실험에서 $$n=100$$일 때 All-pairs 방법이 더 빠름

**4. 데이터 특성에 따른 성능 변동**
- 강건 회귀 실험에서 일부 데이터셋에서 정규화가 도움이 되지 않음

### 4. 일반화 성능 향상과 관련된 분석

#### 4.1 정규화의 일반화 효과

**구조적 정규화**: 순열체로의 사영 자체가 정규화 역할을 수행
- 높은 차원에서의 과적합 방지
- 연속적인 근사를 통한 부드러운 학습 곡선

**엔트로피 정규화 vs 이차 정규화**:
- 엔트로피 정규화: 확률적 해석 제공, 더 부드러운 함수 경향
- 이차 정규화: 계산 효율적, 더 명확한 블록 구조

실험에서 대부분의 경우 이차 정규화($$r_Q$$)가 더 실용적이었다.[1]

#### 4.2 순서 보존의 일반화 영향

순위 작업에서 **순서 보존 특성**은:
- 학습된 순위가 논리적 일관성 유지
- 과도한 진동 방지
- 테스트 시 하드 순위로 전환 가능 (근처성 보장)

#### 4.3 볼록화 효과

이 논문은 처음으로 **정렬/순위 연산의 볼록화 경로(convexification path)** 제시:
- $$\varepsilon$$를 증가시키면 비볼록 문제가 볼록에 수렴
- 이는 SGD의 수렴성 개선
- 더 안정적인 학습 역학(learning dynamics)

### 5. 앞으로의 연구에 미치는 영향과 고려사항

#### 5.1 최근 연구 발전 (2021-2025)

**1. 구조화된 정렬 네트워크**

Petersen et al. (2021)은 비트 정렬 네트워크(bitonic sorting networks)를 기반으로 한 미분 가능 정렬을 제시했으며, 최대 1024개 원소까지 확장 가능하다. 이는 Blondel의 순열체 방식과 상호 보완적이다.[2]

**2. 대규모 순위 작업의 확장**

ARF (Adaptive Neural Ranking Framework)와 LCRON (Listing-wise Learning for Cascade Ranking Optimization Network) 같은 최근 연구들은 cascade ranking 시스템에서 미분 가능 정렬 기술을 활용하고 있다. 특히, DFTopK 연산자는 Full Ranking의 경쟁적 특성으로 인한 기울기 충돌을 완화하는 데 중점을 둔다.[3]

**3. LLM 기반 순위 최적화**

DRPO (Direct Ranking Preference Optimization)는 diffNDCG 손실(미분 가능 NDCG)을 이용하여 정렬 네트워크로 NDCG를 시뮬레이션한다. 이는 LLM 정렬에서 인간 선호도 정렬의 정확도를 60% 이상으로 높이는 데 성공했다.[4]

**4. DAG 학습에의 응용**

Zantedeschi et al. (2023)은 관측 데이터에서 인과 방향성 비순환 그래프(DAG)를 학습하기 위해 순열체를 사용했다. 이는 Blondel의 프레임워크를 인과 추론 분야로 확장한 사례이다.[5]

**5. 강화학습을 통한 정렬 알고리즘 발견**

Nature (2023)에 실린 AlphaDev 연구는 깊은 강화학습을 사용하여 LLVM 표준 C++ 라이브러리에 통합된 새로운 정렬 알고리즘을 발견했다. 이는 미분 가능성과 무관하게 고전적 정렬 알고리즘의 한계를 넘어서는 방향을 보여준다.[6]

#### 5.2 미래 연구 시 고려사항

**1. 이질적 데이터 환경**

Adaptive Hybrid Sort (2025)는 데이터 특성(크기, 범위, 엔트로피)에 따라 정렬 알고리즘을 동적으로 선택한다. 미분 가능 정렬도 입력의 특성을 학습하여 $$\varepsilon$$ 또는 정규화 유형을 적응적으로 선택하는 방향을 탐구할 가치가 있다.

**2. 혼합형 접근법**

순열체 기반 Blondel의 방법과 정렬 네트워크 기반 Petersen의 방법을 결합하면:
- 소규모 문제에서는 정렬 네트워크의 구조 활용
- 대규모 문제에서는 순열체의 $$O(n \log n)$$ 복잡도 활용
- 적응적 선택으로 최적 성능 달성 가능

**3. 확률적 해석**

최근 NeuralSort와 같은 방법들이 확률적 순열 분포를 학습하는 방향으로 발전 중이다. Blondel의 프레임워크에 확률적 정규화(stochastic regularization)를 추가하면:
- 순위의 불확실성 정량화 가능
- 베이지안 순위 학습 가능
- 구간 추정을 통한 순위 신뢰도 평가

**4. 하드웨어 최적화**

PAV 알고리즘의 GPU 구현:
- CUDA 커널로 블록 평균 계산 병렬화
- 메모리 계층 구조 활용으로 캐시 효율성 향상
- 이는 순설정(sorting)이 CPU 제약인 현재 상황 개선 가능

**5. 대규모 검색 시스템 적용**

정보 검색 시스템에서:
- 수백만 개의 문서 순위 지정에 $$O(n \log n)$$ 복잡도는 필수
- 적응형 정규화로 다양한 쿼리 특성에 대응
- 실시간 cascade ranking 시스템 구축 가능

**6. 그래프 구조 학습**

DAG 학습 beyond에서:
- 일반 그래프의 위상 순서 학습
- 상호 연관된 순위 문제(예: 다중 기준 순위)
- 계층적 순위 결합

**7. 도메인 특화 정규화**

다양한 분야의 요구사항에 맞춘 정규화:
- **추천 시스템**: 아이템 인기도를 반영한 정규화
- **자연어 처리**: 문맥 인식 정규화
- **컴퓨터 비전**: 공간적 근처성 고려
- **생물정보학**: 단백질 상호작용 네트워크 특화

#### 5.3 이론적 과제

**1. 수렴성 분석**

미분 가능 정렬/순위를 사용한 신경망의:
- 전역 수렴성 보증 조건
- 국소 최소값의 특성화
- 일반화 경계(generalization bounds)

**2. 샘플 복잡도**

$$\varepsilon$$ 선택에 따른:
- 표본의 크기 요구사항
- 학습 곡선의 특성화
- PAC 학습성

**3. 그래디언트 흐름**

역전파 시:
- 기울기 소실/폭발 문제 분석
- 정규화 유형에 따른 영향
- 신경망 깊이와의 상호작용

### 결론

**Fast Differentiable Sorting and Ranking**은 미분 가능 프로그래밍 분야의 획기적 기여이다. 순열체로의 사영과 등장계 최적화를 연결함으로써 이론과 실무 간 격차를 좁혔다. 제안 방법은 지난 십여 년간 다양한 근사 방법들을 완전히 우월하게 만들었으며, 특히 $$O(n \log n)$$ 복잡도는 대규모 응용에 필수적이다.[1]

향후 연구는:
1. **적응적 정규화**: 데이터와 작업 특성에 맞춘 $$\varepsilon$$ 및 $$\Psi$$ 자동 선택
2. **확률적 확장**: 불확실성 정량화 기능 추가
3. **하드웨어 최적화**: GPU 병렬화로 실시간 대규모 응용 실현
4. **이론 강화**: 수렴성, 표본 복잡도, 일반화 경계 완전 규명
5. **도메인 특화**: 추천, NLP, 생물정보학 등에서의 맞춤 응용 확대

이 논문이 개시한 방향으로의 발전은 신경망이 직접 순서 정보를 학습하는 새로운 시대를 열 것으로 기대된다.[7][8][9][10][4][6][5][3][2]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f301be27-7d55-4816-8079-86eb304ed654/2002.08871v2.pdf)
[2](http://proceedings.mlr.press/v139/petersen21a/petersen21a.pdf)
[3](https://arxiv.org/pdf/2510.11472.pdf)
[4](https://openreview.net/forum?id=Lz5lOSC0zg)
[5](https://arxiv.org/abs/2301.11898)
[6](https://www.nature.com/articles/s41586-023-06004-9)
[7](https://arxiv.org/pdf/2105.04019.pdf)
[8](https://arxiv.org/pdf/2503.06242.pdf)
[9](https://arxiv.org/pdf/2311.01864.pdf)
[10](https://arxiv.org/pdf/2310.10462.pdf)
[11](http://arxiv.org/pdf/2405.17798.pdf)
[12](http://arxiv.org/pdf/2410.10728.pdf)
[13](https://arxiv.org/pdf/2107.03290.pdf)
[14](https://arxiv.org/pdf/2202.00211.pdf)
[15](https://dl.acm.org/doi/10.5555/3524938.3525027)
[16](https://arxiv.org/pdf/2506.20677.pdf)
[17](https://proceedings.mlr.press/v51/lim16.html)
[18](https://philarchive.org/archive/MOHFOD)
[19](http://papers.neurips.cc/paper/5611-beyond-the-birkhoff-polytope-convex-relaxations-for-vector-permutation-problems.pdf)
