# SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm
---

## 1. Executive Summary (10문장 이내)

SiameseNorm은 Transformer 아키텍처에서 Pre-Norm과 Post-Norm의 근본적 트레이드오프를 해소하기 위해 제안된 **이중 스트림(two-stream) 잔차 구조**이다.  
Pre-Norm은 안정적 학습을 제공하지만 깊은 레이어의 유효 활용도가 낮고, Post-Norm은 높은 표현력을 가지나 대규모 학습에서 불안정하다.  
저자들은 단일 스트림에서는 두 패러다임을 동시에 만족시키는 것이 구조적으로 불가능함을 이론적으로 분석한다.  
SiameseNorm은 정규화되지 않은 Pre-Norm 유사 스트림($Y_i$)과 정규화된 Post-Norm 유사 스트림($X_i$)을 공유 잔차 블록으로 결합한다.  
두 스트림은 매 레이어에서 동일한 잔차 변환 출력 $O_i$를 공유하므로 파라미터 오버헤드가 0.1% 미만이다.  
400M·1.3B 밀집 언어 모델, 15B MoE 모델, Vision Transformer, Diffusion Transformer에서 일관된 성능 향상을 보인다.  
학습률 $\eta = 1 \times 10^{-3}$ 설정에서 최고 PPL 10.43을 달성해 가장 강한 베이스라인 대비 0.3 개선했다.  
기존 Pre-Norm 학습 레시피와 완전히 호환되어 아키텍처별 추가 튜닝이 불필요하다.  
깊이가 증가할수록 성능 향상이 더 뚜렷하며(80레이어에서 PPL 감소 2.04), 레이어 활용도(layer utilization)가 개선됨을 레이어 프루닝 실험으로 확인했다.  
SiameseNorm은 멀티스트림 잔차 아키텍처 설계의 실용적 기반을 제시한다.

---

### 1-1. 연구의 목적과 필요성

**목적**: Pre-Norm과 Post-Norm 각각의 장점(안정적 최적화 vs. 높은 표현력)을 단일 구조에서 동시에 실현하는 새로운 Transformer 정규화 아키텍처를 설계한다.

**필요성** (p.1–2):
- **Pre-Norm의 한계**: 깊은 레이어에서 은닉 상태 노름(hidden state norm)이 지수적으로 증가하여 깊은 레이어의 기여가 희석됨 (*Dilution Problem*). Gromov et al. (2025)은 Pre-Norm 모델에서 많은 수의 레이어를 제거해도 성능 저하가 미미하다고 보고.
- **Post-Norm의 한계**: 학습률이 높을 때 발산하거나 손실 스파이크가 빈번하게 발생. 하이퍼파라미터에 극도로 민감 (*Distortion Problem*).
- **기존 하이브리드 방법의 한계**: HybridNorm, SpanNorm 등 기존 혼합 방식은 특정 학습 설정 밖에서 안정성이 부족하며, 이는 단일 스트림 내에서 두 패러다임의 구조적 긴장(structural tension)을 해소할 수 없기 때문.

> 📘 **용어 설명**
> - **Pre-Norm**: 잔차 브랜치 내부(변환 함수 입력 직전)에 Layer Normalization을 배치하는 방식
> - **Post-Norm**: 잔차 덧셈 이후에 Layer Normalization을 배치하는 방식
> - **PPL (Perplexity)**: 언어 모델의 불확실성 지표. 낮을수록 좋음
> - **잔차 연결(Residual Connection)**: 입력을 변환 결과에 더해주는 지름길 경로. 기울기 소실 방지에 핵심적

---

## 2. 핵심 주장과 근거 표

| 구분 | 핵심 주장 | 근거 | 위치 |
|------|-----------|------|------|
| 문제 1 | Pre-Norm은 깊이에 따라 은닉 상태 크기가 폭발적으로 증가 (Dilution Problem) | Figure 2a: 16레이어 모델에서 Hidden State Norm이 최대 175까지 증가 | p.3, Fig. 2a |
| 문제 2 | Post-Norm은 높은 학습률에서 기울기 폭발로 발산 (Distortion Problem) | Table 1: $\eta=10^{-3}$에서 Post-Norm, HybridNorm 모두 발산 | p.5, Table 1 |
| 문제 3 | 단일 스트림에서는 두 패러다임의 요구를 동시에 충족 불가 (Structural Tension) | 이론적 야코비안 분석: Pre-Norm은 항등 경로 필요, Post-Norm은 정규화 경로 필요 | p.3, Sec. 2.3 |
| 제안 | SiameseNorm: 두 스트림을 공유 블록으로 결합 | Algorithm 1, Figure 1c | p.4 |
| 주장 1 | Pre-Norm급 학습 안정성 | Figure 5: 기울기 노름이 0.5 이하로 유지 | p.8, Fig. 5 |
| 주장 2 | Post-Norm급 표현력 | Table 1: 모든 LR 설정에서 최고 PPL 달성 | p.5, Table 1 |
| 주장 3 | 깊은 모델에서 레이어 활용도 향상 | Figure 2c: 레이어 프루닝 시 손실 증가폭이 Pre-Norm 대비 큼 | p.8, Fig. 2c |
| 주장 4 | 다양한 모달리티로 일반화 | Table 2: DeiT, DiT에서도 Pre-Norm 대비 개선 | p.7, Table 2 |
| 주장 5 | 오버헤드 무시 가능 | 파라미터/FLOPs 증가 0.1% 미만, 속도 저하 0.5% | p.5, Sec. 3 |

---

## 2-1. 상세 설명

### 🔴 해결하고자 하는 문제 (p.1–3)

**문제 1: Pre-Norm의 Dilution Problem**

$$X_{i+1} = X_i + F_i(\text{LN}_i(X_i)) \quad \cdots (1)$$

- $X_i \in \mathbb{R}^d$: $i$번째 레이어의 입력 은닉 상태
- $F_i(\cdot)$: Attention 또는 MLP 등 잔차 변환 함수
- $\text{LN}_i(\cdot)$: 레이어 정규화 (LayerNorm 또는 RMSNorm)

항등 경로 $X_i$로 인해 $\|X_i\|_2$가 레이어가 깊어질수록 단조 증가. 깊은 레이어의 블록은 점점 커지는 $X_i$에 대해 상대적으로 작은 기여를 하게 되어 유효 깊이(effective depth)가 제한됨.

> 📘 **용어 설명**
> - **$\ell_2$-노름**: 벡터의 유클리드 크기. $\|X\|_2 = \sqrt{\sum_j x_j^2}$
> - **유효 깊이(Effective Depth)**: 모델이 실제로 활용하는 레이어의 수. 이론적 깊이와 다를 수 있음

**문제 2: Post-Norm의 Distortion Problem**

$$X_{i+1} = \text{LN}_i(X_i + F_i(X_i)) \quad \cdots (5)$$

기울기 역전파 시:

$$\nabla_{\theta_i} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial X_N} \left[ \prod_{j=N-1}^{i+1} J_{\text{LN}_j}\left(\mathbf{I} + J_{F_j}\right) \right] \frac{\partial X_{i+1}}{\partial \theta_i} \quad \cdots (7)$$

- $\mathcal{L}$: 손실 함수
- $\theta_i$: $i$번째 블록의 파라미터
- $J_{\text{LN}_j} \triangleq \frac{\partial \text{LN}(X)}{\partial X}$: 정규화 함수의 야코비안 행렬
- $J_{F_j}$: 잔차 함수 $F_j$의 야코비안
- $\mathbf{I}$: 항등 행렬

$J_{\text{LN}}$이 레이어마다 곱해져 스펙트럼 노름이 누적되면 기울기 소실 또는 폭발 발생.

> 📘 **용어 설명**
> - **야코비안(Jacobian)**: 다변수 함수의 편미분 행렬. 역전파에서 기울기 전파를 결정
> - **스펙트럼 노름(Spectral Norm)**: 행렬의 최대 특이값. 기울기 안정성의 척도

---

### 🟢 제안하는 방법: SiameseNorm (p.4)

**Forward Pass (Algorithm 1)**:

$$O_i = F_i\!\left(X_i + \text{LN}^Y_i(Y_i)\right) \quad \cdots \text{(Alg. 1, Line 4)}$$

$$X_{i+1} = \text{LN}^X_i(X_i + O_i) \quad \cdots \text{(Alg. 1, Line 5)}$$

$$Y_{i+1} = Y_i + O_i \quad \cdots \text{(Alg. 1, Line 6)}$$

$$\text{Output} = X_N + \text{LN}_{\text{final}}(Y_N) \quad \cdots \text{(Alg. 1, Line 8)}$$

- $X_i$: Post-Norm 유사 스트림 (정규화 후 크기 제어)
- $Y_i$: Pre-Norm 유사 스트림 (항등 경로 보존)
- $O_i$: 두 스트림이 공유하는 잔차 변환 출력
- $\text{LN}^X_i, \text{LN}^Y_i$: 각 스트림의 정규화 함수

초기값: $X_0 = Y_0 = \text{Embed}(x)$

**기울기 분석** (p.4, Eq. 8–10):

$$\nabla_{\theta_i} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial S_N} \left(\prod_{j=N-1}^{i+1} \frac{\partial S_{j+1}}{\partial S_j}\right) \begin{bmatrix} J_{\text{LN}^X_i} \\ \mathbf{I} \end{bmatrix} \frac{\partial O_i}{\partial \theta_i} \quad \cdots (9)$$

블록 야코비안 전이 행렬:

$$\frac{\partial S_{j+1}}{\partial S_j} = \begin{bmatrix} J_{\text{LN}^X_j}(\mathbf{I} + J_{F_j}) & J_{\text{LN}^X_j} J_{F_j} J_{\text{LN}^Y_j} \\ J_{F_j} & \mathbf{I} + J_{F_j} J_{\text{LN}^Y_j} \end{bmatrix} \quad \cdots (10)$$

- 우하단 블록 $\mathbf{I} + J_{F_j} J_{\text{LN}^Y_j}$: Pre-Norm 전이(Eq. 4)와 동일 → **안정적 기울기 고속도로**
- 좌상단 블록 $J_{\text{LN}^X_j}(\mathbf{I} + J_{F_j})$: Post-Norm 전이(Eq. 7)와 유사 → **정규화된 잔차 경로**

> 📘 **용어 설명**
> - **$S_i = [X_i, Y_i]^\top$**: 두 스트림을 합친 결합 상태 벡터
> - **블록 야코비안(Block Jacobian)**: 결합 상태에 대한 편미분을 블록 행렬로 표현한 것
> - **기울기 고속도로(Gradient Highway)**: 항등 행렬 $\mathbf{I}$로 인해 기울기가 레이어를 거슬러 올라가는 직접 경로

**보조 메커니즘** (p.4):
1. **Normalized Input**: 공유 블록 $F_i$의 입력에 추가 LN 적용 → 안정적 입력 분포 보장
2. **Depth-wise Scaling**: $X$ 스트림(Post-Norm 유사)에 주입되는 잔차를 $\frac{1}{\sqrt{l+1}}$로 스케일링 ($l$: 레이어 인덱스) → 두 스트림 간 크기 불균형 완화

> 📘 **용어 설명**
> - **Depth-wise Scaling**: DeepNorm(Wang et al., 2024)에서 영감을 얻은 기법으로, 깊은 레이어에서 잔차 업데이트의 크기를 감쇠시켜 안정성 향상
> - **RMSNorm**: Root Mean Square Layer Normalization. 평균을 빼지 않고 RMS로만 정규화하는 경량 변형

---

### 🔵 모델 구조 (p.1, Fig. 1c; p.4, Algorithm 1)

```
Input x
  │
Embed(x) → h
  ├──────────────┐
  X₀ = h        Y₀ = h
  │              │
  ├── [레이어 i = 0,...,N-1] ──┤
  │   O_i = F_i(X_i + LN^Y(Y_i))   (공유 잔차 블록)
  │   X_{i+1} = LN^X(X_i + O_i)    (Post-Norm 유사 스트림)
  │   Y_{i+1} = Y_i + O_i           (Pre-Norm 유사 스트림)
  │
Output = X_N + LN_final(Y_N)
```

언어 사전학습 설정에서는 Post-Norm 스트림으로 **HybridNorm** (Attention의 Q,K,V 선형 변환 후 LN 적용) 사용 (p.6, Figure 3).

---

### 🟡 성능 향상 (p.5–7)

| 설정 | Pre-Norm PPL | SiameseNorm PPL | 개선 |
|------|-------------|----------------|------|
| LR= $4\times10^{-4}$, 100B | 11.21 | **10.57** | △0.64 |
| LR= $1\times10^{-3}$, 100B | 10.84 | **10.43** | △0.41 |
| LR= $2\times10^{-3}$, 100B | 10.89 | **10.48** | △0.41 |
| LR= $2\times10^{-3}$, 350B | 9.67 | **9.42** | △0.25 |
| 15A2B MoE | 7.92 | **7.76** | △0.16 |

산술 과제(Arithmetic): $\eta=2\times10^{-3}$에서 Pre-Norm 28.1% → SiameseNorm **39.6%** (41% 상대 향상)

---

### 🔴 한계 (p.14, Appendix A.1)

- **이론적 보장 부재**: 엄밀한 수렴 이론 증명 없음 (실험적 결과에 의존)
- **특정 조합 의존성**: 실험에서 HybridNorm을 Post-Norm 스트림으로 사용. 다른 Post-Norm 변형 사용 시 성능이 달라질 수 있음 (Table 4 참조)
- **아키텍처 복잡도 소폭 증가**: 개념적으로 두 스트림을 관리해야 함 (구현 복잡도)

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| Pre-Norm의 은닉 상태 노름 지수적 증가 | p.3, **Figure 2a** |
| Pre-Norm 레이어 프루닝 시 성능 저하 미미 | p.3, **Figure 2c** |
| Post-Norm의 높은 LR에서 발산 | p.5, **Table 1** (Setting B, C) |
| SiameseNorm 전 LR에서 최고 PPL | p.5, **Table 1** |
| 깊이 증가 시 SiameseNorm 이점 확대 | p.7, **Table 2** (상단) |
| Vision/Diffusion Transformer 일반화 | p.7, **Table 2** (하단) |
| 기울기 안정성 (노름 0.5 이하) | p.8, **Figure 5** |
| 레이어 활용도 개선 (프루닝 손실 더 큼) | p.8, **Figure 2c** |
| HybridNorm 스트림 우세 (Logit Lens) | p.9, **Figure 6** |
| 두 보조 메커니즘의 기여 분리 | p.8, **Table 3** |
| ResiDual vs. SiameseNorm 구조 비교 | p.14, **Figure 7** |
| 오버헤드 0.1% 미만 | p.5, **Sec. 3 (Computational Overhead)** |

---

## 4. 저자 보고 결과 vs. 분석자 해석 분리

### 📌 저자가 직접 보고한 결과

**연구 주제** (Abstract, p.1):
> "We revisit this dilemma, showing that *single-stream* architectures struggle to reconcile Pre-Norm's stable identity-gradient propagation with Post-Norm's normalization of the main residual path."

**방법** (p.4, Algorithm 1, Eq. 8–10): 두 스트림 분리, 공유 $O_i$를 통한 기울기 동시 수신

**성능 결과** (p.5–7, Tables 1–2):
- Setting B(LR= $10^{-3}$ ): SiameseNorm PPL **10.43** (최고 베이스라인 Hyper-Connections-2×DHC 10.73 대비 0.30 개선)
- Setting C(LR= $2\times10^{-3}$ ): Arithmetic 정확도 **39.6%** (Pre-Norm 28.1% 대비 41% 상대 향상)
- 오버헤드: 파라미터/FLOPs **0.1% 미만** 증가, 학습 속도 **0.5%** 감소, 활성화 메모리 **2%** 증가

**분석 결과** (p.8–9):
- 기울기 노름: Pre-Norm·SiameseNorm 모두 웜업 후 **0.5 이하**, HybridNorm은 **100 초과** 스파이크
- Logit Lens: HybridNorm 스트림이 최종 출력과 **42.6%** 일치, Pre-Norm 스트림 **16.2%**

---

### 🔍 분석자(필자)의 해석

1. **Logit Lens 결과의 함의**: HybridNorm 스트림이 모델 결정을 주도한다는 결과는 단순히 두 스트림의 동등한 기여가 아님을 시사. Pre-Norm 스트림은 주로 기울기 안정화 역할을 담당하고, 표현력은 Post-Norm 스트림에 집중되는 **기능적 분업** 구조일 가능성이 높음.

2. **Arithmetic 과제의 41% 상대 향상**: 이 향상은 순차적 추론에서 효과적 깊이 증가가 직접적으로 성능에 기여함을 보여줌. 다만 이 단일 과제의 점프(28.1%→39.6%)가 다른 과제들의 안정적 향상 패턴(1–3%)과 대비되어, **Arithmetic 특유의 현상**일 가능성도 배제할 수 없음.

3. **EmbedNorm 실험의 의의**: EmbedNorm 추가 시 PPL이 0.4 악화된 결과(p.3)는 Pre-Norm에서 은닉 상태의 크기 성장 자체가 최적화에 일부 역할을 한다는 것을 시사하며, 단순 정규화로는 문제가 해결되지 않음을 보여줌.

---

## 5. 통계적 취약점 및 비교 불가 수치 ⚠️

| 항목 | 취약성 유형 | 세부 내용 |
|------|------------|-----------|
| Arithmetic 41% 상대 향상 ⚠️ | **이상치 가능성** | 다른 7개 벤치마크 향상(0–5%)과 불균형. 단일 LR 설정($2\times10^{-3}$)에서만 관찰. 반복 실험 없음 |
| Setting D (350B) 비교 ⚠️ | **비교 불균형** | Pre-Norm, HC, SiameseNorm 3개만 비교. HybridNorm 등 다른 방법 배제 |
| Table 1 각 셀 ⚠️ | **단일 시드** | 각 실험의 랜덤 시드 수, 표준편차 미보고 |
| HC(Hyper-Connections) Setting D ⚠️ | **신뢰성 의문** | loss spike 발생($9.57^*$) → 정상 학습과 동일선상 비교 부적절 |
| Table 2 (비전/확산 모델) ⚠️ | **제한된 조건** | ImageNet 단일 데이터셋, 표준 설정만 보고. 전이학습, 파인튜닝 성능 미보고 |
| 0.5% 속도 저하 ⚠️ | **하드웨어 의존** | A100 기준 수치. 다른 하드웨어(H100, TPU)에서 달라질 수 있음 |
| PPL 감소 0.3 (Setting B) ⚠️ | **통계적 유의성 미보고** | 대규모 LM에서 0.3 PPL 차이의 통계적 유의성 검증 없음 |

> 📘 **용어 설명**
> - **이상치(Outlier)**: 다른 관측값들과 동떨어진 수치로, 특정 조건에서만 나타나 일반화가 어려울 수 있음

---

## 6. 논문이 답하지 않는 질문 ❓

1. **이론적 수렴 보장**: SiameseNorm이 최적해로 수렴한다는 엄밀한 수학적 증명이 없음 (저자들도 한계로 인정, p.14)
2. **최적 스트림 조합의 선택 기준**: HybridNorm을 Post-Norm 스트림으로 선택한 기준이 실험적 탐색(Table 4) 이상의 이론적 근거를 제시하지 않음
3. **스케일 법칙(Scaling Law)과의 관계**: 1.3B·15B 이상 초대형 모델(70B, 400B+)에서의 거동 미검증
4. **파인튜닝 및 RLHF 호환성**: 사전학습 이후 지시 따르기(instruction tuning), RLHF 등에서의 안정성 미검증
5. **장문맥(Long Context) 설정**: 시퀀스 길이 2048 초과 설정에서의 성능 미검증
6. **두 스트림의 최적 초기화 방법**: LN 스케일을 모두 1.0으로 초기화하는 것이 최적인지 탐색 부족
7. **두 스트림 기여 가중치 학습 동역학**: 왜 HybridNorm 스트림이 Pre-Norm 스트림보다 훨씬 큰 기여를 하게 되는지 메커니즘 미설명
8. **MoE 설정에서 라우팅 안정성**: SiameseNorm이 전문가 라우팅(expert routing) 패턴에 미치는 영향 미분석

---

## 7. 가장 중요한 그림 5개 해석

### 📊 Figure 1 (p.1) — 아키텍처 비교
**해석**: (a) Post-Norm: 잔차 덧셈 후 LN → 크기 제어, 불안정. (b) Pre-Norm: 잔차 브랜치 내부에만 LN → 안정, 크기 비제어. **(c) SiameseNorm**: 두 스트림이 동일한 $F\times N$ 블록을 공유하고, 상단(X스트림)은 출력에 LN, 하단(Y스트림)은 LN 없이 누적. 두 스트림의 출력이 최종 결합되는 구조가 핵심. 이 그림은 SiameseNorm이 단순한 레이어 배치 변경이 아닌 **위상적(topological) 구조 변경**임을 보여준다.

---

### 📊 Figure 2 (p.2) — 세 가지 비교 실험
**2a (Hidden State Norm)**: Pre-Norm은 레이어 증가에 따라 노름이 선형 이상으로 증가(~175). EmbedNorm은 초반 제어에는 성공하나 이후에도 증가. **SiameseNorm은 안정적으로 낮은 노름 유지** → X스트림의 주기적 정규화 효과 실증.

**2b (Training Loss)**: SiameseNorm이 세 방법 중 가장 낮은 손실로 수렴. EmbedNorm은 Pre-Norm보다 오히려 나쁜 손실 → 단순 초기 정규화로는 문제 해결 불가를 입증.

**2c (Layer Pruning Loss)**: SiameseNorm의 각 레이어를 제거하면 Pre-Norm 대비 손실이 더 크게 증가 → SiameseNorm 레이어가 더 능동적으로 활용됨. 특히 중간 레이어(5–10번)에서 차이가 두드러짐.

> 📘 **용어 설명**
> - **레이어 프루닝(Layer Pruning)**: 학습 후 특정 레이어를 제거하여 해당 레이어의 기여도를 측정하는 방법

---

### 📊 Figure 5 (p.8) — 기울기 노름 비교
**해석**: (a) HybridNorm: 초기 수천 스텝에서 기울기 노름이 100 이상으로 폭발. 발산으로 이어짐. (b) SiameseNorm: 기울기 노름이 0.5 이하에서 안정적으로 유지 (Pre-Norm과 동일 수준). (c) HybridNorm-ResiDual: 중간 정도의 스파이크 발생. (d) Pre-Norm: 안정적이나 표현력 제한. **SiameseNorm이 Post-Norm의 불안정성 없이 Pre-Norm의 안정성을 그대로 계승함을 직접 확인**하는 핵심 근거.

---

### 📊 Figure 6 (p.9) — 스트림별 기여 비율
**해석**: (a) Attention 블록, (b) MLP 블록 모두에서 X스트림(HybridNorm, 파란색)과 Y스트림(Pre-Norm, 빨간색)이 **대부분의 레이어에서 유의미한 비율을 유지**. 특정 레이어에서 한 스트림이 지배적인 경우도 있으나, 대체로 두 스트림 모두 활성화됨. 이는 SiameseNorm이 두 스트림 중 하나를 퇴화(degenerate)시키지 않고 실제로 두 스트림을 모두 활용한다는 증거.

---

### 📊 Figure 4 (p.7) — Siamese 토폴로지 효과
**해석**: HybridNorm(노란색)은 학습 초기에 빠르게 발산. HybridNorm-ResiDual(파란색)은 빈번한 loss spike로 불안정. **SiameseNorm(초록색)은 부드럽게 수렴**. Depth-wise Scaling 없는 설정임에도 불구하고 안정적 수렴 → Siamese **토폴로지 자체**가 안정성의 핵심 원천임을 입증. 보조 메커니즘 효과와 분리된 아블레이션으로서 가치가 높음.

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 8-0. 저자 제시 시사점 (p.9, Sec. 7)

> "SiameseNorm improves performance while maintaining robust optimization. We view this approach as a promising foundation for future theoretical and empirical studies on **multi-stream residual architecture design**."

- Pre-Norm 학습 레시피와의 완전한 호환성 → 기존 인프라에 drop-in 적용 가능
- 멀티스트림 잔차 구조 설계의 가능성 공간 확장

### 8-1. 모델 일반화 성능 향상 가능성 🔍

**실증된 일반화 범위** (Table 2):

| 축 | 검증 내용 | 관찰 |
|----|----------|------|
| 깊이 | 10~80레이어, 고정 파라미터 예산 | 깊을수록 향상 폭 확대 (80L: △PPL 2.04) |
| 언어 모달리티 | 밀집 1.3B, MoE 15B | 일관된 PPL 개선 |
| 비전 모달리티 | DeiT-T/S (분류) | Top-1 Acc +1.4%~+1.5% |
| 생성 모달리티 | DiT-B/2, DiT-L/4 (생성) | FID 개선 (더 큰 모델에서 더 큰 향상) |

**일반화 관련 주요 분석**:
1. **깊이 스케일링 일반화**: Pre-Norm이 17레이어에서 최적인 반면, SiameseNorm은 33레이어에서 최적 PPL 달성 → **동일 파라미터 대비 더 깊은 구조를 효율적으로 활용**
2. **MoE 일반화**: 전문가 라우팅을 가진 sparse 모델에서도 안정적으로 동작 → 아키텍처 일반성 시사
3. **학습률 강건성**: $4\times10^{-4}$부터 $2\times10^{-3}$까지 세 LR 설정 모두에서 최고 성능 → 하이퍼파라미터 민감도 낮음

**아직 검증되지 않은 일반화 영역** (본 분석자의 지적):
- 70B+ 초대규모 모델에서의 스케일링 법칙
- 다국어(multilingual) 설정
- 코드, 수학 특화 도메인 파인튜닝
- RLHF/DPO 이후 정렬(alignment) 단계에서의 안정성

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 논문 원문의 참고문헌과 arXiv 공개 정보를 기반으로 합니다. 2025–2026년 일부 논문은 검색 접근이 제한되어 원문 인용에 주로 의존합니다.

#### 주요 관련 연구 타임라인

| 연도 | 논문 | 방법 | SiameseNorm과의 관계 |
|------|------|------|---------------------|
| 2020 | Liu et al., "Understanding the difficulty of training Transformers" (EMNLP) | Post-Norm 불안정성 이론 분석 | SiameseNorm이 해결하려는 근본 문제 정의 |
| 2020 | Xiong et al., "On Layer Normalization in the Transformer Architecture" (ICML) | Pre-Norm 이론적 정당화 | Pre-Norm 안정성의 이론적 토대 |
| 2024 | Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers" (TPAMI) | DeepNorm: $\alpha$-스케일링 + Post-Norm | Depth-wise Scaling의 영감 제공; Table 1에서 비교 |
| 2023 | Xie et al., "ResiDual" (arXiv) | 이중 스트림 구조 (최초) | 구조적으로 가장 유사하나 Y스트림이 블록 입력에 미연결 → 기울기 정보 손실 |
| 2025 | Li et al., "Mix-LN" (ICLR 2025) | 앞 레이어 Post-Norm, 뒷 레이어 Pre-Norm | SiameseNorm이 일반화 가능한 하위 케이스 |
| 2025 | Zhuo et al., "HybridNorm" (arXiv) | Attention 내부 LN 삽입 | SiameseNorm의 Post-Norm 스트림으로 채택 |
| 2025 | Kim et al., "Peri-LN" (arXiv) | LN을 잔차 블록 주변에 배치 | Table 1에서 직접 비교; SiameseNorm에 열세 |
| 2025 | Zhu et al., "Hyper-Connections" (ICLR 2025) | 적응형 폭 확장 + Pre/Post 혼합 | SiameseNorm이 대부분 설정에서 우세 (Table 1, Setting D에서 HC는 spike 발생) |
| 2026 | Wang et al., "SpanNorm" (arXiv) | 레이어 스팬에 걸친 정규화 | $\eta=2\times10^{-3}$에서 발산; SiameseNorm은 안정 |
| 2026 | Chen & Wei, "Post-LayerNorm is Back" (arXiv:2601.19895) | 안정적 Post-Norm 학습 방법 | 유사한 문제 의식; SiameseNorm과 접근법 비교 필요 |

#### 핵심 차별점 분석

**vs. ResiDual (Xie et al., 2023)** (p.14, Appendix A.2):
ResiDual의 야코비안:

$$\frac{\partial S_{j+1}}{\partial S_j} = \begin{bmatrix} J_{\text{LN}^X_j}(\mathbf{I} + J_{F_j}) & \mathbf{0} \\ J_{F_j} & \mathbf{I} \end{bmatrix}$$

우상단 블록이 **$\mathbf{0}$** → Y스트림이 X스트림의 이후 변환으로부터 기울기를 받지 못함. SiameseNorm은 이 블록이 $J_{\text{LN}^X_j} J_{F_j} J_{\text{LN}^Y_j}$로 채워짐 → **더 풍부한 기울기 정보 흐름**.

**vs. Hyper-Connections (Zhu et al., 2025a)**:
HyperConnections는 폭(width) 차원의 확장으로 접근, SiameseNorm은 정규화 배치의 위상적 분리. mHC 연구에서도 $H_\text{res}$ 없이는 불안정하며 Pre-Norm 편향 초기화 필요. SiameseNorm은 균등 초기화(LN scale=1.0)에서도 안정적.

---

#### 앞으로의 연구에 미치는 영향

1. **멀티스트림 잔차 설계의 정당화**: SiameseNorm은 단일 스트림 Transformer의 구조적 한계를 이론+실험으로 명확히 하여, 앞으로의 아키텍처 연구에서 **위상적 설계(topological design)**를 고려해야 함을 확립.

2. **정규화 배치의 재고**: 지금까지 Pre-Norm vs. Post-Norm을 단순 선택 문제로 보던 것에서, **두 정규화 위상의 기능적 역할을 분리**하는 새로운 시각을 제공.

3. **스케일링 연구의 방향**: 파라미터 수 증가보다 **아키텍처 위상 설계**로도 효과적 깊이를 늘릴 수 있음을 시사. 단순한 모델 크기 확장 연구에 보완적 방향 제시.

4. **MoE + SiameseNorm 결합**: 15B MoE 실험이 가능성을 열었으나, 전문가 선택(routing)과 두 스트림 간 상호작용 메커니즘 연구가 필요.

---

#### 앞으로 연구 시 고려할 점

1. **이론적 수렴 분석**: Depth-wise Scaling $\frac{1}{\sqrt{l+1}}$이 두 스트림의 크기 균형을 언제 보장하는지, 어떤 조건에서 발산을 방지하는지 엄밀한 분석 필요.

2. **Sub-stream 선택의 원칙**: Table 4에서 HybridNorm > Post-Norm 임이 확인되었으나, 더 좋은 Post-Norm 변형(예: SpanNorm의 안정화 버전, 미래의 신규 방법)과 결합했을 때의 성능 탐색.

3. **스트림 비대칭성 학습**: HybridNorm 스트림이 Logit Lens에서 지배적임에도 두 스트림을 균등 초기화하는 것이 최적인지, **비대칭 초기화** 탐색 필요.

4. **장문맥(Long Context) 일반화**: RoPE와 결합한 SiameseNorm이 긴 시퀀스에서 두 스트림 간 위치 인코딩 상호작용을 어떻게 처리하는지 미검증.

5. **양자화(Quantization) 호환성**: 두 스트림의 LN 파라미터가 추론 시 양자화(INT8/FP8)에 어떤 영향을 받는지 확인 필요.

6. **지식 증류(Knowledge Distillation)**: SiameseNorm 교사 모델에서 Pre-Norm 학생 모델로의 증류 시 두 스트림의 정보를 어떻게 활용할지.

---

## 📚 참고자료

**주요 참고 논문 (원문 인용 기준)**:
- Li, T. et al. (2025). "SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm." *arXiv:2602.08064v2*
- Vaswani, A. et al. (2017). "Attention is All You Need." *NeurIPS*
- Ba, J.L. et al. (2016). "Layer Normalization." *arXiv:1607.06450*
- Wang, H. et al. (2024). "DeepNet: Scaling Transformers to 1,000 Layers." *TPAMI*
- Xie, S. et al. (2023). "ResiDual: Transformer with Dual Residual Connections." *arXiv:2304.14802*
- Zhuo, Z. et al. (2025). "HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization." *arXiv:2503.04598*
- Zhu, D. et al. (2025a). "Hyper-Connections." *ICLR 2025*
- Li, P. et al. (2025). "Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN." *ICLR 2025*
- Wang, C. et al. (2026). "SpanNorm: Reconciling Training Stability and Performance in Deep Transformers." *arXiv:2601.22580*
- Kim, J. et al. (2025). "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture." *arXiv:2502.02732*
- Gromov, A. et al. (2025). "The Unreasonable Ineffectiveness of the Deeper Layers." *ICLR 2025*
- Sun, W. et al. (2025). "The Curse of Depth in Large Language Models." *arXiv:2502.05795*
- Chen, C. & Wei, L. (2026). "Post-LayerNorm is Back: Stable, Expressive, and Deep." *arXiv:2601.19895*
- Xiong, R. et al. (2020). "On Layer Normalization in the Transformer Architecture." *ICML*
- Liu, L. et al. (2020). "Understanding the Difficulty of Training Transformers." *EMNLP*
- Groeneveld, D. et al. (2024). "OLMo: Accelerating the Science of Language Models." *ACL*
- Geva, M. et al. (2021). "Transformer Feed-Forward Layers Are Key-Value Memories." *EMNLP*
