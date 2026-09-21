# The Emergent Symbolic Structure of Artificial Neural Networks

> **⚠️ 정확도 안내**: 본 분석은 제공된 PDF 원문에 근거하며, 논문의 arXiv 제출일(2026년 8월 30일)을 감안할 때 일부 "최신 연구 비교" 항목은 현재 필자의 지식 한계(2025년 초반)로 인해 불완전할 수 있음을 명시합니다.

---

## 1. Executive Summary (10문장 이내)

현대 AI의 핵심 역설은 **명시적으로 기호 구조를 갖지 않는 신경망이 논리·언어·수학 등 기호 처리가 필요한 영역에서 탁월한 성능을 보인다**는 점이다.  
본 논문은 이 역설에 대해 "신경망의 내부 벡터 표현이 암묵적으로 기호 구조를 실현한다"는 가설을 제시하고 DISCOVER 방법론으로 검증한다.  
DISCOVER는 신경망의 벡터 표현을 **텐서 곱 표현(Tensor Product Representation, TPR)** 기반의 닫힌 형태 방정식으로 근사하는 분석 기법이다.  
분석 결과, MLP·GRU·Transformer 등 세 계열의 소규모 신경망과 7개 대형 언어 모델(LLM) 모두에서 TPR 구조가 창발적으로 발견되었다.  
GPT-OSS를 대상으로 수학·논리·코드·언어 등 4개 상징적 영역에서 실험한 결과, 태스크별 TPR 근사치가 원래 모델의 성능을 거의 그대로 재현했다.  
더 나아가, DISCOVER가 식별한 TPR 구조를 통해 내부 표현에 정밀한 인과 개입(causal intervention)을 가하면 모델 행동이 예측 방향으로 변경됨을 확인했다.  
DISCOVER는 훈련 중 보지 못한 역할-채움자(role-filler) 쌍에도 일반화되어, 신경망이 단순 암기가 아닌 **체계적 결합**을 사용함을 증명한다.  
이 결과는 기호주의와 연결주의의 오랜 대립을 **연결론적 기호주의(limitivism)** 관점에서 조화시킬 가능성을 제시한다.

---

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|---|---|
| **핵심 역설** | 기호 처리가 필요한 영역(언어, 수학, 논리)에서 벡터 기반 신경망이 기호 시스템을 능가함 |
| **기존 설명의 한계** | 선형 표현 가설(Linear Representation Hypothesis)은 구조적 순서 정보를 설명하지 못함 (예: "cats chase dogs"와 "dogs chase cats"를 구분 불가) |
| **필요성** | 기계적 해석가능성(mechanistic interpretability) 연구의 핵심 질문인 "LLM이 내부적으로 어떤 구조로 정보를 인코딩하는가"에 답하기 위함 |
| **목적** | 신경망 내부 표현이 TPR 구조를 암묵적으로 실현하는지 실험적으로 검증하고, 이를 통해 모델 행동에 대한 인과적 통제 방법 제공 |

> **💡 용어 설명: 선형 표현 가설(Linear Representation Hypothesis)**
> 신경망 내부의 각 벡터가 여러 개념 $c_i$의 벡터 합 $\sum_i e(c_i)$으로 표현된다는 가설. 덧셈 연산이 순서에 무관하므로 구조 정보를 포착하지 못하는 한계가 있음.

> **💡 용어 설명: 기계적 해석가능성(Mechanistic Interpretability)**
> AI 시스템, 특히 LLM의 내부 작동 방식을 이해하려는 연구 분야. 어떤 뉴런·벡터 방향·회로(circuit)가 어떤 기능을 담당하는지 규명하는 것이 목표.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 (실험/결과) | 해당 위치 |
|---|---|---|
| **소규모 신경망에서 TPR 창발** | 3개 아키텍처(MLP, GRU, Transformer), 3개 태스크에서 양방향(bidirectional) 역할 체계가 거의 완벽한 근사 정확도 달성 (최저 97.3%) | Section 4, Figure 4.1 |
| **LLM 7개 모두에서 TPR 창발** | 주기(period) 토큰 인코딩을 TPR로 근사 시 적절한 역할 체계에서 높은 approximation accuracy 달성 | Section 5, Figure 5.2 |
| **4개 상징 영역에서 태스크별 TPR** | GPT-OSS에서 `task-specific (all)` 역할 체계가 다른 모든 체계를 압도하며, 원래 모델 정확도와 최대 2.36% 차이 이내 | Section 6, Figure 6.2 |
| **TPR 구조가 행동에 인과적 영향** | 내부 표현의 역할-채움자 성분만 교체하면 GPT-OSS 행동이 예측 방향으로 변경됨 (31개 유형, 평균 개입 정확도 0.903) | Section 7, Figure 7.1–7.4 |
| **체계적 결합(systematic binding) 존재** | 훈련 중 미출현 역할-채움자 쌍에도 DISCOVER가 일반화되며 강한 기회 기준선(strong chance baseline) 초과 | Section 8, Figure 8.1–8.3 |

---

### 2-1. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

#### 해결하고자 하는 문제

벡터 기반 신경망이 기호 처리(언어 구조, 산술, 논리 등)에서 높은 성능을 보이는 이유를 설명하는 **메커니즘 해석** 문제. 특히 **결합 문제(binding problem)**: 신경망이 특징(feature)과 위치(position)를 어떻게 묶는가.

> **💡 용어 설명: 결합 문제(Binding Problem)**
> 인지과학에서 유래한 개념으로, "cats chase dogs"에서 *cats*가 주어 역할을, *dogs*가 목적어 역할을 담당한다는 정보를 어떻게 하나의 표현에 담는가 하는 문제. 단순한 벡터 합으로는 순서 정보를 구별할 수 없어 이 문제가 발생함.

---

#### 제안하는 방법: DISCOVER + 선형변환 TPR

**텐서 곱 표현 (Linearly-transformed TPR)**

기호 구조 $S$가 채움자(filler) 집합 $\{f_i\}$와 역할(role) 집합 $\{r_i\}$의 쌍으로 표현될 때, 전체 인코딩 벡터 $E$는:

$$E = W\left(\sum_i f_i \otimes r_i\right) + b$$

- $f_i \in \mathbb{R}^{d_f}$: 채움자 임베딩 벡터 (예: 단어 *cats*의 벡터)
- $r_i \in \mathbb{R}^{d_r}$: 역할 임베딩 벡터 (예: *subject* 역할의 벡터)
- $\otimes$: 텐서 곱(tensor product) — 두 벡터를 입력받아 행렬 반환
- $W \in \mathbb{R}^{d_{\text{hidden}} \times (d_f \cdot d_r)}$: 선형 변환 행렬 (행렬을 벡터로 리사이즈)
- $b \in \mathbb{R}^{d_{\text{hidden}}}$: 편향(bias) 벡터

> **💡 용어 설명: 텐서 곱(Tensor Product)**
> 두 벡터 $f \in \mathbb{R}^{d_f}$와 $r \in \mathbb{R}^{d_r}$의 텐서 곱 $f \otimes r$는 크기 $d_f \times d_r$의 행렬로, 모든 $(i, j)$ 쌍에 대해 $(f \otimes r)_{ij} = f_i \cdot r_j$로 정의됨. 채움자와 역할의 결합 정보를 하나의 행렬로 인코딩하는 수학적 장치.

**인과 개입 수식 (Constituent Surgery)**

원래 TPR에서 특정 역할-채움자 쌍을 교체하는 조작:

$$TPR(Q, M, Z) = TPR(1^{st}: Q) + TPR(2^{nd}: M) + TPR(3^{rd}: Z) \tag{1}$$

$$TPR(Q, M, Z) - TPR(3^{rd}: Z) + TPR(3^{rd}: U) = TPR(Q, M, U) \tag{2}$$

대상 모델(target model)에 동일한 조작 적용:

$$\text{target}(Q, M, Z) - TPR(3^{rd}: Z) + TPR(3^{rd}: U) = \text{target}(Q, M, U) \tag{3}$$

- $TPR(r:f) = W(r \otimes f)$: DISCOVER 모델이 예측한 역할-채움자 쌍의 표현
- $\text{target}(s)$: 대상 신경망이 입력 $s$에 대해 생성한 내부 표현

**DISCOVER 학습 목표**

$$\hat{\theta} = \arg\min_{\theta} \sum_{s} \|E_s - E^{TPR}_s(\theta)\|^2$$

- $E_s$: 대상 모델이 입력 $s$에 대해 생성한 인코딩 벡터
- $E^{TPR}_s(\theta)$: DISCOVER 모델(파라미터 $\theta$ = 채움자/역할 임베딩 + $W$, $b$)이 생성한 TPR 근사 벡터
- 학습은 MSE(평균 제곱 오차) 최소화

**OOD 일반화를 위한 정규화 손실**:

$$L_{reg} = L_{MSE} + \lambda L_{2,1}(W_{emb,f}) + \lambda L_{2,1}(W_{emb,r}) \tag{4}$$

- $\lambda$: 정규화 강도 하이퍼파라미터
- $L_{2,1}$: 행렬 각 열의 $L_2$ 놈을 먼저 계산한 후 $L_1$ 놈을 계산하는 정규화. 임베딩 공간에서 불필요한 차원을 제거하는 효과.

> **💡 용어 설명: $L_{2,1}$ 정규화**
> 행렬 $W$에 대해 $L_{2,1}(W) = \sum_j \sqrt{\sum_i W_{ij}^2}$ 로 정의. 임베딩 행렬에서 특정 행(차원) 전체를 0으로 만들도록 유도하여 희소하고 체계적인 임베딩 학습을 촉진함.

---

#### 모델 구조

```
[입력 시퀀스]
      │
  ┌───▼────────────────────────────────┐
  │        Target Model (Encoder)       │
  │  (MLP / GRU / Transformer / LLM)   │
  └───────────────┬───────────────────┘
                  │ E (인코딩 벡터)
  ┌───────────────▼───────────────────┐
  │     DISCOVER Model (TPR Encoder)   │
  │  E^TPR = W(Σ fᵢ ⊗ rᵢ) + b       │
  └───────────────┬───────────────────┘
                  │ E^TPR (근사 벡터)
  ┌───────────────▼───────────────────┐
  │        Target Model (Decoder)      │
  │     (원래 모델의 디코더 재사용)       │
  └───────────────┬───────────────────┘
                  │
            [출력 / 평가]
```

**근사 정확도(Approximation Accuracy)**: 디코더에 $E^{TPR}$을 입력했을 때 올바른 출력을 생성하는 테스트 예제의 비율.

---

#### 성능 향상

| 실험 범위 | 최고 성능 | 비교 기준 |
|---|---|---|
| 문자 시퀀스 모델 (bidirectional) | 평균 >99% (최저 97.3%) | bag-of-words: ~0% |
| LLM 주기 인코딩 - SVO 문장 | 100% (모든 LLM, 모든 층) | bag-of-words: ~0% |
| LLM 주기 인코딩 - 복잡 문장 | ~70~96% (모델·층별 상이) | bag-of-words: ~0% |
| GPT-OSS 인과 개입 | 평균 0.903 (31개 유형) | 무작위 기준선 대비 유의미 |
| OOD 일반화 (novel role-filler) | 강한 기회 기준선($\frac{1}{n!}$) 대폭 초과 | 원자적 쌍(Atomic Pair) 모델: 완전 실패 |

---

#### 한계

1. **감독형(supervised) DISCOVER**: 역할 체계(role scheme) 가설을 인간이 사전 정의해야 함 — 자동화 불가
2. **생성 메커니즘 미설명**: 신경망이 어떻게 TPR을 *생성*하는지(연산 메커니즘)는 분석하지 않음
3. **단일 LLM 심층 분석**: GPT-OSS 집중 분석에 >3,000 GPU 시간 소요 — 확장성 제한
4. **부분적 근사**: 평균 개입 정확도 0.903 — 완전한 기호 시스템이 아닌 근사적 실현
5. **복잡 문장 한계**: 복잡한 구문 구조 처리 시 정확도가 단순 SVO 대비 현저히 낮음 (Section 5)
6. **산술 OOD 실패**: 그림 8.3에서 arithmetic 태스크는 unseen role-filler pairs에 대한 일반화가 저조 — ⚠️ **통계적 취약 지점**

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|---|---|
| 신경망과 기호 표현의 외관적 불일치 | p.1, Figure 1.1 |
| TPR의 수학적 형식화 | pp.3–5, Figure 2.1 |
| DISCOVER 절차 설명 | pp.5–8, Figure 3.1 |
| 역할 체계별 성능 비교 (GRU 반전 태스크) | p.7, Figure 3.2 |
| 4개 아키텍처 × 3개 태스크 DISCOVER 결과 | pp.9–10, Figure 4.1 |
| LLM 주기 인코딩 실험 설계 | pp.10–12, Figure 5.1 |
| LLM 7개 모델 DISCOVER 결과 | p.13, Figure 5.2 |
| GPT-OSS 6개 태스크 정의 | pp.15, Table 1 |
| GPT-OSS 5개 역할 체계 정의 | pp.17–18, Table 2 |
| GPT-OSS DISCOVER 결과 | pp.18–19, Figure 6.2 |
| 인과 개입 수식 | pp.19, Eq. (1)–(3) |
| 문자 시퀀스 모델 인과 개입 결과 | p.20, Figure 7.1 |
| LLM 주기 인코딩 인과 개입 결과 | p.20, Figure 7.2 |
| GPT-OSS 인과 개입 예시 | pp.21–22, Figure 7.3, 7.4 |
| 역할-채움자 체계적 결합 테스트 | pp.23–24, Figure 8.1–8.3 |
| 정규화 손실 함수 | p.57, Eq. (4) |
| 화이트박스 모델 검증 | p.58, Figure M.1 |
| 역할 수와 DISCOVER 성능의 관계 | pp.58–59, Table 7 |

---

## 4. 저자 보고 결과 vs. 내 해석 분리

### 연구 주제

**저자 보고**: "우리는 신경망의 벡터 표현이 암묵적으로 TPR로 근사될 수 있는지 테스트한다. 이 연구는 기호적 지능 개념과 현대 AI의 벡터 기반 특성을 조화시킬 가능성을 제공한다." (Abstract)

**내 해석**: 이 연구는 해석가능성(interpretability) 연구의 패러다임 전환을 시도한다. 기존 연구가 "어떤 정보가 벡터 *안에* 있는가(in)"를 묻는 반면, 이 논문은 "벡터의 구조가 무엇인가(*of*)"를 묻는다는 점에서 방법론적으로 독창적이다.

---

### 방법 (수식)

**저자 보고**: TPR 근사 $E^{TPR} = W(\sum_i f_i \otimes r_i) + b$가 대상 모델 인코딩 $E$를 재현하면 TPR 구조가 존재한다고 결론. 평가는 원래 디코더에 $E^{TPR}$을 입력했을 때의 근사 정확도로 정량화. (Section 3.2)

**내 해석**: 이 방법의 핵심 강점은 근사가 성공할 **보장이 없다**는 점이다(논문도 이를 명시, p.7). 즉 성공 자체가 정보를 담고 있다. 그러나 단점은 DISCOVER가 더 풍부한 역할 체계(예: bidirectional은 left-to-right를 포함)를 사용하면 더 단순한 실제 구조를 가린 채로도 성공할 수 있다는 점(p.8, Section 3.5)이다. 저자들도 이를 인정하나 "어떤 TPR이든 근사하면 된다"는 목표를 설정해 이를 회피한다.

---

### 결과

**저자 보고 (GPT-OSS 복잡 문장 예시)**: "GPT-OSS 중간 층에서 주기 인코딩으로부터 문장 복원 시, LLM 원래 인코딩 사용 시 정확도 0.71인 반면 bidirectional DISCOVER 근사치 사용 시 0.96을 기록했다." (p.14, Section 5.4)

**내 해석**: ⚠️ **이 결과는 표면적으로 역설적**이다 — DISCOVER 근사치가 원래 인코딩보다 더 좋은 성능을 낸다는 것은, 원래 LLM 인코딩에 "노이즈 섞인 TPR 구조"가 있음을 시사한다. 저자의 해석(p.26, "limitivism" 지지 증거)은 타당하지만, 대안적 해석도 가능하다: period-unpacking 모델이 LLM 인코딩의 *실제* 구조 대신 DISCOVER가 강요한 편향을 활용하는 것일 수 있다. 이를 통제하기 위한 ablation이 충분하지 않다.

> **💡 용어 설명: Limitivism(한계론)**
> Smolensky(1988)가 제안한 관점으로, 신경망은 기호 시스템의 극한(limit)에 근접하지만 실제로는 근사적으로만 구현한다는 입장. 기호 제거론(eliminativism)과 기호 구현론(implementationalism)의 중간.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 구분 | 내용 | 취약 이유 |
|---|---|---|
| ⚠️ **GPT-OSS 단독 심층 분석** | 6개 태스크 전체를 단 1개 LLM으로 분석 | 모델 선택 편향 가능성, 다른 LLM으로의 일반화 미검증 |
| ⚠️ **산술 OOD 일반화 실패** | Figure 8.3에서 unseen pair 증가 시 기준선 초과 실패 | 저자는 "noise" 탓으로 설명하나 불충분한 검증 |
| ⚠️ **100회 개입 표본 크기** | GPT-OSS 인과 개입 각 유형당 n=100 | 유형별 분산이 클 경우 신뢰구간 불안정 가능 |
| ⚠️ **역할 수 혼재 비교** | task-specific(all) vs. bidirectional(all)에서 역할 어휘 크기 상이 (Table 7) | 파라미터 수 차이가 결과에 기여했을 가능성 (저자 Appendix N에서 부분적으로 반박) |
| ⚠️ **복잡 문장 낮은 기준 정확도** | period-unpacking 정확도 ~50~70%에서 DISCOVER 평가 | 기저(base) 성능이 낮아 개선 여지와 의미 해석이 어려움 |
| ⚠️ **층별 z-scoring 처리** | LLM 실험에서 표현 정규화 후 분석 | z-scoring이 제거한 정보가 DISCOVER 결과에 영향을 미쳤는지 불명확 (저자 Appendix E에서 등가성 논증하나 가정 존재) |
| 🚫 **비교 불가 수치** | 소규모 모델(GRU, hidden=64) vs. GPT-OSS(~20B) 간 근사 정확도 직접 비교 | 모델 규모, 태스크 복잡도, 평가 방식이 모두 상이 |

---

## 6. 논문이 답하지 않는 질문

1. **TPR을 어떻게 생성하는가?** 저자가 명시적으로 답하지 않음: "우리의 목표는 표현의 구조를 이해하는 것이지, 네트워크가 어떻게 그 구조를 생성하는지 설명하는 것이 아니다." (p.8, Section 3.5)

2. **어떤 종류의 VSA(Vector Symbolic Architecture)가 사용되는가?** 선형변환 TPR이 모든 VSA를 포함하므로, DISCOVER 성공이 어느 특정 VSA 계열을 지시하는지 특정 불가. (p.27, Section 9.2)

3. **뇌도 TPR을 사용하는가?** 저자는 가능성을 열어두나 "현재 결과로는 강한 주장을 할 수 없다"고 명시. (p.28, Section 9.6)

4. **합성적 표현이 왜 합성적 일반화로 이어지지 않는가?** TPR 표현이 있음에도 신경망이 새로운 역할-채움자 조합에 행동적으로 일반화하지 못하는 역설을 완전히 설명하지 못함. (p.28, Section 9.5)

5. **비정형(partially systematic) 도메인에서의 표현 구조는?** 논문은 완전 체계적 태스크만 분석하며, 자연어의 퍼지(fuzzy)·통계적 측면을 포함한 도메인은 미분석. (p.30, Section 11)

6. **층별 표현 변화는?** DISCOVER는 모든 층에 동일한 구조 가설을 적용하며, 층별 정보 변화는 후속 연구로 남겨둠. (p.50, Appendix H.3)

7. **비감독(unsupervised) DISCOVER의 확장성?** 역할 체계를 인간이 정의해야 하는 한계를 자동화하는 방법이 미완성. (p.8, Section 3.5)

8. **비영어 언어·다중 모달 모델에서도 TPR이 창발하는가?** 분석 범위가 영어 중심 태스크로 제한됨.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1.1 (p.1) — 신경망과 기호 표현의 외관적 불일치

**내용**: 동일한 문장 "the poet near the judge saw the author"의 두 가지 표현 — (좌) 구문 트리, (우) 49차원 연속 벡터.

**해석**: 논문의 중심 역설을 시각적으로 제시. 트리는 이산적·구조적이고, 벡터는 연속적·비구조적으로 보인다. 그러나 이것이 *외관상의* 차이임을 논문 전체가 반박한다. 이 그림은 "왜 이 연구가 필요한가"를 직관적으로 답한다.

---

### Figure 2.1 (p.4) — 선형변환 TPR의 구조

**내용**: (A) TPR 생성 4단계, (B) 텐서 곱 행렬 예시, (C) 특수·일반 경우의 기하학적 해석 비교.

**해석**: 핵심 통찰은 (C)의 일반 경우(general case)다. 수치 목록으로 보면 무작위처럼 보이는 벡터들이, 기하학적으로는 명확한 구조를 유지한다(회전·신축된 좌표계). 이는 신경망이 인간에게는 불투명해 보이는 방식으로 구조를 인코딩할 수 있다는 논문의 핵심 주장을 지지한다. "좌표계가 뒤틀렸다는 것이 구조가 없다는 뜻이 아니다."

---

### Figure 4.1 (p.10) — 문자 시퀀스 모델 DISCOVER 결과

**내용**: 4개 아키텍처 × 3개 태스크에서 5가지 역할 체계의 근사 정확도 막대 그래프.

**해석**: 세 가지 패턴이 두드러진다:
1. **bidirectional 우세**: 거의 모든 조합에서 압도적으로 높음 → TPR 구조가 아키텍처와 태스크에 무관하게 일반적임
2. **태스크별 비대칭**: 복사(copying)는 left-to-right 우세, 반전(reversing)은 right-to-left 우세 → 태스크 특성이 어떤 위치 체계가 선호되는지 결정
3. **Wickelroles 실패**: 729개 역할에도 불구하고 21개 bidirectional보다 낮음 → DISCOVER 성능이 단순히 파라미터 수로 결정되지 않음을 증명

---

### Figure 6.2 (p.19) — GPT-OSS 6개 상징 태스크 DISCOVER 결과

**내용**: 5개 역할 체계 × 6개 태스크에서 근사 정확도. 점선은 GPT-OSS의 원래 정확도.

**해석**: `task-specific (all)`이 모든 태스크에서 1위이며, 원래 GPT-OSS 정확도(점선)에 근접한다. 특히 주목할 점은 `bidirectional (self)` (자기 자신만 인코딩)가 `bidirectional (all)` (이전 토큰 전체 인코딩)보다 대체로 낮다는 점 → 각 토큰의 표현이 자기 자신뿐 아니라 **문맥 전체의 구조적 정보**를 누적한다는 증거. 또한 `bag-of-words`가 거의 0에 가까워 순서/구조가 없는 표현으로는 복잡한 태스크 재현이 불가능함을 재확인.

---

### Figure 8.2 (p.24) — LLM 주기 인코딩 OOD 일반화

**내용**: 7개 LLM의 목록/문장 실험에서 미출현 역할-채움자 쌍 수가 늘어날수록의 근사 정확도 변화. 점선은 강한 기회 기준선($1/n!$).

**해석**: 모든 LLM에서 근사 정확도가 기준선을 크게 초과하며, 이는 **체계적 결합**의 결정적 증거다. 만약 신경망이 역할-채움자 쌍을 원자적으로 기억한다면, 미출현 조합에 대한 일반화는 불가능하고 정확도는 기준선으로 수렴해야 한다. 그러나 실제로는 새로운 조합에서도 높은 정확도를 유지한다. 이 그림 하나가 논문의 가장 강력한 주장인 "신경망이 단순 암기가 아닌 체계적 구성을 사용한다"를 가장 설득력 있게 지지한다.

---

## 8. 결론, 시사점, 후속 연구

### 저자들이 제시한 시사점

1. **연결론-기호론 통합**: 하나의 시스템이 신경망이면서 동시에 기호적일 수 있음 → limitivism 지지
2. **기계적 해석가능성에의 시사**: 분석 단위가 개별 뉴런이나 방향(direction)이 아닌 **역할-채움자의 곱(multiplicative combination)** 이어야 함 (Section 9.4)
3. **명시적 TPR 통합 연구에의 기여**: 어떤 TPR이 자연스럽게 창발하는지 이해하면, 명시적 TPR 통합 아키텍처 설계에 활용 가능

### 저자들이 제시한 후속 연구 방향

| 방향 | 설명 |
|---|---|
| 생성 메커니즘 분석 | 신경망이 어떻게 TPR을 *계산*하는지 (Transformer 회로 분석 등) |
| 비감독 DISCOVER | 역할 체계를 자동 발견하는 방식으로 확장 |
| 점진적 기호 시스템 | 연속-이산 차원을 통합하는 Gradient Symbol Systems로 확장 |
| 뇌 데이터 적용 | DISCOVER를 뇌 기록 데이터에 적용해 신경 표현 구조 분석 |
| 비정형 도메인 확장 | 자연어의 퍼지·통계적 측면을 포함하는 부분 체계적 도메인 분석 |

---

### 8-1. 모델의 일반화 성능 향상 가능성

본 논문은 일반화와 관련하여 중요한 **역설**을 발견했다:

> "신경망이 합성적 표현을 가지고 있음에도 불구하고, 새로운 역할-채움자 조합에 행동적으로 일반화하지 못하는 이유는 무엇인가?" (Section 9.5)

저자의 잠정적 설명: 신경망이 **훈련 중 만난 역할-채움자 조합에 대해서만** 합성적 표현을 발전시킬 수 있다. 즉, *표현의 합성성*이 있어도 *행동의 합성성*이 따르지 않을 수 있다.

**일반화 향상을 위한 DISCOVER 기반 전략 (내 해석)**:

1. **정규화 유도 일반화**: $L_{2,1}$ 정규화가 OOD 성능을 크게 향상시킨 결과(예: interleaving bottleneck Transformer에서 0.54 → 0.97, Table 5)는, **희소하고 체계적인 임베딩 학습이 일반화의 핵심**임을 시사한다. 훈련 시 TPR 구조를 강제하는 귀납적 편향이 일반화를 향상시킬 수 있다.

2. **명시적 TPR 아키텍처**: DISCOVER가 발견한 역할-채움자 구조를 훈련 시 명시적으로 강제하는 아키텍처(Soulos et al., 2023, 2024 참조)는 새로운 조합에 대한 행동적 일반화를 개선할 가능성이 있다.

3. **정밀 데이터 증강**: DISCOVER로 식별된 역할 체계를 기반으로 훈련 데이터의 미출현 역할-채움자 조합을 타겟으로 증강하면, 모델이 해당 조합에 대해 합성적 표현을 발전시킬 수 있다.

4. **인과 개입 기반 디버깅**: DISCOVER + Constituent Surgery를 활용하여 모델이 실패하는 일반화 사례에서 어떤 역할-채움자 표현이 부재하거나 부정확한지 진단하고, 이를 보완하는 fine-tuning 전략 설계 가능.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 본 논문의 arXiv 제출일이 2026년 8월 30일이므로, 논문 자체가 이미 2025~2026년 연구를 인용하고 있습니다. 아래 비교는 논문이 인용한 연구들을 중심으로 구성하며, 2025년 이후 필자가 확인하지 못한 연구에 대한 추측은 배제합니다.

| 연구 | 본 논문과의 관계 | 주요 차이 |
|---|---|---|
| **Soulos et al. (2020)** "Discovering the Compositional Structure..." | DISCOVER의 전신, Constituent Surgery 도입 | 소규모 RNN에만 적용, LLM 미분석 |
| **Park et al. (2024)** "Linear Representation Hypothesis" | 본 논문이 확장하는 배경 이론 | 구조(순서) 정보를 설명하지 못하는 한계 → 본 논문이 TPR로 해결 |
| **Templeton et al. (2024)** "Scaling Monosemanticity (SAE)" | 표현 분석의 경쟁 방법론 | SAE는 원자적 특징 분해(bottom-up), 본 논문은 역할-채움자 구조 재구성(encoding-based) |
| **Geiger et al. (2024)** "Distributed Alignment Search (DAS)" | 인과 개입 방법론 비교 | DAS는 인과 개입을 위해 직접 훈련, DISCOVER는 표현 가설에서 개입이 파생됨 |
| **Enyan & McCoy (2026)** "A Unifying Perspective..." | 본 논문의 직접 후속 | 기존 해석가능성 방법의 성공이 암묵적 TPR 구조로 설명됨을 보임 |
| **Smolensky et al. (2025)** "Mechanisms of symbol processing for ICL" | 이론적 보완 | TPR이 in-context learning을 설명할 수 있음을 이론적으로 제시 |
| **Yang et al. (2025b)** "Emergent Symbolic Mechanisms..." | 동기 연구 | LLM의 추상적 추론에서 기호 메커니즘 발견 (행동 분석), 본 논문은 표현 분석 |
| **Huang & Hahn (2026)** | 보완적 방법론 | 비감독 방식으로 변수 유사 부분공간(variable-like subspaces) 발견 |

**본 논문이 앞으로의 연구에 미치는 영향**:

1. **해석가능성 연구 방향 전환**: "어떤 특징이 벡터 안에 있는가" → "벡터의 전체 구조가 무엇인가"로의 전환을 촉진
2. **신경-기호 통합 연구**: TPR 창발이 광범위하게 확인되면서, 명시적 기호 구조를 신경망에 통합하는 연구(neuro-symbolic AI)의 이론적 토대 강화
3. **인과 개입 방법론의 정교화**: 단순 패칭(patching)이나 스티어링(steering)보다 이론적으로 기반 있는 개입 방법 제공

**앞으로 연구 시 고려할 점 (내 추가 제안)**:

1. **비감독 역할 체계 발견**: 현재 감독형 DISCOVER의 가장 큰 한계. Soulos et al. (2020)의 비감독 접근을 현대 LLM 스케일로 확장하는 것이 핵심 과제.

2. **다국어·다중 모달 검증**: 영어 중심 태스크에 제한된 현재 분석을 비영어권 언어와 시각-언어 모델로 확장해야 일반성 주장 강화 가능.

3. **계층별(layer-by-layer) 구조 변화 추적**: 층이 깊어질수록 어떻게 TPR 구조가 변화하는지 분석하면, Transformer의 처리 과정(점진적 정제)에 대한 이해 심화.

4. **훈련 동역학(training dynamics) 연구**: TPR이 언제, 어떻게 창발하는가? 초기 훈련부터 추적하면 "왜 신경망이 이 구조를 학습하는가"를 설명 가능.

5. **교차 모델 표현 정렬**: 동일한 DISCOVER 모델이 다른 LLM의 표현도 근사할 수 있는가? 가능하다면 LLM들이 보편적 표현 공간을 공유한다는 증거가 됨.

6. **실용적 안전성 응용**: DISCOVER + Constituent Surgery를 통한 정밀 표현 편집은 모델 편향 제거, 사실 수정(factual editing), 행동 조정 등 AI 안전성 연구에 직접 응용 가능.

---

## 참고 자료

**논문 원문**:
- McCoy, R.T., Soulos, P., Linzen, T., & Smolensky, P. (2026). *The Emergent Symbolic Structure of Artificial Neural Networks*. arXiv:2608.29530v1.

**논문 내 인용 주요 문헌**:
- Smolensky, P. (1990). Tensor product variable binding and the representation of symbolic structures in connectionist systems. *Artificial Intelligence*, 46(1-2), 159–216.
- Soulos, P., McCoy, R.T., Linzen, T., & Smolensky, P. (2020). Discovering the Compositional Structure of Vector Representations with Role Learning Networks. *BlackboxNLP*.
- Park, K., Choe, Y.J., & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. *ICML*.
- Geiger, A., Wu, Z., et al. (2024). Finding alignments between interpretable causal variables and distributed neural representations. *Causal Learning and Reasoning*, PMLR.
- Templeton, A., et al. (2024). Scaling Monosemanticity. *Transformer Circuits Thread*.
- Enyan, Z., & McCoy, R.T. (2026). A Unifying Perspective on Language Model Representations. *arXiv*.
- Smolensky, P., et al. (2025). Mechanisms of symbol processing for in-context learning in transformer networks. *JAIR*, 84.
- Yang, Y., et al. (2025b). Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models. *ICML*.
- Huang, X., & Hahn, M. (2026). Decomposing representation space into interpretable subspaces. *ICLR*.
