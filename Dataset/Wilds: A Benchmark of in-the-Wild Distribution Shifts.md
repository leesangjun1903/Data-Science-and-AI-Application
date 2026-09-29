# Wilds: A Benchmark of in-the-Wild Distribution Shifts

## 1. Executive summary — 8문장

WILDS는 새로운 신경망을 제안한 논문이 아니라, **실제 배포 환경에서 발생하는 분포 이동을 평가하기 위한 10개 데이터셋과 공통 실험·평가 체계를 제안한 벤치마크 논문**이다.  
연구의 목적은 무작위로 나눈 학습·시험 데이터나 인위적인 변형만으로는 실제 환경의 일반화 실패를 충분히 측정하기 어렵다는 문제를 해결하는 데 있다.  
저자들은 병원, 카메라, 실험 배치, 분자 구조, 국가, 시간, 사용자, 코드 저장소 등이 달라지는 문제를 포함하고, 각 응용에서 중요한 평가 지표와 도메인 메타데이터를 제공한다. **[pp. 4–6, Figures 1–2]** :chatgpt-content-reference{index="1"}

연구는 학습에서 보지 못한 환경으로의 **도메인 일반화**, 이미 존재하는 집단 사이의 비중이나 성능이 달라지는 **하위집단 분포 이동**, 그리고 두 문제가 결합한 설정을 다룬다.  
평균 학습 손실을 최소화하는 표준 방법인 ERM은 모든 선정 데이터셋에서 저자들이 구성한 분포 내·외 비교 간극을 보였으며, 예를 들어 Camelyon17의 정확도는 93.2에서 70.3으로, iWildCam의 macro F1은 47.0에서 31.0으로 낮아졌다. **[p. 7; p. 20, Table 1]** :chatgpt-content-reference{index="2"} :chatgpt-content-reference{index="3"}

CORAL·IRM·Group DRO는 전반적으로 ERM을 일관되게 개선하지 못했지만, CivilComments에서는 Group DRO가 최악 집단 정확도를 56.0에서 70.0으로 높였다.  
다만 CivilComments와 Amazon의 대표적인 간극은 서로 다른 집계 지표를 비교한 것이므로, 이를 전부 “새로운 환경 때문에 발생한 순수한 성능 손실”로 해석해서는 안 된다. **[pp. 19–22, Tables 1–2]** :chatgpt-content-reference{index="4"} :chatgpt-content-reference{index="5"}

**제 해석으로 이 논문의 가장 중요한 기여는 일반화 문제를 해결했다는 데 있지 않고, 실제 환경에서 무엇이 실패하며 어떤 개선을 공정하게 비교해야 하는지를 연구 가능한 형태로 만들었다는 데 있다.**

> **용어 설명**  
> **분포 이동(distribution shift)**은 학습할 때와 사용할 때 데이터의 통계적 특성이 달라지는 현상입니다. **ID(in-distribution)**는 기준 학습 분포와 같은 조건, **OOD(out-of-distribution)**는 다른 조건을 뜻합니다. **도메인(domain)**은 병원·카메라·국가처럼 데이터의 환경을 구분하는 단위이며, **일반화**는 학습에 사용하지 않은 데이터에서도 성능을 유지하는 능력입니다.

---

## 2. 핵심 주장과 근거

| 핵심 주장 | 저자가 직접 제시한 근거 | 원문 위치 | 제 해석 및 주의점 |
|---|---|---|---|
| 실제 분포 이동을 반영하는 평가가 부족하다. | 기존의 인위적 변형·상이한 데이터셋 간 전이만으로 실제 배포의 이동을 충분히 대표하기 어렵다고 논의하고, 다양한 응용의 10개 데이터셋을 구성한다. | pp. 5–6, Figure 2 | 기존 벤치마크가 무용하다는 주장이 아니라, **통제 실험과 실제성 높은 평가가 상호 보완적**이라는 주장이다. :chatgpt-content-reference{index="6"} |
| 선정한 실제 분포 이동은 표준 모델의 성능을 저하시킨다. | ERM의 ID·OOD 결과를 비교하며 모든 데이터셋에서 간극을 보고한다. | p. 20, Table 1 | 성능 저하가 큰 데이터셋을 선정했으므로, 이 표로 **모든 실제 분포 이동의 평균적 심각성**을 추정할 수는 없다. :chatgpt-content-reference{index="7"} :chatgpt-content-reference{index="8"} |
| 분포 이동의 영향을 평가하려면 시험 분포를 통제해야 한다. | 다른 분포에서 측정한 ID와 OOD 성능 차이는 데이터 자체의 난이도 차이를 포함할 수 있다고 지적한다. | pp. 17–18, §§5.1–5.2 | 논문의 가장 중요한 방법론적 기여 중 하나다. **“OOD 성능이 낮다”와 “분포 이동 때문에 낮아졌다”는 동일한 명제가 아니다.** :chatgpt-content-reference{index="9"} |
| 기존 강건 학습 방법의 효과는 문제 설정에 따라 다르다. | CORAL·IRM·Group DRO가 대체로 ERM을 개선하지 못하며, CivilComments에서는 개선된다. | pp. 20–22, Table 2 | 불변성 학습이나 강건 최적화가 원리적으로 불가능하다는 증거는 아니다. 시험한 구현·튜닝·도메인 정의에 대한 결과다. :chatgpt-content-reference{index="10"} |
| 여러 분포 이동이 결합하면 취약성이 커질 수 있다. | FMoW와 PovertyMap에서 시간·국가 이동과 하위집단 문제가 결합할 때 성능 저하가 커진다. | p. 23, §7.3; pp. 97–98, Tables 18–20; p. 104, Table 21 | 개별 이동만 검사하면 복합 환경에서의 실패를 놓칠 수 있다. 다만 모든 데이터셋에서 증폭 효과가 나타나는 것은 아니다. :chatgpt-content-reference{index="11"} |
| 좋은 모델을 학습하는 것과 좋은 모델을 선택하는 것은 별개의 문제다. | Camelyon17에서 비슷한 OOD 검증 성능을 보이는 모델들의 OOD 시험 성능이 크게 달라진다. | p. 74, Figure 18 | 일반화 연구에는 손실 함수뿐 아니라 **검증 도메인의 구성과 모델 선택의 신뢰성**도 포함되어야 한다. :chatgpt-content-reference{index="12"} |

> **용어 설명**  
> **강건성(robustness)**은 환경이나 데이터 조건이 달라져도 성능이 크게 무너지지 않는 성질입니다. **불변성(invariance)**은 환경이 달라져도 유지되는 특징이나 예측 관계를 뜻합니다. **메타데이터(metadata)**는 정답 외에 제공되는 병원 ID, 촬영 시각, 위치 등의 부가 정보입니다.

---

## 3. 해결하려는 문제와 제안한 방법: 수식 중심 설명

### 3.1 WILDS의 실제 제안은 ‘문제와 평가의 설계’다

**[저자 보고]** 데이터셋 선정 기준은 세 가지입니다. 실제로 성능을 저하시키는 분포 이동이 있어야 하고, 데이터 분할과 평가 지표가 실제 응용에 부합해야 하며, 여러 학습 도메인과 메타데이터 등 일반화를 학습하는 데 활용할 정보가 있어야 합니다. 여기에 표준 데이터 로더, 기본 모델, 평가 코드, 재현 가능한 실험 환경을 제공합니다. **[p. 6; pp. 29–31]** :chatgpt-content-reference{index="13"} :chatgpt-content-reference{index="14"} :chatgpt-content-reference{index="15"}

**[제 해석]** 따라서 “WILDS가 제안한 모델의 성능 향상”보다는 **“WILDS가 제공한 평가 환경에서 어떤 방법이 일반화에 성공하거나 실패하는가”**가 올바른 독해 방향입니다.

### 3.2 분포 이동의 정식화

**[원문 수식의 표기 정리: p. 7]**

```math
P^{\text{train}}
=
\sum_{d\in\mathcal D}q_d^{\text{train}}P_d,
\qquad
P^{\text{test}}
=
\sum_{d\in\mathcal D}q_d^{\text{test}}P_d.
```

여기서 $\mathcal D$는 전체 도메인의 집합, $d$는 하나의 도메인, $P_d$는 해당 도메인의 데이터 분포입니다. 데이터는 입력 $x$, 정답 $y$, 도메인 정보 $d$로 구성됩니다. $q_d^{\text{train}}$와 $q_d^{\text{test}}$는 학습·시험에서 도메인 $d$가 차지하는 비중이며, 각각 음수가 아니고 전체 합이 1입니다. 즉, 학습과 시험은 **동일한 도메인 목록을 서로 다른 비중으로 섞은 분포**로 표현됩니다. :chatgpt-content-reference{index="16"}

**도메인 일반화**에서는 학습 도메인과 시험 도메인이 겹치지 않습니다.

```math
\mathcal D_{\text{train}}
\cap
\mathcal D_{\text{test}}
=
\varnothing.
```

$\mathcal D_{\text{train}}$과 $\mathcal D_{\text{test}}$는 각각 학습·시험에서 실제로 등장하는 도메인 집합입니다. 예컨대 병원 A·B·C의 데이터로 학습해 병원 E에서 평가합니다. **[p. 7, §3.1]** :chatgpt-content-reference{index="17"}

**하위집단 분포 이동**에서는 시험 집단이 학습에도 존재하지만 비중이 달라질 수 있습니다.

$$
\mathcal D_{\text{test}}
\subseteq
\mathcal D_{\text{train}},
\qquad
q^{\text{test}}\ne q^{\text{train}}.
$$

두 $\mathcal D$의 의미는 위와 같고, $q^{\text{train}}$과 $q^{\text{test}}$는 모든 도메인의 비중을 모은 벡터입니다. 이 경우 평균 성능뿐 아니라 **가장 성능이 낮은 집단**을 중요하게 평가합니다. FMoW처럼 시간에 대해서는 새로운 환경으로 일반화하고, 지역에 대해서는 최악 집단 성능을 평가하는 혼합 설정도 있습니다. **[pp. 7–8, §§3.2–3.3]** :chatgpt-content-reference{index="18"}

**중요한 예외:** CivilComments의 집단은 서로 겹칩니다. 한 댓글이 여러 정체성을 동시에 언급할 수 있으므로, 실제 평가 집단이 위의 단순한 상호 배타적 도메인 분할과 완전히 일치하지는 않습니다. **[p. 91]** :chatgpt-content-reference{index="19"}

### 3.3 ERM: 전체 평균 손실을 줄이는 기준선

다음부터의 목적함수는 **논문이 비교한 기존 방법을 설명하기 위한 정리**입니다. WILDS가 새로 발명한 손실 함수가 아니며, 미니배치 추정이나 가중치 스케줄 등의 구현 세부사항은 단순화했습니다.

도메인별 경험적 위험과 ERM은 다음처럼 나타낼 수 있습니다.

```math
\widehat R_d(f)
=
\frac{1}{n_d}
\sum_{i:d_i=d}
\ell\!\left(f(x_i),y_i\right),
```

$$
\widehat f_{\text{ERM}}
\in
\text{argmin}_{f}
\sum_{d\in\mathcal D_{\text{train}}}
\frac{n_d}{n}\widehat R_d(f).
$$

$f$는 예측 모델, $\ell$은 예측 오류를 수치화하는 손실 함수, $i$는 샘플 번호, $n_d$는 도메인 $d$의 학습 샘플 수, $n=\sum_d n_d$는 전체 학습 샘플 수입니다. $\widehat R_d$는 관측된 샘플로 계산한 평균 손실이며, $\text{argmin}$은 목적함수를 가장 작게 하는 모델을 뜻합니다. WILDS는 분류 문제에 교차엔트로피, PovertyMap 회귀에 평균제곱오차를 사용합니다. **[p. 19; p. 66, Appendix D.3]** :chatgpt-content-reference{index="20"} :chatgpt-content-reference{index="21"}

**[제 해석]** ERM은 샘플이 많은 도메인이 학습 목표에 더 큰 영향을 미칩니다. 따라서 전체 평균은 좋아도 작은 집단에서의 오류가 충분히 줄어들지 않을 수 있습니다. 다만 이러한 가능성이 있다고 해서 모든 ERM 실패의 원인이 집단 불균형인 것은 아닙니다.

> **용어 설명**  
> **위험(risk)**은 이 문맥에서 평균 예측 손실을 뜻하며, 안전사고의 위험이라는 의미는 아닙니다. **교차엔트로피**는 정답에 낮은 확률을 줄수록 큰 벌점을 주고, **평균제곱오차**는 예측값과 정답의 차이를 제곱해 평균합니다.

### 3.4 CORAL: 도메인 간 특징 분포를 맞춘다

WILDS의 CORAL 구현은 도메인마다 추출된 특징의 **평균과 공분산 차이**에 벌점을 줍니다. 또한 각 도메인을 균등하게 샘플링하므로, 기본 손실도 샘플 수가 아니라 도메인에 동일한 비중을 줍니다. **[p. 21; pp. 66–67]** :chatgpt-content-reference{index="22"} :chatgpt-content-reference{index="23"}

이를 설명하는 목적함수는 다음과 같습니다.

$$
\min_{\theta,w}
\left[
\frac{1}{D_{\text{tr}}}
\sum_{d\in\mathcal D_{\text{train}}}
\widehat R_d(h_w\circ\phi_\theta)
+
\lambda
\frac{1}{|\mathcal P|}
\sum_{(d,d')\in\mathcal P}
\left(
\frac{\|\mu_d-\mu_{d'}\|_2^2}{p}
+
\frac{\|\Sigma_d-\Sigma_{d'}\|_F^2}{p^2}
\right)
\right].
$$

$\phi_\theta$는 파라미터 $\theta$를 가진 특징 추출기, $h_w$는 파라미터 $w$를 가진 예측층입니다. $D_{\text{tr}}$는 학습 도메인 수, $\mathcal P$는 서로 다른 학습 도메인 쌍의 집합, $p$는 특징 차원입니다. $\mu_d$와 $\Sigma_d$는 도메인 $d$의 특징 평균과 공분산이고, $\lambda\ge0$는 정렬 벌점의 강도입니다. $\|\cdot\|_2$는 벡터 길이, $\|\cdot\|_F$는 행렬 원소들의 제곱합에 기반한 크기입니다.

**[제 해석]** 이 방법은 병원마다 달라지는 염색처럼 불필요한 차이를 줄이는 데 적합할 수 있습니다. 그러나 도메인 간 차이가 정답 예측에 필요한 정보까지 포함하면, 무조건적인 정렬은 유용한 신호를 없앨 수 있습니다. 실제로 저자들은 RxRx1에서 배치 효과와 생물학적 신호가 얽혀 있을 수 있다고 논의합니다. **[p. 80]** :chatgpt-content-reference{index="24"}

> **용어 설명**  
> **특징 표현(representation)**은 신경망이 원본 입력을 변환한 내부 수치 표현입니다. **공분산**은 특징들이 함께 증가하거나 감소하는 경향을 나타냅니다. 평균·공분산을 맞춘다는 것은 표현의 중심과 퍼짐을 비슷하게 만드는 것이지, 전체 분포가 완전히 같아짐을 보장하는 것은 아닙니다.

### 3.5 IRM: 여러 도메인에서 동일한 예측 규칙이 유효하도록 한다

**[저자 보고]** IRM은 각 도메인에서 최적인 선형 예측기가 달라지는 표현에 벌점을 주는 방법으로 소개됩니다. **[p. 21]** :chatgpt-content-reference{index="25"}

실용적인 IRMv1의 기본 형태를 같은 기호로 쓰면 다음과 같습니다.

$$
\min_{\theta}
\frac{1}{D_{\text{tr}}}
\sum_{d\in\mathcal D_{\text{train}}}
\left[
\widehat R_d(f_\theta)
+
\lambda
\left|
\left.
\frac{\partial}{\partial a}
\widehat R_d(a f_\theta)
\right|_{a=1}
\right|^2
\right].
$$

$f_\theta$는 전체 예측 함수이고, 분류에서는 확률 변환 전 점수를 출력하는 함수로 생각할 수 있습니다. $a$는 출력에 곱하는 보조 스칼라이며, 실제로 학습할 최종 분류기가 아니라 $a=1$에서 최적성의 정도를 검사하는 장치입니다. $\widehat R_d$는 도메인별 평균 손실, $D_{\text{tr}}$는 학습 도메인 수, $\lambda$는 벌점 강도입니다. 미분값이 작으면 해당 도메인에서 출력의 배율을 조금 바꾸어도 손실을 쉽게 줄일 수 없다는 뜻입니다. 이 수식은 IRM 원 논문의 **p. 5, IRMv1**을 바탕으로 한 설명입니다. :chatgpt-content-reference{index="26"}

**[제 해석]** 이 벌점은 “인과적으로 올바른 특징을 반드시 찾는다”는 보증이 아닙니다. WILDS에서의 나쁜 성능은 표현 선택, 최적화, 작은 도메인에서의 추정 문제 등을 구분해서 살펴야 합니다. 저자들 역시 iWildCam의 IRM 실패에 대해 작은 도메인에서의 벌점 추정 편향을 **가능한 원인**으로 제시할 뿐, 확정하지 않습니다. **[p. 70]** :chatgpt-content-reference{index="27"}

> **용어 설명**  
> **인과적 특징**은 환경의 우연한 동반 변화가 아니라 예측 대상의 생성 과정과 관련된 정보를 뜻합니다. **기울기·미분값**은 파라미터를 조금 바꿀 때 손실이 어느 방향으로 얼마나 변하는지 나타냅니다.

### 3.6 Group DRO: 가장 어려운 학습 집단의 손실을 줄인다

```math
\min_f
\max_{q\in\Delta^{D_{\text{tr}}}}
\sum_{d\in\mathcal D_{\text{train}}}
q_d\widehat R_d(f)
=
\min_f
\max_{d\in\mathcal D_{\text{train}}}
\widehat R_d(f).
```

$f$는 예측 모델, $\widehat R_d$는 학습 도메인 $d$의 평균 손실입니다. $\Delta^{D_{\text{tr}}}$는 $q_d\ge0$, $\sum_d q_d=1$을 만족하는 모든 가중치 벡터의 집합입니다. 내부 최대화는 가장 큰 손실을 가진 도메인에 비중을 집중할 수 있으므로, 최악 도메인 손실을 최소화하는 것과 같습니다. **[p. 21, §6.2]** :chatgpt-content-reference{index="28"}

**[제 해석]** 여기서 최대화하는 대상은 **이미 관측한 학습 도메인**입니다. 따라서 이 목적함수만으로 새로운 병원이나 새로운 분자 구조에서의 최악 성능까지 보장할 수는 없습니다. 또한 CivilComments의 Table 2 대표 결과는 학습 시 `label × Black 언급 여부`의 4개 그룹을 이용한 것으로, 평가에 쓰는 16개 중첩 집단을 그대로 최적화한 결과가 아닙니다. **[pp. 92–93, Table 15]** :chatgpt-content-reference{index="29"} :chatgpt-content-reference{index="30"}

> **용어 설명**  
> **DRO(distributionally robust optimization)**는 하나의 고정된 데이터 비중만 고려하지 않고, 불리하게 달라질 수 있는 비중까지 고려하는 최적화입니다. 어떤 분포 변화를 허용하느냐에 따라 보장할 수 있는 강건성의 범위도 달라집니다.

---

## 4. 모델 구조: 하나의 ‘WILDS 모델’은 없다

기본적인 예측 구조는 다음처럼 표현할 수 있습니다.

$$
x
\xrightarrow{\ \phi_\theta\ }
z
\xrightarrow{\ h_w\ }
\widehat y.
$$

$x$는 입력, $\phi_\theta$는 특징 추출기, $z$는 내부 표현, $h_w$는 예측층, $\widehat y$는 예측 결과입니다. 도메인 정보 $d$는 주로 그룹별 손실·평가·모델 선택을 구성하는 데 사용되며, WILDS가 새로운 공통 신경망 모듈을 도입하는 것은 아닙니다. **[p. 19; pp. 66–67]** :chatgpt-content-reference{index="31"} :chatgpt-content-reference{index="32"}

| 데이터셋 | 예측 문제와 이동의 축 | 원문 기본 모델 | 근거 |
|---|---|---|---|
| iWildCam2020 | 카메라가 달라지는 동물 이미지 분류 | ImageNet 사전학습 ResNet-50, 448×448 입력 | p. 69, §E.1.2. :chatgpt-content-reference{index="33"} |
| Camelyon17 | 병원이 달라지는 종양 패치 분류 | **처음부터 학습한 DenseNet-121**, 96×96 입력 | p. 73, §E.2.2. :chatgpt-content-reference{index="34"} |
| RxRx1 | 실험 배치가 달라지는 유전자 처리 분류 | ImageNet 사전학습 ResNet-50 | p. 79, §E.3.2. :chatgpt-content-reference{index="35"} |
| OGB-MolPCBA | 새로운 분자 골격에서 128개 생물학적 활성 예측 | **GIN + 가상 노드**, 5개 그래프 계층, 특징 차원 300 | p. 83, §E.4.2. :chatgpt-content-reference{index="36"} |
| GlobalWheat | 새로운 국가·촬영 세션에서 밀 이삭 위치 검출 | ImageNet 사전학습을 활용한 Faster R-CNN | p. 87, §E.5.2. :chatgpt-content-reference{index="37"} |
| CivilComments | 정체성 언급 집단별 댓글 독성 판별 | DistilBERT-base-uncased 미세조정, 최대 300토큰 | p. 91, §E.6.2. :chatgpt-content-reference{index="38"} |
| FMoW | 미래 시점의 토지 이용 분류와 지역별 성능 | ImageNet 사전학습 DenseNet-121 | p. 98, §E.7.2. :chatgpt-content-reference{index="39"} |
| PovertyMap | 새로운 국가에서 자산 수준 회귀 | 8채널 위성영상 입력의 ResNet-18 | pp. 102–103, §E.8. :chatgpt-content-reference{index="40"} :chatgpt-content-reference{index="41"} |
| Amazon | 새로운 리뷰어의 별점 1–5 예측 | DistilBERT-base-uncased 미세조정, 최대 512토큰 | p. 108, §E.9.2. :chatgpt-content-reference{index="42"} |
| Py150 | 새로운 저장소에서 코드 다음 토큰 예측 | CodeSearchNet 사전학습 CodeGPT, 256토큰 블록 | pp. 111–112, §E.10.2. :chatgpt-content-reference{index="43"} |

> **용어 설명**  
> **사전학습**은 다른 대규모 데이터로 먼저 표현을 학습하는 과정이고, **미세조정(fine-tuning)**은 이를 특정 과제에 맞게 추가 학습하는 과정입니다. **GIN**은 원자와 결합으로 이루어진 그래프를 처리하는 신경망이며, **가상 노드**는 그래프 전체의 정보를 공유하도록 추가한 노드입니다. **토큰**은 언어나 코드를 모델이 처리하는 작은 단위입니다.

**[제 해석]** 사전학습을 활용한 모델도 실패한다는 점은 중요하지만, 이를 “사전학습으로는 해결할 수 없다”로 확대하면 안 됩니다. 원문이 평가한 사전학습 데이터·모델 규모·미세조정 방식은 제한적이며, 이후 연구는 바로 이 선택들이 일반화에 큰 영향을 줄 수 있음을 보여줍니다.

---

## 5. 성능 결과와 올바른 해석

### 5.1 Table 1: ID와 OOD의 간극

아래 수치는 **저자의 직접 보고**입니다. 괄호는 원문의 **표준편차**입니다. 단, PovertyMap은 다른 데이터셋과 달리 주로 **국가 분할 5개 fold 사이의 변동**을 나타냅니다. **[p. 20, Table 1; p. 66; p. 103]** :chatgpt-content-reference{index="44"} :chatgpt-content-reference{index="45"} :chatgpt-content-reference{index="46"}

| 데이터셋·주요 지표 | ID 비교 방식 | ID | OOD | 보고된 간극 |
|---|---|---:|---:|---:|
| iWildCam · macro F1 | 학습 도메인 내 평가 | 47.0 (1.4) | 31.0 (1.3) | 16.0 |
| Camelyon17 · 평균 정확도 | 학습 도메인의 **ID 검증 세트** | 93.2 (5.2) | 70.3 (6.4) | 22.9 |
| RxRx1 · 평균 정확도 | 학습·시험 도메인 혼합 학습 | 39.8 (0.2) | 29.9 (0.4) | 9.9 |
| OGB-MolPCBA · 평균 AP | 분자 단위 무작위 분할 | 34.4 (0.9) | 27.2 (0.3) | 7.2 |
| GlobalWheat · 세션 평균 검출 정확도 | 혼합 학습, 축소한 시험 세트 | 63.3 (1.7) | 49.6 (1.9) | 13.7 |
| CivilComments · 최악 집단 정확도 | **전체 평균 ↔ 최악 집단** | 92.2 (0.1) | 56.0 (3.6) | 36.2 |
| FMoW · 최악 지역 정확도 | 과거·이후 시점 혼합 학습 | 48.6 (0.9) | 32.3 (1.3) | 16.3 |
| PovertyMap · 최악 도시/농촌 Pearson $r$ | 국가 혼합 학습 | 0.60 (0.06) | 0.45 (0.06) | 0.15 |
| Amazon · 리뷰어별 정확도 10백분위 | **전체 평균 ↔ 10백분위** | 71.9 (0.1) | 53.8 (0.8) | 18.1 |
| Py150 · 클래스/메서드 토큰 정확도 | 학습 저장소 내 평가 | 75.4 (0.4) | 67.9 (0.1) | 7.5 |

> **지표 설명**  
> **Macro F1**은 클래스별 F1을 계산한 뒤 동일한 비중으로 평균해 흔한 클래스에만 유리한 평가를 피합니다. **AP(average precision)**는 정답 항목을 얼마나 잘 상위에 배치하는지 측정하는 정밀도·재현율 기반 지표입니다. **Pearson $r$**은 예측과 정답의 선형적 동반 변화를 측정하며, 예측값 자체가 정확히 일치하는지를 보장하지 않습니다. **10백분위 정확도**는 리뷰어별 정확도를 낮은 순서로 정렬했을 때 하위 10% 지점의 값입니다.

**이 표의 간극을 데이터셋 사이에서 평균내거나 크기순으로 단순 비교하면 안 됩니다.** 정확도 차이는 퍼센트포인트, F1·AP 차이는 해당 점수의 차이이며, 상관계수 0.15는 “15% 정확도 감소”가 아닙니다. 특히 CivilComments와 Amazon은 같은 종류의 평균 성능을 두 분포에서 비교한 것이 아닙니다. **[pp. 18–20]** :chatgpt-content-reference{index="47"} :chatgpt-content-reference{index="48"}

### 5.2 Table 2: 무엇이 개선되었고, 무엇이 개선되지 않았는가?

대표적인 결과를 발췌하면 다음과 같습니다. 모두 저자의 보고이며, 괄호는 표준편차입니다. **[p. 20, Table 2]** :chatgpt-content-reference{index="49"}

| 데이터셋 | ERM | CORAL | IRM | Group DRO |
|---|---:|---:|---:|---:|
| iWildCam | 31.0 (1.3) | **32.8 (0.1)** | 15.1 (4.9) | 23.9 (2.1) |
| Camelyon17 | **70.3 (6.4)** | 59.5 (7.7) | 64.2 (8.1) | 68.4 (7.3) |
| RxRx1 | **29.9 (0.4)** | 28.4 (0.3) | 8.2 (1.1) | 23.0 (0.3) |
| OGB-MolPCBA | **27.2 (0.3)** | 17.9 (0.5) | 15.6 (0.3) | 22.4 (0.6) |
| CivilComments | 56.0 (3.6) | 65.6 (1.3) | 66.3 (2.1) | **70.0 (2.0)** |

**[제 해석]** 정확한 결론은 “기존 방법이 모든 경우에 나빴다”가 아니라 **“광범위하고 일관된 개선을 보여주지 못했다”**입니다. iWildCam에서는 CORAL의 평균값이 ERM보다 1.8점 높습니다. 그러나 소수 반복 실험의 평균·표준편차만으로 그 차이를 안정적인 우월성으로 확정할 수는 없습니다.

### 5.3 CivilComments: 개선의 상당 부분은 단순 재가중으로도 얻어진다

**[저자 보고]** 독성·비독성 댓글을 균형 있게 샘플링하는 단순한 방법도 최악 집단 정확도 **69.2 (0.9)**를 얻습니다. Group DRO의 **70.0 (2.0)**은 ERM의 56.0보다 크지만, 이 단순 기준선보다는 평균값으로 **0.8퍼센트포인트** 높습니다. 또한 Group DRO의 전체 평균 정확도는 **89.9**로 ERM의 **92.2**보다 낮습니다. **[p. 92, Table 15]** :chatgpt-content-reference{index="50"}

**[제 해석]** 따라서 14퍼센트포인트의 개선을 전부 복잡한 강건 최적화의 효과라고 설명하면 과장입니다. 적어도 **클래스 균형 조정의 효과와 추가 알고리즘의 효과를 분리**해야 합니다. 0.8퍼센트포인트의 차이에 대해서는 원문 표만으로 통계적 유의성을 확정할 수 없습니다.

또한 이 데이터셋의 집단은 **댓글 작성자의 인구통계학적 정체성**이 아니라 **댓글이 특정 정체성을 언급하는지 여부**입니다. 따라서 이를 곧바로 “특정 인구집단 사용자를 공정하게 대우함”의 증명으로 읽어서는 안 됩니다. **[pp. 90, 93–94]** :chatgpt-content-reference{index="51"} :chatgpt-content-reference{index="52"}

> **용어 설명**  
> **재가중·재샘플링**은 적게 관측된 클래스나 집단이 학습에서 더 큰 비중을 갖도록 조정하는 것입니다. **허위 상관(spurious correlation)**은 학습 데이터에서는 예측에 도움이 되지만, 환경이 바뀌면 유지되지 않을 수 있는 관계입니다.

### 5.4 혼합 학습 비교가 보여주는 일반화 개선의 여지

같은 시험 분포에서 모델을 비교하려는 실험을 다음과 같이 표현할 수 있습니다.

```math
\Delta_{\text{mixed}}
=
M_{P^{\text{test}}}(f_{\text{mixed}})
-
M_{P^{\text{test}}}(f_{\text{source}}).
```

$M_{P^{\text{test}}}$는 동일한 시험 분포에서 계산하는 성능 지표, $f_{\text{source}}$는 원래 학습 도메인만 사용한 모델, $f_{\text{mixed}}$는 시험 도메인에서 따로 확보한 **라벨 있는 학습 자료**도 섞어 학습한 모델입니다. $\Delta_{\text{mixed}}$는 두 모델의 점수 차이입니다.

**[저자 보고]** Camelyon17에서 시험 병원의 슬라이드 10개 중 1개를 학습에 추가하고 나머지 동일한 9개에서 평가하면, 정확도는 **71.0 (6.3) → 82.9 (9.8)**로 높아집니다. FMoW에서도 학습 데이터 수를 유지하면서 이후 시점의 자료를 포함하면 최악 지역 정확도가 **32.3 → 48.6**으로 높아집니다. **[p. 73, Table 6; p. 98, Table 20]** :chatgpt-content-reference{index="53"} :chatgpt-content-reference{index="54"}

**[제 해석]** 이는 낮은 OOD 성능이 전적으로 “그 시험 데이터는 본질적으로 풀기 어렵기 때문”만은 아니라는 근거입니다. 하지만 **시험 도메인의 라벨 없이도 동일한 개선을 달성할 수 있다는 증명은 아닙니다.** 학습 알고리즘의 잠재력과 추가 정보의 효과를 구분해야 합니다.

---

## 6. 중요한 그림의 선정과 해석

### 6.1 Figure 1, p. 4: 새로운 환경과 취약 집단은 다른 문제다

!:chatgpt-content-reference{index="126"}[WILDS Figure 1](sandbox:/mnt/data/wilds_figure_1.png)

**[저자 보고]** 위쪽은 새로운 분자 골격에 대한 도메인 일반화이고, 아래쪽은 지역별 성능을 비교하는 하위집단 설정입니다. 두 설정을 하나의 “OOD 문제”로 뭉뚱그리지 않고 구분합니다. :chatgpt-content-reference{index="55"}

**[제 해석]** Group DRO가 학습에서 이미 관측한 취약 집단을 개선하는 데 효과적이더라도, 새로운 환경의 분포를 예측하는 문제까지 해결한다고 기대할 수 없는 이유가 이 그림에 있습니다. 이 그림은 문제 설명용 도식이며, 정량 비교에는 반복 실험 결과인 Table 1을 사용해야 합니다.

### 6.2 Figure 2, p. 5: 일반화는 이미지 분류 한 종류의 문제가 아니다

**[저자 보고]** Figure 2는 사진·병리·세포 영상·분자 그래프·댓글·위성영상·리뷰·코드를 함께 배치합니다. 입력뿐 아니라 분류·회귀·검출·코드 예측 등 과제와 도메인 구조도 다릅니다. :chatgpt-content-reference{index="56"}

**[제 해석]** 한 가지 영상 증강 기법이 잘 작동했다고 해서 분자 그래프나 코드에서도 일반화 문제가 해결됐다고 주장하기 어렵습니다. 반대로 모든 과제에 적용되는 단일 방법만 가치 있는 것도 아닙니다. **이동의 구조에 맞는 전문적 방법과 범용 방법을 별도로 평가해야 한다**는 논문의 지침과 연결됩니다. **[pp. 28–29]** :chatgpt-content-reference{index="57"}

### 6.3 Figure 18, p. 74: OOD 검증 성능이 좋아도 다른 OOD 환경에서 실패할 수 있다

!:chatgpt-content-reference{index="127"}[WILDS Figure 18](sandbox:/mnt/data/wilds_figure_18.png)

**[저자 보고]** 같은 하이퍼파라미터에서 난수 초기값을 바꾼 모델들의 결과입니다. 가로축은 한 병원의 OOD 검증 정확도이고, 세로축은 다른 병원의 OOD 시험 정확도입니다. 시험 정확도의 변동이 검증 정확도보다 훨씬 큽니다. :chatgpt-content-reference{index="58"}

**[제 해석]** 이 그림은 단순히 “여러 번 학습하라”는 권고보다 강한 메시지를 줍니다. **검증 병원 하나를 잘 맞추는 능력과 새로운 병원들에 일반화하는 능력이 충분히 일치하지 않을 수 있습니다.** 다만 점의 수가 적으므로 이 그림만으로 보편적인 검증·시험 상관계수를 주장할 수는 없습니다.

> **용어 설명**  
> **하이퍼파라미터**는 학습률·정규화 강도처럼 학습 절차를 정하는 설정입니다. **미결정성 또는 underspecification**은 관측된 학습·검증 성능만으로는 배포에서 서로 다르게 행동하는 모델들을 충분히 구별하지 못하는 문제입니다.

### 6.4 Figure 27, p. 108: 평균 정확도는 사용자별 편차를 숨긴다

!:chatgpt-content-reference{index="128"}[WILDS Figure 27](sandbox:/mnt/data/wilds_figure_27.png)

**[저자 보고]** 파란 분포는 Amazon 리뷰어별 ERM 정확도이고, 회색 분포는 모든 리뷰어의 정답 확률이 동일하다고 가정한 무작위 기준선입니다. 실제 10백분위 정확도는 **53.8%**, 기준선에서는 **65.4%**입니다. :chatgpt-content-reference{index="59"}

**[제 해석]** 관측된 리뷰어별 차이는 단순한 유한 표본 변동만으로 설명하기 어려울 정도로 큽니다. 그러나 이 차이가 전부 편향이나 허위 상관 때문이라는 뜻은 아닙니다. 리뷰어마다 별점 사용 방식이나 문장의 해석 가능성 등 과제 난이도 자체가 다를 수 있고, 원문도 이를 분리하지 못했다고 명시합니다. **[pp. 108–109]** :chatgpt-content-reference{index="60"}

### 6.5 Figure 26, p. 104: 공간 정보는 활용 가능한 신호지만 인과적 보증은 아니다

**[저자 보고]** 가까운 위치의 두 마을은 자산 수준 차이가 작은 경향이 있습니다. 그림은 두 지점 사이의 거리와 자산 수준 절대 차이의 평균을 보여줍니다. :chatgpt-content-reference{index="61"}

**[제 해석]** 이는 좌표를 단순한 부가 정보로 버리지 않고 일반화에 활용할 이유를 제공합니다. 그러나 가까운 곳끼리 비슷하다는 상관관계가 국경을 넘어 항상 유지된다는 뜻은 아니며, 이 그림 자체가 공간 정보를 이용한 특정 모델의 성능 향상을 입증하지도 않습니다.

---

## 7. 통계적으로 취약한 부분과 비교 불가능한 수치

### 7.1 통계적 증거의 범위

| 표시 | 문제 | 원문 근거와 해석상 제한 |
|---|---|---|
| **[통계적 주의] 적은 반복 수** | 대부분 3개 난수 시드의 결과다. | Camelyon17은 10회, CivilComments는 5회지만, 다수의 작은 성능 차이에 대해 검정 결과나 신뢰구간이 제공되는 것은 아니다. **[p. 66, Appendix D.2]** :chatgpt-content-reference{index="62"} |
| **[통계적 주의] 반복의 단위가 다르다** | PovertyMap의 변동은 난수 시드 변동과 같지 않다. | 국가 분할 5개 fold에 대해 fold당 1개 시드를 사용한다. 다른 데이터셋의 괄호와 같은 의미로 비교하면 안 된다. **[p. 103]** :chatgpt-content-reference{index="63"} |
| **[일반화 범위 제한] 병원이 적다** | Camelyon17의 최종 OOD 시험 병원은 하나다. | 많은 패치가 있어도 독립적인 시험 병원이 많은 것은 아니다. 해당 병원은 시각적으로 가장 다른 병원으로 선택됐다. **[p. 72]** :chatgpt-content-reference{index="64"} |
| **[독립성 주의] 패치는 서로 연관된다** | Camelyon17의 ID 검증 패치는 학습과 같은 슬라이드에서 나온다. | 패치 단위 ID 성능을 새로운 슬라이드·환자·병원 성능으로 동일시할 수 없다. **[pp. 72, 75]** :chatgpt-content-reference{index="65"} :chatgpt-content-reference{index="66"} |
| **[원인 분리 제한] 튜닝 조건** | 기본 모델의 설정은 주로 ERM으로 고른 뒤 다른 방법에 재사용한다. | CORAL·IRM은 추가 벌점 탐색을 하지만, 모든 방법을 완전히 독립적으로 최적 튜닝한 비교는 아니다. **[p. 66]** :chatgpt-content-reference{index="67"} |
| **[원인 분리 제한] 샘플링도 바뀐다** | CORAL·IRM은 도메인 균등 샘플링을 사용한다. | ERM 대비 변화에는 벌점뿐 아니라 재가중 효과도 포함된다. CivilComments에서 특히 중요하다. **[pp. 66–67, 93]** :chatgpt-content-reference{index="68"} :chatgpt-content-reference{index="69"} |
| **[선정 편향] 어려운 이동을 골랐다** | 성능 저하가 큰 이동이 포함 기준이다. | “실제 이동이 항상 심각하다”거나 “모든 실제 응용의 평균 하락 폭이 이 정도다”라는 추론은 불가능하다. **[pp. 6, 17]** :chatgpt-content-reference{index="70"} :chatgpt-content-reference{index="71"} |

표준편차와 표준오차도 구분해야 합니다.

$$
\text{SE}(\overline m)=\frac{s}{\sqrt n}.
$$

$\overline m$은 반복 결과의 평균, $s$는 반복 결과의 표준편차, $n$은 독립 반복 수입니다. 이 식은 평균 추정의 변동을 설명하지만, **고정된 시험 병원에서 시드만 바꾼 실험의 작은 SE가 새로운 병원들에 대한 불확실성까지 작다는 뜻은 아닙니다.** 원문도 표준편차와 평균의 표준오차를 구분합니다. **[p. 20, Table 1 설명]** :chatgpt-content-reference{index="72"}

> **용어 설명**  
> **표준편차(SD)**는 반복 결과 자체의 퍼짐이고, **표준오차(SE)**는 평균 추정치의 불확실성을 나타냅니다. **신뢰구간**은 특정 표집 가정 아래 추정의 불확실성을 나타내는 구간이며, SD나 SE와 같은 숫자가 아닙니다.

### 7.2 반드시 구별해야 하는 수치

**① GlobalWheat의 49.6과 51.2는 같은 시험 세트의 점수가 아닙니다.**  
Table 1의 49.6은 혼합 학습 비교를 위해 축소한 시험 세트에서 얻은 값이고, Table 2의 51.2는 공식 시험 세트의 값입니다. 이를 방법 변경에 따른 1.6퍼센트포인트 개선으로 읽으면 안 됩니다. **[p. 20, Table 2 설명; p. 88, Table 13]** :chatgpt-content-reference{index="73"} :chatgpt-content-reference{index="74"}

**② Amazon의 18.1퍼센트포인트는 ‘새로운 사용자로 바뀐 효과’가 아닙니다.**  
이는 OOD 사용자들의 전체 평균 71.9와 사용자별 10백분위 53.8의 차이입니다. 같은 10백분위 지표끼리 보면 기존 사용자 57.3과 새로운 사용자 53.8의 차이는 **3.5퍼센트포인트**입니다. 이 3.5는 Table 23 값에서 계산한 차이입니다. **[p. 108, Table 23]** :chatgpt-content-reference{index="75"}

**③ Camelyon17의 혼합 학습 비교는 70.3이 아니라 71.0을 기준으로 해야 합니다.**  
슬라이드 하나를 학습으로 옮긴 뒤 남은 9개 슬라이드에서 양쪽 모델을 평가하기 때문입니다. 올바른 해당 비교는 **71.0 ↔ 82.9**입니다. **[p. 73, Table 6]** :chatgpt-content-reference{index="76"}

**④ RxRx1의 기존 대회 성적과 WILDS 29.9%는 직접 비교할 수 없습니다.**  
입력 채널·해상도·시험 제어 조건의 라벨 접근·예측 집계 단위·실험 구조를 이용한 후처리가 다릅니다. 원문도 기존 대회에서의 매우 높은 정확도와 WILDS 성능 차이의 이유를 별도로 설명합니다. **[p. 81]** :chatgpt-content-reference{index="77"}

**⑤ ID–OOD 간극이 줄어드는 것만으로는 개선이 아닙니다.**  
Py150에서 IRM은 ID와 OOD가 각각 67.3·64.3으로 간극이 3.0이지만, ERM의 OOD 67.9보다 낮습니다. 즉, **ID 성능을 더 많이 떨어뜨려 간극만 줄일 수도 있습니다.** **[p. 112, Table 26]** :chatgpt-content-reference{index="78"}

**문서 내 일관성 주의:** Amazon의 학습 사용자 수는 본문 p. 16에 5,008명, 부록 p. 107에는 1,252명으로 적혀 있습니다. 이 불일치는 임의로 정정할 수 없으며, 재현 실험에서는 실제 데이터 버전과 분할 명세를 확인해야 합니다. :chatgpt-content-reference{index="79"} :chatgpt-content-reference{index="80"}

---

## 8. 2020년 이후 관련 연구 비교: 일반화 향상의 경로는 어떻게 달라졌는가?

아래는 **논문 간 절대 순위표가 아니라 접근법과 증거의 비교**입니다. 특히 원래 WILDS의 기본 모델, 대규모 사전학습 모델, 추가 라벨이나 비라벨 자료를 활용한 모델은 실험 자원이 다릅니다.

### 8.1 선별 비교

| 연구 | 저자가 보고한 방법·결과 | WILDS에 대한 의미와 비교 제한 |
|---|---|---|
| **DomainBed — In Search of Lost Domain Generalization** · 2020 | 7개 데이터셋, 9개 방법, 3개 모델 선택 기준을 통일한다. 강하게 구현한 ERM이 경쟁력 있음을 보고한다. **[pp. 7–8, Table 4]** | **제 해석:** 새로운 손실 함수보다 먼저 강한 ERM과 공정한 선택 절차가 필요하다. 주로 영상 DG 환경이며 WILDS 10개 과제와 같은 평가가 아니다. :chatgpt-content-reference{index="81"} |
| **Accuracy on the Line** · 2021 | FMoW·iWildCam 등을 포함한 여러 이동에서 ID와 OOD 성능이 강하게 연관됨을 보고하지만, Camelyon17 등에서는 관계가 약하다. | **제 해석:** OOD 점수 상승이 일반적 예측력 향상인지 이동에 대한 추가적 강건성인지 분리해야 한다. 이 관계는 모든 문제에서 성립하는 법칙이 아니다. :chatgpt-content-reference{index="82"} |
| **Extending the WILDS Benchmark for Unsupervised Adaptation** · 2022 | 8개 데이터셋에 비라벨 자료를 추가한다. Camelyon17에서 해당 실험의 증강 포함 ERM은 **82.0 (7.4)**, SwAV는 **91.4 (2.0)**이지만, 다른 데이터셋의 개선은 제한적·혼재적이다. **[p. 10, Table 2]** | **제 해석:** 비라벨 자료는 유용한 추가 정보지만 자동적인 해결책은 아니다. 원 논문의 70.3과 비교해 전부 적응 알고리즘의 효과로 계산하면 증강·실험 설정 차이를 섞게 된다. :chatgpt-content-reference{index="83"} |
| **LP-FT — Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution** · 2022 | 좋은 사전학습 표현과 큰 이동에서는 전체 미세조정이 선형 예측층만 학습하는 방법보다 OOD에서 나쁠 수 있다. 먼저 선형층을 학습한 뒤 전체를 미세조정하는 LP-FT를 제안한다. **[pp. 12–13, Tables 1–2]** | **제 해석:** 좋은 표현을 새로 만드는 것만큼 기존 표현을 훼손하지 않는 것이 중요하다. 이 논문의 FMoW는 **지리적 이동을 따로 구성한 설정**으로 WILDS의 공식 시간 이동 결과와 직접 비교할 수 없다. :chatgpt-content-reference{index="84"} |
| **DFR — Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations** · 2022 공개 | ERM 특징을 고정하고 집단 균형 자료로 마지막 층을 다시 학습한다. CivilComments에서 해당 논문의 최악 집단 정확도는 **55.6±0.6 → 70.1±0.8**이다. **[Table 2]** | **제 해석:** 실패 원인이 항상 특징의 부재는 아닐 수 있다. 다만 BERT를 사용하고 검증 자료를 마지막 층 학습에도 쓰므로, 원 WILDS의 DistilBERT·선택 전용 검증 조건과 같지 않다. :chatgpt-content-reference{index="85"} |
| **AutoFT: Learning an Objective for Robust Fine-Tuning** · 2024 | 작은 OOD 검증 세트로 미세조정 목적함수와 하이퍼파라미터를 탐색한다. iWildCam에서 대형 CLIP 모델과 가중치 앙상블을 사용해 **52.0 macro F1**을 보고한다. **[Tables 1–2]** | **제 해석:** 검증 데이터는 단순 조기 종료뿐 아니라 학습 절차 자체를 설계하는 정보다. 대형 사전학습·앙상블을 사용하므로 원 WILDS의 31.0 대비 차이를 AutoFT만의 효과로 해석하면 안 된다. :chatgpt-content-reference{index="86"} |
| **Latent Domain Modeling Improves Robustness to Geographic Shifts** · 2025 초고, 2026 v3 | 위치 인코더와 도메인 예측 보조 손실을 결합한다. 같은 논문의 ERM 대비 FMoW 최악 지역 정확도 **47.6 (SE 0.9) → 55.8 (SE 0.7)**, PovertyMap 최악 상관계수 **0.45 (SE 0.03) → 0.57 (SE 0.03)**를 보고한다. **[p. 5, Table 1]** | **제 해석:** 도메인 정보를 제거하는 대신 구조적으로 활용하는 방향이다. 두 데이터셋의 최고 결과는 서로 다른 결합 방식이며, FMoW의 대형 CLIP·외부 위치 사전학습은 원 기준선과 구별해야 한다. :chatgpt-content-reference{index="87"} |
| **FINO — Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Have** · 2026 공개본 | 메타데이터와 자기지도학습으로 표현을 적응시킨 뒤 고정하고 작은 예측층을 학습한다. iWildCam에서 고정 사전학습 모델의 **51.4 → FINO 53.1**, FMoW에서 **45.0 → 52.9**를 보고한다. **[Table 1]** | **제 해석:** 세밀한 메타데이터를 활용한 표현 학습이 유망하다. **라벨이 없는 것은 표현 적응 단계이며 최종 예측층에는 라벨이 필요하다.** 대표 표는 탐색 중 최고 결과이고 반복 불확실성을 함께 제시하지 않아 작은 차이의 안정성은 별도 검증해야 한다. :chatgpt-content-reference{index="88"} |

> **용어 설명**  
> **선형 프로빙(linear probing)**은 특징 추출기를 고정하고 마지막 선형 예측층만 학습하는 것입니다. **자기지도학습**은 사람이 붙인 과제 정답 대신 입력 자체에서 학습 신호를 만드는 방법입니다. **SwAV**는 여러 변형에서의 군집 할당을 활용하는 자기지도학습 방법입니다. **CLIP**은 이미지와 텍스트의 대응 관계로 사전학습한 모델이고, **앙상블**은 여러 모델 또는 모델 가중치를 결합하는 방식입니다.

### 8.2 후속 연구의 방법을 수식으로 연결하면

#### A. DFR: 특징을 버리지 않고 사용하는 비중을 다시 학습한다

DFR의 핵심은 다음과 같이 요약할 수 있습니다.

```math
\widehat w
=
\text{argmin}_{w}
\left[
\frac{1}{|\mathcal V_{\text{bal}}|}
\sum_{(x,y)\in\mathcal V_{\text{bal}}}
\ell\!\left(h_w(\phi_{\theta_0}(x)),y\right)
+
\lambda\|w\|_1
\right].
```

$\phi_{\theta_0}$는 고정된 ERM 특징 추출기, $\mathcal V_{\text{bal}}$은 집단 균형을 맞춘 재학습 자료, $h_w$는 새로 학습하는 마지막 층입니다. $\ell$은 분류 손실, $\lambda$는 정규화 강도, $\|w\|_1$은 가중치 절댓값의 합입니다. 실제 DFR에는 특징 표준화와 여러 균형 부분집합에서 학습한 가중치 평균도 포함됩니다. **[DFR §5–6, Appendix C]** :chatgpt-content-reference{index="89"}

**[제 해석]** 원 WILDS의 ERM 실패를 볼 때 “필요한 특징을 못 배웠는가?”와 “배운 특징을 잘못 활용하는가?”를 구분해야 합니다.

#### B. AutoFT: OOD 검증 성능을 학습 절차의 선택에 직접 사용한다

개념적으로는 다음과 같습니다.

```math
\theta_\eta
=
\text{Train}(\theta_0,\mathcal S_{\text{train}};\eta),
\qquad
\widehat\eta
=
\text{argmax}_{\eta}
M_{\text{val,OOD}}(\theta_\eta).
```

$\theta_0$는 사전학습 파라미터, $\mathcal S_{\text{train}}$은 미세조정 자료, $\eta$는 목적함수 구성과 하이퍼파라미터, $\theta_\eta$는 그 설정으로 학습한 결과입니다. $M_{\text{val,OOD}}$는 OOD 검증 성능입니다. 실제 탐색은 완전한 학습을 매번 수행하는 대신 짧은 내부 학습을 사용하며, iWildCam·FMoW 설정에서는 검증 예제 1,000개와 500개 외부 탐색 시행을 보고합니다. **[AutoFT Table 1]** :chatgpt-content-reference{index="90"}

**[제 해석]** 이 방향에서는 검증 세트 크기뿐 아니라 **동일 검증 세트를 얼마나 반복적으로 이용했는지**도 재현성과 과적합 평가의 대상이 됩니다.

#### C. 위치 기반 조건부 모델: 도메인을 지우지 않고 예측에 연결한다

2026년 위치 기반 연구의 핵심 구조는 다음과 같이 정리할 수 있습니다.

$$
f(x,s)=F(g(x),e(s)),
$$

```math
\mathcal L
=
\mathcal L_{\text{task}}(f(x,s),y)
+
\alpha\,
\mathcal L_{\text{domain}}(h(e(s)),d).
```

$x$는 이미지, $s$는 좌표, $g$는 이미지 인코더, $e$는 위치 인코더, $F$는 두 표현의 결합 함수입니다. $y$는 과제 정답, $d$는 도메인 라벨, $h$는 도메인 예측기, $\alpha$는 보조 손실의 비중입니다. 시험 시 $h$는 제거하지만, 위치를 조건으로 사용하는 모델에는 좌표 $s$가 필요합니다. **[해당 논문 §3, Figure 1]** :chatgpt-content-reference{index="91"}

**[제 해석]** 이 방향은 “환경 정보를 무조건 없애야 강건하다”는 관점과 다릅니다. 예측 관계가 환경에 따라 달라지는 경우에는 **환경을 이해하고 조건부로 예측하는 것**이 더 적합할 수 있습니다.

### 8.3 이 비교가 보여주는 연구 흐름

**[제 해석]** 이후의 진전은 하나의 새로운 강건 손실 함수로 모이지 않습니다. 강한 기준선과 모델 선택, 좋은 사전학습 표현의 보존, 마지막 층의 재학습, 비라벨 자료, 위치·촬영 조건 같은 메타데이터의 활용이 서로 다른 개선 경로를 제공합니다.

특히 **“일반화 성능 향상”과 “분포 이동을 원리적으로 해결함”을 구분해야 합니다.** 새로운 논문이 기존 WILDS 점수를 크게 높였더라도, 그 변화가 모델 규모·사전학습·추가 정보·튜닝·알고리즘 중 어디에서 비롯되었는지 분해하지 않으면 다음 연구의 설계 원리를 얻기 어렵습니다.

---

## 9. 원 논문이 답하지 않는 질문

| 남아 있는 질문 | 원문이 제공하는 것과 제공하지 않는 것 |
|---|---|
| **같은 라벨 비용이라면 샘플 수와 도메인 다양성 중 무엇이 더 중요한가?** | 많은 도메인과 메타데이터를 학습의 활용 가능 정보로 제공하지만, 고정 예산에서 두 요인의 효과를 체계적으로 분리한 일반적 결론은 없다. **[p. 6; p. 22]** :chatgpt-content-reference{index="92"} :chatgpt-content-reference{index="93"} |
| **CORAL·IRM의 실패는 잘못된 가정, 최적화, 추정 오차 중 무엇 때문인가?** | OGB에서의 과소적합, iWildCam의 벌점 추정 문제 등 후보 설명을 제공하지만 전체 실패 원인을 인과적으로 분리하지는 않는다. **[pp. 70, 84]** :chatgpt-content-reference{index="94"} :chatgpt-content-reference{index="95"} |
| **어떤 도메인·집단 정의가 최적인가?** | 병원·연도·카메라 등의 정의와 일부 대안 실험은 있지만, 새로운 문제에서 적절한 그룹을 선택하는 일반적 방법은 없다. **[p. 93]** :chatgpt-content-reference{index="96"} |
| **적절한 OOD 검증 자료를 얻지 못하면 어떻게 모델을 선택할 것인가?** | ID·OOD 검증 비교를 제공하지만, 실제 배포의 미지 환경을 대신할 검증 분포를 어떻게 구성할지에 대한 보편적 해법은 없다. **[pp. 22–23; p. 66, Table 3]** :chatgpt-content-reference{index="97"} |
| **취약 집단의 성능은 어디까지 개선 가능한가?** | CivilComments·Amazon에서 관측 간극을 제시하지만, 라벨 잡음·본질적 난이도·데이터 부족을 분리한 달성 가능한 상한은 확정하지 않는다. **[p. 92; p. 108]** :chatgpt-content-reference{index="98"} :chatgpt-content-reference{index="99"} |
| **실제 서비스에서 장기적 신뢰성을 유지할 수 있는가?** | 모델이 데이터 분포를 바꾸는 피드백, 장기적 적응, 예측 보류와 사람의 개입 등은 주로 확장 방향으로 남아 있다. **[p. 28; pp. 64–65]** :chatgpt-content-reference{index="100"} :chatgpt-content-reference{index="101"} |

> **용어 설명**  
> **과소적합(underfitting)**은 모델이 학습 자료의 중요한 패턴조차 충분히 학습하지 못한 상태입니다. **라벨 잡음**은 정답에 오류나 해석상의 불일치가 포함된 것을 뜻합니다. **피드백 루프**는 모델의 예측·행동이 이후 수집될 데이터를 바꾸고, 그 데이터가 다시 모델에 영향을 주는 순환입니다.

---

## 10. 결론: 저자의 시사점과 추가 후속 연구 방향

### 10.1 저자들이 제시한 시사점과 후속 방향

**[저자 보고]** WILDS는 실제 분포 이동에서의 강건 학습이 여전히 중요한 과제라고 결론짓지만, 다양한 학습 도메인과 메타데이터가 후속 알고리즘의 학습에 활용될 수 있다고 봅니다. 범용 알고리즘뿐 아니라 특정 이동 구조에 맞춘 방법도 가치가 있으며, 새로운 아키텍처·외부 사전학습은 고정 모델·데이터 조건의 알고리즘 비교와 분리해 평가할 것을 권고합니다. **[p. 22; pp. 28–29]** :chatgpt-content-reference{index="102"} :chatgpt-content-reference{index="103"}

또한 시험 분포에 대한 과적합을 피하고 ID와 OOD를 함께 보고하며, 비지도 도메인 적응·시험 시점 적응·선택적 예측 등으로 벤치마크를 확장할 것을 제안합니다. 이는 이미 완료된 해결책이 아니라 **논문이 제안하는 연구 방향**입니다. **[p. 29; pp. 64–65]** :chatgpt-content-reference{index="104"} :chatgpt-content-reference{index="105"}

> **용어 설명**  
> **비지도 도메인 적응**은 정답 없는 목표 환경 데이터를 활용해 모델을 조정하는 설정입니다. **시험 시점 적응**은 배포 중 들어오는 자료로 모델을 조정하는 방식이고, **선택적 예측**은 확신이 낮은 사례의 판단을 보류하거나 사람에게 넘기는 방식입니다.

### 10.2 제가 우선순위를 두는 추가 연구

**첫째, ‘강한 ERM 대비 무엇이 추가로 좋아졌는가’를 분해해야 합니다.**  
같은 사전학습 체크포인트·입력 해상도·증강·학습 예산을 맞춘 뒤 ERM, 균형 재샘플링, LP-FT, 마지막 층 재학습, 강건 손실을 비교하는 설계가 우선입니다. 이렇게 해야 전체 성능 향상과 분포 이동에 대한 추가적 개선을 구분할 수 있습니다. 또한 Py150의 예처럼 ID를 떨어뜨려 간극만 줄이는 현상을 막기 위해 **OOD 절대 성능·ID 성능·취약 집단 성능을 함께 평가**해야 합니다. 이 제안의 근거는 원문의 평가 지침과 후속 연구의 기준선·표현 보존 결과입니다. :chatgpt-content-reference{index="106"} :chatgpt-content-reference{index="107"} :chatgpt-content-reference{index="108"}

**둘째, 샘플 수준이 아니라 ‘새로운 도메인’ 수준의 불확실성을 측정해야 합니다.**  
여러 병원·카메라·국가를 순환해서 완전히 제외하는 평가와, 개발에 쓰지 않은 별도 도메인 평가를 수행할 필요가 있습니다. 신뢰구간도 패치를 독립적으로 재표집하는 방식만 사용하기보다 슬라이드·병원 등 상위 단위의 의존성을 반영해야 합니다. 이는 Camelyon17에서 관찰된 모델 선택 실패와 단일 시험 병원의 제한을 직접 겨냥하는 제안입니다. :chatgpt-content-reference{index="109"} :chatgpt-content-reference{index="110"}

**셋째, 메타데이터를 ‘제거할 것’과 ‘활용할 것’으로 구분하는 연구가 중요합니다.**  
염색·센서 조건처럼 억제해야 할 변동도 있지만, 생물학적 조건이나 지역적 맥락처럼 예측 관계를 이해하는 데 필요한 정보도 있습니다. 고정된 전체 분포 정렬만 시험하기보다 조건부 정렬, 위치·시간 구조를 이용한 예측, 정보가 누락되거나 부정확할 때의 대체 방식을 비교해야 합니다. 특히 “모든 도메인 정보를 없애면 좋다”는 가정을 먼저 두지 않는 것이 중요합니다. :chatgpt-content-reference{index="111"} :chatgpt-content-reference{index="112"}

**넷째, 데이터 접근 조건을 분리한 연구 트랙이 필요합니다.**  
원래 학습 자료와 허용된 검증 자료만 사용하는 설정, 별도의 비라벨 목표 자료를 사용하는 설정, 소량의 라벨 있는 목표 자료까지 사용하는 설정은 구분해야 합니다. 대규모 사전학습 모델의 데이터 중복 가능성도 함께 확인해야 합니다. 원 WILDS의 Py150는 CodeSearchNet과 겹치는 저장소가 검증·시험에 들어가지 않도록 처리했는데, 후속 대규모 모델 평가에서도 이런 통제가 중요합니다. **[p. 111]** :chatgpt-content-reference{index="113"}

**다섯째, 평균 점수의 개선을 넘어 실제 사용 목적을 검증해야 합니다.**  
병리 패치 정확도가 높아졌다고 환자 수준의 임상적 유용성이 자동으로 높아지는 것은 아니고, 빈곤 예측의 상관계수가 높아졌다고 자산 수준의 절대 오차가 작다는 보장도 없습니다. 후속 평가에서는 원래 과제의 지표를 유지하면서, 사용 목적에 맞는 오류 비용·확률의 신뢰성·예측 보류 시 집단별 영향도 함께 측정하는 것이 바람직합니다. 원문도 데이터의 실제성, 과제·지표의 실제성, 분할의 실제성을 별개로 구분합니다. **[p. 61, Appendix A]** :chatgpt-content-reference{index="114"}

### 최종 판단

**WILDS가 입증한 것은 “일반화가 불가능하다”가 아니라, 당시의 표준 평가와 몇몇 대표적 강건 학습 방법만으로는 실제 분포 이동에서의 성능을 충분히 확보하기 어려웠다는 점입니다.**

후속 연구는 일반화 성능의 개선 가능성을 실제로 보여주지만, 그 개선은 **표현의 품질과 보존, 예측층의 재학습, 검증 전략, 데이터 다양성, 추가 데이터, 도메인 구조의 활용**에서 서로 다른 방식으로 발생합니다. 앞으로의 핵심 질문은 단순히 “WILDS 점수가 올랐는가?”가 아니라, **“어떤 정보와 가정 덕분에 올랐으며, 개발 과정에서 전혀 보지 못한 다른 환경에서도 그 개선이 유지되는가?”**입니다.

---

## 참고자료

아래는 본 분석에 직접 사용한 자료입니다. 다른 논문이 인용한 문헌 전체가 아니라, 위 설명과 비교의 근거가 된 원문을 정리했습니다.

| 자료 | 사용한 출처·버전 |
|---|---|
| **Koh et al., “WILDS: A Benchmark of in-the-Wild Distribution Shifts”** | 사용자 첨부본, arXiv:2012.07421v3, 2021-07-16. 원문 분석·표·그림·실험 조건의 주된 근거입니다. :chatgpt-content-reference{index="115"} |
| **Arjovsky et al., “Invariant Risk Minimization”** | arXiv:1907.02893, 2020 개정본. IRMv1 수식 설명에 사용했습니다. :chatgpt-content-reference{index="116"} |
| **Gulrajani & Lopez-Paz, “In Search of Lost Domain Generalization”** | arXiv:2007.01434, 2020 공개본. DomainBed와 공정한 기준선·모델 선택 비교에 사용했습니다. :chatgpt-content-reference{index="117"} |
| **Miller et al., “Accuracy on the Line: on the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization”** | PMLR, ICML 2021. ID·OOD 상관관계와 예외에 사용했습니다. :chatgpt-content-reference{index="118"} |
| **Sagawa et al., “Extending the WILDS Benchmark for Unsupervised Adaptation”** | arXiv:2112.05090, 2022 개정본. 비라벨 자료 확장과 Table 2 비교에 사용했습니다. :chatgpt-content-reference{index="119"} |
| **Kumar et al., “Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution”** | arXiv:2202.10054, ICLR 2022. LP-FT, 표현 왜곡, FMoW 평가 설정 차이에 사용했습니다. :chatgpt-content-reference{index="120"} |
| **Kirichenko, Izmailov & Wilson, “Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations”** | arXiv:2204.02937. DFR의 목적함수·검증 자료 사용·CivilComments 결과에 사용했습니다. :chatgpt-content-reference{index="121"} |
| **Choi et al., “AutoFT: Learning an Objective for Robust Fine-Tuning”** | arXiv:2401.10220, 2024. OOD 검증 기반 학습 절차 탐색과 WILDS 결과에 사용했습니다. :chatgpt-content-reference{index="122"} |
| **Crasto & Rolf, “Latent Domain Modeling Improves Robustness to Geographic Shifts”** | arXiv:2503.02036v3, 2026-02-09. 위치 기반 조건부 모델과 FMoW·PovertyMap 결과에 사용했습니다. :chatgpt-content-reference{index="123"} |
| **Gardès et al., “Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Have”** | arXiv:2606.05107 공개본, 2026. FINO의 메타데이터 기반 표현 적응, 성능 및 한계에 사용했습니다. :chatgpt-content-reference{index="124"} |
