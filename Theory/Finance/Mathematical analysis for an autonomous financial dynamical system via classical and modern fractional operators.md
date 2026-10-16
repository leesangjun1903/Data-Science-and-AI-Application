# Mathematical analysis for an autonomous financial dynamical system via classical and modern fractional operators

**대상 논문:** Abdullahi Yusuf, Sania Qureshi, Syed Feroz Shah, *Chaos, Solitons & Fractals*, 132, 109552, 2020. 아래 페이지 번호는 첨부 PDF의 인쇄 페이지 번호를 따릅니다. **[저자 보고]**는 원문에 명시된 내용, **[검토·도출]**은 원문의 수식과 그림을 바탕으로 한 제 분석, **[후속 연구]**는 별도로 확인한 문헌을 뜻합니다. 원문의 시뮬레이션을 재실행한 결과는 포함하지 않습니다. Mathematical analysis for an au…

## 1. Executive summary

이 논문은 금리·투자수요·물가지수의 상호작용을 기술하는 기존 금융 동역학 모형을 분수계 미분방정식으로 확장한 **수학적·수치적 연구이며, 금융 데이터를 학습하는 AI 예측모형 연구는 아닙니다**〔pp. 4–7〕.  
연구 목적은 경제 주체의 과거 경험과 기억 효과를 모형에 반영하고, 기억을 표현하는 방식에 따라 금융 변수의 움직임이 어떻게 달라지는지 비교하는 것입니다〔pp. 1–3〕. Mathematical analysis for an au… Mathematical analysis for an au…  
이를 위해 Caputo, Caputo–Fabrizio–Caputo, Atangana–Baleanu–Caputo 연산자를 각각 적용하고, 세 상태변수에 서로 다른 미분 차수를 부여한 세 모형을 구성합니다〔p. 4, 식 (2.2)–(2.4)〕.  
저자들은 특정 시뮬레이션에서 ABC 모형의 진동이 상대적으로 작다는 점을 근거로 더 좋은 안정성을 주장합니다〔pp. 8–11, Figs. 5–7〕. Mathematical analysis for an au… Mathematical analysis for an au…  
그러나 이 결과는 실제 금융 데이터에 대한 예측오차 감소나 일반화 성능 향상을 입증한 것이 아닙니다〔§4〕.  
해의 존재·유일성에 관한 상세한 논증은 ABC 모형을 중심으로 제시되며, 조건부 정리를 모든 설정에서의 안정성 보장으로 읽어서는 안 됩니다〔pp. 4–5, Theorem 1〕. Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au…  
특히 원문의 적분방정식과 초기값을 대조하면 ABC·CFC의 초기조건 적합성 문제가 드러나므로, 보고된 곡선을 원문 초기값 문제의 해로 받아들이기 전에 수학적 재검토가 필요합니다〔p. 4, 식 (3.2); 아래 §4.2의 직접 도출〕.  
따라서 이 논문의 의의는 **기억 구조가 금융 모형의 궤적을 바꾼다는 탐색적 사례**에 있으며, “ABC가 가장 정확하거나 일반화가 잘된다”는 결론까지는 뒷받침하지 못합니다. Mathematical analysis for an au… [arXiv](https://arxiv.org/html/2006.15237v1)

> **용어 풀이 — 분수계 미분:** 미분의 차수를 1, 2 같은 정수가 아닌 값으로 확장한 것입니다. 이 논문에서는 과거 변화에 가중치를 주어 현재의 동역학에 반영합니다.  
> **일반화 성능:** 모형을 설정하거나 학습할 때 사용하지 않은 기간·시장·상황에서도 정확하게 작동하는 능력입니다. 단순히 곡선이 매끄럽다는 뜻이 아닙니다.

---

## 2. 연구 목적과 핵심 주장

### 2.1 무엇을 해결하려는가?

**[저자 보고]** 저자들의 출발점은 경제 주체가 과거의 외생·내생 변수 변화를 기억한다는 것입니다.  
이에 따라 금리, 투자수요, 물가지수의 현재 변화가 현재 상태뿐 아니라 과거 경로에도 의존하도록 기존 모형을 확장하려 합니다.  
또한 세 변수의 실제 움직임이 서로 다르다고 보고, 각 방정식에 서로 다른 분수 차수를 부여합니다〔p. 1, 초록·서론; p. 11, 결론〕. Mathematical analysis for an au… Mathematical analysis for an au…

**[검토·도출]** 따라서 논문의 질문은 다음과 같이 정리하는 것이 정확합니다.

> “같은 비선형 금융 방정식에서 미분 연산자와 기억 가중 방식을 바꾸면, 계산되는 궤적과 진동이 어떻게 달라지는가?”

반면 “어느 모형이 실제 시장을 가장 잘 예측하는가?”는 이 논문의 실험으로 답한 질문이 아닙니다.  
원문의 수치실험은 초기값·모수·차수를 지정해 궤적을 계산하는 방식입니다〔pp. 6–11, §4〕. Mathematical analysis for an au… Mathematical analysis for an au…

> **용어 풀이 — 외생·내생 변수:** 외생 변수는 모형 밖에서 주어지는 요인, 내생 변수는 모형 안의 상호작용으로 결정되는 변수입니다.  
> **궤적:** 시간이 흐르면서 상태변수들이 지나가는 값들의 경로입니다.

### 2.2 핵심 주장과 근거 표

| 핵심 주장 | 저자가 제시한 근거와 위치 | 검토 결과 |
|---|---|---|
| 기존 금융 모형에 서로 다른 기억 구조를 도입할 수 있다. | 세 연산자로 구성한 식 (2.2)–(2.4), **p. 4**. | **모형 구성 자체는 확인됩니다.** 다만 해당 기억 구조가 실제 금융 데이터에 적합한지는 별도 검증이 필요합니다. Mathematical analysis for an au… |
| 분수계 금융 모형에 유일한 해가 존재한다. | ABC 적분방정식과 고정점 논증, **pp. 4–5, Theorem 1, 식 (3.10)**. | **조건부 주장입니다.** 상세 증명은 ABC 중심이며, 초기조건 적합성과 결합계 전체의 수축 조건에 보완이 필요합니다. Mathematical analysis for an au… Mathematical analysis for an au… |
| ABC 모형은 진동이 적어 안정성이 더 좋다. | **Fig. 1, p. 2; Fig. 5, p. 8; p. 11의 해석**. | 선택된 설정에서의 시각적 감쇠는 관찰할 수 있지만, **일반적인 안정성 정리나 정량적 우월성은 아닙니다.** Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au… |
| 저축·투자비용·수요탄력성을 높이면 ABC 모형이 더 안정화된다. | **Figs. 2–4, pp. 3, 6–7; p. 11**. | 제한된 모수 조합에 대한 관찰입니다. 모든 모수 영역에서 성립하는 단조 관계나 경제적 인과관계는 입증하지 않습니다. Mathematical analysis for an au… |
| ABC는 다른 연산자와 다른 혼돈 패턴을 표현한다. | **Figs. 6–7, pp. 9–10; p. 11**. | 복잡한 궤적의 차이는 보이지만, **그림만으로 혼돈을 확정하거나 ‘더 많은 혼돈’을 정량화할 수 없습니다.** Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au… |
| 금융 예측 성능이 향상된다. | 원문에는 해당 성능 지표나 데이터 검증이 제시되지 않습니다. **§4의 결과는 시뮬레이션 그림 중심**입니다. | **입증되지 않은 해석입니다.** 예측오차 개선율·시장 간 일반화·실거래 성과를 이 논문에서 도출할 수 없습니다. Mathematical analysis for an au… Mathematical analysis for an au… |

> **용어 풀이 — 고정점 논증:** 어떤 변환을 적용해도 바뀌지 않는 함수를 찾는 방식으로, 미분방정식의 해가 존재하는지 분석하는 방법입니다.  
> **혼돈:** 결정론적 방정식에서도 초기값의 작은 차이가 크게 증폭되어 장기 예측이 어려워지는 현상입니다. 복잡해 보이는 그림 자체와는 구별해야 합니다.

---

## 3. 제안 방법과 모델 구조

### 3.1 출발점: 세 변수로 구성된 자율 금융 동역학계

**[저자 보고: p. 4, 식 (2.1), Table 1]** 원문의 기본 모형은 다음과 같습니다. 특히 첫 식은 $P+R(I-A)$이고, 마지막 식은 $-(CP+R)$입니다. Mathematical analysis for an au…

$$
\begin{aligned}
\frac{dR(t)}{dt}
&=P(t)+R(t)\bigl(I(t)-A\bigr),\\
\frac{dI(t)}{dt}
&=1-BI(t)-R(t)^2,\\
\frac{dP(t)}{dt}
&=-\bigl(CP(t)+R(t)\bigr).
\end{aligned}
$$

$$
R(0)=R_0,\qquad I(0)=I_0,\qquad P(0)=P_0.
$$

| 기호 | 원문의 정의 |
|---|---|
| $t$ | 시간 |
| $R(t)$ | 금리 |
| $I(t)$ | 투자수요 |
| $P(t)$ | 물가지수 |
| $A$ | 저축액 |
| $B$ | 투자당 비용 |
| $C$ | 수요의 탄력성 |
| $R_0,I_0,P_0$ | 세 변수의 초기값 |

**[검토·도출]** 모형 구조는 다음과 같이 읽을 수 있습니다. $R(I-A)$는 금리와 투자수요의 비선형 상호작용을 만들고, $-BI$는 투자수요를 감소시키는 항이며, $-R^2$는 금리의 크기에 따라 투자수요 변화율을 낮춥니다. 마지막 방정식에서는 $-CP$와 $-R$가 물가지수의 변화에 작용합니다. 이는 **주어진 방정식 안의 관계**이지, 데이터를 통해 식별한 경제적 인과관계가 아닙니다〔식 (2.1)에서 직접 해석〕. Mathematical analysis for an au…

> **용어 풀이 — 자율계:** 방정식의 우변에 시간이 직접 나타나지 않고 상태변수만 나타나는 계입니다. 시간이 지나도 상태가 변하지 않는다는 뜻은 아닙니다.  
> **비선형:** 변수끼리 곱해지거나 제곱되는 등, 입력 변화에 출력이 단순 비례하지 않는 구조입니다.

### 3.2 세 분수계 연산자

아래 식에서는 원문의 정규화 함수 $B(q)$를 투자비용 $B$와 구분하기 위해 **$\mathcal N(q)$로 표기**합니다. 이는 설명을 위한 기호 변경이며, 새 함수를 제안하는 것이 아닙니다. 원문은 $\mathcal N(0)=\mathcal N(1)=1$에 해당하는 조건을 제시합니다〔p. 4〕. Mathematical analysis for an au… Mathematical analysis for an au…

#### A. Caputo: 거듭제곱 커널

**[저자 보고: p. 4, 식 (2.2)]**

```math
{}^{\text{C}}D_t^\alpha u(t)
=
\frac{1}{\Gamma(1-\alpha)}
\int_0^t (t-s)^{-\alpha}u'(s)\,ds,
\qquad 0<\alpha<1.
```

여기서 $u(t)$는 미분 대상 함수, $s$는 과거 시점을 나타내는 적분 변수, $u'(s)$는 보통의 1차 미분, $\alpha$는 분수 차수, $\Gamma$는 감마함수입니다. 과거 변화 $u'(s)$에 $(t-s)^{-\alpha}$라는 가중치를 부여합니다. Mathematical analysis for an au…

> **용어 풀이 — 커널:** 과거 정보에 얼마만큼의 가중치를 줄지 정하는 함수입니다.  
> **특이 커널:** 여기서는 $s$가 $t$에 가까워질 때 가중치가 커지는 커널입니다. 다만 적분 자체는 적절한 조건에서 유한할 수 있습니다.  
> **감마함수:** 정수에서의 계승을 더 넓은 수의 범위로 확장하는 함수입니다.

#### B. Caputo–Fabrizio–Caputo: 지수 커널

**[저자 보고: p. 4, 식 (2.3)]**

```math
{}^{\text{CFC}}D_t^\beta u(t)
=
\frac{\mathcal N(\beta)}{1-\beta}
\int_0^t
\exp\!\left[-\frac{\beta}{1-\beta}(t-s)\right]
u'(s)\,ds,
\qquad 0<\beta<1.
```

$\beta$는 CFC의 차수이며, $\exp$는 지수함수입니다. 과거 변화에 지수적으로 감소하는 가중치를 부여하고, 커널은 $s=t$에서 유한합니다. Mathematical analysis for an au…

> **용어 풀이 — 비특이 커널:** 위 적분의 커널처럼 현재 시점 부근에서도 값이 무한대로 발산하지 않는 함수입니다.

#### C. Atangana–Baleanu–Caputo: Mittag–Leffler 커널

**[저자 보고: p. 4, 식 (2.4)]**

```math
{}^{\text{ABC}}D_t^\gamma u(t)
=
\frac{\mathcal N(\gamma)}{1-\gamma}
\int_0^t
E_\gamma\!\left[
-\frac{\gamma}{1-\gamma}(t-s)^\gamma
\right]
u'(s)\,ds,
\qquad 0<\gamma<1.
```

여기서 $\gamma$는 ABC의 차수입니다. $E_\gamma$는 다음 Mittag–Leffler 함수입니다. Mathematical analysis for an au… [arXiv](https://arxiv.org/html/2006.15237v1)

```math
E_\gamma(z)
=
\sum_{m=0}^{\infty}
\frac{z^m}{\Gamma(\gamma m+1)}.
```

$z$는 함수의 입력, $m$은 합산 인덱스입니다.

> **용어 풀이 — Mittag–Leffler 함수:** 지수함수를 일반화한 함수로, 분수계 모형에서 기억과 감쇠를 표현할 때 사용됩니다.

**[검토·도출]** 세 정의의 핵심 차이는 **동일한 과거 변화를 어떤 시간 가중치로 누적하는가**입니다. 따라서 $\alpha=\beta=\gamma=0.9$처럼 차수의 숫자가 같더라도 세 모형의 기억 길이나 감쇠 특성이 같다는 뜻은 아닙니다. 또한 CFC의 식에도 과거에 대한 적분이 있으므로, 원문 p. 2의 “local”이라는 표현을 “과거 의존성이 전혀 없다”로 받아들여서는 안 됩니다. Mathematical analysis for an au… Mathematical analysis for an au…

위 적분식은 $0<q<1$의 표현입니다. 분모가 0이 되는 $q=1$에 그대로 대입하는 대신, 정수 차수는 별도로 정의하거나 해당 극한으로 다루어야 합니다.

### 3.3 세 연산자를 섞은 하나의 모델이 아니라, 세 개의 별도 모델

상태벡터와 우변을 다음처럼 묶겠습니다.

$$
\mathbf x(t)=
\begin{bmatrix}
R(t)\\I(t)\\P(t)
\end{bmatrix},
\qquad
\mathbf f(\mathbf x)=
\begin{bmatrix}
P+R(I-A)\\
1-BI-R^2\\
-(CP+R)
\end{bmatrix}.
$$

원문의 구성은 다음과 같습니다.

$$
{}^{\text{C}}D_t^{\alpha_i}x_i=f_i(\mathbf x),
\qquad
{}^{\text{CFC}}D_t^{\beta_i}x_i=f_i(\mathbf x),
\qquad
{}^{\text{ABC}}D_t^{\gamma_i}x_i=f_i(\mathbf x),
\quad i=1,2,3.
$$

$i=1,2,3$은 각각 $R,I,P$에 대응합니다. 예를 들어 ABC 모델은 **세 방정식 모두 ABC를 사용하되 $\gamma_1,\gamma_2,\gamma_3$를 다르게 설정**합니다. 금리에는 Caputo, 투자에는 CFC, 물가에는 ABC를 섞는 구조가 아닙니다〔p. 4, 식 (2.2)–(2.4)〕. Mathematical analysis for an au…

---

## 4. 수학적 주장과 수치방법의 검토

### 4.1 저자가 보고한 존재·유일성 결과

**[저자 보고: pp. 4–5, 식 (3.2), Theorem 1]** 저자들은 ABC 모델을 다음 적분방정식으로 바꾸고 반복해를 구성합니다. 아래는 원문의 세 식을 성분별 벡터 표기로 묶은 것입니다. Mathematical analysis for an au…

$$
\begin{aligned}
x_i(t)-x_i(0)
={}&
\frac{1-\gamma_i}{\mathcal N(\gamma_i)}
f_i(\mathbf x(t))\\
&+
\frac{\gamma_i}{\mathcal N(\gamma_i)\Gamma(\gamma_i)}
\int_0^t
(t-s)^{\gamma_i-1}f_i(\mathbf x(s))\,ds.
\end{aligned}
$$

첫 항은 현재 상태의 우변에 직접 의존하고, 두 번째 항은 과거 우변의 가중 적분입니다.

Theorem 1의 조건은 원문 표기를 유지하면 다음과 같습니다.

$$
\frac{1-\gamma_j}{\mathcal N(\gamma_j)}\zeta_j
+
\frac{\zeta_j b^{\gamma_j}}
{\mathcal N(\gamma_j)\Gamma(\gamma_j)}
<1,
\qquad j=1,2,3.
$$

$\zeta_j$는 원문에서 사용하는 Lipschitz 상수입니다. **$b$는 해당 정리 주변에서 명확하게 정의되지 않아, 여기서 임의로 $T$와 동일시하지 않겠습니다.** 원문은 이 조건 아래에서 반복함수열의 수렴과 해의 유일성을 주장합니다〔p. 5, 식 (3.10)–(3.12)〕. Mathematical analysis for an au…

> **용어 풀이 — Lipschitz 조건:** 입력이 조금 달라졌을 때 출력 차이가 일정 배수 이상 급격히 커지지 않는다는 조건입니다.  
> **수축:** 반복할수록 두 후보 해 사이의 거리가 줄어드는 성질입니다. 적절한 함수공간에서 수축이 성립하면 유일한 고정점을 얻을 수 있습니다.

### 4.2 가장 중요한 문제: 초기조건과 ABC·CFC 방정식의 양립성

**[검토·도출]** 원문의 식 (3.2)를 연속해에 대한 식으로 받아들이고 $t\downarrow0$으로 보내면, 왼쪽과 적분항은 0으로 갑니다. 따라서 모든 $\gamma_i<1$에 대해 다음 필요조건이 생깁니다.

$$
0=
\frac{1-\gamma_i}{\mathcal N(\gamma_i)}
f_i(\mathbf x_0),
\qquad\text{즉}\qquad
f_i(\mathbf x_0)=0.
$$

이것은 저자가 명시적으로 점검한 조건이 아니라, **원문 적분방정식에서 직접 도출되는 조건**입니다. Mathematical analysis for an au…

그런데 Fig. 1의 설정은 다음과 같습니다.

$$
\mathbf x_0=(0.5,1,2)^\mathsf T,
\qquad A=C=1,\qquad B=10^{-3}.
$$

이를 원문의 우변에 직접 대입하면 다음을 얻습니다.

```math
\mathbf f(\mathbf x_0)
=
\begin{bmatrix}
2+0.5(1-1)\\
1-10^{-3}\cdot1-0.5^2\\
-(1\cdot2+0.5)
\end{bmatrix}
=
\begin{bmatrix}
2\\
0.749\\
-2.5
\end{bmatrix}
\neq\mathbf0.
```

**즉, Fig. 1의 초기값은 위 필요조건을 만족하지 않습니다.** 원문 식 (3.2)와 연속적인 초기값 충족을 동시에 유지할 수 없다는 뜻입니다〔p. 2, Fig. 1; p. 4, 식 (2.1), (3.2)〕. Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au…

**[후속 연구]** Diethelm 등의 2020년 논문은 보다 일반적으로, 유계 커널을 사용하는 해당 유형의 연산자는 절대연속 함수에 대해 초기 시점의 우극한이 0이 되어 초기조건 제약을 유발함을 보입니다〔§3.2, Theorem 3.1〕. 이 논리로 CFC에도 같은 종류의 문제가 발생합니다. [arXiv](https://arxiv.org/html/2006.15237v1)

> **용어 풀이 — 유계:** 함수의 크기에 유한한 상한이 있다는 뜻입니다.  
> **절대연속 함수:** 대략적으로, 점프 없이 변하고 그 변화량을 적분 가능한 도함수의 적분으로 복원할 수 있는 함수입니다.  
> **초기조건 적합성:** 시작값이 방정식이 요구하는 조건과 충돌하지 않아야 한다는 뜻입니다.

여기서 필요한 결론은 **“ABC를 사용하는 모든 연구가 무효다”가 아닙니다.** Fernandez와 Al-Refai의 2023년 연구는 함수공간과 연산자의 정의를 적절히 다루어 비특이 커널 연산자의 수학적 기반을 구성하는 방향을 제시합니다. 다만 그러한 확장은 이 원문의 초기조건 문제를 자동으로 해결하지는 않습니다. **원문의 정의·연속해 설정·초기값을 그대로 유지한 상태에서 보고된 수치곡선의 정당성은 재검토되어야 합니다.** [i-rep.emu.edu.tr](https://i-rep.emu.edu.tr/items/f5e18756-2025-4210-afd9-2259c161f05d)

### 4.3 결합된 세 방정식 전체에 대한 증명이 충분한가?

**[저자 보고]** p. 5의 식 (3.4)–(3.6)은 첫 번째 우변에서 다른 변수들을 고정하고 $R$만 변화시키는 방식으로 Lipschitz 조건을 설명합니다. Mathematical analysis for an au…

**[검토·도출]** 그러나 실제 두 후보 궤적을 비교하면 $R,I,P$가 함께 달라집니다. 두 상태를 $\mathbf x=(R,I,P)$와 $\widetilde{\mathbf x}=(\widetilde R,\widetilde I,\widetilde P)$로 놓고, $\Delta R=R-\widetilde R$ 등으로 정의하면 다음과 같습니다.

$$
\begin{aligned}
\Delta f_1
&=\Delta P+(I-A)\Delta R+\widetilde R\,\Delta I,\\
\Delta f_2
&=-B\Delta I-(R+\widetilde R)\Delta R,\\
\Delta f_3
&=-C\Delta P-\Delta R.
\end{aligned}
$$

따라서 $\Delta R,\Delta I,\Delta P$ 사이의 **교차 영향까지 포함하는 벡터 부등식**이 필요합니다. 원문처럼 다른 변수를 고정한 성분별 부등식만으로는 결합계 전체의 수축이 자동으로 증명되지 않습니다. 또한 원문은 해의 유계성을 당연한 것으로 취급하지만, 그 영역을 유지한다는 별도의 분석은 제시하지 않습니다〔p. 5, 식 (3.4)–(3.10)〕. Mathematical analysis for an au…

**해가 하나 존재한다는 주장, 작은 교란에도 안정적이라는 주장, 새로운 금융 데이터에 정확하다는 주장은 서로 다른 주장입니다.**

### 4.4 수치계산은 어떻게 이루어지는가?

**[저자 보고]** 수치계산의 고정 시간 간격은 $h=10^{-2}$입니다. 연산자별로 서로 다른 기존 수치방법을 적용합니다〔p. 6, §4〕. Mathematical analysis for an au…

예를 들어 Caputo의 식 (4.4)을 성분별로 묶으면 다음과 같습니다.

```math
x^{\text{C}}_{i,n+1}
=
x_i(0)
+
\frac{h^{\alpha_i}}{\Gamma(\alpha_i+1)}
\sum_{k=0}^{n}
\left[
(n+1-k)^{\alpha_i}-(n-k)^{\alpha_i}
\right]
f_i(\mathbf x_k).
```

여기서 $t_n=nh$, $\mathbf x_k$는 $t_k$에서의 근사 상태, $n$은 현재 계산 단계, $k$는 과거 단계의 인덱스입니다. 과거 단계들을 합산한다는 점에서 기억 효과가 계산식에 직접 나타납니다〔p. 7, 식 (4.4)〕. Mathematical analysis for an au…

**[검토·도출: 재현성 문제]** 같은 $h$를 썼다는 사실만으로 수치오차가 같다고 볼 수는 없습니다. 원문에는 시간 간격을 줄여도 결론이 유지되는지 확인하는 비교가 없으며, 정규화 함수의 실제 선택도 충분히 명확하지 않습니다. 특히 ABC의 식 (4.6)은 $k=0$부터 합산하면서 $k-1$ 시점의 값을 사용하므로, **첫 계산에 필요한 시작 규칙을 별도로 명시해야 합니다**〔p. 7, 식 (4.5); p. 11, 식 (4.6)〕. Mathematical analysis for an au… Mathematical analysis for an au…

따라서 현재 결과만으로는 **모델의 기억 구조가 만든 차이와 수치방법이 만든 차이를 완전히 분리하기 어렵습니다.**

---

## 5. 가장 중요한 그림의 해석

### 5.1 Fig. 5, p. 8 — 연산자 간 차이를 가장 직접적으로 보여 주는 그림

**설정:** 시간 구간 $[0,10]$, 초기값 $(0.91,0.85,0.98)$, 차수 $(0.97,0.91,0.88)$, $A=B=0.5$, $C=1.5$입니다. 금리·투자수요·물가지수별로 네 모형을 같은 패널에 겹쳐 보여 줍니다. Mathematical analysis for an au…

**[저자 보고]** ABC가 작은 시간 구간에서도 가장 안정적이고 진동이 적다고 해석합니다〔p. 11〕. Mathematical analysis for an au…

**[그림에 대한 제 해석]** ABC의 금리와 투자수요 곡선은 정수계 모형보다 일부 봉우리와 골이 완화되어 보이며, Caputo도 유사한 완화 경향을 보이는 구간이 있습니다. 하지만 이것은 **선택된 초기값에서의 단기 과도응답 비교**입니다. 참값이나 관측값이 없기 때문에 더 평탄한 곡선이 더 정확한 곡선인지는 판단할 수 없습니다.

> **용어 풀이 — 과도응답:** 초기 상태에서 출발한 뒤 장기적인 움직임에 도달하기 전까지 나타나는 변화입니다.

### 5.2 Fig. 4, p. 7 — 안정화가 모수에 의존한다는 점이 중요하다

**설정:** $C=0.1,0.6,0.7$을 비교하고, $A=1$, $B=0.55$를 고정합니다. Mathematical analysis for an au…

**[저자 보고]** 수요탄력성이 커지면 물가지수가 감소하고 ABC 모형이 더 안정적이라고 설명합니다〔p. 11〕. Mathematical analysis for an au…

**[그림에 대한 제 해석]** ABC의 $C=0.6,0.7$에서는 진동이 상당히 작아지는 반면, $C=0.1$에서는 진동이 유지됩니다. 따라서 이 그림의 중요한 메시지는 **“ABC라서 언제나 안정적”이 아니라 “같은 ABC 안에서도 모수에 따라 움직임이 크게 달라진다”**입니다. 또한 진동 폭의 감소와 물가지수 수준의 하락은 서로 다른 개념입니다.

### 5.3 Fig. 6, p. 9 — 수렴처럼 보이는 나선과 혼돈을 구별해야 한다

**설정:** 시간 구간 $[0,200]$, 초기값 $(1,-1,1)$, 차수 $(0.89,0.91,0.95)$, $A=0.98$, $B=0.02$, $C=1$입니다. Mathematical analysis for an au…

**[저자 보고]** 저자들은 Figs. 6–7에서 혼돈적 거동과 ABC 고유의 패턴을 관찰했다고 설명합니다. Mathematical analysis for an au… Mathematical analysis for an au…

**[그림에 대한 제 해석]** ABC 패널은 작은 영역을 향해 감기는 나선처럼 보입니다. 이는 감쇠나 평형점 접근의 가능성을 시사하지만, **혼돈의 증거라고 단정하기는 어렵습니다.** 반대로 나머지 패널의 복잡한 고리 역시 그것만으로 혼돈의 증명이 되지 않습니다. 패널마다 축 범위도 다르므로 그림이 작게 보인다는 이유만으로 변동 폭을 정량 비교해서는 안 됩니다.

혼돈을 주장하려면, 예를 들어 초기값 교란 실험과 적절하게 계산된 Lyapunov 지수 등의 추가 근거가 필요합니다.

> **용어 풀이 — 평형점:** 모든 상태변수의 변화가 멈추는 상태입니다.  
> **Lyapunov 지수:** 매우 가까운 두 궤적이 시간이 지나며 얼마나 빨리 벌어지거나 가까워지는지 나타내는 값입니다. 적절한 조건과 계산 아래 양의 최대 지수는 혼돈을 판단하는 중요한 근거가 됩니다.

### 5.4 Fig. 7, p. 10 — Fig. 6의 결론을 일반화할 수 없는 이유

Fig. 7은 초기값·차수·모수 모두 Fig. 6과 다릅니다. 특히 $A=0.2$, $B=10^{-6}$, $C=10^{-4}$이며, ABC에서도 여러 고리 형태의 궤적이 나타납니다. Mathematical analysis for an au…

**[그림에 대한 제 해석]** Fig. 6에서 보인 수렴 유사 형태를 ABC의 보편적 특성으로 확대할 수 없음을 보여 주는 비교 사례입니다. 이 두 그림은 **같은 실험을 반복한 것이 아니라 서로 다른 설정의 사례**이므로, 고리 수를 세어 안정성의 개선율을 계산하는 것도 적절하지 않습니다.

---

## 6. 성능 향상 주장, 통계적 취약점, 비교 불가능한 수치

### 6.1 이 논문에서 확인되는 개선은 무엇인가?

**[검토]** 원문에서 확인되는 것은 일부 설정에서의 **진동 감소 또는 다른 궤적 형성**입니다. 예측오차, 금융 손실, 수익률, 불확실성 보정, 계산시간에 대한 정량적 개선은 보고하지 않습니다. Table 1도 성능 비교표가 아니라 변수·모수 설명표입니다〔p. 4, Table 1; pp. 6–11, §4〕. Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au…

따라서 “ABC가 몇 % 우수하다”는 수치를 원문의 그림에서 만들어내서는 안 됩니다.

### 6.2 취약점과 비교 범위

| 항목 | 구분 | 정확한 평가 |
|---|---|---|
| 실제 금융 데이터 검증 | **미검증** | 관측 데이터 적합과 미래 구간 예측이 없어 금융 예측의 일반화를 판단할 수 없습니다〔§4〕. |
| 통계적 유의성 | **적용 범위 주의** | 결정론적 수치실험에 유의확률이 없다는 것 자체는 오류가 아닙니다. 문제는 제한된 실험으로 보편적인 금융 우월성을 주장하는 데 있습니다. |
| 많은 시간 단계 | **독립 표본으로 해석 불가** | 한 궤적의 수천 계산점은 수천 개의 독립된 시장 실험이 아닙니다〔p. 6, $h=10^{-2}$〕. |
| 같은 차수의 연산자 비교 | **조건부 비교 가능** | 같은 숫자의 차수에서 모델 차이를 비교할 수는 있으나, 같은 기억 길이나 같은 최적 적합 조건의 비교는 아닙니다〔식 (2.2)–(2.4)〕. |
| Fig. 1과 Fig. 5 비교 | **직접 비교 불가** | 초기값·차수·모수·관측 구간이 다릅니다〔pp. 2, 8〕. |
| Fig. 6과 Fig. 7 비교 | **직접 비교 불가** | 서로 다른 초기값과 모수 설정이며 축 범위도 동일하지 않습니다〔pp. 9–10〕. |
| 연산자별 수치오차 | **분리되지 않음** | 서로 다른 수치방법을 사용하므로 동일한 $h$만으로 동일 정확도의 비교가 되지는 않습니다〔pp. 6–7, 11〕. |

위 실험 설정과 수치방법의 근거는 원문 §4 및 각 그림 캡션입니다. Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au…

### 6.3 경제적 해석에서도 해결되지 않은 문제가 있다

**[검토·도출]** 원문은 $P$를 물가지수로 정의하지만 그림에는 음수인 $P$가 나타납니다. 또한 §4에서는 상태벡터를 양의 영역에 놓으면서, Fig. 6에서는 투자수요의 초기값을 $I_0=-1$로 설정합니다. 이를 평균에서의 편차나 표준화된 값으로 해석하면 설명할 여지가 있지만, **원문이 그렇게 정의하지 않았으므로 제가 임의로 보정해서 읽을 수는 없습니다.** Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au…

기본 정수계 방정식에서도 $P=0$, $R>0$이면 다음과 같습니다.

$$
\frac{dP}{dt}=-R<0.
$$

따라서 이 경계에서는 물가지수가 음수 영역으로 이동합니다. 이는 **양의 물가지수 수준을 직접 모형화한 것인지, 어떤 기준값에서의 편차를 모형화한 것인지 명확히 해야 한다**는 뜻입니다〔식 (2.1)에서 직접 도출〕. Mathematical analysis for an au…

분수계로 바꾸면 시간 차원도 달라집니다. 보통의 변화율은 ‘상태/시간’인 반면 차수 $q$의 변화율은 ‘상태/시간 $^q$ ’에 해당하므로, 시간축을 일 단위로 해석하려면 무차원화 또는 시간척도 계수에 대한 설명도 필요합니다. 원문에는 이를 충분히 명시한 설명이 없습니다〔p. 4, 식 (2.1)–(2.4)〕.

> **용어 풀이 — 무차원화:** 변수와 시간을 기준 크기로 나누어 단위가 없는 형태로 바꾸는 작업입니다. 다른 미분 차수와 시간 단위를 일관되게 비교하는 데 중요합니다.

---

## 7. 2020년 이후 관련 연구 비교

아래는 **2020년 이후부터 2026년 발표분까지 확인한 관련성 높은 연구**입니다. 비교는 수학적 정당성, 실제 금융 데이터 검증, 학습 가능한 기억 구조라는 세 축으로 나누었습니다. 이들이 모두 대상 논문에서 직접 파생되었거나 이를 직접 인용했다는 의미는 아닙니다.

### 7.1 비교 표

| 연구 | 저자가 직접 보고한 방법·결과와 확인 위치 | 대상 논문과의 차이 및 한계 |
|---|---|---|
| **Diethelm et al., 2020** — *Why Fractional Derivatives with Nonsingular Kernels Should Not Be Used* | 비특이 커널 연산자의 초기값 제약과 역연산 문제를 분석합니다〔§3, 특히 Theorem 3.1〕. | 대상 논문의 ABC·CFC 초기값 문제를 점검할 수 있는 수학적 기준을 제공합니다. 금융 예측 비교 연구는 아닙니다. [arXiv](https://arxiv.org/html/2006.15237v1) |
| **Fernandez & Al-Refai, 2023** — *A Rigorous Analysis of Integro-Differential Operators with Non-Singular Kernels* | 함수공간과 사상 관계를 분석해 비특이 커널 연산자의 엄밀한 구성 방향을 제시합니다〔저자 소속기관 저장소의 초록〕. | 비특이 커널을 단순히 배제하기보다 정의·적용 공간을 명확히 해야 한다는 관점입니다. 원문의 초기조건을 자동으로 정당화하지는 않습니다. [i-rep.emu.edu.tr](https://i-rep.emu.edu.tr/items/f5e18756-2025-4210-afd9-2259c161f05d) |
| **An et al., 2024; 온라인 2023** — *Parameter estimation for time-fractional Black-Scholes equation with S &P 500 index option* | Caputo 기반 옵션가격 방정식에서 차수와 변동성을 추정하고 실제 S&P 500 옵션 자료에 적합합니다〔§6; Fig. 8, p. 22〕. | 원문의 지정 모수 시뮬레이션과 달리 **데이터로 모수를 추정**합니다. 다만 개선은 옵션 종류·조건에 따라 다르며, 적합도 향상이 곧 시계열 미래예측 일반화는 아닙니다. [Springer](https://link.springer.com/article/10.1007/s11075-023-01563-4)

[Download PDF](https://libkey.io/libraries/4091/articles/576259766/full-text-file?utm_source=nomad) |
| **Zimmering et al., 2024** — *Optimising Neural Fractional Differential Equations for Performance and Efficiency* | 효율적인 Neural FDE 계산과 공정한 Neural ODE 비교를 다루며, 구현 비교에서 최대 570배 속도와 최대 79배 정확도 개선을 보고합니다〔PMLR 출판사 초록〕. | **수치해법·구현 벤치마크 수치**이며 금융 예측 개선율이 아닙니다. 동일한 계산방법과 조정 조건이 중요하다는 점이 원문의 비교 설계에 시사점을 줍니다. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v255/zimmering24a.html) |
| **Coelho et al., 2025; 공개본 2024** — *Neural Fractional Differential Equations* | 신경망으로 동역학과 분수 차수를 학습하고 합성·실제 시계열에서 평가합니다〔공개본 §3–4; Table 8〕. | 기억 구조를 데이터로 학습한다는 진전이 있지만, 실제 금융 실험은 제한적이며 학습된 차수가 실제 생성 차수를 복원한다고 보장하지 않습니다. 아래 수치는 **2024 공개본 v2** 기준입니다. [arXiv](https://arxiv.org/pdf/2403.02737v2) |
| **Waheed et al., 2026** — *Data-driven neural modeling and chaos control in fractional-order financial dynamical systems* | 금융 동역학의 수치 생성 궤적으로 신경망을 학습하고, Lyapunov 지수 및 제어 결과를 보고합니다〔§III; Tables I–II〕. | 궤적 그림에 더해 정량적 동역학 지표를 사용하지만, **합성 궤적 학습이 실제 금융 시장의 미래예측 검증을 대신하지는 않습니다.** [AIP Publishing](https://pubs.aip.org/aip/adv/article/16/1/015323/3378142/Data-driven-neural-modeling-and-chaos-control-in) |

### 7.2 실제 금융 데이터 적합과 미래예측을 구별해야 한다

An 등의 연구는 실제 옵션가격을 사용한다는 점에서 대상 논문보다 실증적입니다. 그러나 Fig. 8에서는 콜옵션의 차이가 작고 일부 풋옵션에서 더 뚜렷한 차이가 나타납니다. 따라서 **“분수계가 모든 금융 상품에서 우수하다”가 아니라 “특정 자료와 조건에서 적합도를 개선할 수 있다”**가 적절한 해석입니다〔Fig. 8, p. 22〕. [Springer](https://link.springer.com/content/pdf/10.1007/s11075-023-01563-4.pdf)

> **용어 풀이 — 적합도:** 이미 관측된 데이터를 모형이 얼마나 잘 설명하는지입니다. 관측하지 않은 미래를 잘 맞히는 능력과는 다릅니다.  
> **모수 추정:** 차수나 변동성처럼 미지의 값을 관측 데이터에 맞추어 결정하는 작업입니다.

### 7.3 Neural FDE가 보여 준 일반화 가능성과 작은 표본의 한계

Coelho 등의 공개본은 다음과 같은 구조를 학습합니다. 아래는 원문의 구조를 요약한 표기입니다.

```math
{}^{\text{C}}D_t^{q_\psi}\mathbf z(t)
=
F_\theta(t,\mathbf z(t)).
```

$\mathbf z(t)$는 모델 상태, $F_\theta$는 동역학을 표현하는 신경망, $\theta$는 그 가중치, $q_\psi$는 별도 학습 매개변수 $\psi$로 결정되는 분수 차수입니다〔공개본 §3〕. [arXiv](https://arxiv.org/pdf/2403.02737v2)

> **용어 풀이 — Neural FDE:** 분수계 미분방정식의 일부를 신경망으로 표현하고 데이터로 학습하는 모델입니다.  
> **Neural ODE:** 같은 아이디어를 보통의 정수계 미분방정식에 적용한 모델입니다.

공개본의 실제 주식 실험은 **Altaba 한 회사**를 대상으로 하며, 외삽 실험은 처음 365개 관측값으로 학습한 뒤 다음 30개를 평가합니다. Table 8의 결과는 다음과 같습니다〔공개본 pp. 16–17, §4.3〕. [arXiv](https://arxiv.org/pdf/2403.02737v2)

| 모델 | 미래 구간 외삽 MSE: 평균 ± 표준편차 |
|---|---:|
| Neural ODE | $0.0793\pm0.00428$ |
| Neural FDE | $0.0542\pm0.00162$ |

**[검토]** 이 표는 해당 실험에서 Neural FDE의 오차가 작았음을 보여 줍니다. 다만 표준편차는 **세 번의 학습 실행 사이 변동**이며, 시장 전체에 대한 신뢰구간이 아닙니다. 또한 같은 논문의 재구성 실험은 학습·평가 데이터를 공유하므로 미래예측과 구별해야 합니다. [arXiv](https://arxiv.org/pdf/2403.02737v2)

> **용어 풀이 — 외삽:** 학습한 시간 범위 밖으로 예측하는 것입니다.  
> **MSE:** 예측값과 실제값 차이의 제곱 평균입니다. 값이 작을수록 해당 평가 자료에서의 오차가 작습니다.  
> **신뢰구간:** 추정값의 통계적 불확실성을 일정한 절차로 표현한 범위입니다. 단순한 여러 실행의 표준편차와 같지 않습니다.

같은 공개본은 학습된 차수가 실제 생성 차수와 일치하지 않을 수 있다고 설명합니다〔Table 7 및 Table 9 주변 논의〕. 따라서 **“예측이 잘된다”와 “경제적 기억의 길이를 정확히 알아냈다”도 구별해야 합니다.** [arXiv](https://arxiv.org/pdf/2403.02737v2)

### 7.4 최신 연구의 작은 손실값도 같은 종류의 수치가 아니다

Waheed 등의 2026년 연구는 수치적으로 생성한 자료를 학습·시험 자료로 나누고, 일부 설정에서 약 $10^{-8}$ 수준의 훈련손실을 보고합니다〔§III, Table I〕. 그러나 이는 실제 시장의 미래 예측오차가 아닙니다. [AIP Publishing](https://pubs.aip.org/aip/adv/article/16/1/015323/3378142/Data-driven-neural-modeling-and-chaos-control-in)

따라서 다음 세 수치는 **동일한 성능 순위표에 놓을 수 없습니다.**

| 수치 또는 관찰 | 측정 대상 |
|---|---|
| 대상 논문의 “진동 감소” | 지정한 방정식의 시뮬레이션 형태 |
| Neural FDE의 미래 구간 MSE | 특정 실제 시계열의 제한된 시험 구간 |
| 계산 속도 배수 또는 매우 작은 훈련손실 | 구현 효율 또는 학습 자료 재현 정확도 |

**[종합 해석]** 이후 연구가 제시하는 중요한 방향은 단순히 새로운 연산자를 추가하는 것보다, **수학적으로 잘 정의된 초기화 → 데이터 기반 모수 추정 → 공정한 수치 비교 → 독립된 미래 구간 평가**로 연구의 근거를 강화하는 것입니다.

---

## 8. 문서가 답하지 않는 질문

다음은 원문의 모형 정의, 정리, 수치실험을 바탕으로 남는 질문입니다〔pp. 4–11〕. Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au…

| 미해결 질문 | 왜 중요한가? |
|---|---|
| $R,I,P$는 실제 수준인가, 기준값에서의 편차인가? | 음수 물가지수·투자수요를 경제적으로 해석할 수 있는지 결정합니다. |
| 선택한 분수 차수는 어떤 자료나 기준으로 정했는가? | 기억 구조가 임의 설정인지 관측에 근거한 것인지 구별해야 합니다. |
| ABC·CFC의 초기조건 적합성은 어떻게 확보하는가? | 계산할 대상인 초기값 문제가 제대로 정의되어야 합니다. |
| Theorem 1의 조건은 각 수치실험에서 실제로 성립하는가? | 조건부 정리를 실험 결과의 정당화에 사용할 수 있는지 결정합니다. |
| $A,B,C$와 차수의 어느 영역에서 어떤 안정성이 성립하는가? | 몇 개의 선택된 궤적을 넘어서는 결론에 필요합니다. |
| 그림의 복잡성이 혼돈인지, 과도응답인지 어떻게 구별하는가? | 금융 불안정성에 관한 해석의 근거가 달라집니다. |
| 시간 간격·수치방법·정규화·시작 규칙을 바꾸어도 결과가 유지되는가? | 결과의 수치적 재현성과 신뢰성 문제입니다. |
| 다른 시장·시기·정책 충격에서도 더 정확한가? | 일반화 성능에 대한 핵심 질문이지만 원문은 평가하지 않습니다. |

---

## 9. 결론과 일반화 성능 중심의 후속 연구 방향

### 9.1 저자들이 실제로 제시한 시사점과 계획

**[저자 보고]** 저자들은 분수계 연산자가 기억과 비국소성을 통해 복잡한 과정을 더 유연하게 표현할 수 있고, 각 변수에 서로 다른 차수를 부여하는 접근이 유용하다고 결론짓습니다. 후속 연구로는 Atangana $\beta$-derivative, truncated $M$-derivative, Atangana–Koca derivative, bi-order Atangana operator, Atangana–Gomez operator 등 **다른 미분 연산자를 같은 금융 모형에 적용하는 방향**을 제안합니다〔p. 11, §5〕. Mathematical analysis for an au…

실제 금융 데이터 기반 학습, 시장 간 일반화 평가, Neural FDE 결합은 원문이 명시한 계획이 아닙니다. 다음 내용은 **제가 제안하는 추가 연구 방향**입니다.

### 9.2 우선순위 1: 새로운 연산자보다 초기화와 재현성부터 해결

**[제안]** 첫 단계는 원문 모델을 그대로 재현하고, 다음을 분리해 검증하는 것입니다.

| 검증 단계 | 확인할 내용 |
|---|---|
| 수학적 정의 | 함수공간, 초기값 또는 과거 이력, 정규화 함수, 시간 단위 |
| 초기화 | ABC·CFC가 요구하는 초기조건과 선택한 시작 상태의 적합성 |
| 해석적 조건 | 결합계 전체의 Lipschitz 상계, 해가 머무는 영역, 정리 적용 범위 |
| 수치 재현 | 첫 단계 계산법, 시간 간격 축소, 독립적인 수치해법 비교 |

여기서 중요한 원칙은 **기억 모형의 시작점을 단순한 숫자 세 개로 처리할 수 있는지부터 확인하는 것**입니다. 모형을 수정해 과거 이력이나 다른 초기화 개념을 도입한다면, 그것은 원문과 구별되는 새 모형으로 명시해야 합니다.

### 9.3 우선순위 2: 기억 차수를 데이터로 추정하되 설명 가능성을 과신하지 않기

**[제안]** 실제 자료를 이용한다면 $A,B,C$와 분수 차수를 함께 추정하되, 예측오차만 최소화하지 말고 여러 매개변수 조합이 같은 자료를 비슷하게 설명하는지도 확인해야 합니다.

> **용어 풀이 — 식별 가능성:** 서로 다른 매개변수 값이 관측 데이터에서 구분되는 성질입니다. 여러 값이 거의 같은 예측을 만들면, 특정 값에 강한 경제적 의미를 부여하기 어렵습니다.

세 변수의 차수를 모두 다르게 하면 표현력은 늘지만 추정해야 할 자유도도 늘어납니다. 따라서 **공통 차수 하나를 사용하는 모형과 세 차수를 사용하는 모형을 함께 비교**해야 합니다. 세 차수가 미래 자료에서도 일관되게 유리할 때에만, 추가 복잡성이 정당화됩니다.

### 9.4 우선순위 3: 구조를 유지하는 하이브리드 AI 모형

**[제안: 원문에 없는 확장]** 무차원화된 변수와 시간을 사용한다는 전제에서, 다음처럼 원문의 구조에 작은 학습 잔차를 추가하는 연구를 고려할 수 있습니다.

```math
{}^{\text{C}}D_t^{q_i}x_i(t)
=
f_i(\mathbf x(t);\boldsymbol\eta)
+
r_{\phi,i}\bigl(\mathbf x(t),\mathbf u(t)\bigr),
\qquad 0<q_i\le1.
```

여기서 $\boldsymbol\eta=(A,B,C)$는 기존 모수, $q_i$는 상태별 차수, $r_{\phi,i}$는 가중치 $\phi$를 갖는 신경망 잔차, $\mathbf u(t)$는 관측 가능한 외생 입력입니다. $f_i$는 원문의 기본 구조를 유지합니다.

> **용어 풀이 — 하이브리드 모형:** 알려진 수학적 구조와 데이터 학습을 결합한 모형입니다.  
> **잔차 신경망:** 기존 방정식이 설명하지 못한 부분만 학습하도록 추가한 신경망입니다.

예시 학습 목적함수는 다음처럼 설계할 수 있습니다.

```math
\mathcal L
=
\frac1N\sum_{n=1}^{N}
\left\|
W\bigl(
\widehat{\mathbf x}(t_n)-\mathbf x_{\text{obs}}(t_n)
\bigr)
\right\|_2^2
+
\lambda_r\mathcal R(\phi)
+
\lambda_c\mathcal C(\widehat{\mathbf x}).
```

$N$은 학습 관측 수, $\widehat{\mathbf x}$는 모형 예측, $\mathbf x_{\text{obs}}$는 관측값, $W$는 변수별 척도 차이를 조정하는 행렬입니다. $\mathcal R$은 신경망의 복잡성을 제한하는 항, $\mathcal C$는 명확하게 정당화된 제약의 위반을 벌점화하는 항이며, $\lambda_r,\lambda_c$는 그 강도를 정합니다.

**이 구조가 반드시 더 잘 일반화한다는 보장은 없습니다.** 학습 잔차가 지나치게 유연하면 기존 금융 구조와 분수 차수의 의미를 흡수해 버릴 수 있으므로, 단순한 모델부터 단계적으로 검증해야 합니다.

### 9.5 우선순위 4: 일반화 평가 자체를 실험의 중심에 놓기

**[제안]** 평가 설계는 다음처럼 구성하는 것이 적절합니다.

| 평가 축 | 제안하는 설계 | 방지하려는 문제 |
|---|---|---|
| 미래 시간 | 과거로 학습하고 뒤의 연속 구간으로 평가하는 순차 검증 | 미래 정보가 학습에 섞이는 문제 |
| 다른 상황 | 저변동·고변동 시기와 정책 변화 전후를 분리 | 한 시장 상황에만 맞춘 모델 |
| 다른 시장 | 특정 국가·시장·자산을 학습에서 제외하고 평가 | 한 자료에 한정된 우월성 |
| 기억 유지 | 학습 종료 시점까지의 이력을 시험 구간에 일관되게 전달 | 시험 구간마다 기억을 임의로 초기화하는 문제 |
| 비교 공정성 | 모형 복잡도, 조정 예산, 수치오차 또는 계산 예산을 통제 | 더 많은 계산을 쓴 모델을 구조적으로 우수하다고 오인 |
| 불확실성 | 시간 의존성을 고려한 오차 구간과 여러 평가 기간 사용 | 학습 실행 간 편차를 시장 불확실성으로 오인 |

기본 비교군에는 원문의 정수계 모형, 공통 차수 Caputo, 상태별 차수 Caputo, 적합한 초기화가 명확한 비특이 커널 모형, 단순한 시계열 모형을 포함하는 것이 좋겠습니다.

또한 실제 자료가 크게 진동한다면, 이를 무조건 매끄럽게 만드는 것은 개선이 아닙니다. **좋은 모형의 목적은 진동을 줄이는 것이 아니라, 존재하는 진동과 불확실성을 정확하게 설명하고 예측하는 것입니다.**

### 최종 판단

이 논문은 **“금융 동역학에 어떤 기억 구조를 넣느냐에 따라 계산 결과가 달라진다”는 비교 실험**으로 읽을 때 의미가 있습니다. 그러나 핵심적인 초기조건 문제, 제한된 존재·유일성 논증, 정량적 안정성 검증의 부재, 실제 금융 데이터 평가의 부재 때문에 **ABC의 보편적 우월성이나 일반화 성능 향상을 입증한 논문으로 평가하기는 어렵습니다**. Mathematical analysis for an au… Mathematical analysis for an au… Mathematical analysis for an au…

**향후 연구의 가장 중요한 전환은 ‘새로운 연산자를 더 적용하기’에서 ‘초기화가 정당하고, 데이터로 검증되며, 새로운 상황에서도 유지되는 기억 모형을 만들기’로 이동하는 것입니다.**

---

## 참고문헌 및 확인한 출처

1. **Yusuf, A., Qureshi, S., & Shah, S. F. (2020).** *Mathematical analysis for an autonomous financial dynamical system via classical and modern fractional operators.* **Chaos, Solitons & Fractals, 132**, 109552. DOI: `10.1016/j.chaos.2019.109552`.  
   **확인 출처:** 사용자가 첨부한 Elsevier 논문 PDF 전체, pp. 1–12. Mathematical analysis for an au… Mathematical analysis for an au…

2. **Diethelm, K., Garrappa, R., Giusti, A., & Stynes, M. (2020).** *Why Fractional Derivatives with Nonsingular Kernels Should Not Be Used.* **Fractional Calculus and Applied Analysis, 23**, 610–634. DOI: `10.1515/fca-2020-0032`.  
   **확인 출처:** arXiv 공개 전문 `2006.15237`, 특히 §3의 초기값 제약 분석. [arXiv](https://arxiv.org/html/2006.15237v1)

3. **Fernandez, A., & Al-Refai, M. (2023).** *A Rigorous Analysis of Integro-Differential Operators with Non-Singular Kernels.* **Fractal and Fractional, 7(3)**, 213. DOI: `10.3390/fractalfract7030213`.  
   **확인 출처:** Eastern Mediterranean University 기관 저장소의 서지정보와 초록. 이 답변에서는 확인하지 못한 전문의 개별 정리를 인용하지 않았습니다. [i-rep.emu.edu.tr](https://i-rep.emu.edu.tr/items/f5e18756-2025-4210-afd9-2259c161f05d)

4. **An, X., Wang, Q., Liu, F., Anh, V. V., & Turner, I. W. (2024; 온라인 2023).** *Parameter estimation for time-fractional Black-Scholes equation with S &P 500 index option.* **Numerical Algorithms, 95**, 1–30. DOI: `10.1007/s11075-023-01563-4`.  
   **확인 출처:** Springer Nature 전문 및 PDF, 특히 §6와 Fig. 8. [Springer](https://link.springer.com/article/10.1007/s11075-023-01563-4)
   
   [Download PDF](https://libkey.io/libraries/4091/articles/576259766/full-text-file?utm_source=nomad)

5. **Zimmering, B., Coelho, C., & Niggemann, O. (2024).** *Optimising Neural Fractional Differential Equations for Performance and Efficiency.* **Proceedings of Machine Learning Research, 255**, 1–22.  
   **확인 출처:** PMLR 출판사 페이지와 초록. 속도·정확도 배수는 해당 초록의 보고이며, 세부 표를 재분석한 값이 아닙니다. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v255/zimmering24a.html)

6. **Coelho, C., Costa, M. F. P., & Ferrás, L. L. (2025).** *Neural Fractional Differential Equations.* **Applied Mathematical Modelling, 144**, 116060. DOI: `10.1016/j.apm.2025.116060`.  
   **확인 출처:** 저널 서지정보와 arXiv 공개본 `2403.02737v2`〔2024〕. 본문에 제시한 실험 설정과 Table 8의 수치는 공개본 v2를 기준으로 했습니다. [arXiv](https://arxiv.org/abs/2403.02737)

7. **Waheed, S., Qayyum, M., Khan, O., & Chambashi, G. (2026).** *Data-driven neural modeling and chaos control in fractional-order financial dynamical systems.* **AIP Advances, 16**, 015323. DOI: `10.1063/5.0311667`.  
   **확인 출처:** AIP Publishing 전문, 특히 §III와 Tables I–II. [AIP Publishing](https://pubs.aip.org/aip/adv/article/16/1/015323/3378142/Data-driven-neural-modeling-and-chaos-control-in)
