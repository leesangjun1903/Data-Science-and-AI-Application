# Reproducing Kernel Hilbert Space와 주요 정리들: 수식과 개념 설명

본 논문에서 제시된 재생핵 힐베르트 공간(Reproducing Kernel Hilbert Space, RKHS)의 핵심 정리들을 수식과 함께 자세히 설명드리겠습니다.

## RKHS의 정의와 기본 개념

### RKHS란 무엇인가?

RKHS는 **모든 점 평가(point evaluation) 함수가 유계 선형 함수**인 힐베르트 공간입니다[1]. 이는 일반적인 $$L^2$$ 공간과는 다른 중요한 특성입니다[2].

정식으로 표현하면, 어떤 영역 $$T$$에서 정의된 함수들의 힐베르트 공간 $$H$$에 대해, 모든 $$t \in T$$에 대하여 다음을 만족하는 원소 $$\eta_t \in H$$가 존재합니다[1]:

$$ f(t) = \langle \eta_t, f \rangle, \quad \forall f \in H $$

여기서 $$\langle \cdot, \cdot \rangle$$는서의 내적입니다. 이때 $$\eta_t$$를 **점 $$t$$에서의 평가 표현자(representer of evaluation)**라고 합니다[2].

### 재생핵의 정의

$$\langle \eta_s, \eta_t \rangle = K(s,t)$$로 정의하면, $$K(s,t)$$는 $$T \times T$$에서 **양정부호(positive definite)** 함수가 됩니다[1]. 이 함수를 **재생핵(reproducing kernel)**이라 하며, 다음 조건을 만족합니다[3][4]:

1. **대칭성**: $$K(s,t) = K(t,s)$$
2. **양정부호성**: 임의의 $$t_1, \ldots, t_n \in T$$와 $$a_1, \ldots, a_n \in \mathbb{R}$$에 대해

$$ \sum_{i,j} a_i a_j K(t_i, t_j) \geq 0 $$

$$\eta_t = K(t, \cdot)$$이므로, 다음의 핵심적인 **재생 성질(reproducing property)**을 얻습니다[1][2]:

$$ \langle K(t, \cdot), K(s, \cdot) \rangle = K(s,t) $$

이것이 "재생핵"이라는 이름의 유래입니다.

## Moore-Aronszajn 정리

이 정리는 RKHS 이론의 핵심으로, **양정부호 핵과 RKHS 사이의 일대일 대응**을 보장합니다[1][5][6].

### 정리 내용

**정리**: 임의의 집합 $$X$$에 대하여, 함수 $$K: X \times X \rightarrow \mathbb{R}$$가 양정부호 핵인 것과 어떤 RKHS의 재생핵인 것은 동치이며, 이러한 대응은 유일합니다[6].

### 구성 방법

양정부호 핵 $$K$$가 주어졌을 때, 대응하는 RKHS는 다음과 같이 구성됩니다[6][7]:

1. **초기 공간 구성**: 
   $$H_0 = \text{span}\{K(\cdot, x) : x \in X\} $$

2. **내적 정의**: $$f = \sum_{i=1}^n a_i K(\cdot, x_i)$$, $$g = \sum_{j=1}^m b_j K(\cdot, y_j)$$에 대해

$$ \langle f, g \rangle = \sum_{i=1}^n \sum_{j=1}^m a_i b_j K(x_i, y_j) $$

3. **완비화**: RKHS $$H$$는 $$H_0$$의 완비화입니다.

이 구성에서 중요한 점은 **노름 수렴이 점별 수렴을 함의**한다는 것입니다[1]:

$$ |f_n(t) - f_m(t)| = |\langle K(t, \cdot), f_n - f_m \rangle| \leq \sqrt{K(t,t)} \|f_n - f_m\| $$

## 표현자 정리 (Representer Theorem)

표현자 정리는 RKHS에서의 최적화 문제의 해가 항상 **유한 차원 부공간**에 존재함을 보장하는 핵심 결과입니다[1][8][9].

### 기본 형태

다음 최적화 문제를 고려합니다[1]:

```math
\min_{f \in H} \left\{ \sum_{i=1}^n C(y_i, f(t_i)) + \lambda \|f\|^2 \right\}
```

여기서 $$C$$는 볼록 함수이고, $$\lambda > 0$$는 정규화 매개변수입니다.

**정리**: 위 문제의 해 $$f_\lambda$$는 다음과 같이 표현됩니다[1]:

$$ f_\lambda(\cdot) = \sum_{i=1}^n c_i K(t_i, \cdot) $$

### 증명의 핵심 아이디어

임의의 $$f \in H$$를 다음과 같이 분해할 수 있습니다[10]:

$$ f = \sum_{i=1}^n c_i K(t_i, \cdot) + \rho $$

여기서 $$\rho$$는 $$\text{span}\{K(t_1, \cdot), \ldots, K(t_n, \cdot)\}$$에 직교합니다. 재생 성질에 의해 $$f(t_i) = \sum_{j=1}^n c_j K(t_j, t_i)$$이므로, $$\rho$$는 목적함수의 첫 번째 항에 영향을 주지 않습니다. 따라서 $$\rho = 0$$으로 설정할 수 있습니다[10].

### 일반화된 형태

관측이 유계 선형 함수 $$L_i$$를 통해 이루어지는 경우[1]:

$$ y_i = L_i f + \epsilon_i $$

이때 해는 다음과 같이 표현됩니다:

$$ f_\lambda(\cdot) = \sum_{i=1}^n c_i \eta_i(\cdot) $$

여기서 $$\eta_i$$는 $$L_i$$의 표현자입니다[1].

## 가우시안 과정과의 관계

RKHS와 가우시안 과정 사이에는 밀접한 관계가 있습니다[1][11][12]. 

모든 양정부호 핵 $$K(\cdot, \cdot)$$에 대해, $$K$$를 공분산 함수로 하는 평균이 0인 가우시안 과정이 존재합니다[1]. 이를 통해 베이즈 추정, 가우시안 과정, RKHS에서의 최적화 문제 사이의 연결고리가 형성됩니다[11].

## 편향-분산 트레이드오프

RKHS에서 정규화 매개변수 $$\lambda$$는 **편향-분산 트레이드오프**를 제어합니다[1]:

```math
\min_{f \in H} \left\{ \sum_{i=1}^n C(y_i, f(t_i)) + \lambda \|f\|^2 \right\}
```

- **$$\lambda$$ 증가**: 더 매끄러운 함수, 높은 편향, 낮은 분산
- **$$\lambda$$ 감소**: 더 복잡한 함수, 낮은 편향, 높은 분산

이 매개변수는 **일반화 교차검증(Generalized Cross Validation, GCV)**이나 **일반화 최대우도(Generalized Maximum Likelihood, GML)** 등의 방법으로 선택할 수 있습니다[1][13].

## 스플라인과의 연결

다양한 스플라인들이 RKHS의 특별한 경우로 이해될 수 있습니다[1][14]:

### 다항 평활 스플라인
단위 구간에서 다음 벌칙 함수를 가집니다[1]:

$$ J(f) = \int_0^1 [f^{(m)}(u)]^2 du $$

### 얇은 판 스플라인
평면에서 다음 벌칙 함수를 사용합니다[1]:

$$ J(f) = \iint \left[ \left(\frac{\partial^2 f}{\partial x_1^2}\right)^2 + 2\left(\frac{\partial^2 f}{\partial x_1 \partial x_2}\right)^2 + \left(\frac{\partial^2 f}{\partial x_2^2}\right)^2 \right] dx_1 dx_2 $$

## Mercer 정리와 방사형 기저 함수

$$\int_T \int_T K^2(s,t) ds dt < \infty$$ 이면, $$K$$는 가산개의 고유값과 고유함수를 가집니다[1]. 하지만 이 조건이 항상 성립하지는 않습니다. 예를 들어, 가우시안 핵 $$K(s,t) = e^{-\|s-t\|^2}$$은 무한 실직선에서 이 조건을 만족하지 않지만 여전히 유용한 재생핵입니다[1].

방사형 기저 함수들은 $$\|s-t\|$$에만 의존하는 함수들로, 양정부호인 경우 Micchelli(1986)에 의해 특성화되었습니다[1].

## 베이즈 신뢰구간

Wahba(1983)는 RKHS에서 베이즈 "신뢰구간"을 제안했습니다[1]. 이는 점별 신뢰구간이 아니라 **함수 전체에 걸친** 성질을 가집니다. 즉, $$n$$개의 95% 신뢰구간 중 평균적으로 약 95%가 참값을 포함합니다[1][15].

## 실제 응용

RKHS 기반 방법들은 다음과 같은 다양한 비용 함수와 함께 사용됩니다[1]:

- **회귀**: $$C(y,f) = (y-f)^2$$
- **베르누이 데이터**: $$C(y,f) = -yf + \log(1+e^f)$$
- **서포트 벡터 머신**: $$C(y,f) = (1-yf)_+$$
- **로버스트 함수**: 이상치에 강건한 함수들

이러한 이론적 기반들이 현대 기계학습에서 커널 방법, 서포트 벡터 머신, 가우시안 과정 등의 강력한 도구들을 가능하게 만들었습니다[1][16].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c4d3fc82-241d-43eb-9974-c88b608a3fc9/1-s2.0-S1474667017348152-main.pdf
[2] https://lee-jaejoon.github.io/stat-rkhs/
[3] https://ko.wikipedia.org/wiki/%EC%9E%AC%EC%83%9D%ED%95%B5_%ED%9E%90%EB%B2%A0%EB%A5%B4%ED%8A%B8_%EA%B3%B5%EA%B0%84
[4] https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space
[5] https://stats.stackexchange.com/questions/576815/moore-aronszajn-theorem-and-mercer-theorem-for-the-kernel-trick
[6] https://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/notes/aronszajn.pdf
[7] https://www.ism.ac.jp/~fukumizu/H20_kernel/Kernel_2_elements.pdf
[8] http://papers.neurips.cc/paper/4841-the-representer-theorem-for-hilbert-spaces-a-necessary-and-sufficient-condition.pdf
[9] https://en.wikipedia.org/wiki/Representer_theorem
[10] https://pages.stat.wisc.edu/~wahba/ftp1/wahba.wang.2019submit.pdf
[11] https://ieeexplore.ieee.org/document/10121704/
[12] https://arxiv.org/abs/2506.17366
[13] https://ieeexplore.ieee.org/document/8814879/
[14] https://pages.stat.wisc.edu/~wahba/stat860public/bigpicture/wahba.wang.overview2015.pdf
[15] https://link.springer.com/10.1007/s40314-022-01790-w
[16] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/65b5dd75-dca4-4697-9527-946f5eaa0de7/sysid1.pdf
[17] https://www.tandfonline.com/doi/full/10.1080/01630563.2023.2221857
[18] https://www.mdpi.com/2073-8994/14/11/2227
[19] https://www.semanticscholar.org/paper/a3305c4ae89142b97dd42324eb18bba2e0c298a3
[20] https://iopscience.iop.org/article/10.1088/1402-4896/ac8958
[21] https://pubsonline.informs.org/doi/10.1287/opre.2020.2069
[22] https://www.semanticscholar.org/paper/cb9a4886e6880a82304ddbc969477e89dab61744
[23] https://elementary-physics.tistory.com/210
[24] https://enginius.tistory.com/587
[25] https://www.themoonlight.io/ko/review/distributed-learning-and-function-fusion-in-reproducing-kernel-hilbert-space
[26] https://www.themoonlight.io/ko/review/general-reproducing-properties-in-rkhs-with-application-to-derivative-and-integral-operators
[27] https://nzer0.github.io/reproducing-kernel-hilbert-space.html
[28] https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/RKHS2013_slides1.pdf
[29] https://www.themoonlight.io/ko/review/policy-newton-algorithm-in-reproducing-kernel-hilbert-space
[30] https://hihunjin.tistory.com/20
[31] https://arxiv.org/html/2210.11855v3
[32] https://arxiv.org/pdf/1901.01002.pdf
[33] http://arxiv.org/pdf/2504.06754.pdf
[34] http://arxiv.org/pdf/2412.11473.pdf
[35] http://arxiv.org/pdf/2009.02989.pdf
[36] http://arxiv.org/pdf/2107.11148.pdf
[37] http://arxiv.org/pdf/2312.01961.pdf
[38] https://arxiv.org/pdf/2106.08443.pdf
[39] https://www.youtube.com/watch?v=juTjfwTZ7XQ
[40] https://hgmin1159.github.io/dimension/nonlineardr/
[41] https://www.reddit.com/r/MachineLearning/comments/1xvwjs/why_do_kernel_functions_had_to_be_positive/
[42] https://velog.io/@_voirmer/Reproducing-Kernel-Hilbert-Space
[43] https://math.stackexchange.com/questions/2187587/is-positive-definite-in-context-of-kernels-and-inner-products-the-same-thing
[44] https://www.themoonlight.io/ko/review/integral-representation-of-translation-invariant-operators-on-reproducing-kernel-hilbert-spaces
[45] https://velog.io/@ddochi132/Maximum-Mean-Discrepancy
[46] https://en.wikipedia.org/wiki/Positive-definite_kernel
[47] https://minye-lee19.gitbook.io/sw-engineer/deep-learning/class/untitled
[48] https://lee-soohyun.tistory.com/61
[49] https://junstar92.github.io/mml-study-note/2022/07/07/ch3-2.html
[50] https://www.youtube.com/watch?v=lgvcyCPxNDQ
[51] https://www.semanticscholar.org/paper/be51e9141ae2af4daf3a1ba745ad3ff66a5990f3
[52] https://aircconline.com/mlaij/V11N4/11424mlaij01.pdf
[53] https://ieeexplore.ieee.org/document/10711948/
[54] https://onlinelibrary.wiley.com/doi/10.1111/cogs.13241
[55] https://linkinghub.elsevier.com/retrieve/pii/S0951833924001102
[56] https://link.springer.com/10.1007/s11340-022-00928-5
[57] https://linkinghub.elsevier.com/retrieve/pii/S2352012422009018
[58] https://arxiv.org/abs/2011.08651
[59] https://www.themoonlight.io/ko/review/a-gap-between-the-gaussian-rkhs-and-neural-networks-an-infinite-center-asymptotic-analysis
[60] https://hofe-rnd.tistory.com/entry/interpolation-Spline-method-1
[61] https://proceedings.mlr.press/v119/yang20j.html
[62] https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
[63] https://cdm98.tistory.com/27
[64] https://stats.stackexchange.com/questions/491501/is-the-idea-of-a-bias-variance-tradeoff-a-false-construct
[65] https://hihunjin.tistory.com/23
[66] https://swrush.tistory.com/221
[67] https://www.pnas.org/doi/10.1073/pnas.1903070116
[68] https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/splines_rkhs.pdf
[69] https://koppel.netlify.app/assets/papers/c_2019_koppel_b.pdf
[70] https://www.koreascience.kr/article/JAKO202210261460699.pdf
[71] https://www.sciencedirect.com/science/article/pii/S2211124721016855
[72] https://www.semanticscholar.org/paper/97847c5baccad5c50ca8a411c323f130ef53cdb1
[73] https://arxiv.org/abs/2402.04613
[74] https://arxiv.org/pdf/2401.01295.pdf
[75] https://arxiv.org/pdf/2011.14821.pdf
[76] http://arxiv.org/pdf/2412.18360.pdf
[77] http://arxiv.org/pdf/1602.05350.pdf
[78] http://arxiv.org/pdf/1601.07380.pdf
[79] http://arxiv.org/pdf/1602.00760.pdf
[80] https://arxiv.org/pdf/1707.08492.pdf
[81] https://www.mdpi.com/2504-3110/7/5/357/pdf?version=1683855575
[82] http://arxiv.org/pdf/2011.03360.pdf
[83] https://arxiv.org/pdf/1202.4443.pdf
[84] https://math.stackexchange.com/questions/855751/how-to-prove-the-semi-parametric-representer-theorem
[85] https://www.maxwellsci.com/announce/RJASET/5-507-509.pdf
[86] https://www.mdpi.com/2227-7390/3/3/615/pdf
[87] https://arxiv.org/pdf/0910.1013.pdf
[88] https://downloads.hindawi.com/journals/aaa/2012/514103.pdf
[89] https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D6C85487ABC9D29C3F79208BEED1F1B9/S0008414X24000488a.pdf/div-class-title-a-reproducing-kernel-approach-to-lebesgue-decomposition-div.pdf
[90] https://downloads.hindawi.com/journals/mpe/2015/518406.pdf
[91] http://downloads.hindawi.com/journals/aaa/2012/414612.pdf
[92] https://arxiv.org/pdf/2412.13901.pdf
[93] http://arxiv.org/pdf/1204.3573.pdf
[94] https://downloads.hindawi.com/journals/aaa/2013/959346.pdf
[95] https://www.tandfonline.com/doi/pdf/10.1080/25765299.2021.1891678?needAccess=true
[96] https://arxiv.org/abs/2308.02870
[97] https://www.semanticscholar.org/paper/83e815f7c4c6dd068ccec80d94c10a6dd21c0da9
[98] https://arxiv.org/pdf/2002.11328.pdf
[99] http://arxiv.org/pdf/2407.10418.pdf
[100] http://arxiv.org/pdf/1103.5538.pdf
[101] http://arxiv.org/pdf/2203.05443.pdf
[102] https://arxiv.org/pdf/1812.11118.pdf
[103] http://arxiv.org/pdf/1810.08591.pdf
[104] https://arxiv.org/pdf/2310.09250.pdf
[105] https://arxiv.org/abs/2006.00278
[106] https://arxiv.org/html/2405.15403
[107] https://arxiv.org/pdf/2302.04525.pdf
[108] https://www.themoonlight.io/ko/review/new-duality-in-choices-of-feature-spaces-via-kernel-analysis
