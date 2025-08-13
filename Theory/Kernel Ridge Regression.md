# Kernel Ridge Regression: 자세한 이해와 수식 설명

Kernel Ridge Regression(커널 릿지 회귀)은 전통적인 ridge regression과 kernel trick을 결합한 강력한 머신러닝 기법입니다. 비선형 데이터를 효과적으로 처리할 수 있으면서도 overfitting을 방지하는 regularization 효과를 동시에 얻을 수 있습니다[1][2].

## Ridge Regression의 기본 개념

Ridge regression은 일반적인 선형 회귀에 regularization(규제화) 항을 추가한 방법입니다[3][4].

### 일반 선형 회귀의 문제점

일반적인 최소제곱법(Ordinary Least Squares, OLS)은 다음 비용 함수를 최소화합니다[1]:

$$ C(w) = \frac{1}{2} \sum_i (y_i - w^T x_i)^2 $$

여기서:
- $$y_i$$: 실제 목표값
- $$w$$: 가중치 벡터  
- $$x_i$$: 입력 특성 벡터

하지만 이 방법은 **overfitting**(과적합)습니다. 특히 특성의 개수가 데이터 개수보다 많거나, 특성들 간에 높은 상관관계(multicollinearity)가 있을 때 문제가 심각해집니다[5][6].

### Ridge Regression의 해결책

Ridge regression은 비용 함수에 **regularization term**(정규화 항)을 추가하여 이 문제를 해결합니다[1][4]:

$$ C = \frac{1}{2} \sum_i (y_i - w^T x_i)^2 + \frac{\lambda}{2} ||w||^2 $$

여기서 $$\lambda$$는 **regularization parameter**(정규화 매개변수)로, 정규화의 강도를 조절합니다[7][8].

### Lambda(λ)의 역할

- **$$\lambda = 0$$**: 일반적인 선형 회귀와 동일
- **$$\lambda$$ 증가**: 가중치들이 0에 가까워지면서 모델이 단순해짐
- **$$\lambda$$ 매우 큰 값**: 가중치들이 거의 0에 가까워져 모델이 과도하게 단순해질 수 있음[9][10][11]

이는 **bias-variance tradeoff**(편향-분산 트레이드오프)의 원리에 기반합니다. λ가 증가하면 bias(편향)는 증가하지만 variance(분산)는 감소합니다[5][4].

## Kernel Trick의 개념

### 비선형 문제의 해결

많은 실제 데이터는 선형으로 분리되지 않습니다. 이를 해결하기 위해 **kernel trick**(커널 트릭)을 사용합니다[12][13][14].

Kernel trick의 핵심 아이디어는 다음과 같습니다[15][16]:

1. **원본 공간**에서 **고차원 특성 공간**으로 데이터를 암시적으로 매핑
2. 고차원 공간에서는 선형 분리가 가능해짐
3. 실제로는 고차원 공간으로 변환하지 않고, **커널 함수**를 통해 내적(inner product)만 계산

### 수학적 표현

특성 매핑 함수를 $$\Phi(x)$$라 하면, 커널 함수는[13][14]:

$$ K(x, z) = \Phi(x)^T \Phi(z) $$

즉, 두 데이터 포인트의 고차원 공간에서의 내적을 원본 공간에서 직접 계산할 수 있습니다.

### 주요 커널 함수들

**1. 선형 커널 (Linear Kernel)**[17]:

$$ K(x, z) = x^T z $$

**2. 다항식 커널 (Polynomial Kernel)**[17]:

$$ K(x, z) = (x^T z + c)^d $$

**3. RBF(Gaussian) 커널**[17]:

$$ K(x, z) = \exp(-\gamma ||x-z||^2) $$

## Kernel Ridge Regression의 수학적 도출

### 1단계: 특성 공간으로의 확장

원본 데이터 $$x_i$$를 특성 벡터 $$\Phi(x_i)$$로 매핑합니다[1]:

$$ x_i \rightarrow \Phi_i = \Phi(x_i) $$

### 2단계: 특성 공간에서의 Ridge Regression

특성 공간에서 ridge regression 문제는[1]:

$$ w = (\lambda I + \Phi\Phi^T)^{-1}\Phi y $$

여기서 $$\Phi$$는 모든 특성 벡터들을 행으로 가지는 행렬입니다.

### 3단계: 차원 축소 트릭

핵심적인 수학적 항등식을 사용합니다[1]:

$$ (P^{-1} + B^T R^{-1}B)^{-1}B^T R^{-1} = PB^T(BPB^T + R)^{-1} $$

이를 적용하면:

$$ w = \Phi(\Phi^T\Phi + \lambda I_n)^{-1}y $$

### 4단계: 듀얼 표현 (Dual Representation)

가중치 벡터는 훈련 데이터의 선형 결합으로 표현됩니다[1]:

$$ w = \sum_i \alpha_i \Phi(x_i) $$

여기서 $$\alpha = (\Phi^T\Phi + \lambda I_n)^{-1}y$$입니다.

### 5단계: 예측 함수

새로운 테스트 포인트 $$x$$에 대한 예측은[1]:

$$ y = w^T\Phi(x) = y^T(K + \lambda I_n)^{-1}\kappa(x) $$

여기서:
- $$K_{ij} = K(x_i, x_j) = \Phi(x_i)^T\Phi(x_j)$$: **Gram matrix**
- $$\kappa(x) = [K(x_1, x), K(x_2, x), ..., K(x_n, x)]^T$$

## Reproducing Kernel Hilbert Space (RKHS)

Kernel Ridge Regression의 이론적 기반은 **RKHS**(재생 커널 힐베르트 공간)입니다[18][19].

### RKHS의 정의

RKHS는 다음 **재생 성질**(reproducing property)을 만족하는 함수 공간입니다[18][20]:

$$ \langle f, K_x \rangle_H = f(x) $$

여기서 $$K_x(\cdot) = K(x, \cdot)$$는 reproducing kernel입니다.

### Representer Theorem

RKHS에서의 최적화 문제의 해는 항상 다음 형태로 표현됩니다[21]:

$$ f^* = \sum_{i=1}^n \alpha_i K(x_i, \cdot) $$

이것이 kernel ridge regression에서 가중치가 훈련 데이터의 선형 결합으로 표현되는 이론적 근거입니다[1].

## Kernel Ridge Regression의 장단점

### 장점

1. **비선형 관계 모델링**: Kernel trick을 통해 복잡한 비선형 패턴을 포착할 수 있습니다[22][2]
2. **Overfitting 방지**: Regularization을 통해 과적합을 효과적으로 방지합니다[3][23]
3. **닫힌 형태 해**: SVM과 달리 닫힌 형태의 해를 가지므로 중간 크기 데이터셋에서 빠른 학습이 가능합니다[2]
4. **수치적 안정성**: Regularization이 역행렬 계산을 안정화시킵니다[1]

### 단점

1. **희소성 부족**: SVM과 달리 support vector 개념이 없어 모든 훈련 데이터가 예측에 기여합니다[1]
2. **메모리 사용량**: Gram matrix $$K$$를 저장해야 하므로 $$O(n^2)$$ 메모리가 필요합니다[2]
3. **하이퍼파라미터 선택**: $$\lambda$$와 커널 매개변수들을 적절히 선택해야 합니다[8]

## 실제 구현 시 고려사항

### 하이퍼파라미터 선택

1. **교차 검증**: $$\lambda$$ 값은 주로 cross-validation을 통해 선택합니다[1][8]
2. **그리드 서치**: 여러 $$\lambda$$ 값을 시도해보며 최적값을 찾습니다[8]
3. **일반적 범위**: 보통 0.0001, 0.001, 0.01, 0.1, 1, 10 등의 값들을 시도합니다[11]

### 계산 복잡도

- **학습 시간**: $$O(n^3)$$ (Gram matrix 역행렬 계산)
- **예측 시간**: $$O(n)$$ (모든 훈련 데이터와의 커널 계산)
- **메모리**: $$O(n^2)$$ (Gram matrix 저장)

## 결론

Kernel Ridge Regression은 전통적인 ridge regression의 regularization 효과와 kernel trick의 비선형 모델링 능력을 결합한 강력한 방법입니다. 수학적으로 우아하고 이론적 기반이 탄탄하며, 중간 규모의 비선형 회귀 문제에서 매우 효과적입니다. 다만 대용량 데이터에서는 계산 비용과 메모리 사용량이 문제가 될 수 있으므로, 데이터 크기와 문제의 특성을 고려하여 사용해야 합니다[24][2].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/fe9234b3-e550-4293-b59d-f21f17c47b85/welling-notes-on-kernel-ridge.pdf
[2] https://scikit-learn.org/stable/modules/kernel_ridge.html
[3] https://www.ibm.com/think/topics/ridge-regression
[4] https://pmc.ncbi.nlm.nih.gov/articles/PMC9410599/
[5] https://www.geeksforgeeks.org/machine-learning/what-is-ridge-regression/
[6] https://corporatefinanceinstitute.com/resources/data-science/ridge/
[7] https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/lambda.html
[8] https://stackoverflow.com/questions/12182063/how-to-calculate-the-regularization-parameter-in-linear-regression
[9] https://www.youtube.com/watch?v=Q81RR3yKn30
[10] https://developers.google.com/machine-learning/crash-course/overfitting/regularization
[11] https://community.deeplearning.ai/t/large-value-of-lambda-in-regularization/180194
[12] https://www.geeksforgeeks.org/machine-learning/kernel-trick-in-support-vector-classification/
[13] https://en.wikipedia.org/wiki/Kernel_method
[14] https://bommbom.tistory.com/entry/%EC%BB%A4%EB%84%90-%ED%95%A8%EC%88%98%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0
[15] https://blog.damavis.com/en/the-kernel-trick-in-machine-learning/
[16] https://yololifestudy.tistory.com/entry/%EC%BB%A4%EB%84%90-%EB%B0%A9%EB%B2%95-%EC%9D%B4%EB%A1%A0kernel-method-kernel-trick
[17] https://direction-f.tistory.com/89
[18] https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space
[19] https://people.eecs.berkeley.edu/~bartlett/courses/281b-sp08/7.pdf
[20] https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf
[21] https://mlweb.loria.fr/book/en/kernelridgeregression.html
[22] https://www.geeksforgeeks.org/machine-learning/understanding-kernel-ridge-regression-with-sklearn/
[23] https://www.geeksforgeeks.org/machine-learning/overfitting-and-regularization-in-ml/
[24] https://www.nature.com/articles/s41524-018-0096-5
[25] https://link.springer.com/10.1007/s10994-022-06165-0
[26] https://arxiv.org/abs/2405.09362
[27] https://link.springer.com/10.1007/s41365-024-01379-4
[28] https://link.aps.org/doi/10.1103/PhysRevC.109.024310
[29] https://arxiv.org/abs/2403.08938
[30] https://dl.acm.org/doi/10.1145/3580305.3599398
[31] https://ieeexplore.ieee.org/document/10405861/
[32] https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=statistics-kernel-ridge-regression
[33] https://www.reddit.com/r/MachineLearning/comments/1joh9v/can_someone_explain_kernel_trick_intuitively/
[34] https://web2.qatar.cmu.edu/~gdicaro/10315-Fall19/additional/welling-notes-on-kernel-ridge.pdf
[35] https://sanghyu.tistory.com/13
[36] https://www.youtube.com/watch?v=Q7vT0--5VII
[37] https://dailyheumsi.tistory.com/57
[38] https://sanghyu.tistory.com/14
[39] https://iajit.org/upload/files/Semi-Supervised-Kernel-Discriminative-Low-Rank-Ridge-Regression-for-Data-Classification.pdf
[40] https://iopscience.iop.org/article/10.1088/2632-2153/ad40fc
[41] https://ieeexplore.ieee.org/document/8462186/
[42] https://osf.io/75s8v
[43] https://ieeexplore.ieee.org/document/10084201/
[44] https://ieeexplore.ieee.org/document/10743968/
[45] https://www.jenrs.com/v03/i10/p004/
[46] https://www.semanticscholar.org/paper/171de320c6b3a0a276bddf9af4a536e74497a4fa
[47] https://ko.eitca.org/artificial-intelligence/eitc-ai-mlp-machine-learning-with-python/support-vector-machine/soft-margin-svm-and-kernels-with-cvxopt/examination-review-soft-margin-svm-and-kernels-with-cvxopt/can-you-explain-the-concept-of-the-kernel-trick-and-how-it-enables-svm-to-handle-complex-data/
[48] https://www.youtube.com/watch?v=tz9XwX_mfVE
[49] https://svivek.com/teaching/lectures/slides/svm/kernels.pdf
[50] https://happydias1.tistory.com/entry/6%EB%B2%88%EC%A7%B8-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5
[51] https://lee-jaejoon.github.io/stat-rkhs/
[52] https://thebook.io/080223/0156/
[53] https://nzer0.github.io/reproducing-kernel-hilbert-space.html
[54] https://arxiv.org/abs/1408.0952
[55] https://direction-f.tistory.com/88
[56] https://www.frontiersin.org/articles/10.3389/fgene.2020.581594/full
[57] https://www.semanticscholar.org/paper/afeef60e3e2255ac6560424b919ad425ebe3e2e3
[58] https://www.semanticscholar.org/paper/71cb3563cb92fa837d1775e8fb0e8235bcb283d4
[59] https://onlinelibrary.wiley.com/doi/10.1002/sta4.540
[60] https://link.springer.com/10.1007/978-3-030-77094-5_12
[61] https://ijmge.ut.ac.ir/article_81246.html
[62] https://dx.plos.org/10.1371/journal.pone.0302221
[63] https://ieeexplore.ieee.org/document/10715451/
[64] https://www.statlect.com/fundamentals-of-statistics/ridge-regression
[65] https://daeson.tistory.com/entry/18-%EB%AA%A8%EB%8D%B8%EC%9D%98-%EA%B3%BC%EC%B5%9C%EC%A0%81%ED%99%94%EB%A5%BC-%ED%94%BC%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95-overfitting-regularization
[66] https://velog.io/@jiyeah3108/AI-Overfitting-Regularization
[67] https://stats.stackexchange.com/questions/508984/why-regularization-parameter-called-as-lambda-in-theory-and-alpha-in-python
[68] https://en.wikipedia.org/wiki/Ridge_regression
[69] https://gm-note.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Overfitting-Regularization-with-Polynomial-function
[70] https://kimlog.me/machine-learning/2016-01-30-4-regularization/
[71] https://www.publichealth.columbia.edu/research/population-health-methods/ridge-regression
[72] https://velog.io/@som_3/5.-Regularization-The-problem-of-overfitting
[73] https://arxiv.org/abs/2401.01270
[74] https://ieeexplore.ieee.org/document/10793171/
[75] http://arxiv.org/pdf/2403.09907.pdf
[76] https://arxiv.org/abs/1305.5029
[77] https://arxiv.org/pdf/2102.00760.pdf
[78] https://arxiv.org/abs/2502.11665
[79] https://arxiv.org/abs/1501.03854v2
[80] http://arxiv.org/pdf/2306.16838.pdf
[81] https://arxiv.org/pdf/1611.03220.pdf
[82] http://arxiv.org/pdf/2405.07791.pdf
[83] http://arxiv.org/pdf/2410.02680.pdf
[84] https://arxiv.org/abs/2301.07172
[85] https://sikmulation.tistory.com/13
[86] https://teazrq.github.io/SMLR/kernel-ridge-regression.html
[87] https://velog.io/@claude_ssim/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5-Linear-Regression-Ridge-Regression
[88] https://link.springer.com/10.1007/978-3-030-89010-0_8
[89] https://dl.acm.org/doi/10.1145/3552326.3567486
[90] https://arxiv.org/pdf/2307.04112.pdf
[91] http://arxiv.org/pdf/2007.11643.pdf
[92] http://arxiv.org/pdf/1609.06473.pdf
[93] http://arxiv.org/pdf/2401.17035.pdf
[94] http://arxiv.org/pdf/2410.12635.pdf
[95] https://arxiv.org/pdf/2106.08443.pdf
[96] https://arxiv.org/pdf/1808.04475.pdf
[97] https://arxiv.org/pdf/2502.08470.pdf
[98] https://arxiv.org/pdf/2009.04614.pdf
[99] https://arxiv.org/pdf/1812.03155.pdf
[100] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13219/3036681/Implementation-of-linear-regression-lasso-ridge-regression-and-kernel-trick/10.1117/12.3036681.full
[101] https://arxiv.org/abs/1509.09169
[102] https://arxiv.org/abs/2210.08571
[103] https://comptes-rendus.academie-sciences.fr/mathematique/item/10.5802/crmath.367.pdf
[104] https://arxiv.org/pdf/1910.02373.pdf
[105] https://arxiv.org/pdf/2006.05800.pdf
[106] http://arxiv.org/pdf/2203.08564.pdf
