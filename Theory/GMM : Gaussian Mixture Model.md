
# Gaussian Mixture Model, GMM 과 EM 알고리즘 가이드

## 1. 개요  
가우시안 혼합 모델(GMM)은 여러 개의 정규분포를 합쳐 데이터의 복잡한 분포를 근사하는 알고리즘입니다.  
딥러닝을 공부하는 대학생이 이해하기 쉽도록, 이론부터 구현 예시까지 차근차근 설명합니다.  

***

## 2. GMM의 기본 개념  
GMM은 데이터 $$\mathbf{x}$$가 $$K$$개의 가우시안 분포 중 하나에서 생성되었다고 가정합니다.  
각 분포는 평균 $$\boldsymbol{\mu}_k$$와 공분산 $$\Sigma_k$$를 가집니다.  
혼합비 $$\pi_k$$는 각 분포가 선택될 확률입니다.  
다음 식으로 모델을 정의합니다.  

$$
p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x};\boldsymbol{\mu}_k,\Sigma_k)
$$  

- $$\pi_k \ge 0$$, $$\sum_{k=1}^K \pi_k = 1$$  
- $$\mathcal{N}(\mathbf{x};\boldsymbol{\mu}_k,\Sigma_k)$$는 정규분포 밀도함수  

GMM을 학습한다는 것은 데이터 샘플들의 **로그-가능도**를 최대화하는 매개변수 $$\{\pi_k,\boldsymbol{\mu}_k,\Sigma_k\}$$를 찾는 과정입니다.

***

## 3. 최대 우도 추정과 한계  
로그-가능도는 다음과 같습니다.  

$$
L(\mathcal{X};\theta) = \sum_{n=1}^N \log \Bigl(\sum_{k=1}^K \pi_k\,\mathcal{N}(\mathbf{x}_n;\boldsymbol{\mu}_k,\Sigma_k)\Bigr)
$$

이 함수를 직접 미분하여 해를 구하기에는 로그 안에 합이 있어서 해석적 해가 나오지 않습니다.  
따라서 다른 최적화 기법이 필요합니다.

***

## 4. EM 알고리즘  
EM 알고리즘은 **Expectation(기댓값)** 단계와 **Maximization(최대화)** 단계를 번갈아 수행하며 로그-가능도의 하한을 최대화합니다.

### 4.1 E-step  
현재 매개변수를 고정하고, 잠재 변수 $$z_n$$의 분포  

$$\gamma_k(\mathbf{x}_n) = q(z_n=k|\mathbf{x}_n)$$를 계산합니다.

$$
\gamma_k(\mathbf{x}_n)
= \frac{\pi_k \mathcal{N}(\mathbf{x}_n;\boldsymbol{\mu}_k,\Sigma_k)}
       {\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n;\boldsymbol{\mu}_j,\Sigma_j)}
$$

### 4.2 M-step  
E-step에서 계산한 $$\gamma_k(\mathbf{x}_n)$$를 이용해 매개변수를 갱신합니다.

1) 평균 $$\boldsymbol{\mu}_k$$ 업데이트  

$$
\boldsymbol{\mu}_k
= \frac{\sum_{n=1}^N \gamma_k(\mathbf{x}_n)\,\mathbf{x}_n}
       {\sum_{n=1}^N \gamma_k(\mathbf{x}_n)}
$$

2) 공분산 $$\Sigma_k$$ 업데이트  

$$
\Sigma_k
= \frac{\sum_{n=1}^N \gamma_k(\mathbf{x}_n)(\mathbf{x}_n-\boldsymbol{\mu}_k)(\mathbf{x}_n-\boldsymbol{\mu}_k)^T}
       {\sum_{n=1}^N \gamma_k(\mathbf{x}_n)}
$$

3) 혼합비 $$\pi_k$$ 업데이트  

$$
\pi_k = \frac{1}{N}\sum_{n=1}^N \gamma_k(\mathbf{x}_n)
$$

이 과정을 반복하여 로그-가능도가 수렴할 때까지 학습합니다.

***

## 5. 구현 예시 (Python + scikit-learn)  
```python
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 샘플 데이터 생성
np.random.seed(0)
X1 = np.random.randn(100,2) + np.array([0,0])
X2 = np.random.randn(100,2) + np.array([5,5])
X = np.vstack([X1,X2])

# GMM 모델 학습
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
gmm.fit(X)

# 예측 및 시각화
labels = gmm.predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=20)
plt.title('GMM Clustering 결과')
plt.show()
```

- `n_components`: 가우시안 분포의 개수 $$K$$  
- `covariance_type`: 공분산 행렬 형태 (`full`, `diag` 등)  

***

## 6. 분류에의 응용  
학습된 GMM은 각 입력 $$\mathbf{x}$$에 대해 어느 분포에서 나왔을 확률 $$\gamma_k(\mathbf{x})$$를 계산합니다.  
최댓값을 가지는 $$k$$를 클래스 레이블로 사용하면 분류 문제에 응용할 수 있습니다.

$$
y = \arg\max_{k}\;\gamma_k(\mathbf{x})
$$

***

## 7. 결론  
가우시안 혼합 모델은 **복잡한 데이터 분포**를 정규분포의 합으로 표현합니다.  
EM 알고리즘을 통해 **효율적으로** 매개변수를 학습할 수 있습니다.  
실습을 통해 직접 모델을 구성해 보세요. 이해가 더욱 깊어집니다.

[1](https://untitledtblog.tistory.com/133)

# Reference
https://untitledtblog.tistory.com/133
