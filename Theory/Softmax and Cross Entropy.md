# Softmax와 Cross Entropy의 관계 및 손실함수 유도

## 핵심 요약  
Softmax 함수는 다중 클래스 분류에서 예측 확률 분포를 생성하고, Cross Entropy는 이 확률 분포와 실제 라벨 분포의 차이를 측정하는 손실함수이다.  
- **Binary class**: Sigmoid + Binary Cross Entropy  
- **Multiclass**: Softmax + Categorical Cross Entropy  

## 1. Binary Classification 손실함수 유도

### 1.1 Sigmoid 함수  
이진 분류에서 모델의 출력 $$z\in\mathbb{R}$$를 확률로 변환하는 함수:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

여기서 $$\sigma(z)\in(0,1)$$은 양성 클래스일 확률을 나타낸다.

### 1.2 Binary Cross Entropy (Log Loss)  
실제 레이블 $$y\in\{0,1\}$$, 예측 확률 $$\hat{y}=\sigma(z)$$에 대해 손실함수:

$$
L(y, \hat{y}) = -\bigl[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\bigr]
$$  

이 식은 Bernoulli 분포의 로그우도(log-likelihood)에 음수를 붙인 형태로 유도된다[1].

### 1.3 Gradient 유도  
파라미터 $$w$$에 대한 기울기를 구하기 위해 연쇄법칙(chain rule) 적용:

$$
\frac{\partial L}{\partial z}
= -\Bigl[y\frac{1}{\hat{y}} ( \hat{y}(1-\hat{y}) ) - (1-y)\frac{1}{1-\hat{y}}( \hat{y}(1-\hat{y}) )\Bigr]
= \hat{y} - y
$$

$$
\frac{\partial L}{\partial w}
= \frac{\partial L}{\partial z}\frac{\partial z}{\partial w}
= (\hat{y}-y)\,x
$$

즉, 이진 교차 엔트로피의 gradient는 **예측 확률–실제 레이블** 형태이다[2].

## 2. Multiclass Classification 손실함수 유도

### 2.1 Softmax 함수  
$$K$$개의 클래스 로짓 벡터 $$\mathbf{z}=(z_1,\dots,z_K)$$에 대해:

$$
\mathrm{softmax}(z)\_i
= \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}},\quad i=1,\dots,K
$$

출력 벡터 $$\mathbf{p}\in(0,1)^K$$, $$\sum_i p_i=1$$이다[3].

### 2.2 Categorical Cross Entropy  
실제 라벨을 one-hot 벡터 $$\mathbf{y}$$로 표현할 때 손실함수:

$$
L(\mathbf{y}, \mathbf{p})
= -\sum_{i=1}^K y_i \log(p_i)
$$

one-hot 특성상 $$y_k=1$$인 클래스 $$k$$만 활성화된다.

### 2.3 Gradient 유도  

$$
\frac{\partial L}{\partial z_k}
= \sum_{i=1}^K \frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_k}
= -\sum_{i=1}^K y_i\frac{1}{p_i}\,p_i(\delta_{ik}-p_k)
= p_k - y_k
$$

따라서 Softmax + Cross Entropy 결합 시 gradient도 **예측 확률–실제 레이블** 형태로 단순화된다[4][5].

## 3. 정리

|구분|활성화 함수|손실함수|기울기(예시)|  
|--|--|--|--|  
|Binary class|Sigmoid $$\sigma(z)$$|$$-[y\ln\hat y+(1-y)\ln(1-\hat y)]$$[6]| $$\hat y-y$$ [7]|  
|Multiclass|Softmax $$\frac{e^{z_i}}{\sum_j e^{z_j}}$$[3]|$$-\sum_i y_i\ln p_i$$[8]| $$p_k - y_k$$[9]|

- Binary: Sigmoid 변환 → Binary Cross Entropy 유도 → gradient = $$\hat{y}-y$$  
- Multiclass: Softmax 변환 → Categorical Cross Entropy 유도 → gradient = $$p_k - y_k$$  

이로써 Softmax와 Cross Entropy는 **확률 분포 생성**과 **분포 간 차이 측정**을 결합하여, 학습 시 직관적인 gradient 형태를 제공함을 알 수 있다.

[1] https://www.numberanalytics.com/blog/binary-cross-entropy-deep-dive-machine-learning
[2] https://www.python-unleashed.com/post/derivation-of-the-binary-cross-entropy-loss-gradient
[3] https://peterroelants.github.io/posts/cross-entropy-softmax/
[4] https://people.tamu.edu/~sji/classes/LR.pdf
[5] https://stackoverflow.com/questions/41663874/cs231n-how-to-calculate-gradient-for-softmax-loss-function
[6] https://www.sec.gov/Archives/edgar/data/1172178/000164117225015042/form10-q.htm
[7] https://www.sec.gov/Archives/edgar/data/2071486/000110465925056683/tm2517117-1_s1.htm
[8] https://www.sec.gov/Archives/edgar/data/846377/000147793225003912/totaligent_10q.htm
[9] https://www.sec.gov/Archives/edgar/data/846377/000147793225003416/totaligent_s1a.htm
[10] https://www.sec.gov/Archives/edgar/data/1172178/000164117225008127/form10-k.htm
[11] https://www.sec.gov/Archives/edgar/data/1754581/000141057825000706/futu-20241231x20f.htm
[12] https://www.semanticscholar.org/paper/3e9e37ef38a305618bb56fc6ad5d670ed00bf12b
[13] https://link.springer.com/10.1007/s10554-025-03369-y
[14] https://ieeexplore.ieee.org/document/10394113/
[15] https://journals.rta.lv/index.php/HET/article/view/8251
[16] https://ieeexplore.ieee.org/document/10649122/
[17] https://www.iieta.org/journals/ria/paper/10.18280/ria.380304
[18] https://discuss.pytorch.org/t/how-can-i-calculate-correct-softmax-gradient/212975
[19] https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function
[20] https://stats.stackexchange.com/questions/347254/deriving-binary-cross-entropy-loss-function
[21] https://www.youtube.com/watch?v=f-nW8cSa_Ec
[22] https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
[23] https://www.sec.gov/Archives/edgar/data/1172178/000149315224050159/form10-q.htm
[24] https://linkinghub.elsevier.com/retrieve/pii/S0952197623012447
[25] https://ieeexplore.ieee.org/document/10274084/
[26] https://ieeexplore.ieee.org/document/10317335/
[27] https://ieeexplore.ieee.org/document/10273491/

https://wikidocs.net/181819

https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
