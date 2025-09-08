# Likelihood

다음 글은 Likelihood, MLE, Log-Likelihood, Negative Log-Likelihood를 딥러닝 관점에서 한 번에 이해하고, 수식과 함께 바로 실습할 수 있도록 구성한 글입니다. 핵심은 “확률모형을 세우고, 관측 데이터를 가장 그럴듯하게 만드는 파라미터를 찾는 과정”입니다.[1][2][3]

## 한눈에 핵심
- **Likelihood**는 관측 데이터가 주어졌을 때, 파라미터가 데이터를 만들어냈을 “그럴싸함”을 나타내는 **함수**입니다.[3][1]
- **MLE**는 Likelihood를 **최대화**하는 파라미터 추정 방법입니다.[2][1]
- **Log-Likelihood**는 Likelihood에 로그를 취해 곱을 합으로 바꿔 계산을 단순화합니다.[2][3]
- **Negative Log-Likelihood(NLL)** 을 **최소화**하는 것은 MLE와 동치이며, 분류 문제에서는 **교차엔트로피**와 사실상 같은 목적을 가집니다.[4][5][6]

## Likelihood란?
Likelihood $L(θ)$ 는 “분포 파라미터 θ가 주어졌을 때 데이터를 관측할 가능도”를 데이터의 함수가 아니라 파라미터의 함수로 본 값입니다.[1][3]

독립동일분포(i.i.d.) 표본 x1,…,xn과 모형 밀도 $f(x|θ)$ 에 대해 $$L(\theta)=\prod_{i=1}^{n} f(x_i\mid \theta)$$ 로 정의합니다 [2][3].  

연속형/이산형 모두 같은 아이디어이며, 연속형에서 확률은 면적이지만 Likelihood는 “밀도값의 곱”으로 표현됩니다.[3][2]

### i.i.d.
i.i.d.는 "independent and identically distributed"의 약자로, 서로 독립적이고 동일한 확률분포를 따르는 랜덤 변수들의 집합을 의미합니다.
즉, 각 변수가 같은 분포에서 무작위로 추출되며 서로 영향을 주지 않는다는 뜻입니다.

더 구체적으로,

- Identically distributed (동일 분포): 모든 변수들이 같은 확률 분포를 가지고 있어, 분포의 형태나 파라미터가 변하지 않습니다.  
- Independent (독립): 한 변수의 값이 다른 변수의 값에 영향을 미치지 않아, 변수들 간 상관관계가 전혀 없습니다.  
이 개념은 통계, 기계학습 등 여러 분야에서 기본 가정으로 사용되며, 빅데이터의 샘플링과 분석, 추론의 정확도를 보장하는 데 중요합니다.

## MLE: 최대가능도추정
MLE는 $$\hat{\theta}=\arg\max_{\theta} L(\theta) $$ 또는 동치로 $$\hat{\theta}=\arg\max_{\theta} \log L(\theta) $$ 를 풉니다.[1][2]
로그를 취하면 곱이 합으로 바뀌어 수치적 안정성과 미분·최적화가 쉬워집니다.[2][3]  
정규분포 $$\mathcal{N}(\mu,\sigma^2) $$의 경우, i.i.d. 표본에서 $$\hat{\mu}=\bar{x} $$, $$\hat{\sigma}^2=\frac{1}{n}\sum_i (x_i-\bar{x})^2 $$가 MLE로 유도됩니다.[7][3]

## Log-Likelihood와 미분
로그가능도는 $$\ell(\theta)=\log L(\theta)=\sum_{i=1}^{n}\log f(x_i\mid \theta) $$ 입니다.[3][2]
정규분포의 로그가능도는 $$\ell(\mu,\sigma^2)= -\frac{n}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_i (x_i-\mu)^2 $$로 정리됩니다.[2][3]
이에 대한 1차 조건을 풀면 앞서의 $$\hat{\mu},\hat{\sigma}^2$$ 식이 도출됩니다. 로그는 단조 증가이므로 최대점은 Likelihood와 같습니다.[3][2]

## Negative Log-Likelihood와 최적화
최적화는 보통 “최소화”로 쓰기 때문에 $$\text{NLL}(\theta)=-\ell(\theta) $$를 최소화합니다.[8][4]
즉, $$\hat{\theta}=\arg\min_{\theta} -\sum_i \log f(x_i\mid \theta) $$이며, 경사하강법으로 쉽게 학습할 수 있습니다.[4][8]
딥러닝에서 NLL은 분류 손실의 기본 형태이며, 계산 그래프와 자동미분에 잘 맞습니다.[5][9]

## 분류에서의 NLL와 교차엔트로피
이산 분류에서 정답 분포 p(one-hot)와 예측 분포 q가 있으면 교차엔트로피 $$H(p,q)=-\sum_i p(i)\log q(i)$$가 됩니다.[5][4]
정답이 원-핫이면 실제로는 “정답 클래스의 로그확률만” 더한 NLL과 같아집니다. 즉, 교차엔트로피와 NLL은 본질적으로 같은 목적입니다.[6][4]

$(p(i))$ 는 정답 클래스 인덱스에서는 1이고, 나머지 인덱스에서는 0(즉, one-hot)입니다. 따라서 이 합은 정답 클래스에 해당하는 하나의 항만 남게 되어,

```math
[H(p, q) = - \log q(\text{정답 클래스})
]
```

로 단순화됩니다. 즉, 교차엔트로피는 정답 클래스에 대한 예측 확률의 로그 값의 음수입니다.

프레임워크에서는 “LogSoftmax + NLLLoss”가 “CrossEntropyLoss”와 동등하며, CrossEntropyLoss는 내부적으로 Softmax와 로그를 함께 처리합니다.[6][4]

## 왜 같은가?
CrossEntropyLoss는 내부적으로 LogSoftmax와 NLLLoss(Negative Log Likelihood Loss)를 결합한 형태입니다. 즉, CrossEntropyLoss는 먼저 모델의 로짓(logits)에 대해 LogSoftmax를 계산하고, 그 결과에 대해 NLLLoss를 적용하는 수식을 사용합니다.

수식적으로 설명하면,

Softmax 함수: ( $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ )

LogSoftmax: ( $\log \sigma(z_i)$ )

NLLLoss: ( $-\sum_i y_i \log p_i$ ), 여기서 ( $p_i$ )는 LogSoftmax의 출력

그래서 CrossEntropyLoss는 다음과 같이 표현할 수 있습니다:

```math
[\text{CrossEntropyLoss}(z, y) = \text{NLLLoss}(\log \sigma(z), y)
]
```

PyTorch에서 CrossEntropyLoss는 위 과정을 내부에서 자동으로 수행하기 때문에, 모델의 출력은 raw logits이어야 합니다. 반면, NLLLoss는 LogSoftmax로 변환된 값을 입력으로 받기 때문에, 별도로 LogSoftmax를 적용한 후 사용해야 합니다.

## 예시 1: 정규분포 MLE 파이썬 구현
아래 예시는 표준편차를 미지수로 두고 $$\mu$$와 $$\sigma$$를 동시에 MLE로 추정합니다. 수치최적화로 NLL을 최소화합니다.[2][3]
```python
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(7)
# 예시 데이터: 키(cm)라고 가정
x = rng.normal(loc=168, scale=5, size=10)

def nll_normal(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)  # sigma>0 보장
    return 0.5*len(x)*np.log(2*np.pi) + len(x)*np.log(sigma) + np.sum((x - mu)**2)/(2*sigma**2)

res = minimize(nll_normal, x0=np.array([np.mean(x), np.log(np.std(x))]))
mu_hat, log_sigma_hat = res.x
sigma_hat = np.exp(log_sigma_hat)
print(mu_hat, sigma_hat)
```
- 핵심: 로그-표준편차를 최적화해 양수 제약을 처리하고, 정규분포의 NLL 공식을 사용합니다.[3][2]
- 해석: 해는 표본평균과 비편향 보정 없는 분산에 가까운 값으로 수렴합니다(정규 MLE 성질).[7][3]

## 예시 2: 이진 로지스틱 회귀와 NLL
로지스틱 회귀에서 한 표본의 로그가능도는 $$y\log \sigma(z)+(1-y)\log(1-\sigma(z))$$이며, 전체 NLL은 그 음수 합입니다.[4][5]
여기서 $$\sigma(z)=1/(1+e^{-z})$$, $$z=w^\top x + b$$ 입니다. 이는 “로지스틱 손실=음의 로그우도”와 같은 말입니다.[10][5]
직접 구현 예시는 아래와 같습니다.
```python
import numpy as np

rng = np.random.default_rng(0)
n, d = 200, 3
X = rng.normal(size=(n,d))
w_true = rng.normal(size=d)
b_true = 0.5
logits = X @ w_true + b_true
p = 1/(1+np.exp(-logits))
y = rng.binomial(1, p)

w = np.zeros(d)
b = 0.0
lr = 0.1
for _ in range(2000):
    z = X @ w + b
    q = 1/(1+np.exp(-z))
    # NLL = -sum[y*log q + (1-y)*log(1-q)]
    grad_w = X.T @ (q - y) / n
    grad_b = np.mean(q - y)
    w -= lr * grad_w
    b -= lr * grad_b

print(w, b)
```
- 핵심: 로지스틱 회귀의 최대우도는 NLL 최소화와 동치이며, 위 갱신식은 바로 그 경사입니다.[5][4]
- 직관: 정답 클래스의 로그확률을 크게 만드는 방향으로 가중치를 조정합니다.[4][5]

## 예시 3: PyTorch 분류 모델과 NLL/교차엔트로피
PyTorch에서는 다음 두 방식이 사실상 같습니다.[6][4]
- 방식 A: 마지막 층 출력에 LogSoftmax를 적용하고 NLLLoss 사용.  
- 방식 B: 마지막 층 출력(logit)에 CrossEntropyLoss 사용(내부적으로 softmax+log).  
간단 예시는 아래와 같습니다.
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 더미 데이터
X = torch.randn(256, 20)
y = torch.randint(0, 5, (256,))

# 방식 A: LogSoftmax + NLLLoss
model_a = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 5), nn.LogSoftmax(dim=1)
)
loss_a = nn.NLLLoss()  # 정답 클래스의 로그확률 음수 합
opt_a = optim.Adam(model_a.parameters(), lr=1e-3)

# 방식 B: CrossEntropyLoss (logit 입력)
model_b = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 5)
)
loss_b = nn.CrossEntropyLoss()  # 내부적으로 LogSoftmax + NLL
opt_b = optim.Adam(model_b.parameters(), lr=1e-3)

# 한 스텝 학습 루프 예시
def train_step(model, loss_fn, opt):
    model.train()
    opt.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()
    return loss.item()

print(train_step(model_a, loss_a, opt_a))
print(train_step(model_b, loss_b, opt_b))
```
- 구현 차이: CrossEntropyLoss는 로짓을 입력으로 받고 내부에서 Softmax와 로그를 처리하여 수치적으로 안정적입니다.[9][4]
- 개념 동일성: 둘 다 “정답 클래스의 로그확률을 최대화(=NLL 최소화)”한다는 점에서 같게 작동합니다.[6][4]

## 왜 로그를 쓰나?
- 수치 안정성: 작은 확률의 곱은 언더플로우를 유발하므로 로그로 합으로 바꿉니다.[2][3]
- 최적화 용이성: 합은 미분·평균화·정규화 등과 결합하기 좋고, 경사 계산이 간단합니다.[3][2]
- 해석성: NLL은 “정답의 부정확도”로 볼 수 있어, 손실로 직관적입니다.[5][4]

## 자주 하는 실수와 팁
- LogSoftmax+NLLLoss와 CrossEntropyLoss를 동시에 쓰지 않습니다(중복 Softmax 문제).[9][6]
- 이진 분류에서 다중분류용 CrossEntropyLoss를 쓰면 레이블·출력 형태가 맞지 않을 수 있으니 BCE/BCEWithLogits를 상황에 맞게 선택합니다.[9][5]
- 정규분포 추정 시 $$\sigma>0$$ 제약 처리를 log-parameterization으로 안정화하면 좋습니다.[2][3]

## 수학 박스: 정규분포의 MLE 요약
- 정규분포 로그가능도: $$\ell(\mu,\sigma^2)= -\frac{n}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_i (x_i-\mu)^2$$ 입니다.[3][2]
- 일차조건: $$\partial \ell/\partial \mu=0 \Rightarrow \hat{\mu}=\bar{x}$$, $$\partial \ell/\partial \sigma^2=0 \Rightarrow \hat{\sigma}^2=\frac{1}{n}\sum_i (x_i-\bar{x})^2$$ 입니다.[7][3]
- 해석: 평균은 표본평균, 분산은 비편향 보정 없는 표본분산이 MLE입니다(큰 표본에서 좋은 성질).[1][3]

## 마무리
딥러닝 관점에서 학습은 “모형을 가정하고 NLL을 최소화”하는 과정이며, 이는 MLE와 동치입니다.[4][5]
분류 문제에서는 교차엔트로피가 곧 NLL로 이해되며, 프레임워크에서는 CrossEntropyLoss 하나로 실용적이고 안정적으로 처리합니다.[6][4]

참고: 본 글의 개념적 골자는 위 링크(정규분포 MLE 유도, NLL-교차엔트로피 관계, 프레임워크 구현 관행)에 기반합니다.[4][6][3]

부록: 원문 포스트 맥락
- 제공된 원문은 Likelihood, MLE, Log/Negative Log-Likelihood의 핵심을 간결히 정리하고 있으며, 본 초안은 수식 전개와 실습 코드를 추가하여 딥러닝 학습에 바로 연결되도록 확장했습니다.[11]
- 표준편차 고정 사례의 직관(평균 168에서 최대/최소)은 정규분포 로그가능도의 형태와 일치하는 현상으로 해석됩니다.[11]

출처  
- Maximum likelihood estimation(정의, 정규분포 유도, 성질).[1][2][3]
- NLL과 교차엔트로피의 관계, 프레임워크 구현 관행(LogSoftmax+NLL vs CrossEntropyLoss).[5][6][4]
- 정규 MLE 해 유도(표본평균, 비보정 분산).[7][3]
- 원문 맥락 확인(티스토리 제공 파일).[11]

[1](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
[2](https://web.stanford.edu/class/archive/cs/cs109/cs109.1206/lectureNotes/LN20_parameters_mle.pdf)
[3](https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood)
[4](https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81/)
[5](https://sebastianraschka.com/blog/2022/losses-learned-part1.html)
[6](https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816)
[7](https://statproofbook.github.io/P/ug-mle.html)
[8](https://supermemi.tistory.com/entry/Loss-Cross-Entropy-Negative-Log-Likelihood-%EB%82%B4%EC%9A%A9-%EC%A0%95%EB%A6%AC-Pytorch-Code)
[9](https://wandb.ai/capecape/classification-techniques/reports/Classification-Loss-Functions-Comparing-SoftMax-Cross-Entropy-and-More--VmlldzoxODEwNTM5)
[10](https://sebastianraschka.com/faq/docs/negative-log-likelihood-logistic-loss.html)
[11](https://byeonggeuk.tistory.com/19)
[12](https://airsbigdata.tistory.com/202)
[13](https://velog.io/@mmodestaa/Log-Probability-Negative-Log-Likelihood-and-Cross-Entropy-%EC%84%A4%EB%AA%85)
[14](https://fenzhan.tistory.com/21)
[15](https://faculty.washington.edu/yenchic/17Sp_403/Lec3_MLE_MSE.pdf)
[16](https://engineering.purdue.edu/ChanGroup/ECE645Notes/StudentLecture08.pdf)
[17](https://paul-hyun.github.io/nlp-tutorial-02-04-negative-log-likelihood/)
[18](https://ai-com.tistory.com/entry/DL-%EB%B6%84%EB%A5%98-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98-Cross-Entropy-Loss-Focal-Loss)
[19](https://keepdev.tistory.com/48)
[20](https://gaussian37.github.io/dl-concept-nll_loss/)
# Reference
https://byeonggeuk.tistory.com/19
