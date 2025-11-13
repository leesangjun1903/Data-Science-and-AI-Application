## Cross Entropy (크로스 엔트로피) 이해하기

Cross Entropy는 머신러닝과 딥러닝에서 가장 널리 사용되는 손실함수 중 하나로, 정보이론(Information Theory)에서 유래된 개념입니다[1][2]. 분류 문제에서 실제 확률 분포와 모델이 예측한 확률 분포 간의 차이를 측정하는 핵심적인 지표입니다.

### 정보이론의 기초 개념

Cross Entropy를 이해하기 위해서는 먼저 정보이론의 기본 개념들을 살펴볼 필요가 있습니다.

#### 정보량 (Information)

**정보량**은 특정 사건이 발생했을 때 얻는 정보의 양을 수치적으로 나타낸 것입니다[3][4]. 확률이 낮은 사건일수록 더 많은 정보를 제공합니다.

특정 사건 $$x_i$$의 정보량은 다음과 같이 정의됩니다[5]:

$$ I(x_i) = -\log_2 p(x_i) $$

여기서 $$p(x_i)$$는 사건 $$x_i$$가 발생할 확률입니다. 로그의 밑이 2일 때 정보량의 단위는 비트(bit)입니다[4].

예를 들어, 공정한 동전던지기에서:
- 앞면이 나올 확률: 1/2, 정보량: $$-\log_2(1/2) = 1$$ bit
- 확률이 1/8인 사건의 정보량: $$-\log_2(1/8) = 3$$ bit

#### 엔트로피 (Entropy)

**엔트로피**는 정보량의 평균값으로, 확률 분포의 불확실성을 나타냅니다[1][6]. 섀넌(Shannon)이 정의한 엔트로피는 다음과 같습니다[7]:

$$ H(P) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i) $$

엔트로피는 다음과 같은 특성을 가집니다[8]:
- **모든 사건의 확률이 동일할 때 최대값**을 가집니다
- **한 사건의 확률이 1이고 나머지가 0일 때 최소값(0)**을 가집니다
- 엔트로피가 높을수록 예측이 어려우며 불확실성이 큽니다

### Cross Entropy의 정의와 수식

Cross Entropy는 **두 개의 서로 다른 확률 분포 간의 차이**를 측정하는 지표입니다[1][2]. 실제 분포 $$P$$와 예측 분포 $$Q$$에 대한 Cross Entropy는 다음과 같이 정의됩니다[9]:

$$ H(P, Q) = -\sum_{i=1}^{n} p(x_i) \log q(x_i) $$

여기서:
- $$P$$: 실제 데이터의 확률 분포 (Ground Truth)
- $$Q$$: 모델이 예측한 확률 분포
- $$p(x_i)$$: 실제 데이터에서 사건 $$x_i$$의 확률
- $$q(x_i)$$: 예측된 사건 $$x_i$$의 확률

### Cross Entropy의 의미와 해석

Cross Entropy는 **$$P$$ 분포를 따르는 데이터를 $$Q$$ 분포로 표현할 때 필요한 평균 비트 수**를 의미합니다[10][11]. 다시 말해, 실제 분포 $$P$$에 대해 예측 분포 $$Q$$의 전략을 사용할 때의 질문 개수의 기댓값입니다[11].

#### 중요한 성질

1. **Cross Entropy ≥ Entropy**: Cross Entropy는 항상 Entropy보다 크거나 같습니다[1][9].
   $$ H(P, Q) \geq H(P) $$

2. **최소값 조건**: $$P = Q$$일 때 Cross Entropy는 Entropy와 같아지며, 이때 최소값을 가집니다[8].
   $$ H(P, P) = H(P) $$

3. **비대칭성**: $$H(P, Q) \neq H(Q, P)$$로 Cross Entropy는 비대칭적입니다[1].

### 머신러닝에서의 Cross Entropy

#### KL Divergence와의 관계

Cross Entropy는 KL Divergence(Kullback-Leibler Divergence)와 밀접한 관련이 있습니다[12][13]. 분류 문제에서 실제 분포와 예측 분포 간의 KL Divergence는:

$$ D_{KL}(P||Q) = H(P, Q) - H(P) $$

분류 문제에서 실제 레이블은 원-핫 인코딩된 형태로 $$[0, 0, 1, 0, \ldots, 0]$$와 같이 표현되며, 이 경우 실제 분포의 엔트로피 $$H(P) = 0$$이 됩니다[12]. 따라서:

$$ D_{KL}(P||Q) = H(P, Q) $$

즉, **분류 문제에서 Cross Entropy를 최소화하는 것은 KL Divergence를 최소화하는 것과 동일**합니다[12][8].

#### 손실함수로서의 Cross Entropy

머신러닝에서 Cross Entropy가 손실함수로 널리 사용되는 이유는 다음과 같습니다[14]:

1. **확률 분포 간 유사성 측정**: 모델의 예측 분포가 실제 분포에 얼마나 가까운지 정량적으로 측정할 수 있습니다.

2. **미분 가능성**: 경사하강법 등의 최적화 알고리즘에 적합한 연속적이고 미분 가능한 함수입니다.

3. **수치적 안정성**: 로그 함수의 특성상 확률이 0에 가까울 때 큰 페널티를 주어 학습을 안정화합니다.

### Binary Cross Entropy

**이진 분류 문제**에서 사용되는 Binary Cross Entropy는 다음과 같이 정의됩니다[15][16]:

$$ BCE = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)] $$

여기서:
- $$y_i$$: 실제 레이블 (0 또는 1)
- $$\hat{y}_i$$: 예측 확률 (0과 1 사이의 값)
- $$N$$: 전체 샘플 수

Binary Cross Entropy는 **시그모이드 활성화 함수와 함께 사용**되어 이진 분류 문제를 효과적으로 해결합니다[15].

### Categorical Cross Entropy

**다중 클래스 분류 문제**에서 사용되는 Categorical Cross Entropy는 다음과 같이 정의됩니다[17][16]:

$$ CCE = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij}) $$

여기서:
- $$C$$: 클래스의 개수
- $$y_{ij}$$: 샘플 $$i$$가 클래스 $$j$$에 속하는지를 나타내는 원-핫 인코딩된 값
- $$\hat{y}_{ij}$$: 샘플 $$i$$가 클래스 $$j$$에 속할 예측 확률

Categorical Cross Entropy는 **소프트맥스 활성화 함수와 함께 사용**되어 다중 클래스 분류 문제를 해결합니다[17].

### Cross Entropy의 미분

Cross Entropy 손실함수의 그래디언트는 역전파 알고리즘에서 중요한 역할을 합니다[18][19]. 소프트맥스 함수와 Cross Entropy를 결합한 경우, 그래디언트는 매우 간단한 형태를 가집니다[18]:

$$ \frac{\partial L}{\partial z_i} = \hat{y}_i - y_i $$

여기서:
- $$z_i$$: 클래스 $$i$$에 대한 로짓 값
- $$\hat{y}_i$$: 소프트맥스를 통해 계산된 예측 확률
- $$y_i$$: 실제 레이블

이러한 간단한 그래디언트 형태는 **계산 효율성과 수치적 안정성**을 제공합니다[20].

### Negative Log-Likelihood와의 관계

Cross Entropy는 **Negative Log-Likelihood와 수학적으로 동일**합니다[21][22]. 분류 문제에서 최대우도추정(Maximum Likelihood Estimation)을 수행하는 것은 Cross Entropy를 최소화하는 것과 같습니다[21].

### 실제 구현에서의 고려사항

#### PyTorch에서의 구현

PyTorch에서 `CrossEntropyLoss`는 내부적으로 LogSoftmax와 NLLLoss를 결합한 형태로 구현됩니다[23][24]:

```python
# CrossEntropyLoss는 다음과 같이 작동
logits = model(input)  # 원시 로짓 값
loss = nn.CrossEntropyLoss()(logits, targets)

# 이는 다음과 동일
log_probs = F.log_softmax(logits, dim=1)
loss = F.nll_loss(log_probs, targets)
```

#### 수치적 안정성

실제 구현에서는 수치적 안정성을 위해 다음과 같은 기법들이 사용됩니다[25]:

1. **Log-Sum-Exp 트릭**: 큰 로짓 값에서 오버플로우를 방지
2. **Label Smoothing**: 과신(overconfidence) 문제를 완화
3. **클리핑**: 극한 확률 값에서의 불안정성 방지

### 활용 분야와 확장

Cross Entropy는 다양한 분야에서 활용됩니다:

1. **컴퓨터 비전**: 이미지 분류, 객체 탐지
2. **자연어 처리**: 텍스트 분류, 언어 모델링
3. **추천 시스템**: 사용자 선호도 예측
4. **의료 진단**: 질병 분류 및 예측

### 한계와 대안

Cross Entropy의 한계점과 이를 해결하기 위한 대안들도 연구되고 있습니다:

1. **클래스 불균형 문제**: Weighted Cross Entropy, Focal Loss 등으로 해결[26]
2. **노이즈 레이블**: Generalized Cross Entropy 등의 robust 손실함수 사용[27]
3. **과신 문제**: Label Smoothing, Temperature Scaling 등으로 완화[28]

Cross Entropy는 이론적 배경이 탄탄하고 실용적 효과가 검증된 손실함수로, 현대 머신러닝과 딥러닝에서 핵심적인 역할을 하고 있습니다. 정보이론에서 출발한 이 개념이 어떻게 실제 문제 해결에 적용되는지 이해하는 것은 효과적인 모델 설계와 학습에 매우 중요합니다.

[1] https://tinyarchive.tistory.com/17
[2] https://wikidocs.net/157190
[3] https://89douner.tistory.com/28
[4] http://www.ktword.co.kr/test/view/view.php?no=649
[5] https://lcyking.tistory.com/entry/%ED%86%B5%EA%B3%84-%EC%A0%95%EB%B3%B4%EB%9F%89%EA%B3%BC-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC
[6] https://blog.naver.com/ssm6410/221704953922
[7] https://www.jaenung.net/tree/3057
[8] https://3months.tistory.com/436
[9] https://memesoo99.tistory.com/39
[10] https://jins-sw.tistory.com/entry/Cross-Entropy-%EC%9D%B4%EC%95%BC%EA%B8%B0
[11] https://blog.naver.com/qbxlvnf11/223018634868
[12] https://ladun.tistory.com/81
[13] https://velog.io/@min0731/Entropy-Cross-Entropy-KL-Divergence
[14] https://bommbom.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%97%90%EC%84%9C-%ED%81%AC%EB%A1%9C%EC%8A%A4-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BCCross-Entropy%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0
[15] https://wikidocs.net/235864
[16] http://gombru.github.io/2018/05/23/cross_entropy_loss/
[17] https://wordbe.tistory.com/46
[18] https://xogns7652.tistory.com/entry/Cross-Entropy-Loss-and-gradient
[19] https://jmlb.github.io/ml/2017/12/26/Calculate_Gradient_Softmax/
[20] https://blog.naver.com/jjys9047/222074525554
[21] https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81/
[22] https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816
[23] https://discuss.pytorch.org/t/softmax-cross-entropy-loss/125383
[24] https://discuss.pytorch.kr/t/cross-entropy-softmax/1286
[25] https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning
[26] https://ieeexplore.ieee.org/document/9319440/
[27] https://www.semanticscholar.org/paper/1e1855ca80e8ac3de0e169871f320416902e9ad1
[28] https://arxiv.org/abs/2402.03979
[29] https://www.sec.gov/Archives/edgar/data/2027172/0002027172-24-000002-index.htm
[30] https://www.sec.gov/Archives/edgar/data/2001575/0002001575-24-000001-index.htm
[31] https://www.sec.gov/Archives/edgar/data/2007386/0002007386-24-000001-index.htm
[32] https://www.sec.gov/Archives/edgar/data/1976738/0001976738-23-000001-index.htm
[33] https://www.sec.gov/Archives/edgar/data/1975749/0001976738-23-000002-index.htm
[34] https://www.sec.gov/Archives/edgar/data/1898395/0001898395-22-000002-index.htm
[35] https://arxiv.org/abs/2401.02058
[36] https://www.mdpi.com/1099-4300/26/6/491
[37] https://arxiv.org/abs/2304.07288
[38] https://www.mdpi.com/1099-4300/26/7/576
[39] https://www.isca-archive.org/interspeech_2023/plaquet23_interspeech.html
[40] https://nuguziii.github.io/dev/dev-002/
[41] https://www.sec.gov/Archives/edgar/data/1898395/0001898395-21-000002-index.htm
[42] https://www.mdpi.com/1099-4300/26/7/560
[43] https://ieeexplore.ieee.org/document/10377280/
[44] https://arxiv.org/abs/2306.03288
[45] https://www.machinelearningmastery.com/cross-entropy-for-machine-learning/
[46] https://en.wikipedia.org/wiki/Cross-entropy
[47] https://blog.naver.com/nonezerok/221431459060
[48] https://www.sec.gov/Archives/edgar/data/1001082/000155837025002818/dish-20241231x10k.htm
[49] https://www.sec.gov/Archives/edgar/data/1707919/000114036125011607/ef20038940_10k.htm
[50] https://www.sec.gov/Archives/edgar/data/1610601/000119312525098265/d744071d20f.htm
[51] https://www.sec.gov/Archives/edgar/data/1758009/000121390024028799/ea0202448-10k_quantum.htm
[52] https://ieeexplore.ieee.org/document/10317335/
[53] https://journals.rta.lv/index.php/HET/article/view/8251
[54] http://www.warse.org/IJATCSE/static/pdf/file/ijatcse175942020.pdf
[55] https://www.webology.org/abstract.php?id=436
[56] https://link.springer.com/10.1007/978-3-030-96308-8_130
[57] https://link.springer.com/10.1007/s00521-022-07091-x
[58] https://ieeexplore.ieee.org/document/9739948/
[59] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12456/2659662/An-object-classification-and-detection-method-with-faster-R-CNN/10.1117/12.2659662.full
[60] https://www.semanticscholar.org/paper/5068b47b98fca6387e88d15812faa76c624416d1
[61] https://discuss.pytorch.org/t/can-we-use-cross-entropy-loss-for-binary-classification/159501
[62] https://www.youtube.com/watch?v=HmXJ6JPQyI8
[63] https://stats.stackexchange.com/questions/357963/what-is-the-difference-between-cross-entropy-and-kl-divergence
[64] https://rain-bow.tistory.com/entry/CrossEntropy
[65] https://www.sec.gov/Archives/edgar/data/1728205/000172820525000047/pll-20241231.htm
[66] https://www.sec.gov/Archives/edgar/data/1285785/000161803425000003/mos-20241231.htm
[67] https://www.sec.gov/Archives/edgar/data/2031069/000121390024112539/ea0209567-05.htm
[68] https://www.semanticscholar.org/paper/5648c62aa784451472b71ce0430e166409e6f2ad
[69] https://www.semanticscholar.org/paper/50aed1d96f1d41f9ec36027c27ce151519979f5f
[70] https://scholar.kyobobook.co.kr/article/detail/4010028909100
[71] https://www.semanticscholar.org/paper/ca12b9ea229635f12844a881879d5c0c70b91e65
[72] https://www.semanticscholar.org/paper/c262f1a98b26e595ffcde8dadc895088d8de4932
[73] https://www.semanticscholar.org/paper/7d5a72492fdc663663536b7f35a8629fd1be78e9
[74] https://www.semanticscholar.org/paper/f1e6ac5115d5757a20d8d3c1644f5ff80e582553
[75] https://www.semanticscholar.org/paper/628a2c02776cdd3e2456e9e578cdbc71e7d55467
[76] https://www.semanticscholar.org/paper/f9df0af3178063d0f3b0cc59c1d2884daba7370d
[77] https://www.semanticscholar.org/paper/b58c127ebcd048dd6077596ef4494a0bdcd518f4

# Reference
https://wordbe.tistory.com/46
