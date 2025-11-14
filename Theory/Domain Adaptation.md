# Domain Adaptation Methods 

Domain Adaptation은 기계학습과 딥러닝 분야의 핵심적인 기법으로, 하나의 도메인(소스 도메인)에서 학습된 모델을 다른 도메인(타겟 도메인)에서도 잘 작동하도록 적응시키는 방법론입니다[1][2]. 데이터 분포의 차이로 인해 발생하는 성능 저하를 해결하기 위해 다양한 수학적 접근법과 기법들이 개발되었습니다.

## 기본 개념 및 정의

Domain Adaptation의 핵심은 소스 도메인 $$\mathcal{D}\_s = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}$$ 

에서 학습된 모델을 타겟 도메인 

$$\mathcal{D}\_t = \{x\_j^t\}_{j=1}^{n_t}$$ 에 적용할 때 발생하는 분포 차이를 최소화하는 것입니다[3]. 

여기서 소스 도메인은 레이블이 있는 데이터이고, 타겟 도메인은 일반적으로 레이블이 없는 데이터입니다[1][4].

도메인 간 분포 차이는 다음과 같이 분류됩니다[1][3]:

- **Covariate Shift**: 입력 특성의 분포는 변하지만 조건부 레이블 분포는 동일
- **Prior Shift (Label Shift)**: 레이블 분포는 변하지만 특성의 조건부 분포는 동일  
- **Concept Drift**: 입력이 주어졌을 때 레이블의 조건부 분포가 변화

## 주요 도메인 적응 방법론

### 1. 분포 정렬 방법론 (Distribution Alignment Methods)

#### Maximum Mean Discrepancy (MMD)
MMD는 두 분포 간의 차이를 측정하는 대표적인 방법으로, 다음과 같이 정의됩니다[3][5]:

$$
\text{MMD}(P_s, P_t) = \left\| \frac{1}{n_s} \sum_{i=1}^{n_s} \phi(x_i^s) - \frac{1}{n_t} \sum_{j=1}^{n_t} \phi(x_j^t) \right\|^2_{\mathcal{H}}
$$

여기서 $$\phi(\cdot)$$는 특성 매핑 함수이고, $$\mathcal{H}$$는 재생핵 힐베르트 공간(RKHS)입니다[5][6].

가중치가 적용된 MMD(Weighted MMD)는 클래스 불균형 문제를 해결하기 위해 제안되었습니다[5][7]:

$$
\text{WMMD} = \sum_{c=1}^{C} w_c \left\| \frac{1}{n_s^c} \sum_{i \in D_s^c} \phi(x_i) - \frac{1}{n_t^c} \sum_{j \in D_t^c} \phi(x_j) \right\|^2
$$

#### CORAL (CORrelation ALignment)
CORAL은 소스와 타겟 도메인의 2차 통계량을 정렬하는 방법입니다[8][9][10]:

$$
\min_A \|A^T C_s A - C_t\|_F^2
$$

여기서 $$A$$는 특성 변환 행렬, $$C_s$$와 $$C_t$$는 각각 소스와 타겟 도메인의 공분산 행렬입니다[8].

CORAL의 해는 다음 네 단계로 계산됩니다[8]:
1. $$C_s = \text{Cov}(X_s) + \lambda I_p$$
2. $$C_t = \text{Cov}(X_t) + \lambda I_p$$  
3. $$X_s = X_s C_s^{-1/2}$$
4. $$X_s^{enc} = X_s C_t^{1/2}$$

### 2. 적대적 도메인 적응 (Adversarial Domain Adaptation)

#### Domain-Adversarial Neural Network (DANN)
DANN은 도메인 적응 분야의 획기적인 방법론으로, 적대적 학습을 통해 도메인 불변 특성을 학습합니다[11][12][13].

DANN의 핵심 구조는 다음과 같습니다[11][12]:
- **특성 추출기** (Feature Extractor): 도메인 불변 특성 생성
- **레이블 예측기** (Label Predictor): 분류 작업 수행
- **도메인 분류기** (Domain Classifier): 도메인을 구별

DANN의 목적 함수는 다음과 같이 정의됩니다[12]:

$$
E(\theta_f, \theta_y, \theta_d) = \frac{1}{n_s} \sum_{i=1}^{n_s} L_y(G_y(G_f(x_i^s)), y_i^s) - \lambda \left[ \frac{1}{n_s} \sum_{i=1}^{n_s} L_d(G_d(G_f(x_i^s)), 0) + \frac{1}{n_t} \sum_{j=1}^{n_t} L_d(G_d(G_f(x_j^t)), 1) \right]
$$

#### 기울기 역전 층 (Gradient Reversal Layer)
DANN의 핵심 구성요소인 기울기 역전 층은 순전파에서는 입력을 그대로 전달하고, 역전파에서는 기울기에 -1을 곱합니다[14][12]:

순전파: $$y = x$$
역전파: $$\frac{\partial L}{\partial x} = -\alpha \frac{\partial L}{\partial y}$$

이를 통해 특성 추출기가 도메인 분류기를 혼동시키도록 학습됩니다[14][15].

### 3. 고급 도메인 적응 기법

#### Multi-Adversarial Domain Adaptation (MADA)
MADA는 다중 도메인 분류기를 사용하여 각 클래스별로 세밀한 정렬을 수행합니다[16]:

$$
L_{adv}^k = -\frac{1}{n_s} \sum_{i=1}^{n_s} p_k^s(x_i^s) \log[G_d^k(G_f(x_i^s))] - \frac{1}{n_t} \sum_{j=1}^{n_t} p_k^t(x_j^t) \log[1-G_d^k(G_f(x_j^t))]
$$

#### Source-Free Domain Adaptation (SFDA)
SFDA는 소스 데이터에 접근할 수 없는 상황에서의 도메인 적응을 다룹니다[17][18]. 사전 학습된 소스 모델과 타겟 데이터만을 사용하여 적응을 수행합니다.

## 손실 함수 및 최적화

### 복합 손실 함수
도메인 적응에서는 여러 손실 함수를 결합하여 사용합니다[19][20]:

$$
L_{total} = L_{cls} + \lambda L_{domain} + \gamma L_{reg}
$$

여기서:
- $$L_{cls}$$: 분류 손실 (Cross-entropy, Focal loss 등)
- $$L_{domain}$$: 도메인 정렬 손실 (MMD, CORAL 등)
- $$L_{reg}$$: 정규화 손실

### 최적화 전략
도메인 적응 모델의 학습은 다음과 같은 minimax 최적화 문제로 공식화됩니다[21][22]:

$$
\min_{\theta_f, \theta_y} \max_{\theta_d} L_{cls}(\theta_f, \theta_y) - \lambda L_{domain}(\theta_f, \theta_d)
$$

## 이론적 기반

### 도메인 적응 이론
Ben-David 등의 이론에 따르면, 타겟 도메인에서의 오류는 다음과 같이 상한이 설정됩니다[4]:

$$
\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(D_S, D_T) + \lambda
$$

여기서:
- $$\epsilon_T(h)$$: 타겟 도메인 오류
- $$\epsilon_S(h)$$: 소스 도메인 오류  
- $$d_{\mathcal{H}\Delta\mathcal{H}}$$: H-divergence
- $$\lambda$$: 이상적인 공동 가설의 오류

### 수렴성 및 일반화 보장
최근 연구들은 Rényi divergence 기반 일반화 경계[20]와 optimal transport 이론[23]을 활용하여 도메인 적응의 이론적 보장을 제공합니다.

## 실제 응용 및 성능

도메인 적응은 다양한 분야에서 성공적으로 적용되고 있습니다:

- **컴퓨터 비전**: 이미지 분류, 객체 탐지, 의료 영상 분석[4][24]
- **자연어 처리**: 감정 분석, 기계 번역[25][26]
- **음성 인식**: 화자 적응, 환경 적응[27]
- **생체 신호**: EEG 기반 감정 인식[28], 뇌-컴퓨터 인터페이스[29]

실험 결과에 따르면, DANN 기반 방법들은 Office-31 데이터셋에서 77.62%의 정확도를 달성하여 기존 방법 대비 6.47% 향상된 성능을 보였습니다[30].

## 결론

Domain Adaptation은 실세계 기계학습 문제의 핵심 해결책으로, 수학적으로 엄밀한 이론과 실용적인 알고리즘을 제공합니다. MMD, CORAL과 같은 분포 정렬 방법부터 DANN과 같은 적대적 학습까지, 다양한 접근법이 각기 다른 상황에서 효과적입니다. 특히 기울기 역전 층을 활용한 DANN은 end-to-end 학습이 가능하여 실제 적용에서 큰 성공을 거두었습니다. 향후 연구는 source-free 적응, 다중 소스 적응, 그리고 더욱 강건한 이론적 보장 방향으로 진행될 것으로 예상됩니다.

[1] https://en.wikipedia.org/wiki/Domain_adaptation
[2] https://www.numberanalytics.com/blog/mastering-domain-adaptation-deep-learning
[3] https://www.numberanalytics.com/blog/ultimate-guide-domain-adaptation-machine-learning
[4] https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/
[5] https://paperswithcode.com/paper/mind-the-class-weight-bias-weighted-maximum
[6] https://arxiv.org/abs/2007.00689
[7] https://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Mind_the_Class_CVPR_2017_paper.pdf
[8] https://adapt-python.github.io/adapt/generated/adapt.feature_based.CORAL.html
[9] https://github.com/VisionLearningGroup/CORAL
[10] https://arxiv.org/abs/1612.01939
[11] https://www.elderresearch.com/blog/introduction-to-domain-adversarial-neural-networks/
[12] https://jmlr.org/papers/volume17/15-239/15-239.pdf
[13] https://arxiv.org/abs/1505.07818
[14] https://www.linkedin.com/pulse/gradient-reversal-layers-yeshwanth-n
[15] https://www.reddit.com/r/MachineLearning/comments/166tq43/d_why_did_the_authors_design_this_gradient/
[16] https://cdn.aaai.org/ojs/11767/11767-13-15295-1-2-20201228.pdf
[17] https://arxiv.org/abs/2302.11803
[18] https://arxiv.org/abs/2403.10834
[19] https://ieeexplore.ieee.org/document/9477825/
[20] https://ieeexplore.ieee.org/document/10102307/
[21] https://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf
[22] http://papers.neurips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
[23] https://ieeexplore.ieee.org/document/10877795/
[24] https://www.mdpi.com/2072-4292/14/17/4380
[25] https://www.taus.net/resources/blog/domain-adaptation-types-and-methods
[26] https://aclanthology.org/2020.coling-main.603/
[27] https://ieeexplore.ieee.org/document/10734810/
[28] https://arxiv.org/abs/2305.07446
[29] https://ieeexplore.ieee.org/document/10423987/
[30] https://dl.acm.org/doi/10.1145/3627631.3627635
[31] https://www.sec.gov/Archives/edgar/data/1786205/000095017025029150/aclx-20241231.htm
[32] https://www.sec.gov/Archives/edgar/data/1790340/000179034025000042/imrx-20241231.htm
[33] https://www.sec.gov/Archives/edgar/data/1860657/000143774925010198/allr20241231_10k.htm
[34] https://www.sec.gov/Archives/edgar/data/1785279/000095017025040217/mgx-20241231.htm
[35] https://www.sec.gov/Archives/edgar/data/1701478/000170147825000012/aztr-20241231.htm
[36] https://www.sec.gov/Archives/edgar/data/1705843/000162828025014139/cbus-20241231.htm
[37] https://xlink.rsc.org/?DOI=D3DD00162H
[38] https://linkinghub.elsevier.com/retrieve/pii/S016819232300343X
[39] https://linkinghub.elsevier.com/retrieve/pii/S0956713523000221
[40] https://linkinghub.elsevier.com/retrieve/pii/S1746809423003567
[41] https://ieeexplore.ieee.org/document/10477388/
[42] https://ieeexplore.ieee.org/document/9894051/
[43] https://www.lucentinnovation.com/blogs/it-insights/understanding-domain-adaptation-with-machine-learning
[44] https://s-space.snu.ac.kr/handle/10371/169184
[45] https://www.sciencedirect.com/topics/computer-science/domain-adaptation
[46] https://datascience0321.tistory.com/34
[47] https://www.sec.gov/Archives/edgar/data/721994/000072199425000087/lkfn-20241231.htm
[48] https://www.sec.gov/Archives/edgar/data/1766260/0001766260-23-000001-index.htm
[49] https://www.sec.gov/Archives/edgar/data/1772695/000177269524000008/nova-20231231.htm
[50] https://www.sec.gov/Archives/edgar/data/1843714/000119312523265265/d469525ds4.htm
[51] https://www.sec.gov/Archives/edgar/data/1053352/000155837023003189/htbk-20221231x10k.htm
[52] https://www.sec.gov/Archives/edgar/data/1272842/000095017023008785/airg-20221231.htm
[53] https://ieeexplore.ieee.org/document/10535297/
[54] https://ieeexplore.ieee.org/document/10769571/
[55] https://ieeexplore.ieee.org/document/9744510/
[56] https://journals.bilpubgroup.com/index.php/aia/article/view/6718
[57] https://paperswithcode.com/task/unsupervised-domain-adaptation
[58] https://www.numberanalytics.com/blog/mathematics-behind-domain-adaptation
[59] https://openreview.net/forum?id=Y08yLPFm1z
[60] https://arxiv.org/abs/2004.10618
[61] https://www.sec.gov/Archives/edgar/data/1944831/000194483125000023/wt-20250331.htm
[62] https://www.sec.gov/Archives/edgar/data/1212545/000121254525000141/wal-20250331.htm
[63] https://www.sec.gov/Archives/edgar/data/1325702/000119312525103977/d920126ddef14a.htm
[64] https://www.sec.gov/Archives/edgar/data/1001290/000114036125015768/ef20038949_20f.htm
[65] https://www.sec.gov/Archives/edgar/data/1756699/000095017025056953/tigr-20241231.htm
[66] https://www.sec.gov/Archives/edgar/data/1973832/000162828025017760/au-20241231.htm
[67] https://ieeexplore.ieee.org/document/8892738/
[68] https://arxiv.org/abs/2001.01046
[69] https://ojs.aaai.org/index.php/AAAI/article/view/5757/5613
[70] https://openaccess.thecvf.com/content_CVPR_2019/papers/Roy_Unsupervised_Domain_Adaptation_Using_Feature-Whitening_and_Consensus_Loss_CVPR_2019_paper.pdf
[71] https://www.linkedin.com/advice/1/what-most-effective-loss-functions-metrics-domain
[72] https://www.sec.gov/Archives/edgar/data/1053352/000155837022002901/htbk-20211231x10k.htm
[73] https://www.sec.gov/Archives/edgar/data/29905/000002990522000009/dov-20211231.htm
[74] https://www.sec.gov/Archives/edgar/data/1288469/000128846922000029/mxl-20211231.htm
[75] https://www.sec.gov/Archives/edgar/data/1562463/000156246322000039/inbk-20211231.htm
[76] https://www.sec.gov/Archives/edgar/data/1272842/000095017022004238/airg-20211231.htm
[77] https://www.sec.gov/Archives/edgar/data/1697412/000121390022032605/f10k2021_migomglobal.htm
[78] https://ieeexplore.ieee.org/document/10424420/
[79] https://ieeexplore.ieee.org/document/10725470/
[80] https://www.mdpi.com/2077-1312/11/11/2128
[81] https://ieeexplore.ieee.org/document/10295968/
[82] https://papers.phmsociety.org/index.php/phmap/article/view/3651
[83] https://www.numberanalytics.com/blog/applying-transfer-learning-to-advanced-mathematical-concepts
[84] https://proceedings.neurips.cc/paper/2021/file/db9ad56c71619aeed9723314d1456037-Paper.pdf
[85] https://arxiv.org/abs/1409.7495
[86] https://en.wikipedia.org/wiki/Transfer_learning
[87] https://blog.naver.com/dmsquf3015/222029881504
[88] https://www.sec.gov/Archives/edgar/data/1703647/000095017025040759/krro-20241231.htm
[89] https://www.mdpi.com/2032-6653/15/6/255
[90] https://academic.oup.com/jamiaopen/article/3/2/146/5831556
[91] https://ieeexplore.ieee.org/document/10050808/
[92] https://www.mdpi.com/2504-4990/6/3/84
[93] https://academic.oup.com/bioinformatics/article/38/18/4369/6649678
[94] https://www.mdpi.com/2072-4292/15/18/4562
[95] https://www.mdpi.com/1424-8220/23/9/4436
[96] https://arxiv.org/abs/2502.10694
[97] https://www.sec.gov/Archives/edgar/data/1971381/000119312525068918/d888030d10k.htm
[98] https://ieeexplore.ieee.org/document/10677898/
[99] https://ieeexplore.ieee.org/document/10868910/
[100] https://ieeexplore.ieee.org/document/10657858/
[101] https://ieeexplore.ieee.org/document/10843723/
[102] https://link.springer.com/10.1007/s42979-023-02090-8
[103] https://ieeexplore.ieee.org/document/9309429/
[104] https://www.mdpi.com/2075-1702/10/8/610

## Contrastive Adaptation Network for Unsupervised Domain Adaptation
https://arxiv.org/abs/1901.00976

대조적 도메인 불일치(Contrastive Domain Discrepancy, CDD)는 주로 기계 학습 및 컴퓨터 비전 분야에서 사용되는 개념입니다. 이는 서로 다른 도메인 간의 분포 차이를 최소화하며, 특정 작업을 수행할 때 모델의 일반화 성능을 향상시키기 위한 방법입니다.

주요 포인트는 다음과 같습니다:

도메인 불일치: 서로 다른 데이터셋이나 환경에서 학습된 모델은 일반적으로 성능 저하를 경험합니다.

대조적 학습: CDD는 서로 다른 도메인 간의 특징을 비교하고 대조함으로써 이러한 불일치를 줄이려는 전략입니다.

응용 분야: 이미지 분류, 객체 탐지 등 다양한 컴퓨터 비전 작업에 활용됩니다.


# Reference
https://ballentain.tistory.com/m/60
