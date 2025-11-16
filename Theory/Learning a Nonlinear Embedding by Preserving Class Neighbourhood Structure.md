# Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure

### 핵심 주장과 주요 기여

이 논문은 **Neighbourhood Component Analysis (NCA)를 비선형 신경망으로 확장**하여 저차원 특징 공간에서 K-최근접 이웃(KNN) 분류 성능을 최적화하는 방법을 제시합니다. 핵심 기여는 다음과 같습니다.[1]

**주요 기여:**

선형 방법(선형 NCA, LDA, PCA)의 한계를 극복하기 위해 **다층 신경망을 사용한 비선형 변환 함수**를 학습합니다. 비지도 사전학습(Unsupervised Pretraining) 단계에서 **제한된 볼츠만 머신(Restricted Boltzmann Machine, RBM)**의 층별 스택을 이용하여 네트워크를 초기화하고, 이후 지도학습 미세조정(Supervised Fine-tuning) 단계에서 NCA 목적함수를 통해 최적화합니다. **준지도학습(Semi-supervised Learning)** 성능을 향상시키기 위해 자동 부호화기 재구성 오류를 정규화 항으로 사용하며, 이를 통해 미표지 데이터를 활용할 수 있습니다.[1]

***

### 해결하는 문제와 제안 방법

**문제 정의:**

전통적인 거리 학습 방법들은 두 가지 제약이 있습니다. 선형 변환은 입력 차원 간의 고차 상관관계를 모델링할 수 없으며, 대부분의 비선형 접근법은 다중 차원 출력의 엔트로피 추정 문제로 인해 어려움을 겪습니다. 특히 상호정보량 극대화 기반 방법은 최적화 과정에서 가우시안 근사를 악용하여 실제 엔트로피와 괴리가 발생합니다.[1]

**제안 방법:**

논문은 쌍(pair)의 구조에만 집중하여 엔트로피 추정 문제를 회피합니다. 입력 공간 $$x_i, x_j$$에 대해 저차원 부호 공간에서의 거리에 기반한 확률 분포를 정의합니다:[1]

$$P(i|j) = \frac{\exp(-D[y_i, y_j]^2)}{\sum_{k \neq i} \exp(-D[y_i, y_k]^2)}$$

여기서 $$D[\cdot, \cdot]$$는 유클리드 거리이고, $$y_i = f_\theta(x_i)$$는 신경망 변환입니다.[1]

**NCA 목적함수:**

점 $$i$$가 이웃 $$j$$를 선택할 확률은:

$$p_{ij} = \frac{\exp(-D[y_i, y_j]^2)}{\sum_{k \neq i} \exp(-D[y_i, y_k]^2)}$$

클래스 $$c$$에 속할 확률:

$$P(c_i) = \sum_{j \in c_i} p_{ij}$$

최대화 목적함수:

$$\mathcal{L}_{NCA} = \sum_{i=1}^{n} \sum_{j \in c_i} p_{ij}$$

또는 로그 확률 버전:

$$\mathcal{L}_{log} = \sum_{i=1}^{n} \log \left( \sum_{j \in c_i} p_{ij} \right)$$

***

### 모델 구조 및 사전학습 전략

**네트워크 아키텍처:**

MNIST 실험에서 $$784 \to 500 \to 500 \to 2000 \to 30$$ 구조를 사용합니다. 인코더-디코더 설계로 구성되어, 인코더는 저차원 특징 추출을, 디코더는 원본 데이터 재구성을 담당합니다.[1]

**비지도 사전학습 (RBM 기반):**

**1단계: 제한된 볼츠만 머신 학습**

가시층(visible layer) $$v$$와 은닉층(hidden layer) $$h$$ 간의 조건부 확률:[1]

$$P(h_i|v) = \sigma(b_i^h + \sum_j W_{ij}v_j)$$

$$P(v_j|h) = \sigma(b_j^v + \sum_i W_{ij}h_i)$$

여기서 $$\sigma(\cdot)$$는 로지스틱 함수입니다.[1]

에너지 함수:

$$E(v, h) = -\sum_i b_i^h h_i - \sum_j b_j^v v_j - \sum_{i,j} W_{ij} h_i v_j$$

**2단계: 대조 발산(Contrastive Divergence) 학습**

1-스텝 대조 발산을 사용한 가중치 업데이트:[1]

$$\Delta W_{ij} = \alpha \left( \langle h_i v_j \rangle_{data} - \langle h_i v_j \rangle_{recon} \right)$$

**3단계: 층별 재귀 학습**

하위 층의 활성화 확률을 다음 층의 입력 데이터로 사용하여 깊은 계층 구조를 학습합니다. 이러한 그리디(Greedy) 층별 학습은 각 층이 입력의 주요 변이를 포착하도록 합니다.[1]

**미세조정 (Supervised Fine-tuning):**

사전학습된 RBM 스택을 '펼쳐' 인코더-디코더 구조로 변환합니다. 지도학습 NCA 목적함수와 함께 경합제곱법(Conjugate Gradients)을 사용하여 최적화합니다.[1]

***

### 성능 향상 및 실험 결과

**비선형 NCA의 성능:**

MNIST 데이터셋에서 1.08%, 1.00%, 1.03%, 1.01%의 오류율을 달성합니다(1, 3, 5, 7-최근접 이웃), 이는 SVM(1.4%)과 표준 역전파(1.6%)보다 우수합니다. 선형 NCA, PCA는 각각 약 2.5-3% 수준으로 비선형 방법의 우월성을 입증합니다.[1]

**정규화된 비선형 NCA (반지도학습):**

정규화 매개변수 $$\lambda = 0.99$$일 때:

$$\mathcal{L}' = \mathcal{L}_{NCA} + \lambda \cdot \mathcal{R}$$

여기서 $$\mathcal{R}$$은 재구성 오류이며, 교차 엔트로피로 정의됩니다:[1]

$$\mathcal{R} = -\sum_i [v_i^{(u)} \log(\hat{v}_i) + (1-v_i^{(u)}) \log(1-\hat{v}_i)]$$

미표지 데이터 활용 결과:[1]

| 표지 비율 | 정규화 NCA ($$\lambda=0.99$$) | 비선형 NCA ($$\lambda=1$$) | 선형 NCA |
|---------|---------------------------|----------------------|---------|
| 1% | 약 2.5-3% | 약 8-9% | 약 11-12% |
| 5% | 약 2% | 약 4-5% | 약 6-7% |
| 10% | 약 1.5% | 약 3% | 약 4-5% |

**클래스 관련성 코드 분할:**

50차원 코드를 사용하되, 첫 30차원은 NCA 목적함수에만 적용하고 나머지 20차원은 자동 부호화기 재구성에만 사용합니다. 이를 통해 30개 코드 단위만 사용한 성능(1.00% 오류)이 선택적으로 적용할 때도 우수하며, 마지막 20개 단위로만 분류 시 4.3% 오류를 보여 클래스 정보의 집중도를 입증합니다.[1]

***

### 일반화 성능 향상 가능성

**일반화 메커니즘:**

**1. 비지도 사전학습의 정규화 효과:**

최근 연구에 따르면 비지도 사전학습은 매개변수 초기화를 통해 최적화 궤적을 제어합니다. 사전학습은 더 나은 일반화를 지원하는 손실함수의 극소점 지역(basin of attraction)으로 학습자를 유도합니다. 이는 초기 단계의 매개변수 변화가 최종적으로 도달하는 지역에 큰 영향을 미치기 때문입니다. 따라서 **비지도 사전학습은 암묵적 정규화 역할**을 수행하여, 훈련 데이터의 일반적 구조를 학습하도록 합니다.[2]

**2. 반지도학습을 통한 일반화 향상:**

제안된 정규화 항 $$\mathcal{R}$$은 재구성 목적함수로 작동하여 저차원 표현이 입력의 중요한 구조를 보존하도록 강제합니다. 이는 **분류와 무관한 변동성(예: 필기체 기울기, 두께)**을 별도 차원에 인코딩하여 분류기 혼동을 줄입니다. 미표지 데이터를 활용하면 표현이 더 강건해져, 소량의 표지 데이터만으로도 높은 정확도를 유지할 수 있습니다.[1]

**3. 깊은 신경망의 표현 학습:**

깊은 계층 구조는 저수준 특징(선 및 모서리)에서 고수준 개념(얼굴 특징)으로의 계층적 추상화를 가능합니다. 이러한 다층 표현은 입력 데이터의 내재적 다양체(manifold) 구조를 더 잘 캡처하여 **미지 데이터에 대한 전이 가능성을 높입니다.[2][1]

**4. 거리 메트릭 학습의 이론적 기초:**

2024년 최신 연구에서 깊은 ReLU 신경망을 사용한 메트릭 학습의 일반화 분석 결과, **초과 위험도(Excess Risk)** 경계를 유도했습니다. 이는 네트워크 복잡도(깊이, 비영 가중치 수, 계산 단위)에 따라 근사 오류와 추정 오류의 균형을 설명하며, 적절한 네트워크 용량 선택으로 최적 초과 위험 속도를 달성할 수 있음을 보여줍니다.[3]

**5. 지역 근처 구조 보존:**

NCA는 **글로벌 클래스 분리 대신 지역 이웃 구조 보존**을 목표로 합니다. 이는 각 클래스의 다중성(multimodality)을 허용하면서도 각 샘플이 동일 클래스의 K개 이웃 가까이 있도록 강제합니다. 이는 강한 기하학적 제약으로 작용하여 **이웃 패턴의 학습 일반화**를 향상시킵니다.[4][5][1]

**6. 토폴로지 인식성(Topology Awareness) 향상:**

최근 2024년 연구는 신경망이 토폴로지 인식성(예: 그래프 구조, 근처 관계)을 강화하면 일반화 성능이 다양한 그룹 간 개선됨을 보였습니다. 비선형 NCA는 클래스 근처 구조를 명시적으로 학습하므로, 토폴로지 인식성이 강화되어 더 견고한 표현을 형성합니다.[6]

***

### 한계

**1. 계산 복잡도:**

NCA 목적함수 계산 시 모든 훈련 데이터 쌍에 대한 확률 정규화가 필요하며, 이는 $$O(n^2)$$ 복잡도를 야기합니다. 대규모 데이터셋에서 실제 적용은 제한적입니다.[1]

**2. 초매개변수 민감성:**

정규화 가중치 $$\lambda$$, 학습률, 모멘텀, 가중치 감쇠 등 다수의 매개변수 튜닝이 필요합니다. 논문에서는 이들이 상대적으로 견고하다고 주장하지만, 데이터셋 특성에 따른 최적값 탐색 비용이 높습니다.[1]

**3. 깊은 네트워크 학습의 어려움:**

당시(2007년) 기술로는 매우 깊은 네트워크 학습이 어려웠으며, 사전학습이 필수적이었습니다. 더 깊은 구조일수록 그래디언트 소실 문제가 심화될 가능성이 있습니다.[1]

**4. 표지 데이터 의존성:**

반지도학습 성능이 초기 표지 데이터 품질에 크게 좌우됩니다. 잘못된 표지 데이터는 전체 모델 성능 저하를 야기합니다.[1]

**5. 시각화 및 해석성:**

저차원 코드가 직관적으로 해석하기 어려울 수 있으며, 특히 높은 차원에서는 표현의 의미를 파악하기 어렵습니다.

***

### 앞으로의 연구에 미치는 영향

**1. 깊은 메트릭 학습의 기초:**

이 논문은 **깊은 신경망을 거리 메트릭 학습에 적용**한 선구적 연구입니다. 현대의 메트릭 학습, 얼굴 인식, 이미지 검색 등 많은 응용 분야의 이론적 토대를 제공했습니다.[7]

**2. 반지도학습 발전:**

제안된 정규화 항 $$\mathcal{R} = \mathcal{L}\_{NCA} + \lambda \mathcal{R}_{recon}$$ 구조는 현대 반지도 학습의 **결합 손실함수(Joint Loss) 패러다임**을 선도했습니다.[38-46] 최근의 의사 라벨링(Pseudo-labeling)과 일관성 정규화(Consistency Regularization) 기반 방법들이 이를 발전시킨 형태입니다.[44-46][2]

**3. 표현 학습(Representation Learning):**

비지도 사전학습 → 지도 미세조정 패러다임은 현대 자기 지도학습(Self-supervised Learning), 대조 학습(Contrastive Learning)으로 발전했습니다. 최근 2024-2025년 연구는 이를 고도화하여 토폴로지 보존과 일반화 성능을 동시에 향상시키고 있습니다.[8][9][10][6]

**4. 일반화 이론 발전:**

논문의 경험적 일반화 성공이 이론적 연구를 자극했습니다.[20-28] 2023-2024년의 깊은 학습 일반화 분석은 이 논문의 비선형 메트릭 학습 접근을 정량적으로 입증합니다.[22-24][3][8]

**5. 최신 확장 연구:**

**ModernNCA (2025):** 심화된 비선형 임베딩과 확률론적 근처 샘플링을 결합한 확장 버전입니다. 배치 정규화, 드롭아웃, 검색 기반 메커니즘을 추가하여 고차원 데이터에서 최신 성능을 달성합니다.[11]

**깊은 메트릭 학습 분석 (2024):** ReLU 신경망으로 메트릭 학습의 초과 위험도 경계를 첫 도출했으며, 네트워크 복잡도와 일반화의 정량적 관계를 입증합니다.[3]

***

### 앞으로 연구 시 고려할 점

**1. 계산 효율성:**

대규모 데이터셋 처리를 위해 근사 최근접 이웃 검색(ANN)이나 확률론적 근처 샘플링(Stochastic Neighborhood Sampling, SNS)의 적용이 필수적입니다. 최신 ModernNCA는 이를 성공적으로 구현했습니다.[12][11]

**2. 데이터 분포 불균형:**

클래스 불균형, 아웃오브디스트리뷰션(Out-of-Distribution) 샘플 처리 등 실제 문제를 고려한 견고성 강화가 필요합니다. 최근 개방형 반지도 학습(Open-set SSL) 연구가 이 방향으로 진행 중입니다.[13]

**3. 이론-실제 간극 축소:**

깊은 신경망의 일반화 경계가 여전히 느슨하다는 비판이 있습니다. 메트릭 학습 맥락에서 더 타이트한 경계 유도와 실제 성능 연관성 검증이 필요합니다.[14][3]

**4. 멀티모달 및 고차원 데이터:**

현대적으로 이미지, 텍스트, 그래프 등 다양한 데이터 타입에 대한 통합 메트릭 학습이 중요합니다. 교차 모달 매니폴드 정렬(Cross-modal Manifold Alignment) 연구가 이 방향을 보여줍니다.[15]

**5. 적응형 정규화:**

고정된 $$\lambda$$가 아닌 학습 동역학에 따라 자동 조정되는 적응형 정규화 계수 개발이 필요합니다.

**6. 전이 학습과 도메인 적응:**

표현 학습의 전이 가능성을 향상시키기 위한 도메인 불변 메트릭 학습, 메타-학습 적용 등이 유망합니다.[16]

***

### 결론

이 논문은 **선형 메트릭 학습의 한계를 극복하고 비선형 신경망 기반 거리 학습의 길을 개척**한 중요한 연구입니다. 비지도 사전학습과 지도 미세조정의 결합, 그리고 반지도 학습의 정규화 전략은 현대 딥러닝의 핵심 패러다임으로 자리 잡았습니다. 특히 깊은 표현 학습의 일반화 성능 향상 메커니즘이 명확히 제시되어, 이후 메트릭 학습, 대조 학습, 자기 지도학습으로의 발전을 촉발했습니다.

그러나 계산 복잡도, 초매개변수 민감성, 이론과 실제의 간극 등 해결해야 할 과제도 남아 있습니다. 최신 연구(2023-2025)는 이러한 한계를 보완하면서도 일반화 이론의 증진, 실무적 확장성 강화, 다양한 데이터 양식으로의 적용 확대를 추진 중입니다. 비선형 메트릭 학습은 오늘날의 AI 시스템—얼굴 인식, 추천 시스템, 시각 검색, 언어 모델 정렬 등—의 기저를 이루고 있으며, 그 중요성은 계속 증가하고 있 중요성은 계속 증가하고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/200b053f-f733-41e6-8239-8c275c5e38d9/salakhutdinov07a.pdf)
[2](https://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf)
[3](https://arxiv.org/abs/2405.06415)
[4](http://papers.neurips.cc/paper/5310-dimensionality-reduction-with-subspace-structure-preservation.pdf)
[5](https://www.cs.toronto.edu/~fritz/absps/nca.pdf)
[6](https://eccv.ecva.net/virtual/2024/poster/2355)
[7](https://proceedings.neurips.cc/paper/2010/file/d56b9fc4b0f1be8871f5e1c40c0067e7-Paper.pdf)
[8](http://arxiv.org/pdf/2402.03254.pdf)
[9](https://arxiv.org/pdf/2302.12383.pdf)
[10](https://arxiv.org/pdf/2505.04937.pdf)
[11](https://www.emergentmind.com/topics/modernnca)
[12](https://arxiv.org/html/2407.03257v2)
[13](https://pure.korea.ac.kr/en/publications/unknown-aware-graph-regularization-for-robust-semi-supervised-lea/)
[14](https://arxiv.org/html/2409.01498v1)
[15](https://pmc.ncbi.nlm.nih.gov/articles/PMC9038085/)
[16](http://arxiv.org/pdf/2302.06874.pdf)
[17](https://www.tandfonline.com/doi/full/10.1080/09540091.2022.2133082)
[18](http://repositori.uji.es/xmlui/bitstream/10234/190848/1/fernandez_2020_deep.pdf)
[19](https://www.mdpi.com/2079-9292/8/2/219/pdf)
[20](http://arxiv.org/pdf/2103.06383.pdf)
[21](https://arxiv.org/pdf/2209.01984.pdf)
[22](http://arxiv.org/pdf/2205.11720v1.pdf)
[23](https://arxiv.org/pdf/1905.00987.pdf)
[24](https://pmc.ncbi.nlm.nih.gov/articles/PMC12009717/)
[25](https://arxiv.org/html/2504.16335v1)
[26](https://peerj.com/articles/cs-3025/)
[27](https://arxiv.org/pdf/2211.13609.pdf)
[28](http://arxiv.org/pdf/2404.03176.pdf)
[29](https://arxiv.org/pdf/1710.05468.pdf)
[30](http://arxiv.org/pdf/2310.06182.pdf)
[31](https://arxiv.org/pdf/2203.09082.pdf)
[32](https://openreview.net/pdf?id=OaVi1yjdEc)
[33](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhu_LDMNet_Low_Dimensional_CVPR_2018_paper.pdf)
[34](https://arxiv.org/abs/1711.06246)
[35](https://research.google.com/pubs/archive/35536.pdf)
[36](https://proceedings.iclr.cc/paper_files/paper/2025/file/b3b2c656de99c016f02a7f6b476efaae-Paper-Conference.pdf)
[37](https://89douner.tistory.com/340)
[38](https://proceedings.neurips.cc/paper_files/paper/2024/file/bd8058b8580eb7f54dbacd8c8c1eb5ce-Paper-Conference.pdf)
[39](http://arxiv.org/pdf/2208.08631.pdf)
[40](https://arxiv.org/pdf/1705.07219.pdf)
[41](http://arxiv.org/pdf/2311.04055.pdf)
[42](http://arxiv.org/pdf/2402.13505.pdf)
[43](http://arxiv.org/pdf/1906.10343.pdf)
[44](http://arxiv.org/pdf/2306.01222.pdf)
[45](http://arxiv.org/pdf/2312.16892.pdf)
[46](https://arxiv.org/abs/1606.04586)
[47](https://www.ijcai.org/proceedings/2024/0580.pdf)
[48](https://arxiv.org/abs/2107.12521)
[49](https://www.sciencedirect.com/science/article/abs/pii/S0893608024001643)
[50](https://viso.ai/deep-learning/deep-belief-networks/)
[51](https://www.nature.com/articles/s41598-024-64205-2)
[52](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4d2aa4c034745f558bfea34643c8d6a6-Abstract-Conference.html)
[53](https://2024.sci-hub.box/6197/34ab60e63ffc3751a728df71d2634225/fischer2014.pdf)
[54](https://gram-blogposts.github.io/blog/2024/contrast-learning/)
[55](https://arxiv.org/abs/2403.10658)
