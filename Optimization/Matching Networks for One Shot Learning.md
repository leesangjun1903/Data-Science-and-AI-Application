# Matching Networks for One Shot Learning | Image classification, One shot learning, Meta-Learning

## 1. 논문 핵심 주장과 주요 기여

### 핵심 주장

**Matching Networks**는 인간과 같이 극소량의 데이터(single example)로부터 새로운 개념을 학습할 수 있는 one-shot learning 시스템을 제안합니다[1]. 논문의 핵심 주장은 **"test와 train 조건을 일치시켜야 한다"**는 간단한 머신러닝 원칙에 기반하여, episodic training을 통해 실제 테스트 환경과 동일한 조건에서 모델을 학습시켜야 한다는 것입니다[2][3].

```
###  1. One-shot learning과 Zero-shot learning의 차이점

### One-shot learning
**한 개의 예시**로부터 새로운 클래스를 학습하는 방법입니다[1][2]. 예를 들어:
- 고양이 한 장의 사진만 보고도 다른 고양이 사진들을 구별할 수 있게 하는 것
- 얼굴 인식에서 한 사람의 사진 한 장만으로 그 사람을 인식하는 기술[1]

### Zero-shot learning  
**예시 없이** 새로운 클래스를 인식하는 방법입니다[1][3]. 특징:
- 학습 과정에서 본 적 없는 새로운 클래스를 **이전에 학습된 개념과의 관계**를 통해 인식[1]
- 예: 소와 말을 학습한 모델이 "얼룩말은 소와 말의 특징을 합친 것"이라는 설명만으로 얼룩말을 인식[4]

**핵심 차이**: One-shot은 **최소한의 예시**(1개)가 필요하지만, zero-shot은 **예시 없이** 의미적 관계만으로 학습합니다[5].
```

### 주요 기여

1. **Non-parametric 접근법**: Parametric 모델의 느린 학습 속도와 Non-parametric 모델의 metric 의존성 문제를 동시에 해결하는 새로운 접근법을 제시[1][2]

```
## 4. Non-parametric 접근법 상세 설명

### Parametric vs Non-parametric 차이

**Parametric 모델**[9][10]:
- 고정된 수의 파라미터
- 데이터 분포에 대한 가정 필요
- 예: Linear regression, Neural networks

**Non-parametric 모델**[9][10]:
- 파라미터 수가 데이터 크기에 따라 변함
- 분포 가정 없음
- 예: K-NN, Decision Trees

### Matching Networks의 Non-parametric 특성

1. **동적 분류기**: Support set 크기에 따라 분류기가 변화[6]
   
   Support set이 5개 → 5-way classifier
   Support set이 10개 → 10-way classifier
   

2. **메모리 기반**: 외부 메모리(Support set)에 의존[6]
   - 새로운 클래스 추가 시 fine-tuning 불필요
   - Support set만 교체하면 즉시 새로운 분류 가능

3. **연상 메모리**: Query와 가장 유사한 Support 예시를 "가리키는" 방식[6]

### 장점
- **빠른 적응**: 새로운 클래스에 즉시 적응
- **Catastrophic forgetting 방지**: 이전 지식을 잃지 않음[6]
- **유연성**: Support set 변경만으로 다양한 태스크 수행
```

2. **Set-to-set 프레임워크**: One-shot learning 문제를 set-to-set 프레임워크로 재정의하여, support set에서 분류기로의 매핑을 학습[4]

3. **Episodic Training 도입**: 전체 데이터셋으로 학습하는 대신, 테스트와 동일한 few-shot 환경에서 episode 단위로 학습하는 새로운 훈련 전략[3]

```
## 3. Episodic Training과 동일한 조건

### Episodic Training의 개념
**"학습하는 방법을 학습"**하는 메타러닝 방식입니다[7][8]. 

### 작동 원리

Episode 생성 과정:
1. Task T에서 레이블 세트 L 샘플링 (예: {고양이, 개})
2. L에서 Support set S와 Batch B 샘플링
3. S를 조건으로 B의 레이블 예측
4. 예측 오류로 모델 파라미터 업데이트


### 테스트와 동일한 조건
- **훈련 시**: 5-way 1-shot 에피소드로 학습
- **테스트 시**: 5-way 1-shot 상황에서 평가
- **핵심**: "train과 test 조건이 일치해야 한다"는 원칙[6]

### 구체적 예시

# 훈련 에피소드 예시
Support Set: [고양이1, 개1, 새1, 물고기1, 햄스터1]  # 5-way 1-shot
Query: 고양이2  # 예측해야 할 이미지
목표: Query가 고양이1과 유사함을 학습

```

4. **새로운 벤치마크 제공**: ImageNet과 Penn Treebank에서 one-shot learning 태스크를 정의하고 평가 기준 제시[1]

## 2. 해결하고자 하는 문제와 제안 방법

### 문제 정의

기존 딥러닝 시스템은 수백 또는 수천 개의 예시가 필요한 반면, 인간은 단 하나의 예시(예: 책의 기린 그림 한 장)만으로도 새로운 개념을 일반화할 수 있습니다[1]. 또한 표준 지도학습 패러다임은 적은 데이터로부터 새로운 개념을 빠르게 학습하는 데 만족할만한 해결책을 제공하지 못합니다[1].

### 제안 방법

#### 핵심 수식

모델의 예측 출력은 다음과 같이 정의됩니다:

$$ \hat{y} = \sum_{i=1}^{k} a(\hat{x}, x_i)y_i $$

여기서:
- $$\hat{x}$$: 테스트 이미지 (query)
- $$x_i, y_i$$: support set의 샘플과 레이블
- $$a(\cdot, \cdot)$$: attention mechanism[2][5]

#### Attention Kernel

Attention mechanism은 softmax와 cosine distance의 조합으로 구현됩니다:

$$ a(\hat{x}, x_i) = \frac{e^{c(f(\hat{x}), g(x_i))}}{\sum_{j=1}^{k} e^{c(f(\hat{x}), g(x_j))}} $$

여기서 $$f$$와 $$g$$는 각각 query와 support set을 위한 embedding 함수입니다[2].

```
## 5. Attention Mechanism의 장점과 적용

### Attention Mechanism의 장점

1. **선택적 집중**[11][12]: 
   - 중요한 정보에만 집중하여 계산 효율성 향상
   - 인간의 주의집중 방식을 모방

2. **정보 손실 방지**[13][14]:
   - 모든 입력 정보를 보존하면서 중요도에 따라 가중치 부여
   - 기존 압축 방식의 정보 손실 문제 해결

3. **장거리 의존성 해결**[13]:
   - 시퀀스의 모든 위치 정보에 직접 접근 가능

### CNN과 LSTM에서의 활용

**CNN에서의 Visual Attention**[11][15]:

# 이미지 캡셔닝 예시
1. CNN이 이미지를 n개 영역으로 분할
2. LSTM이 단어 생성 시 관련 영역에 attention
3. "고양이"라는 단어 생성 시 → 고양이가 있는 영역에 높은 가중치

**LSTM에서의 Sequence Attention**[11][13]:

# 기계번역 예시
Input: "나는 학생이다"
Output: "I am a student"
- "student" 생성 시 → "학생"에 높은 attention weight
- "am" 생성 시 → "는"에 높은 attention weight

**Matching Networks에서의 활용**:
- Query와 Support set 간의 **cosine similarity** 계산
- Softmax로 attention weights 정규화
- 가장 유사한 Support 예시에 높은 가중치 부여
```

#### Full Context Embeddings (FCE)

모델의 주요 혁신은 embedding 함수가 전체 support set의 맥락을 고려하도록 하는 것입니다:
- $$g(x_i, S)$$: support set $$S$$의 맥락에서 $$x_i$$를 embedding
- $$f(\hat{x}, S)$$: support set $$S$$를 고려하여 query $$\hat{x}$$를 embedding[1]

이는 bidirectional LSTM과 attention mechanism을 통해 구현됩니다[1].

```
## 6. Full Context Embeddings (FCE) 상세 설명

### Support Set의 정의
**Support Set**은 각 클래스의 소수 예시들로 구성된 참조 데이터입니다[16][17].

### FCE의 핵심 아이디어
기존의 "myopic" embedding을 벗어나 **전체 맥락을 고려한 embedding**을 생성[6][17].

### 구체적 작동 방식

#### 1. 일반적 Embedding (문제점)

# 개별적으로 embedding
cat_embedding = g(cat_image)  # 고양이만 고려
dog_embedding = g(dog_image)  # 개만 고려
# 문제: 다른 클래스와의 관계 정보 손실

#### 2. Full Context Embeddings (해결책)

# Support Set 전체를 고려한 embedding
S = [cat_image, dog_image, bird_image]  # Support Set

cat_embedding = g(cat_image, S)  # 전체 맥락에서 고양이 embedding
dog_embedding = g(dog_image, S)  # 전체 맥락에서 개 embedding

# 장점: 고양이가 개, 새와 어떻게 다른지 명시적으로 학습

### 실제 구현 방법

**Support Set Embedding (g function)**[17]:
1. **Bidirectional LSTM** 사용
2. 각 Support 이미지를 전체 세트의 맥락에서 인코딩
3. 순방향과 역방향 정보를 결합

**Query Embedding (f function)**[17]:
1. **LSTM with Read-Attention** 사용
2. Support Set에 대한 attention을 K번 수행
3. 각 step에서 가장 관련성 높은 Support 정보 선택

### 성능 향상 예시
- **Omniglot**: FCE 효과 미미 (단순한 태스크)
- **miniImageNet**: **약 2% 성능 향상**[6] (복잡한 태스크에서 효과적)

### 직관적 예시
Support Set: [스핑크스 고양이, 페르시안 고양이, 골든 리트리버]
Query: 새로운 고양이 품종

일반 embedding: 각각 독립적으로 특징 추출
FCE: "이 이미지는 개보다는 고양이들과 유사하다"는 맥락 정보 활용
→ 더 정확한 분류 가능
```

### 모델 구조

1. **Embedding Networks**: CNN 기반 feature extractor (VGG, Inception 등)
2. **Attention Mechanism**: Support set과 query 간의 유사도 계산
3. **LSTM with Attention**: Full context embedding을 위한 구조
4. **Non-parametric Classifier**: Support set 크기에 따라 동적으로 확장되는 분류기[1]

## 3. 성능 향상 및 한계

### 성능 향상

#### 실험 결과
- **Omniglot**: 5-way 1-shot에서 88.0% → 93.8% 성능 향상[1]
- **ImageNet**: 5-way 1-shot에서 87.6% → 93.2% 성능 향상[1]
- **miniImageNet**: FCE 적용 시 1-shot에서 46.6%, 5-shot에서 60.0% 달성[1]

```
## 2. One-shot learning 시스템의 학습 데이터

Matching Networks는 다음과 같은 데이터로 학습됩니다:

### 학습 데이터 구성
- **Base classes**: 대량의 라벨링된 데이터를 가진 기본 클래스들 (예: 1200개 문자 클래스)
- **Support set**: 각 에피소드에서 사용되는 소수의 예시들 (K-shot의 K개)
- **Query set**: 실제 테스트할 데이터

### 실제 데이터셋 예시
1. **Omniglot**: 50개 알파벳, 1623개 문자, 각 문자당 20개 샘플[6]
2. **ImageNet**: 1000개 클래스 중 일부를 제외하고 학습
3. **miniImageNet**: 100개 클래스 중 80개로 학습, 20개로 테스트[6]

핵심은 **테스트 시 등장할 클래스는 학습 시 보지 않았다**는 점입니다.
```

#### 성능 향상 요인
1. **Episodic Training**: 테스트와 동일한 조건에서 훈련하여 overfitting 방지[3]
2. **Full Context Embeddings**: Support set 전체 맥락을 고려한 embedding으로 약 2% 성능 향상[1]
3. **Non-parametric 특성**: 새로운 클래스에 대해 fine-tuning 없이 즉시 적응 가능[1]

### 한계점

1. **계산 복잡도**: Support set 크기가 증가할수록 각 gradient update의 계산 비용이 증가[1]

2. **Fine-grained Classification 어려움**: ImageNet dogs subtask에서 성능 저하 (1% 감소)를 보임. 이는 훈련 시 랜덤 분포에서 샘플링하지만 테스트 시 유사한 클래스들이 포함되는 경우 발생[1]

3. **Domain Gap 민감성**: 훈련 분포 $$T$$와 테스트 분포 $$T'$$ 간의 차이가 클수록 성능 저하[1][6]

```
## 7. Domain Gap 민감성과 구체적 예시

### Domain Gap의 정의
**훈련 분포 T**와 **테스트 분포 T'** 간의 차이가 클수록 모델 성능이 저하되는 현상[18][19].

### 논문의 구체적 사례

#### ImageNet Dogs Subtask 실패 사례[6]
훈련 설정:
- 훈련 분포: ImageNet의 랜덤한 클래스들 (다양한 카테고리)
- Support Set 샘플링: 완전히 랜덤한 클래스들

테스트 설정:
- 테스트 분포: 모두 개 품종 (fine-grained classification)
- 결과: 1% 성능 저하 발생

#### 성능 저하 원인
1. **훈련 시**: 완전히 다른 카테고리 (개, 자동차, 비행기, 과일...)
2. **테스트 시**: 유사한 카테고리 (골든 리트리버, 불독, 푸들...)

### 다양한 Domain Gap 예시

#### 1. 시각적 도메인 차이[18][19]
Source Domain: 낮에 촬영된 도로 이미지
Target Domain: 밤에 촬영된 도로 이미지
Gap: 조명, 색상, 대비의 차이

#### 2. 해상도/품질 차이
Source: 고해상도 전문 사진
Target: 저해상도 스마트폰 사진
Gap: 이미지 품질, 노이즈 수준

#### 3. 스타일 차이
Source: 실제 사진 (Real photos)
Target: 그림/일러스트 (Synthetic images)
Gap: 텍스처, 색감, 스타일

### 성능 저하 정도
- **유사한 분포**: 성능 저하 미미 (1-2%)
- **중간 차이**: 5-10% 성능 저하
- **큰 차이**: 20% 이상 성능 저하 가능[18]

### 해결 방안 (논문 제안)
Fine-grained 분류를 위해서는 **훈련 전략 개선** 필요:
- 랜덤 샘플링 대신 **유사한 클래스들로 구성된 Support Set**으로 훈련
- 예: 개 품종들끼리, 자동차 모델들끼리 그룹화하여 훈련
```

4. **언어 모델링 한계**: Penn Treebank 태스크에서 LSTM 언어모델 대비 상당한 성능 격차 (72.8% vs 38.2%)[1]

## 4. 일반화 성능 향상 가능성

### 일반화 메커니즘

1. **메타러닝 접근**: "학습하는 방법을 학습"하여 새로운 태스크에 빠른 적응[7][8]

## 8. Meta Learning인 이유

### Meta Learning의 정의
**"Learning to Learn"** - 학습하는 방법 자체를 학습하는 것[7][20].

```
### Matching Networks가 Meta Learning인 이유

#### 1. 이중 수준 최적화

# 일반적인 학습
θ* = argmin_θ Loss(θ, Training_Data)

# Meta Learning (Matching Networks)
θ* = argmax_θ E[E[log P_θ(y|x, S)]]
#          ↑     ↑
#      episodes  batches

#### 2. 학습 과정의 메타적 특성[8]

**Inner Loop (각 에피소드 내)**:
- Support Set S가 주어짐
- Query 데이터의 레이블 예측
- **분류 방법을 학습**

**Outer Loop (에피소드 간)**:
- 여러 에피소드에서의 성능을 종합
- **더 나은 학습 방법을 학습**

#### 3. 일반화된 학습 능력

기존 학습: 특정 클래스들을 구분하는 방법 학습
Meta Learning: 새로운 클래스들을 빠르게 구분하는 방법 학습

### 구체적 예시로 이해하기

#### 전통적인 학습
# 개와 고양이 분류기 학습
model = train(images=[dog1, dog2, cat1, cat2], labels=[0,0,1,1])
# 결과: 개와 고양이만 구분 가능

#### Meta Learning (Matching Networks)
# Episode 1: 개 vs 고양이 구분 학습
# Episode 2: 새 vs 물고기 구분 학습  
# Episode 3: 자동차 vs 비행기 구분 학습
# ...

# 결과: "두 클래스를 구분하는 일반적 방법" 학습
# → 새로운 클래스 쌍(예: 사과 vs 오렌지)도 빠르게 학습 가능

### Meta Learning의 핵심 통찰
1. **태스크 분포에서 학습**: 특정 태스크가 아닌 **태스크들의 패턴** 학습
2. **빠른 적응**: 새로운 태스크에 **몇 번의 예시만으로** 적응
3. **일반화**: 학습한 **메타 지식**을 다양한 새로운 상황에 적용

이러한 특성들로 인해 Matching Networks는 단순한 분류 모델이 아닌, **학습 자체를 학습하는 메타러닝 모델**로 분류됩니다[8][20].
```

2. **Non-parametric 메모리**: 외부 메모리를 활용한 연상 기억 방식으로 catastrophic forgetting 방지[1]

3. **Metric Learning 기반**: 판별적 분류기로서 충분한 alignment만으로도 분류 가능[1]

### 일반화 향상 전략

1. **훈련 분포 다양화**: Fine-grained 분류를 위해 유사한 클래스들로 구성된 support set으로의 훈련[1]

2. **Domain Adaptation**: 훈련과 테스트 도메인 간 격차 줄이기 위한 추가 연구 필요[1]

3. **Hybrid 접근**: Parametric 모델(LSTM-LM)과 non-parametric 구성요소의 결합 탐색[1]

## 5. 연구 영향과 향후 고려사항

### 연구 영향

#### 학술적 영향
1. **Few-shot Learning 분야 개척**: Meta-learning 연구의 핵심 논문으로 자리잡음[9][10]
2. **후속 연구 촉진**: Prototypical Networks, Relation Networks 등 metric-based 방법론들의 기반[11][6]
3. **Episodic Training 표준화**: Few-shot learning의 표준 훈련 방법으로 채택[3]

#### 산업 응용
1. **의료 영상**: 희귀 질병 진단에서 적용[12]
2. **결함 진단**: 제한된 데이터로 산업 장비 고장 진단[13][14]
3. **자율주행**: 새로운 시나리오에 빠른 적응[12]

### 향후 연구 고려사항

#### 기술적 개선 방향
1. **계산 효율성**: Sparse attention이나 sampling 기반 방법으로 계산 복잡도 해결[1]
2. **Multi-modal Learning**: 이미지-텍스트 결합 one-shot learning[12]
3. **Continual Learning**: 새로운 태스크 학습 시 기존 지식 보존[8]

#### 이론적 발전
1. **Generalization Bound**: Few-shot learning의 이론적 보장 연구[15]
2. **Domain Transfer**: 도메인 간 전이 성능 향상 방법[6]
3. **Interpretability**: 모델 결정 과정의 설명가능성 확보[16]

#### 응용 확장
1. **Zero-shot Learning**: 예시 없이도 작동하는 시스템으로 확장[17]
2. **Reinforcement Learning**: Few-shot 환경에서의 강화학습 적용[8]
3. **Natural Language Processing**: 언어 태스크에서의 성능 개선 필요[1]

이 논문은 few-shot learning과 meta-learning 분야의 출발점을 제공했으며, 현재까지도 이 분야 연구의 핵심 기반을 이루고 있습니다. 특히 "test와 train 조건 일치"라는 단순하지만 강력한 원칙은 현재의 대규모 foundation model 시대에도 여전히 유효한 통찰을 제공합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/317c8e93-186f-4908-b336-d3ddad914356/1606.04080v2.pdf
[2] https://aistudy9314.tistory.com/69
[3] https://deep-learning-study.tistory.com/941
[4] http://www.navisphere.net/6014/matching-networks-for-one-shot-learning/
[5] https://seunghan96.github.io/meta/study/study-(meta)(paper-2)Matching-Networks-for-One-Shot-Learning/
[6] http://dmqa.korea.ac.kr/activity/seminar/301
[7] https://huidea.tistory.com/252
[8] https://jjhdata.tistory.com/30
[9] https://www.semanticscholar.org/paper/be1bb4e4aa1fcf70281b4bd24d8cd31c04864bb6
[10] https://ieeexplore.ieee.org/document/8578527/
[11] https://justmajor.tistory.com/22
[12] https://duckyoh.tistory.com/entry/Few-shot-Learning-%EC%A0%9C%ED%95%9C%EB%90%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A1%9C-%ED%9A%A8%EA%B3%BC%EC%A0%81%EC%9D%B8-%ED%95%99%EC%8A%B5
[13] https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-023-01063-6
[14] https://royalsocietypublishing.org/doi/10.1098/rsos.230706
[15] https://www.themoonlight.io/ko/review/rethinking-the-key-factors-for-the-generalization-of-remote-sensing-stereo-matching-networks
[16] https://koreascience.kr/article/JAKO202514232406192.pdf
[17] https://gloomysky.tistory.com/5
[18] https://www.nature.com/articles/s41467-023-42981-1
[19] https://link.springer.com/10.1007/s11042-023-14386-1
[20] https://ieeexplore.ieee.org/document/10529130/
[21] https://ieeexplore.ieee.org/document/9151263/
[22] https://arxiv.org/abs/2305.09552
[23] https://ieeexplore.ieee.org/document/8953585/
[24] https://arxiv.org/pdf/1606.04080.pdf
[25] https://arxiv.org/abs/1804.08281
[26] https://www.aclweb.org/anthology/D18-1223.pdf
[27] https://arxiv.org/pdf/1612.02192.pdf
[28] http://arxiv.org/pdf/2405.13178.pdf
[29] https://arxiv.org/abs/1606.05233
[30] https://arxiv.org/pdf/1906.00820.pdf
[31] http://arxiv.org/pdf/1804.07275.pdf
[32] https://arxiv.org/abs/1712.01867
[33] https://pmc.ncbi.nlm.nih.gov/articles/PMC4224099/
[34] https://scienceon.kisti.re.kr/srch/selectPORSrchTrend.do?cn=GTB2017002752
[35] https://www.youtube.com/watch?v=SW0cgNZ9eZ4
[36] https://www.aitimes.kr/news/articleView.html?idxno=24919
[37] https://repository.kisti.re.kr/bitstream/10580/15547/1/(%EA%B8%B0%EC%88%A0)%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%82%B9%EC%9D%84%20%EC%9C%84%ED%95%9C%20AI%20%EC%97%B0%EA%B5%AC%20%EB%8F%99%ED%96%A5.pdf
[38] https://cs-coffee-story.tistory.com/71
[39] https://www.mk.co.kr/news/culture/9322843
[40] https://velog.io/@gunny1254/Few-shot-learning-basic
[41] https://www.kist.re.kr/_attach/kist/file/2023/12/DMhWPIheMMuchsvsPDEEtEEbSg.pdf
[42] https://familia-89.tistory.com/8
[43] https://www.ksam.co.kr/p_base.php?action=story_base_view&no=2761&s_category=_3_11_
[44] https://bigwaveai.tistory.com/18
[45] https://news.skhynix.co.kr/decode-ai-2/
[46] http://koreascience.or.kr/journal/view.jsp?kj=GCSHCI&py=2014&vnc=v39Bn1&sp=8
[47] https://arxiv.org/html/2503.05122v1
[48] https://www.mdpi.com/1424-8220/24/9/2689/pdf?version=1713939685
[49] https://arxiv.org/html/2312.08704v2
[50] https://pmc.ncbi.nlm.nih.gov/articles/PMC11045109/
[51] https://www.mdpi.com/2076-3417/12/7/3416/pdf
[52] http://arxiv.org/pdf/2406.13573.pdf
[53] http://arxiv.org/pdf/2405.18872.pdf
[54] https://arxiv.org/abs/2109.10380
[55] https://pmc.ncbi.nlm.nih.gov/articles/PMC9270151/
[56] https://arxiv.org/html/2502.06825v2
[57] http://arxiv.org/pdf/2406.05959.pdf
[58] https://arxiv.org/pdf/2302.02533.pdf
[59] https://arxiv.org/abs/2501.14945
[60] http://arxiv.org/pdf/2408.16871.pdf
[61] http://arxiv.org/pdf/2201.06621.pdf
[62] http://arxiv.org/pdf/1502.05760.pdf
[63] https://www.mdpi.com/2313-433X/11/5/164
[64] https://arxiv.org/pdf/2201.10945.pdf
[65] http://arxiv.org/pdf/2310.12515.pdf
[66] https://proceedings.neurips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf
[67] https://supkoon.tistory.com/27
[68] https://s-space.snu.ac.kr/bitstream/10371/188745/1/000000172759.pdf
[69] https://kicj.re.kr/boardDownload.es?bid=0007&list_no=11807&seq=1
[70] https://lsjsj92.tistory.com/640
[71] https://turingpost.co.kr/p/topic-42-meta-learning
[72] https://koreascience.kr/article/JAKO201809258121099.pdf
[73] https://www.manuscriptlink.com/society/kips/conference/ask2023/file/downloadSoConfManuscript/abs/KIPS_C2023A0240F
[74] https://www.kiep.go.kr/gallery.es?mid=a10103050100&bid=0001&tag=&b_list=10&act=view&list_no=2439&nPage=45&vlist_no_npage=0&keyField=&orderby=
[75] https://dsba.snu.ac.kr/seminar/?mod=document&uid=63
[76] https://devbasket.tistory.com/71
[77] https://www.codil.or.kr/filebank/original/RK/OTKCRK200197/OTKCRK200197.pdf
[78] https://greeksharifa.github.io/computer%20vision/2019/04/17/Dual-Attention-Networks/
[79] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09874525


# QnA Reference

[1] https://cartinoe5930.tistory.com/entry/Zero-shot-One-shot-Few-shot-Learning%EC%9D%B4-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C
[2] https://dodonam.tistory.com/452
[3] https://madang-ai.tistory.com/2
[4] https://ds-jungsoo.tistory.com/20
[5] https://velog.io/@euisuk-chung/%EC%83%9D%EC%84%B1-AI%EC%9D%98-%ED%95%99%EC%8A%B5-%EB%B0%A9%EC%8B%9D-%EC%A0%9C%EB%A1%9C%EC%83%B7%EC%9B%90%EC%83%B7%ED%93%A8%EC%83%B7-%EB%9F%AC%EB%8B%9D
[6] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/317c8e93-186f-4908-b336-d3ddad914356/1606.04080v2.pdf
[7] https://huidea.tistory.com/252
[8] http://dmqa.korea.ac.kr/activity/seminar/301
[9] https://process-mining.tistory.com/131
[10] https://newsight.tistory.com/40
[11] https://sbspace.tistory.com/entry/Attention-Mechanism
[12] https://www.ibm.com/kr-ko/think/topics/attention-mechanism
[13] https://jomuljomul.tistory.com/entry/Deep-Learning-Attention-Mechanism-%EC%96%B4%ED%85%90%EC%85%98
[14] https://bigdaheta.tistory.com/67
[15] https://ys-cs17.tistory.com/46
[16] https://velog.io/@gunny1254/Few-shot-learning-basic
[17] https://talkingaboutme.tistory.com/entry/DL-Meta-Learning-Learning-to-Learn-Fast
[18] http://dmqa.korea.ac.kr/activity/seminar/448
[19] https://jkimst.org/upload/pdf/KIMST-2021-24-4-382.pdf
[20] https://blog.outta.ai/55
[21] https://ieeexplore.ieee.org/document/11051380/
[22] https://ieeexplore.ieee.org/document/10417178/
[23] https://ieeexplore.ieee.org/document/10802181/
[24] https://dl.acm.org/doi/10.1145/3626253.3635400
[25] https://www.ijcai.org/proceedings/2023/123
[26] https://ieeexplore.ieee.org/document/10657218/
[27] https://arxiv.org/abs/2502.06150
[28] https://academic.oup.com/bib/article/doi/10.1093/bib/bbae354/7739674
[29] https://arxiv.org/abs/2305.12477
[30] https://ieeexplore.ieee.org/document/9763640/
[31] http://arxiv.org/pdf/2103.10741.pdf
[32] https://arxiv.org/pdf/2209.14935.pdf
[33] https://arxiv.org/pdf/1712.05972.pdf
[34] https://arxiv.org/pdf/2102.11856.pdf
[35] https://arxiv.org/pdf/2301.00998.pdf
[36] https://aclanthology.org/2022.emnlp-main.474.pdf
[37] https://arxiv.org/pdf/2402.01264.pdf
[38] http://arxiv.org/pdf/1711.06025v2.pdf
[39] https://arxiv.org/html/2306.16623v1
[40] https://arxiv.org/abs/2401.15657
[41] https://danden.tistory.com/95
[42] https://jieun121070.github.io/posts/Meta-Learning/
[43] https://velog.io/@mmodestaa/parametric-vs.-non-parametric-model
[44] https://velog.io/@yun_haaaa/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-2-parametric-model-and-a-non-parametric-model
[45] https://rahites.tistory.com/371
[46] https://www.nepirity.com/blog/parametric-and-nonparametric-machine-learning-algorith/
[47] https://velog.io/@yjkim0520/Few-Shot-Learning-One-shot-Zero-shot-
[48] https://ineed-coffee.github.io/posts/Parametric-VS.-Non-Parametric-method/
[49] https://rimiyeyo.tistory.com/entry/%EB%8B%A4%EC%96%91%ED%95%9C-%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81Prompt-Engineering%EC%97%90-%EB%8C%80%ED%95%B4-%EC%82%B4%ED%8E%B4%EB%B3%B4%EC%9E%901-Zero-shot-One-shot-Few-shot-CoT
[50] https://blog.si-analytics.ai/3
[51] https://sh-avid-learner.tistory.com/114
[52] https://alpha.velog.io/@sjinu/Few-Shot-Learning
[53] https://www.mdpi.com/2072-4292/16/13/2474
[54] https://www.mdpi.com/2079-9292/13/7/1229
[55] https://ieeexplore.ieee.org/document/10109957/
[56] https://linkinghub.elsevier.com/retrieve/pii/S0360544224033127
[57] https://linkinghub.elsevier.com/retrieve/pii/S1674862X24000247
[58] https://linkinghub.elsevier.com/retrieve/pii/S100107422400175X
[59] https://www.frontiersin.org/articles/10.3389/fnins.2024.1379495/full
[60] https://onlinelibrary.wiley.com/doi/10.1002/cjce.25318
[61] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13181/3031564/Dual-stream-CNNLSTM-model-based-on-attention-mechanism/10.1117/12.3031564.full
[62] https://ieeexplore.ieee.org/document/10839982/
[63] https://arxiv.org/pdf/1709.04696.pdf
[64] https://onlinelibrary.wiley.com/doi/10.1155/2021/6627588
[65] http://arxiv.org/pdf/2206.02203.pdf
[66] https://www.mdpi.com/2076-3417/11/24/12019/pdf?version=1639724203
[67] https://onlinelibrary.wiley.com/doi/10.1002/eng2.70163
[68] https://www.frontiersin.org/articles/10.3389/fphys.2021.700655/pdf
[69] http://arxiv.org/pdf/1808.05578.pdf
[70] https://pmc.ncbi.nlm.nih.gov/articles/PMC9097568/
[71] https://arxiv.org/pdf/2104.06934.pdf
[72] https://pmc.ncbi.nlm.nih.gov/articles/PMC11679432/
[73] https://velog.io/@mingqook/A-Survey-on-Contextual-Embeddings
[74] https://aws.amazon.com/ko/blogs/tech/deploying-embedding-model-on-sagemaker-endpoint-for-genai/
[75] https://www.kibme.org/resources/journal/20220819151239539.pdf
[76] https://velog.io/@rlaehghks5/Data-Distributional-Properties-Drive-Emergent-In-Context-Learning-in-Transformers
[77] http://dmqa.korea.ac.kr/activity/seminar/438
[78] https://discuss.pytorch.kr/t/the-full-guide-to-embeddings-in-machine-learning/1708
[79] https://velog.io/@hsbc/sim2real-gap-ideation
[80] https://glee1228.tistory.com/3
[81] https://cloud.google.com/alloydb/docs/ai/migrate-data-from-langchain-vector-stores-to-alloydb?hl=ko
[82] https://velog.io/@hsbc/sim2real-gap-overview
[83] https://wikidocs.net/22893
[84] https://www.couchbase.com/blog/ko/twitter-thread-tldr-with-ai-part-2/
