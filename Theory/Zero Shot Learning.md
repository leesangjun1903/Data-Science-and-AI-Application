# Zero-shot Learning

핵심 요약: Zero-Shot Learning은 학습 시 보지 못한 클래스에 대해, 텍스트 속성·설명 같은 의미 정보(semantic)를 매개로 추론하는 학습 패러다임입니다. 최근에는 멀티모달 사전학습(예: CLIP)과 프롬프트 기반 접근이 실전 적용을 견인하고 있습니다.[1][2][3]

## 무엇이 Zero-Shot Learning인가
Zero-Shot Learning(ZSL)은 학습 단계에 등장하지 않은 “미지의 클래스”를, 클래스 간 의미적 관계를 활용해 분류·추론하는 방법입니다. 핵심은 시각·언어 등 보조 정보(속성, 텍스트 정의, 관계 그래프)를 임베딩 공간에서 정렬(alignment)하여, 보지 못한 라벨로도 일반화하는 능력입니다. 데이터 레이블링 비용이 큰 도메인(의료, 보안, 특수 산업)에서 특히 유용하며, 새로운 클래스가 계속 생기는 환경에도 확장성이 좋습니다.[3]

## 왜 지금 중요한가
대규모 사전학습(Foundation Models)과 멀티모달 표현학습의 확산으로, “텍스트로 과제 정의 → 즉시 추론”이 가능해졌습니다. 예를 들어 CLIP은 이미지와 텍스트를 같은 임베딩 공간에 정렬해 “설명 문구만으로” 제로샷 분류가 가능합니다. 임상·산업 도메인에서도 프롬프트 기반 제로샷 접근이 레이블 부족 문제를 완화하며 빠른 실전 적용을 보입니다.[4][5][2][6]

## 기본 개념 정리
- 의미 공간(semantic space): 클래스 속성·텍스트 설명을 벡터로 임베딩하여, 보이는(Seen)·보이지 않는(Unseen) 클래스를 연결합니다.[3]
- 정렬(alignment): 이미지·텍스트 임베딩을 같은 공간으로 매핑해 코사인 유사도로 매칭합니다.[6]
- GZSL(Generalized ZSL): 테스트 시 Seen+Unseen이 함께 등장하는 현실적 설정으로, “Seen 편향” 완화를 위한 캘리브레이션이 중요합니다.[1]

### Semantic space
Semantic space는 자연어에서 단어와 개념의 의미를 수치적 벡터로 표현해 의미 관계를 포착하는 공간입니다.  
이는 단어들의 의미적 유사성이나 관련성을 계산하고 자연어 처리에서 어휘의 다양성과 모호성을 극복하는 데 사용됩니다.

처음에는 잠재 의미 분석(LSA)과 유사 언어 공간(HAL) 같은 기법으로 시작했으며, 최근에는 Word2Vec(구글), GloVe(스탠포드), fastText(페이스북) 같은 신경망 기반 임베딩 기술을 통해 발전했습니다.  
이 공간에서 각 단어는 고차원 벡터로 표현되고, 벡터들 간 거리나 각도로 의미적 관계를 분석합니다.

또한, 의미 공간 이론은 감정 연구에도 활용되며, 인간 감정을 다차원적으로 표현하는 데이터 기반 접근법으로 확장되고 있습니다.  
개념의 의미는 감각운동, 정서, 사회적 경험 등 다양한 차원으로 구성되어 있는데, 최신 연구에 따르면 내면 경험과 사회적 상호작용이 의미 공간 형성에서 중요한 역할을 한다고 밝혀졌습니다.

요약하면, 의미 공간은 자연어의 의미를 정량적으로 다루는 도구로, 자연어 처리와 인지과학에서 중요한 역할을 하는 개념입니다.

#### Latent Semantic Analysis, LSA
잠재 의미 분석(LSA)은 문서와 단어 간의 관계에서 숨겨진 의미(잠재 의미)를 수학적으로 추출하는 자연어 처리 기법입니다.  
문서-단어 행렬을 만들고 특이값 분해(SVD)를 통해 차원을 축소하여 문서나 단어 간의 유사성을 파악합니다.

LSA는 비슷한 문맥에서 함께 나타나는 단어들이 비슷한 의미를 가진다고 가정하며, 이를 활용해 문서들 간 의미적 관계를 분석하거나 정보 검색 등에 사용됩니다. 예를 들어, 인터넷, 소프트웨어, 혁신과 같은 관련 단어들의 연관성을 포착해 의미적 관계를 찾아냅니다.

핵심 처리 과정은 다음과 같습니다:

- 문서-단어 행렬 생성
- 특이값 분해(SVD)로 중요한 의미 토픽만 남기고 차원 축소
- 축소된 행렬에서 문서 간 코사인 유사도 계산으로 의미적 유사성 평가.

LSA는 1988년 정보 검색 맥락에서 특허를 받았고, Latent Semantic Indexing(LSI)라고도 불립니다.

#### HAL, Hyperspace Analog to Language
유사 언어 공간(HAL, Hyperspace Analog to Language)은 단어 간 공기 빈도(co-occurrence frequency)를 측정하여 단어 유사성을 벡터 공간 모델로 표현하는 방법입니다.  
일정 크기의 문맥 윈도우 내에서 단어들이 함께 등장하는 빈도를 기반으로 단어 간 관련성을 수치화한 행렬을 만들어, 이를 통해 단어들의 의미적 유사성을 분석할 수 있습니다.

좀 더 구체적으로, HAL은 문서에서 일정 크기의 윈도우(예: 10 단어 내)로 단어들의 동시 출현 빈도를 측정해 co-occurrence matrix을 구성하는데, 이 행렬은 단어 벡터를 형성하며 단어 간 유사도 계산에 활용됩니다. 이러한 방식은 심리 언어학 실험 결과와도 높은 상관관계를 가진다고 알려져 있습니다.

(Co-occurrence matrix는 주어진 문맥이나 윈도우 크기 내에서 요소들이 함께 등장하는 빈도를 수치화한 행렬입니다. 예를 들어, 단어들을 대상 윈도우 내에서 함께 나타난 횟수를 기록해 단어 간 관계를 표현합니다.)

요약하자면, HAL은 단순한 동시 출현 빈도 기반의 벡터 공간 모델로, 단어 의미 유사성을 정량화하는 대표적인 분포 기반(count-based) 자연어 처리 기법 중 하나입니다.

### GZSL(Generalized Zero-Shot Learning)
**GZSL(Generalized Zero-Shot Learning)** 은 학습 시 일부 클래스(보이지 않는 클래스)는 데이터가 없지만, 테스트 시에는 보이는 클래스와 보이지 않는 클래스 모두를 포함하여 분류하는 학습 기법입니다.  
즉, 기존 제로샷 학습(ZSL)이 보이지 않는 클래스만 예측하는 데 반해, GZSL은 현실 세계와 유사하게 보이는 클래스와 보이지 않는 클래스를 모두 다루는 문제를 다룹니다.

GZSL은 보이는 클래스의 학습 데이터와 보이지 않는 클래스의 의미 정보(예: 클래스 임베딩, 속성 벡터)를 활용해 두 클래스 간의 관계를 학습하며, 이를 통해 보이지 않는 클래스도 인식할 수 있게 됩니다.  
일반 ZSL과 비교해 현실 세계에 더 적합하고 도전적인 문제로 평가받고 있습니다.

요약하자면:

- ZSL은 테스트 시 오직 보이지 않는 클래스만 예측함.
- GZSL은 테스트 시 보이는 클래스와 보이지 않는 클래스 모두 예측함.
- GZSL은 의미적 정보(semantic embedding)를 통해 보이는 데이터로부터 보이지 않는 클래스를 예측 가능하도록 학습함.

이러한 접근은 실제 분류 작업의 범위를 넓히고, 기존 ZSL의 과적합 문제를 극복하면서 현실적 응용에 적합합니다.

## 대표 접근법 맵
- 속성 기반: 클래스-속성(예: “날개 있음”, “물에서 삶”) 벡터를 학습해 전이합니다.[3]
- 텍스트 정의 기반: 클래스 설명 문장을 임베딩하여 이미지와 비교합니다(언어모델/멀티모달).[2]
- 멀티모달 정렬: CLIP처럼 대규모 웹 이미지-텍스트 쌍으로 학습한 공통 공간에서 제로샷 추론을 수행합니다.[6]
- 프롬프트 기반 NLP: PLM에 템플릿을 설계해 레이블 없이 과제를 조건화합니다(HealthPrompt 등).[5]

## 응용 분야 한눈에
- 컴퓨터 비전: 제로샷 이미지 분류·세분화·검색 등에서 레이블 비용을 절감합니다.[7]
- 임상 NLP: 레이블 제약 환경에서 프롬프트 기반 제로샷 분석을 시도합니다.[5]
- 산업 현장: 도메인 특화 QA·검색·이상탐지에서 데이터 부족을 완화합니다.[4]

## 핵심 수학: 제로샷 점수화
CLIP류 멀티모달 정렬에서, 이미지 임베딩 $$v=f_{\text{img}}(x)$$와 텍스트 임베딩 $$t_c=f_{\text{text}}(\text{prompt}(c))$$를 정규화 후 코사인 유사도로 점수화합니다.[6]
- 유사도: $$\text{sim}(v,t_c)=\frac{v^\top t_c}{\|v\|\|t_c\|}$$ 입니다 [6].  
- 확률화: $$\displaystyle p(c\mid x)=\frac{\exp(\tau\,\text{sim}(v,t_c))}{\sum_{c'}\exp(\tau\,\text{sim}(v,t_{c'}))}$$ 로 소프트맥스 정규화합니다.[6]

## 연구 과제와 한계
- 비시각적/환각 텍스트 신호: LLM 기반 텍스트 설명이 비시각적 속성으로 흐르면 오분류가 늘 수 있습니다.[8]
- Seen 편향·캘리브레이션: GZSL에서 Unseen 리콜을 끌어올리면서 Seen 정확도를 유지하는 균형이 어렵습니다.[1]
- 도메인 전이: 웹 사전학습 → 의료·공업 데이터로 전이 시, 의미 불일치·분포 편차를 다룹니다.[7]

## 실전 가이드: CLIP 제로샷 분류
아래는 PyTorch와 OpenAI CLIP을 활용한 “연구 수준” 제로샷 분류 파이프라인입니다. 실무 팁(프롬프트 엔지니어링, 온도, 앙상블)까지 포함합니다.[9][6]

### 1) 환경 준비
- pip install git+https://github.com/openai/CLIP, torch, torchvision, Pillow를 준비합니다.[6]
- GPU 가용 시 cuda를 사용합니다.[6]

### 2) 기본 추론 코드
- 공개 예시는 CIFAR-100에 대해 제로샷 예측을 수행하는 간결한 레퍼런스입니다.[6]
- 핵심 절차: 모델 로드→전처리→텍스트 토큰화→임베딩→정규화→유사도 소프트맥스→Top-k 출력입니다.[6]

```python
import os
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)  # 다양한 백본 선택 가능입니다. [30]

# 1) 이미지 전처리
img = preprocess(Image.open("test.jpg")).unsqueeze(0).to(device)  # 표준화/리사이즈 포함입니다. [30]

# 2) 제로샷 라벨 문구 설계(프롬프트 엔지니어링)
labels = ["cat", "dog", "horse", "car"]
templates = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a close-up photo of a {}",
    "a cropped photo of a {}",
]
prompts = [t.format(c) for c in labels for t in templates]  # 다중 템플릿 앙상블입니다. [33]

# 3) 텍스트 임베딩
with torch.no_grad():
    text_tokens = clip.tokenize(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # 정규화입니다. [30]

# 4) 이미지 임베딩
with torch.no_grad():
    img_feats = model.encode_image(img)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # 정규화입니다. [30]

# 5) 템플릿 평균 앙상블
text_feats = text_feats.view(len(labels), len(templates), -1).mean(dim=1)  # 라벨별 평균입니다. [33]

# 6) 유사도 및 소프트맥스(온도 조절 가능)
logits = 100.0 * img_feats @ text_feats.t()  # 스케일 팩터(온도 역수)입니다. [30]
probs = logits.softmax(dim=-1).squeeze(0)    # 확률 변환입니다. [30]

# 7) 결과 출력
topk = min(5, len(labels))
values, indices = probs.topk(topk)
for v, i in zip(values.tolist(), indices.tolist()):
    print(f"{labels[i]}: {v*100:.2f}%")
```

- 템플릿 앙상블은 프롬프트 민감도를 낮추고 견고성을 올립니다.[9]
- 100.0 스케일은 CLIP 예시의 온도 계수 역할을 하며, 조정으로 샤프니스/칼리브레이션을 조절할 수 있습니다.[6]

### 3) 데이터셋·태스크로 확장
- 도메인 용어에 맞춘 라벨 문구를 세밀하게 작성하면 효과가 큽니다(예: “a chest X-ray showing …”).[9]
- OpenVINO 등 추론 최적화 도구를 활용해 경량·가속 배포가 가능합니다.[10]

## GZSL를 위한 캘리브레이션 팁
- 로짓 오프셋: Seen/Unseen에 상이한 바이어스를 적용해 균형을 맞춥니다.[1]
- 스코어 온도 스케일링: Unseen 감도를 올리되 과신(overconfidence)을 줄입니다.[1]
- 프롬프트 다변화: Unseen 라벨을 다양한 문구로 기술해 커버리지를 확장합니다.[9]

## NLP 제로샷: 프롬프트 패턴
- 템플릿 기반 분류: “문장: {x}. 질문: 이 문장은 {label-set} 중 무엇에 해당합니까?”처럼 과제를 언어화합니다.[5]
- 도메인 적응: 임상 용어를 포함한 템플릿과 설명(지시문)을 추가해 의미 정밀도를 높입니다.[5]
- In-context 변형: 소수 예시를 추가하는 Few-shot으로 전환해 안정성을 개선할 수 있습니다.[11]

## 멀티모달·세분화 확장
- 세분화·비디오 객체·3D에서 ZSL/FSL 결합 연구가 활발하며, 의미 공간 정렬과 데이터 분할 정의가 중요합니다.[7]
- 데이터 제약 하의 생성 모델(GAN/확산)도 ZSL/FSL 설정을 다루는 연구가 증가하고 있습니다.[12]

## 연구 트렌드와 고급 주제
- 해석가능 ZSL: LLM이 생성한 클래스 개념을 필터링/선정해 전이성과 판별성을 높이는 접근이 제안됩니다.[8]
- 도메인 특화 QA/검색: 제로샷 인컨텍스트 절차로 외부 문서 증거를 결합하는 파이프라인이 보고됩니다.[4]
- 프롬프트 설계 지침: 제로샷 프롬프트의 구조·토큰화 전략·연속 프롬프트 등 실전 지침이 정리되고 있습니다.[11]

## 체크리스트: 실무 적용 전 점검
- 라벨 문구 품질: “시각적으로 판별 가능한” 텍스트 표현인가를 검토합니다.[9]
- 분포 차이: 소스(웹)와 타깃(의료·산업) 간 도메인 갭을 인지합니다.[7]
- 캘리브레이션: GZSL 기준(Seen/Unseen 동시)에서 밸런스 메트릭을 모니터링합니다.[1]

## 참고 코드·리소스
- OpenAI CLIP 공식 저장소: 제로샷 예제가 포함되어 있습니다.[6]
- 튜토리얼(응용): 프롬프트 엔지니어링과 벡터 유사도 실습을 단계별로 다룹니다.[9]
- OpenVINO 노트북: 배포·최적화 흐름을 재현할 수 있습니다.[10]

이 글의 출발점이 된 용어 설명은 링크의 정의와 동일한 맥락을 공유합니다. 의미 정보로 보지 못한 클래스를 일반화한다는 점, 그리고 멀티모달·프롬프트 기반 확장이 현재 실전의 핵심이라는 점을 명확히 이해하면 좋습니다.[2][3][1]

[1](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1488)
[2](https://encord.com/blog/zero-shot-learning-explained/)
[3](https://www.ibm.com/think/topics/zero-shot-learning)
[4](https://onepetro.org/SPEDC/proceedings/24DC/24DC/D011S002R004/542912)
[5](https://arxiv.org/abs/2203.05061)
[6](https://github.com/openai/CLIP)
[7](https://ieeexplore.ieee.org/document/10084414/)
[8](https://arxiv.org/abs/2505.03361)
[9](https://www.pinecone.io/learn/series/image-search/zero-shot-image-classification-clip/)
[10](https://docs.openvino.ai/2024/notebooks/clip-zero-shot-classification-with-output.html)
[11](https://arxiv.org/abs/2309.13205)
[12](https://arxiv.org/abs/2307.14397)
[13](https://encord.com/glossary/zero-shot-learning-definition/)
[14](https://mrforum.com/product/9781644903513-23)
[15](https://academic.oup.com/eurpub/article/doi/10.1093/eurpub/ckae144.1121/7843734)
[16](https://arxiv.org/abs/2402.11142)
[17](https://onepetro.org/JPT/article/77/01/92/620338/Zero-Shot-Learning-With-Large-Language-Models)
[18](https://www.mdpi.com/2218-6581/13/7/109)
[19](https://arxiv.org/pdf/2010.13320.pdf)
[20](https://arxiv.org/pdf/2011.08641.pdf)
[21](http://arxiv.org/pdf/2103.10741.pdf)
[22](https://arxiv.org/pdf/2301.00998.pdf)
[23](https://arxiv.org/pdf/1604.07093.pdf)
[24](https://arxiv.org/pdf/2406.03032.pdf)
[25](https://arxiv.org/pdf/2207.03824.pdf)
[26](https://arxiv.org/pdf/2203.15310.pdf)
[27](http://arxiv.org/pdf/2404.09640.pdf)
[28](http://arxiv.org/pdf/2410.20215.pdf)
[29](https://redfield.ai/zero-shot-learning/)
[30](https://dataforest.ai/glossary/zero-shot-learning)
[31](https://arxiv.org/html/2505.09188v1)
[32](https://hostman.com/tutorials/zero-shot-image-classification-using-openai-clip/)
[33](https://www.komtas.com/en/glossary/zero-shot-learning-nedir)
[34](https://telnyx.com/learn-ai/zero-shot-learning)
[35](https://arxiv.org/abs/2111.09794)
[36](https://aiola.ai/glossary/zero-shot-learning/)
[37](https://blog.roboflow.com/zero-shot-learning-computer-vision/)
[38](https://arxiv.org/abs/2312.04997)
[39](https://www.kaggle.com/code/aisuko/zero-shot-image-classification-with-clip)
[40](https://arxiv.org/pdf/1712.09300.pdf)

https://encord.com/glossary/zero-shot-learning-definition/
