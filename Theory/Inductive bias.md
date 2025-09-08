# Inductive bias 가이드

Inductive bias는 보지 못한 데이터의 출력을 예측하기 위해 학습 알고리즘이 미리 가지고 있는 **추가 가정들의 집합**입니다. 이 글은 개념을 선명하게 정리하고, CNN·RNN·GNN·Transformer의 사례와 연구 맥락까지 연결하여 실전적으로 이해하도록 구성하였습니다.[1][2][3][4][5][6]

## 왜 필요한가
현실의 데이터로는 가능한 가설이 무한히 많기 때문에, 아무 가정도 없다면 일반화는 원리적으로 불가능합니다. Inductive bias는 “단순한 가설을 선호한다” 같은 규범(예: 오컴의 면도날)이나, 아키텍처 제약, 정규화, 사전분포처럼 모델 선택을 유도하는 원리로 구현됩니다. 덕분에 학습 알고리즘은 같은 훈련 성능을 내는 여러 해 중 하나를 택해, 보지 못한 입력에 대해 일관된 출력을 내릴 수 있습니다.[2][4][1]

## 한 줄 정의
Inductive bias는 학습 데이터 밖의 사례에 대해 추론할 때 모델이 의존하는 **선호·제약·사전지식의 묶음**입니다. 바꿔 말하면, 같은 데이터라도 어떤 해를 우선시하도록 만드는 알고리즘의 “편향”입니다.[1][2]

## 이론적 관점
- 탐색 관점: 가설공간 탐색의 우선순위를 정해 특정 패턴을 먼저 찾게 만듭니다.[1]
- 베이지안 관점: 사전분포 $$p(\omega)$$로 표현되어 사후 $$p(\omega\mid X,Y)$$를 통해 일반화를 유도합니다.[4]
- 정규화/아키텍처 관점: 가중치 감쇠, 희소성, 합성곱·그래프 구조 같은 설계 자체가 편향을 구현합니다.[4][1]

## Bias–Variance와의 연결
Bias가 높으면 단순 가정을 강하게 밀어 과소적합이 쉬워지고, Variance가 높으면 데이터 잡음에 민감해 과적합 위험이 큽니다. Inductive bias는 이 균형을 조절하는 레버로 작동하며, 문제 구조에 맞을수록 일반화 성능이 좋아집니다.[2][1]

## 대표적 예시
- 선형회귀: 선형 관계를 선호한다는 가정이 내재되어 있습니다.[7]
- 결정트리: 더 짧고 단순한 트리를 선호하는 경향으로 일반화를 유도합니다.[7]
- 베이지안 모델: 사전의 선택 자체가 편향을 명시합니다.[4]

## CNN의 편향: 국소성과 변위 등가성
CNN은 인접 픽셀 간 상관이 중요하다는 Locality와, 위치가 달라도 같은 패턴이면 같은 의미라는 Translation equivariance/ invariance를 구조적으로 갖습니다. 합성곱과 가중치 공유로 이러한 편향을 구현해, 이미지처럼 국소 구조가 중요한 데이터에서 강점을 보입니다. 이 편향 덕분에 파라미터 효율과 데이터 효율이 높아지고, 작은 변화에 견고한 표현을 학습합니다.[5][8]

## RNN의 편향: 순차성과 시간 불변성
RNN은 입력이 순서 구조를 가진다는 Sequential 가정과, 같은 순서의 패턴이면 같은 출력을 낸다는 Temporal invariance를 전제로 설계됩니다. 이로 인해 시계열·문장처럼 순서 의존성이 핵심인 데이터에서 일반화가 쉬워집니다.[9]

## GNN의 편향: 순열 불변/등가성과 관계성
GNN은 노드 순서와 무관하게 같은 그래프 구조면 같은 출력을 내야 한다는 Permutation invariance/equivariance를 핵심 편향으로 삼습니다. 메시지 패싱과 그래프 집계는 “엔티티–관계–규칙”을 구성적으로 다루게 하여, 조합적 일반화 능력을 높입니다.[3][6]

## Transformer와 편향의 약함
Transformer는 전역 Self-Attention으로 모든 토큰 쌍을 연결하되, CNN에 비해 locality·translation 등 내장 편향이 약합니다. 위치 정보도 임베딩 학습에 크게 의존하므로, 구조적 가정은 적고 데이터로부터 더 많이 학습합니다. 그 결과 글로벌 문맥이 중요한 과제나 대규모 데이터에서는 매우 강력하지만, 편향이 필요한 소데이터·국소구조 과제에서는 CNN이 더 효율적일 수 있습니다.[10][11][5]

## 연구 관점: Relational Inductive Bias
Battaglia et al.는 “엔티티(노드), 관계(엣지), 구성 규칙”을 명시적으로 다루는 편향이 조합적 일반화의 열쇠라 주장하며, Graph Network 프레임워크를 제시했습니다. 이 관점은 딥러닝이 관계 추론을 더 잘 하도록 아키텍처 수준의 편향을 제공하고, 해석 가능성과 유연성을 함께 추구합니다.[6][3]

## 언제 어떤 편향이 유리한가
- 지역 구조가 강한 이미지 인식: CNN의 **Locality**와 **Translation** 편향이 유리합니다.[5]
- 순서가 본질인 언어/시계열: RNN류의 **Sequential**·**Temporal** 편향이 적합합니다.[9]
- 관계/상호작용 중심의 과제: GNN의 **Permutation** 편향과 메시지 패싱이 적합합니다.[3]
- 전역 문맥·대규모 데이터: 편향이 약한 Transformer가 더 큰 표현력을 발휘합니다.[10]

## 실전 팁
- 데이터 크기와 구조 먼저: 데이터가 작고 구조가 뚜렷하면 강한 편향(CNN/GNN)을, 데이터가 크고 패턴이 다양하면 약한 편향(Transformer)도 고려합니다.[10][5]
- 편향 주입의 스펙트럼: 아키텍처(합성곱/그래프/어텐션), 정규화(가중치 감쇠·희소성), 데이터 증강(변위·회전), 사전학습(프리어) 등으로 선택적으로 주입합니다.[5][4]
- 하이브리드 설계: CNN 백본 + Transformer 헤드, CNN 특징으로 ViT 시퀀스를 구성하는 식으로 상보적 편향을 결합합니다.[10]

## 연구 예시 아이디어
- 저데이터 이미지 분류: 강한 Locality 편향(CNN) 기반에 소량의 Self-Attention을 상위 레벨에만 추가하여 전역 문맥을 보강합니다.[5][10]
- 상호작용 예측(물리/분자): GNN로 엔티티–관계 구조를 모델링하고, 그래프 레벨 어텐션으로 장거리 상호작용을 강화합니다.[6][3]
- 시계열 원인 탐색: 순열 불변 집계(GNN)와 시간 어텐션(Transformer)을 결합해 관계성과 시간적 인과를 동시에 포착합니다.[3][10]

## 한 페이지 요약
- 정의: “보지 못한 입력을 다룰 때 모델이 의존하는 가정들의 묶음”입니다.[2][1]
- 구현: 사전분포, 정규화, 아키텍처 제약, 데이터 증강 등으로 구현됩니다.[1][4]
- 사례: CNN(Locality/Translation), RNN(Sequential/Temporal), GNN(Permutation), Transformer(약한 내장 편향)입니다.[9][3][10][5]
- 실전: 데이터 구조·크기에 맞춰 편향의 강도를 조절하고, 필요하면 하이브리드로 조합합니다.[10][5]

## 참고/더 읽을거리
- Wikipedia: 인덕티브 바이어스의 정통적 정의와 동기.[1]
- Battaglia et al., Relational inductive biases, deep learning, and graph networks: 관계 편향과 그래프 네트워크 프레임워크.[6][3]
- CNN의 Locality/Translation 강의 슬라이드: 이미지 도메인 편향의 직관적 설명.[5]
- ViT 리뷰: Transformer의 약한 내장 편향과 데이터 의존성 논의.[10]
- 입문형 가이드: 다양한 예시와 용어 정리.[7][2]

[1](https://en.wikipedia.org/wiki/Inductive_bias)
[2](https://www.geeksforgeeks.org/machine-learning/what-is-inductive-bias-in-machine-learning/)
[3](https://arxiv.org/abs/1806.01261)
[4](https://aifrenz.github.io/present_file/Inductive%20biases,%20graph%20neural%20networks,%20attention%20and%20relational%20inference.pdf)
[5](https://vds.sogang.ac.kr/wp-content/uploads/2024/01/2024-%EB%8F%99%EA%B3%84%EC%84%B8%EB%AF%B8%EB%82%98_%EC%9C%A0%ED%98%84%EC%9A%B0.pdf)
[6](https://research.google/pubs/relational-inductive-biases-deep-learning-and-graph-networks/)
[7](https://www.appliedaicourse.com/blog/inductive-bias-in-machine-learning/)
[8](https://velog.io/@qtly_u/ViT%EC%9D%98-Inductive-Bias%EA%B0%80-%EB%8F%84%EB%8C%80%EC%B2%B4-%EC%96%B4%EB%96%BB%EB%8B%A4%EB%8A%94-%EA%B1%B0%EC%95%BC)
[9](https://ga02-ailab.tistory.com/150)
[10](https://kubig-2023-1.tistory.com/30)
[11](https://yooniverse1007.tistory.com/4)
[12](https://re-code-cord.tistory.com/entry/Inductive-Bias%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C)
[13](https://velog.io/@euisuk-chung/Inductive-Bias%EB%9E%80)
[14](https://www.tencentcloud.com/techpedia/100066)
[15](https://www.reddit.com/r/MLQuestions/comments/egof3l/explanation_of_inductive_bias/)
[16](https://www.semanticscholar.org/paper/Relational-inductive-biases,-deep-learning,-and-Battaglia-Hamrick/3a58efcc4558727cc5c131c44923635da4524f33)
[17](https://robot-vision-develop-story.tistory.com/29)
[18](https://www.mlgdansk.pl/wp-content/uploads/2018/11/MLGdansk53_29.10.2018_RobertRozanski_DeepLearningAndGraphNetworks.pdf)
[19](https://www.baeldung.com/cs/ml-inductive-bias)
[20](https://indico.ijclab.in2p3.fr/event/5551/attachments/13969/17293/2019.11.02-lal-relational-inductive-bias.pdf)

# Reference
https://re-code-cord.tistory.com/entry/Inductive-Bias%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C
