# Opportunities and Challenges in Explainable Artificial Intelligence (XAI): A Survey

**핵심 주장 및 주요 기여 요약**  
이 논문은 딥러닝 모델의 ‘블랙박스’ 특성으로 인한 신뢰·공정성·윤리 문제를 해결하기 위해 XAI(Explainable AI) 기법을 체계적으로 분류·정리하고, 수학적 개요 및 평가 전략을 제공한다. 주요 기여는 다음과 같다.[1]
- XAI 기법을 **Scope**(국소·전역), **Methodology**(백프로파게이션·퍼터베이션), **Usage**(내재적·사후적) 세 축으로 체계적 분류.[1]
- 대표적인 XAI 알고리즘에 대한 **수학적 정리** 및 **타임라인**(2007–2020) 제시.[1]
- 8가지 XAI 알고리즘을 이미지 데이터에 적용해 설명맵을 비교·평가하고, 한계 및 향후 방향 제안.[1]

***

## 문제 정의  
딥신경망은 수많은 파라미터로 구성되어 의사결정 과정을 이해하기 어려워, 의료·자율주행·법률 등 **미션 크리티컬** 분야에서 신뢰·투명성·공정성 문제가 제기된다. GDPR 등 규제 속에서 “왜 이런 예측을 했는가?”에 대한 인간 친화적 설명이 필수적이다.[1]

***

## 제안 방법  
### 1. XAI 기법의 분류  
- **Scope**: 국소(Local)·전역(Global)  
- **Methodology**: 백프로파게이션(Back­Prop)·퍼터베이션(Perturbation)  
- **Usage**: 내재적(Intrinsic)·사후적(Post-hoc)  
세 축 교차로 평가하며, 각 알고리즘의 특징을 매트릭스로 정리.[1]

### 2. 수학적 개요  
- **국소 설명(Local)**  

- *Integrated Gradients*:  

$$ \mathrm{IG}_j(x, x') \;=\; (x_j - x'_j)\int_{0}^{1} \frac{\partial f\big(x' + \alpha(x - x')\big)}{\partial x_j}\,d\alpha $$  
  
- *LIME*:  

$$ g^* = \arg\min_{g\in G} L\big(f,g,x\big) + \Omega(g) $$  

- **전역 설명(Global)**  
  - *SP-LIME(Submodular Pick)*: 각 인스턴스 설명 후, 비중복성 최대화 예산 B만큼 대표 인스턴스 선택  
  - *TCAV(Concept Activation Vectors)*: 개념 방향 v에 대한 방향 도함수  

$$ S_{C,k,j}(x) = \lim_{\epsilon\to0} \frac{h_{j,k}\big(z + \epsilon v_j^C\big) - h_{j,k}(z)}{\epsilon} $$  

### 3. 모델 구조  
논문은 특정 모델 아키텍처 제안 대신, XAI 알고리즘을 **모델 내부**(예: NAM) 및 **사후적**(예: LIME, SHAP)으로 구분하여 폭넓은 적용성 보장.[1]
- *NAM(Neural Additive Model)*: 각 특성별 신경망 $$f_i(x_i)$$ 합산 후 링크 함수 적용.[1]

### 4. 성능 향상 및 한계  
- **성능 향상**:  
  - XAI 기법을 통해 설명맵을 **정성·정량** 평가함으로써 모델 신뢰도·투명도 증가  
  - NAM 등 내재적 모델로 해석력을 유지하면서 예측 성능 확보  
- **한계**:  
  - 설명맵의 **정확도·안정성** 부족(입력 소량 변형 시 해석맵 왜곡)  
  - **정량 평가 지표** 미비, 인간 편향 개입 위험  
  - 개념 기반 방법(TCAV)에서 개념 선택·상관관계 편향 문제.[1]

***

## 일반화 성능 향상 가능성  
- **설명 기반 규제(Attribution Prior)**: Integrated Gradients에 도메인 지식 반영하여 학습 과정에서 해석력·일반화 동시 개선  
- **개념 기반 강화**: ACE, CaCE를 활용해 **상위 개념**에서 모델 취약점 발견 및 보완  
- **스펙트럴·흐름 분석**: SpRAy, Global Attribution Mapping으로 설명맵 클러스터링해 다양한 입력 분포 대응력 강화  

***

## 향후 연구 영향 및 고려사항  
향후 XAI 연구는 **정량적 평가 프레임워크** 구축에 집중해야 하며, **사람-모델 상호작용**을 고려한 인간 중심 평가가 중요하다. 또한, **개념 자동 발굴**과 **도메인 제약 통합**을 통해 설명력과 일반화 성능을 동시에 개선하는 하이브리드 접근이 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ddf5143-755e-4b60-bf0c-d7190f2710f6/2006.11371v2.pdf)

XAI survey paper review

https://velog.io/@tobigs_xai/3%EC%A3%BC%EC%B0%A8-XAI-survey-paper-review-2

