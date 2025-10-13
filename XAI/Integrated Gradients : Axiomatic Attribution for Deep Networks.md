# Integrated Gradients : Axiomatic Attribution for Deep Networks

## **핵심 주장과 주요 기여**

이 논문은 딥 네트워크의 예측을 입력 특징에 **귀속(attribution)**시키는 문제를 해결하기 위해 **공리적 접근법(axiomatic approach)**을 제시합니다[1]. **핵심 주장**은 기존의 귀속 방법들이 두 가지 기본 공리인 **민감성(Sensitivity)**과 **구현 불변성(Implementation Invariance)**을 만족하지 않는다는 것이며, 이를 해결하기 위해 **통합 기울기(Integrated Gradients)** 방법을 제안합니다[1][2].

**주요 기여**는 다음과 같습니다:
- **공리적 프레임워크** 확립: 귀속 방법이 만족해야 할 근본적 원리들을 명확히 정의[1]
- **기존 방법들의 한계** 증명: DeepLift, LRP, Deconvolutional networks 등이 핵심 공리를 위반함을 수학적으로 증명[1]
- **Integrated Gradients 방법** 개발: 공리들을 만족하면서 구현이 간단한 새로운 귀속 방법 제안[1][3]
- **이론적 유일성** 증명: 대칭성 보존을 포함한 여러 공리를 동시에 만족하는 유일한 방법임을 증명[1]

## **해결하고자 하는 문제**

논문이 해결하려는 **핵심 문제**는 딥 네트워크의 **"블랙박스" 특성**으로 인한 해석 가능성 부족입니다[1][4]. 구체적으로:

1. **기존 귀속 방법들의 결함**: 대부분의 기존 방법들이 기본적인 공리를 위반[1]
2. **실증적 평가의 한계**: 모델의 오작동과 귀속 방법의 오작동을 구분하기 어려움[1]
3. **일관성 없는 해석**: 같은 기능을 하는 다른 네트워크 구현에 대해 다른 해석을 제공[1]

## **제안하는 방법 (수식 포함)**

### **통합 기울기(Integrated Gradients) 공식**

기준점 x'에서 입력 x까지의 직선 경로를 따라 기울기를 적분하여 계산합니다[1]:

$$ IntegratedGrads_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha \times (x - x'))}{\partial x_i} d\alpha $$

여기서:
- F: 딥 네트워크 함수
- x: 입력
- x': 기준점 (예: 검은 이미지)
- α: 경로 매개변수 (0에서 1까지)

### **실제 구현을 위한 근사 공식**

적분을 유한 합으로 근사하여 계산합니다[1]:

$$ IntegratedGrads^{approx}\_i(x) = (x_i - x'_i) \times \sum\_{k=1}^m \frac{\partial F(x' + \frac{k}{m} \times (x - x'))}{\partial x_i} \times \frac{1}{m} $$

여기서 m은 적분 근사를 위한 단계 수입니다[1].

### **핵심 공리들**

1. **민감성(Sensitivity)**: 입력과 기준점이 한 특징에서만 다르고 예측이 다르면, 그 특징은 0이 아닌 귀속값을 가져야 함[1]
2. **구현 불변성(Implementation Invariance)**: 기능적으로 동등한 네트워크들은 동일한 귀속값을 가져야 함[1]
3. **완전성(Completeness)**: 모든 특징의 귀속값의 합은 출력과 기준점 출력의 차이와 같아야 함[1]

## **모델 구조**

Integrated Gradients는 **모델에 구애받지 않는(model-agnostic)** 방법으로, 기존 네트워크 구조를 수정할 필요가 없습니다[1][3]. 단지 **표준 기울기 연산**을 여러 번 호출하여 구현됩니다[1].

### **적용된 모델들**
- **이미지 모델**: GoogleNet을 이용한 객체 인식[1]
- **텍스트 모델**: 질문 분류, 신경 기계 번역[1]
- **화학 모델**: 분자 그래프 컨볼루션 네트워크[1]
- **의료 모델**: 당뇨성 망막병증 예측[1]

## **성능 향상**

### **정성적 개선**
- **더 명확한 시각화**: 기존 기울기 방법보다 입력 이미지의 구별되는 특징을 더 잘 반영[1]
- **의료 응용**: 망막 병변의 경계에 집중하여 의학적으로 의미 있는 해석 제공[1]
- **규칙 추출**: 자연어 처리에서 새로운 트리거 구문 식별[1]

### **이론적 보장**
- **공리 만족**: 기존 방법들과 달리 모든 핵심 공리를 만족[1][2]
- **유일성**: 대칭성 보존을 포함한 공리들을 만족하는 유일한 경로 방법[1]

## **모델의 일반화 성능 향상 가능성**

### **디버깅을 통한 성능 개선**
- **네트워크 결함 발견**: 화학 모델에서 원자 특징의 잘못된 처리 발견[1]
- **편향 탐지**: 질문 분류에서 바람직하지 않은 상관관계 식별[1]
- **특징 중요도 분석**: 모델이 학습한 패턴의 올바름 검증[1]

### **모델 개선 방향 제시**
귀속 분석을 통해 모델이 **올바른 특징에 집중하는지** 확인할 수 있어, 훈련 데이터나 모델 구조 개선에 활용 가능합니다[1][5]. 최근 연구들은 해석 가능성 기반 모델 개선이 **실제 성능 향상**으로 이어질 수 있음을 보여줍니다[5].

## **한계**

1. **계산 비용**: 20-300번의 기울기 계산이 필요하여 단일 기울기 계산보다 비쌈[1]
2. **기준점 선택**: 적절한 기준점 선택이 중요하지만 항상 명확하지 않음[1]
3. **직선 경로 가정**: 다른 경로가 더 적절할 수 있으나 직선 경로만 사용[1]
4. **노이즈 문제**: 일부 연구에서 통합 과정에서 발생하는 노이즈 문제 지적[6]

## **앞으로의 연구에 미치는 영향**

### **연구 방향 제시**
1. **공리적 접근법의 확산**: XAI 분야에서 공리 기반 방법론이 표준이 됨[2][7]
2. **평가 방법론 개선**: 해석 방법의 객관적 평가를 위한 새로운 메트릭 개발[8][9]
3. **다중 경로 방법**: 단일 경로의 한계를 극복하기 위한 무작위 경로 샘플링 방법[6]

### **응용 분야 확장**
- **의료 AI**: 신뢰할 수 있는 진단 시스템 개발[10]
- **금융**: 위험 관리를 고려한 해석 방법[7]
- **자연어 처리**: 텍스트 분류의 신뢰성 평가[8]
- **분자 과학**: 약물 발견에서의 해석 가능한 예측[11]

## **앞으로 연구 시 고려할 점**

### **방법론적 고려사항**
1. **일관성 있는 평가**: 다양한 설정에서의 **메타 평가 프레임워크** 필요[12]
2. **계산 효율성**: 대규모 모델에 적용 가능한 **경량화된 버전** 개발
3. **적대적 강건성**: 해석 방법 자체에 대한 **적대적 공격** 대응[13]

### **이론적 발전 방향**
1. **새로운 공리 탐색**: 도메인 특화된 공리 개발[7]
2. **전역적 해석**: 개별 예측뿐만 아니라 **모델 전체의 행동** 이해
3. **인과관계 분석**: 단순한 상관관계를 넘어선 **인과적 해석** 방법 개발[10]

이 논문은 **XAI 분야의 기초를 확립**하고 **이론적으로 보장된 해석 방법**을 제공함으로써, 신뢰할 수 있는 AI 시스템 개발에 중요한 기여를 했으며, 향후 연구의 방향을 제시하는 **핵심 참고문헌**으로 자리잡았습니다[2][14][4].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4f044a9e-d936-43fa-8d25-f5ee557f9e78/1703.01365v2.pdf
[2] https://arxiv.org/abs/2306.13753
[3] https://captum.ai/docs/extension/integrated_gradients
[4] https://arxiv.org/pdf/2012.14261.pdf
[5] https://iopscience.iop.org/article/10.1088/2632-2153/ace0a1
[6] https://ieeexplore.ieee.org/document/10377784/
[7] https://ieeexplore.ieee.org/document/10772752/
[8] https://ieeexplore.ieee.org/document/10543008/
[9] https://www.ijcai.org/proceedings/2024/0059.pdf
[10] https://ieeexplore.ieee.org/document/10385466/
[11] https://link.springer.com/10.1007/s10994-023-06369-y
[12] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08871.pdf
[13] https://ieeexplore.ieee.org/document/10935253/
[14] https://arxiv.org/abs/1703.01365
[15] https://www.sec.gov/Archives/edgar/data/2064860/0002064860-25-000001-index.htm
[16] https://www.sec.gov/Archives/edgar/data/2064254/0002062351-25-000007-index.htm
[17] https://www.sec.gov/Archives/edgar/data/1914365/0001914365-25-000002-index.htm
[18] https://www.sec.gov/Archives/edgar/data/1763597/0001763597-24-000002-index.htm
[19] https://www.sec.gov/Archives/edgar/data/1675149/000095017025024242/aa-20241231.htm
[20] https://www.sec.gov/Archives/edgar/data/1725057/000172505725000064/day-20250313.htm
[21] https://docs.seldon.io/projects/alibi/en/latest/methods/IntegratedGradients.html
[22] https://openaccess.thecvf.com/content/CVPR2021/papers/Kapishnikov_Guided_Integrated_Gradients_An_Adaptive_Path_Method_for_Removing_Noise_CVPR_2021_paper.pdf
[23] https://www.nature.com/articles/s41540-023-00310-8
[24] https://www.sec.gov/Archives/edgar/data/2064397/0002064397-25-000001-index.htm
[25] https://www.sec.gov/Archives/edgar/data/1527728/000173112225000765/e6599_10-q.htm
[26] https://www.sec.gov/Archives/edgar/data/803578/000143774925002056/wavd20250122_s1a.htm
[27] https://www.sec.gov/Archives/edgar/data/803578/000143774925000977/wavd20241216_s1a.htm
[28] https://www.sec.gov/Archives/edgar/data/1925873/0001925873-24-000003-index.htm
[29] https://www.sec.gov/Archives/edgar/data/1527728/000173112225000258/e6372_10q.htm
[30] https://pubs.acs.org/doi/10.1021/jacs.3c07513
[31] https://www.nature.com/articles/s42256-022-00592-3
[32] https://journals.agh.edu.pl/csci/article/view/4551
[33] https://arxiv.org/abs/2206.13983
[34] https://www.xcally.com/news/interpretability-vs-explainability-understanding-the-importance-in-artificial-intelligence/
[35] https://arxiv.org/html/2412.18036v1
[36] https://www.ibm.com/think/topics/deep-learning
[37] https://www.splunk.com/en_us/blog/learn/explainability-vs-interpretability.html
[38] https://openaccess.thecvf.com/content/CVPR2022/papers/Rao_Towards_Better_Understanding_Attribution_Methods_CVPR_2022_paper.pdf
[39] https://www.coursera.org/articles/deep-learning-models
[40] https://ikramchraibik.com/2021/04/28/interpretabilite-vs-explicabilite-comprendre-vs-expliquer-son-reseau-de-neurones-1-3/
[41] https://www.sec.gov/Archives/edgar/data/1914365/0001914365-23-000002-index.htm
[42] https://link.aps.org/doi/10.1103/PhysRevE.110.054310
[43] https://arxiv.org/abs/2409.01610
[44] https://arc.aiaa.org/doi/10.2514/6.2024-4148
[45] https://ieeexplore.ieee.org/document/10796298/
[46] https://jjdeeplearning.tistory.com/17
[47] https://www.sec.gov/Archives/edgar/data/1829512/0001829512-24-000001-index.htm
[48] https://ieeexplore.ieee.org/document/9892567/
[49] https://onlinelibrary.wiley.com/doi/10.1111/1556-4029.14978
[50] https://ieeexplore.ieee.org/document/9576746/
[51] https://ieeexplore.ieee.org/document/9149804/
[52] https://www.sciencedirect.com/science/article/abs/pii/S0925231224009755
[53] https://woogi-tech.tistory.com/entry/MLDL-Explainability-vs-Interpretability-in-AI

# Reference
- https://captum.ai/docs/extension/integrated_gradients
