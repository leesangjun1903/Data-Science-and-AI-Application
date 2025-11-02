# SHAP : A Unified Approach to Interpreting Model Predictions

# 핵심 요약 및 주요 기여

**주요 주장:** 복잡한 모델(ensemble, 딥러닝)의 예측을 해석하기 위한 다양한 기법들이 존재하지만, 이들 사이의 관계 및 선호 기준이 명확하지 않다. 이를 해결하기 위해 SHAP(SHapley Additive exPlanations)라는 통합 프레임워크를 제시한다.

**주요 기여:**  
1. **Additive Feature Attribution Methods**라는 새로운 기법 클래스를 정의하고, 기존 6가지 해석 기법(LIME, DeepLIFT, Layer‐Wise Relevance Propagation, Shapley Regression, Shapley Sampling, Quantitative Input Influence)을 이 클래스 내에서 통합.  
2. 게임이론의 샤플리 값(Shapley value)에 기반해 **local accuracy, missingness, consistency**라는 세 가지 성질을 만족하는 유일한 해석값이 있음을 증명.  
3. 기존 방법의 한계를 극복하는 **Kernel SHAP**(모델 무관, 샘플 효율 개선)과 **Deep SHAP**(딥러닝 모델 대상, 계층적 근사를 통한 계산 속도 향상)이라는 새로운 근사 기법 제안 및 사용자 연구와 실험을 통해 직관적 일관성 및 샘플 효율성에서 기존 기법 대비 우수성 입증.  

# 상세 설명

## 해결하려는 문제  
- **정확도 vs. 해석 가능성**: 대규모 데이터에서 최고 성능을 내는 모델은 복잡해 일반 사용자가 예측 근거를 파악하기 어려움.  
- **기법 간 상관 관계 및 신뢰성 결여**: LIME, DeepLIFT 등 다양한 해석 기법이 제안되었으나, 서로 어떻게 연관되고 각 기법이 어떠한 성질(accuracy, consistency 등)을 보장하는지 불명확.

## 제안 방법

1. **설명 모델 정의**  
   - 원본 모델 $$f$$의 단일 예측 $$f(x)$$를 설명하는 선형 해석 모델 $$g(z)$$를 도입.  
   - 이진 벡터 $$z\in\{0,1\}^M$$를 통해 단순화된 입력으로 변환하고,  

$$
       g(z) = \phi_0 + \sum_{i=1}^M \phi_i z_i
     $$
     
  으로 정형화(Equation 1).

2. **샤플리 값 특성**  
   - **Local Accuracy**: $$f(x)=g(x')$$  
   - **Missingness**: $$z_i=0$$인 경우 $$\phi_i=0$$  
   - **Consistency**: 기여도가 증가한 특성의 $$\phi_i$$는 감소하지 않음  
   - 이 세 성질을 모두 만족하는 해석값은 **단 하나**, 바로 **Shapley value**임을 증명(Equation 8).

3. **SHAP 값 정의**  
   - 각 특성 기여도를  

$$
       \phi_i = \sum_{S\subseteq N\setminus\{i\}} \frac{|S|!(M-|S|-1)!}{M!}\bigl[E[f(z_{S\cup\{i\}})] - E[f(z_S)]\bigr]
     $$
     
  으로 정의. 여기서 $$E[f(z_S)]$$는 특성 집합 $$S$$를 고정했을 때의 조건부 기댓값(Equation 12).

4. **근사 기법**  
   - **Kernel SHAP**: 가중 선형 회귀를 통해 샘플 수를 줄여 SHAP 값을 추정(Equation Theorem 2).  
   - **Deep SHAP**: DeepLIFT의 계층적 전파 방식을 샤플리 근사로 재정의하여 딥 네트워크에 적용(Equations 13–16).  

## 모델 구조 및 수식  
- **설명 모델**: $$g(z)=\phi_0+\sum_i\phi_i z_i$$  
- **클래식 샤플리**: $$\displaystyle \phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!(M-|S|-1)!}{M!}\bigl[f_{S\cup\{i\}}(x)-f_S(x)\bigr]$$  
- **조건부 기대**: $$f_{S\cup\{i\}}(x)\approx E[f(z_{S\cup\{i\}})]$$  
- **Kernel SHAP 가중치**:  

$$
    \pi(z)=\frac{M-1}{\binom{M}{|z|}\,|z|\,(M-|z|)}
  $$

## 성능 향상 및 사용자 연구  
- **샘플 효율성**: Kernel SHAP이 LIME, Shapley sampling 대비 동일 정확도에 필요한 모델 평가 횟수를 크게 절감.  
- **직관 일관성**: Amazon Mechanical Turk 실험에서 SHAP 값이 인간 직관과 가장 높은 상관을 보임.  
- **딥 모델 적용**: Deep SHAP이 오리지널 DeepLIFT보다 복합함수(예: max pooling)에서 더 정확한 기여도 할당을 달성.

## 한계  
- **계산 복잡도**: 변수 수 $$M$$가 커지면 정확한 샤플리 계산 비용이 급증. 근사에도 비용 부담이 존재.  
- **특성 독립성 가정**: 조건부 기대 근사 시 독립성 가정을 사용하면 실제 상관된 특성에서 설명 왜곡 가능.  
- **모델 비선형성 제한**: 일부 근사(Linear SHAP)는 모델 선형성 가정이 필요해 비선형 모델에 부정확.

# 일반화 성능 향상 관련  
SHAP 자체는 모델 학습 성능을 직접 개선하지 않지만, **특성 중요도 기반 피처 선택**이나 **데이터 편향 탐지**에 활용되어 모델의 **일반화 능력**을 향상시킬 여지가 있다. 특히, 일관성(consistency) 특성 덕분에 다양한 데이터 분포 변화에 강건한 중요도 평가가 가능하여 오버피팅 감소와 도메인 일반화 연구에 기여할 수 있다.

# 향후 연구 영향 및 고려 사항  
SHAP 프레임워크는 해석 기법 통합과 원칙 확립을 통해 추후 연구 기반을 제공한다.  
- **상호작용 효과 확장**: 특성 간 상호작용을 설명하는 SHAP Interaction Values 연구.  
- **모델 특화 근사 고도화**: CNN, 트리 기반 모델 등에 대한 더욱 효율적인 근사 알고리즘 개발.  
- **불확실성 정량화**: SHAP 값의 신뢰구간 제시를 통한 해석 신뢰도 평가.  
- **데이터 의존성 완화**: 조건부 기대 시 독립성 가정 완화를 위한 밀도 추정 기법 통합.  

이와 같은 방향은 해석 가능성뿐 아니라, 해석 기반 특성 공학(feature engineering)과 피처 선택에도 중요한 통찰을 제공할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/be8f6122-a535-427d-a252-912007ef187e/NIPS-2017-a-unified-approach-to-interpreting-model-predictions-Paper.pdf)


25. Shapley Value와 SHAP에 대해서 알아보자 with Python :
https://zephyrus1111.tistory.com/271
