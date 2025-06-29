# Transformer 해석가능성: 주의집중 시각화를 넘어선 새로운 접근법

본 연구는 Transformer 모델의 해석가능성 분야에서 기존 주의집중 맵 기반 방법들의 한계를 극복하고, 딥 테일러 분해(Deep Taylor Decomposition) 원리에 기반한 새로운 관련성 전파 방법을 제안한 중요한 연구이다[1]. 저자들은 기존 방법들이 클래스 무관적이고 부분적인 정보만을 제공하는 문제를 해결하여, 보다 정확하고 클래스 특화된 시각화를 구현했다는 점에서 상당한 기여를 보여준다.

## 연구 동기 및 해결하고자 하는 문제

Transformer 모델이 자연어처리와 컴퓨터 비전 분야에서 최첨단 성능을 달성하면서, 이들 모델의 의사결정 과정을 이해하는 것이 중요한 과제로 대두되었다[1]. 기존의 해석가능성 방법들은 여러 근본적인 한계를 가지고 있었다. 첫째, 주의집중 점수만을 고려하는 방식은 Transformer의 다른 구성요소들을 무시한다는 문제가 있다[1]. 둘째, 기존 LRP(Layer-wise Relevance Propagation) 방법의 부분적 적용은 입력까지 관련성을 전파하지 못해 제한적인 정보만을 제공한다[1]. 셋째, 대부분의 방법들이 클래스 무관적이어서 다중 객체가 포함된 이미지에서 특정 클래스에 대한 구별된 설명을 제공하지 못한다[1].

더 심각한 문제는 Transformer 모델의 구조적 특성에서 발생한다. ReLU가 아닌 GELU와 같은 활성화 함수의 사용, 빈번한 스킵 연결, 그리고 자가 주의집중에서의 행렬 곱셈 연산은 기존 CNN용 해석가능성 방법들의 직접적인 적용을 어렵게 만든다[1]. 특히 스킵 연결에서 발생하는 수치적 불안정성과 행렬 곱셈에서의 보존 법칙 위반은 해결해야 할 핵심 기술적 과제였다.

## 제안하는 방법론 및 핵심 기술

### 관련성 및 기울기 전파

저자들은 딥 테일러 분해 원리를 기반으로 한 새로운 관련성 전파 방법을 제안한다. 기본적인 관련성 전파는 다음과 같이 정의된다[1]:

$$R_j^{(n)} = G(X, Y, R^{(n-1)}) = \sum_i \frac{X_j \frac{\partial L_i^{(n)}(X,Y)}{\partial X_j} R_i^{(n-1)}}{L_i^{(n)}(X,Y)}$$

ReLU가 아닌 활성화 함수를 다루기 위해, 양의 가중 관련성만을 고려하는 수정된 전파 규칙을 도입한다[1]:

$$R_j^{(n)} = G_q(x, w, q, R^{(n-1)}) = \sum_{\{i|(i,j) \in q\}} \frac{x_j w_{ji}}{\sum_{\{j'|(j',i) \in q\}} x_{j'} w_{j'i}} R_i^{(n-1)}$$

여기서 $$q = \{(i,j)|x_j w_{ji} \geq 0\}$$는 양의 기여도를 가진 요소들의 집합이다.

### 비모수적 관련성 전파 및 정규화

Transformer의 스킵 연결과 행렬 곱셈 연산을 위한 이중 텐서 처리 방법을 제안한다. 두 텐서 $$u$$와 $$v$$에 대해[1]:

$$R_j^{u(n)} = G(u, v, R^{(n-1)}), \quad R_k^{v(n)} = G(v, u, R^{(n-1)})$$

수치적 불안정성과 보존 법칙 위반 문제를 해결하기 위해 정규화 기법을 도입한다[1]:

$$\bar{R}_j^{u(n)} = R_j^{u(n)} \frac{|\sum_j R_j^{u(n)}|}{|\sum_j R_j^{u(n)}| + |\sum_k R_k^{v(n)}|} \cdot \frac{\sum_i R_i^{(n-1)}}{\sum_j R_j^{u(n)}}$$

### 관련성과 기울기 확산

최종 출력은 가중된 주의집중 관련성을 통해 계산된다[1]:

$$\bar{A}^{(b)} = I + E_h(\nabla A^{(b)} \odot R^{(n_b)})_+$$

$$C = \bar{A}^{(1)} \cdot \bar{A}^{(2)} \cdot ... \cdot \bar{A}^{(B)}$$

여기서 $$\odot$$는 아다마르 곱, $$E_h$$는 헤드 차원에서의 평균, $$(·)_+$$는 양수 부분만을 취하는 연산이다.

## 실험 결과 및 성능 향상

### 정량적 성능 평가

제안된 방법은 여러 벤치마크에서 기존 방법들을 크게 상회하는 성능을 보였다. ImageNet 검증 세트에서의 섭동 테스트에서, 음성 섭동에서는 54.16% AUC를 달성하여 기존 최고 성능인 53.1%를 초과했다[1]. 양성 섭동에서는 17.03% AUC로 가장 우수한 성능을 보였다[1]. 

ImageNet-Segmentation 데이터셋에서의 분할 성능에서는 픽셀 정확도 79.70%, mAP 86.03%, mIoU 61.95%를 달성하여 모든 메트릭에서 기존 방법들을 상당한 차이로 앞섰다[1]. 특히 mIoU에서는 rollout 방법의 55.42%보다 6.53% 포인트 향상된 결과를 보였다.

자연어처리 영역에서는 Movie Reviews 데이터셋의 rationale 추출 과제에서 지속적으로 최고 성능을 달성했다[1]. 토큰 F1 점수에서 모든 토큰 수 범위(10-80)에서 기존 방법들을 초과하는 성능을 보였다.

### 클래스 특화 시각화 능력

제안된 방법의 가장 중요한 기여 중 하나는 클래스 특화 시각화 능력이다. 다중 객체 이미지에서 서로 다른 클래스에 대해 구별되는 시각화를 생성할 수 있음을 실험을 통해 입증했다[1]. 예를 들어, 개와 고양이가 함께 있는 이미지에서 각각의 클래스에 대해 서로 다른 관련 영역을 정확히 강조하는 것을 확인할 수 있었다.

## 모델 일반화 성능 향상 가능성

### 해석가능성과 일반화의 연관성

본 연구에서 제안된 방법은 여러 측면에서 모델의 일반화 성능 향상에 기여할 수 있는 잠재력을 보인다. 첫째, 정확한 관련성 할당을 통해 모델이 실제로 의미 있는 특징에 집중하고 있는지 검증할 수 있어, 과적합이나 편향을 조기에 발견할 수 있다[1]. 둘째, 클래스 특화 시각화를 통해 각 클래스별 결정 경계를 더 명확히 이해할 수 있어, 모델의 결정 과정을 개선하는 데 활용할 수 있다.

실험에서 나타난 segmentation과 perturbation 테스트에서의 우수한 성능은 제안된 방법이 모델의 진정한 의사결정 과정을 더 정확히 반영함을 시사한다[1]. 이는 모델 디버깅과 개선 과정에서 더 신뢰할 수 있는 피드백을 제공할 수 있음을 의미한다.

### 약지도 학습에서의 활용 가능성

정확한 관련성 맵은 약지도 의미 분할과 같은 하위 과제에서 활용될 수 있다. 연구에서 언급된 바와 같이, GradCAM과 같은 클래스 특화 방법들이 이미 약지도 학습에 활용되고 있는 점을 고려할 때[1], 더 정확한 관련성 정보를 제공하는 본 방법은 이러한 응용 분야에서 더 나은 성능을 달성할 수 있을 것으로 예상된다.

## 연구의 한계점

### 계산 복잡도 문제

제안된 방법은 각 레이어에서 기울기와 관련성을 모두 계산해야 하므로 기존의 단순한 주의집중 시각화 방법보다 계산 비용이 높다[1]. 특히 대규모 모델이나 긴 시퀀스를 다룰 때 이러한 계산 부담이 실용성을 제한할 수 있다.

### 이론적 가정의 제약

딥 테일러 분해와 관련성 전파의 이론적 기반은 특정 가정들에 의존한다[1]. 실제 모델의 복잡한 비선형성이 이러한 가정들을 위반할 경우, 방법의 해석가능성과 신뢰성에 영향을 줄 수 있다. 또한 정규화 기법이 도입되면서 원래의 보존 법칙이 근사적으로만 유지되는 점도 고려해야 할 한계이다.

### 평가 메트릭의 한계

현재 사용된 평가 메트릭들이 해석가능성의 모든 측면을 포괄하지 못한다는 점도 한계로 지적될 수 있다. 특히 인간의 직관과 일치하는 정도나 실제 의사결정 과정과의 일치도를 정량적으로 평가하는 것은 여전히 도전적인 과제이다[1].

## 미래 연구에 미치는 영향 및 고려사항

### 해석가능성 연구의 새로운 방향

본 연구는 Transformer 해석가능성 분야에서 여러 중요한 방향을 제시한다. 첫째, 단순한 주의집중 시각화를 넘어 전체 네트워크의 정보 흐름을 고려하는 포괄적 접근법의 중요성을 입증했다[1]. 둘째, 클래스 특화 해석가능성의 필요성을 강조하여, 향후 연구들이 이 방향으로 발전할 수 있는 기반을 마련했다.

### 기술적 기여의 확장 가능성

제안된 정규화 기법과 비모수적 관련성 전파 방법은 다른 아키텍처에도 적용될 수 있는 일반적 기법이다[1]. 특히 스킵 연결을 사용하는 다양한 모델들(ResNet, DenseNet 등)에서도 유사한 수치적 안정성 문제가 발생할 수 있으므로, 이러한 기법들의 활용 범위는 상당히 넓을 것으로 예상된다.

### 향후 연구에서 고려해야 할 점들

미래 연구자들은 다음과 같은 점들을 고려해야 한다. 첫째, 계산 효율성을 개선하는 방법을 모색해야 한다. 근사 기법이나 선택적 계산 방법 등을 통해 실용성을 높일 필요가 있다. 둘째, 더 다양한 Transformer 아키텍처(BERT, GPT 시리즈, ViT 변형들)에 대한 적용 가능성을 검증해야 한다. 셋째, 해석가능성의 품질을 평가하는 더 포괄적이고 신뢰할 수 있는 메트릭들을 개발해야 한다.

## 결론

"Transformer Interpretability Beyond Attention Visualization" 연구는 Transformer 해석가능성 분야에서 중요한 이정표를 세운 연구이다. 기존 방법들의 근본적 한계를 해결하고 클래스 특화된 해석을 가능하게 함으로써, 모델의 신뢰성과 디버깅 능력을 크게 향상시켰다[1]. 비록 계산 복잡도와 이론적 가정 등의 한계가 존재하지만, 제안된 기법들은 향후 해석가능 AI 연구의 발전에 중요한 기여를 할 것으로 전망된다. 특히 모델 일반화 성능 향상과 약지도 학습 등의 응용 분야에서 실질적인 가치를 제공할 수 있을 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/caddc19b-c323-41d0-9c35-226a025877b4/2012.09838v2.pdf
[2] https://www.sec.gov/Archives/edgar/data/2000819/0002000820-24-000002-index.htm
[3] https://www.sec.gov/Archives/edgar/data/2000820/0002000820-24-000001-index.htm
[4] https://www.sec.gov/Archives/edgar/data/2034520/000164117225010269/form20-f.htm
[5] https://www.sec.gov/Archives/edgar/data/1888886/000155837025001826/gpcr-20241231x10k.htm
[6] https://www.sec.gov/Archives/edgar/data/1773383/000177338325000065/dt-20250331_htm.xml
[7] https://www.sec.gov/Archives/edgar/data/1707753/000170775325000021/estc-20250430.htm
[8] https://ieeexplore.ieee.org/document/9577970/
[9] https://www.ewadirect.com/proceedings/ace/article/view/13745
[10] https://journals.flvc.org/FLAIRS/article/view/128399
[11] https://ieeexplore.ieee.org/document/9991178/
[12] https://arxiv.org/abs/2412.14231
[13] https://ieeexplore.ieee.org/document/10939406/
[14] https://github.com/hila-chefer/Transformer-Explainability
[15] https://proceedings.mlr.press/v162/ali22a/ali22a.pdf
[16] https://www.ibm.com/think/topics/attention-mechanism
[17] https://www.comet.com/site/blog/explainable-ai-for-transformers/
[18] https://arxiv.org/abs/2311.06786
[19] https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)
[20] https://www.youtube.com/watch?v=XCED5bd2WT0
[21] https://www.youtube.com/watch?v=eMlx5fFNoYc
[22] https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf
[23] https://www.sec.gov/Archives/edgar/data/2000819/0002000819-23-000001-index.htm
[24] https://ieeexplore.ieee.org/document/10167142/
[25] https://www.semanticscholar.org/paper/d97c9e6bf61efebc6e5a4ea1b134094e16315185
[26] http://biorxiv.org/lookup/doi/10.1101/2021.01.28.428629
[27] https://ieeexplore.ieee.org/document/10823114/
[28] https://www.mdpi.com/2073-431X/13/4/92
