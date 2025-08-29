# Root Mean Square Layer Normalization

## 1. 핵심 주장과 주요 기여 (간결 요약)  
**Root Mean Square Layer Normalization (RMSNorm)**은 기존 LayerNorm이 수행하는 “평균(µ) 제거” 단계만 제거하고, 오직 표준편차 대신 RMS(root mean square)로 정규화하여 연산량을 획기적으로 줄이면서도 LayerNorm과 동등한 성능을 달성할 수 있음을 보였다.  
- **주장**: LayerNorm의 재중심화(리센터링) 불변성(re-centering invariance)은 훈련 안정성에 필수적이지 않으며, 재스케일링 불변성(re-scaling invariance)만으로 충분하다.  
- **기여**:  
  1. 평균 중심화 연산 제거로 계산량 감소  
  2. RMSNorm 수식 제안: $$\displaystyle \bar a_i = \frac{a_i}{\sqrt{\tfrac1n\sum_j a_j^2}}\odot g$$  
  3. 부분 집합만 사용해 RMS를 추정하는 pRMSNorm 제안 (p% 요소 이용)  
  4. 다양한 모델·프레임워크(RNN, Transformer, CNN 등)에서 7∼64% 속도 향상 입증  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능, 한계  
### 2.1 해결하고자 하는 문제  
- **연산 오버헤드**: LayerNorm은 입력 벡터 평균·분산을 모두 계산하므로, 깊고 큰 모델일수록 연산 비용이 크게 증가하여 전체 훈련 시간이 느려진다.  
- **가설**: LayerNorm의 핵심은 분산 스케일링(variance scaling)이지, 평균 재중심(mean centering) 단계가 아니므로 이를 제거해도 안정성이 유지된다.  

### 2.2 제안 방법  
1. **RMSNorm 수식**  

$$  
   \bar a_i = \frac{a_i}{\mathrm{RMS}(a)} \odot g,\quad
   \mathrm{RMS}(a)=\sqrt{\frac1n\sum_{j=1}^n a_j^2},  
   $$  
   
여기서 $$a=W x$$는 층별 선형 변환 출력, $$g$$는 학습 가능 이득 파라미터.  
2. **pRMSNorm**  

$$\mathrm{RMS}(a)$$ 계산 시 전체 $$n$$ 차원 대신 상위 $$p\%$$만 사용하여 근사.  

```math
     \mathrm{RMS}_p(a)=\sqrt{\frac1{k}\sum_{i=1}^k a_i^2},\quad k=\lceil n\cdot p\rceil.
``` 

### 2.3 모델 구조 적용  
- **RNN 기반 RNNSearch**, **Transformer**, **CNN (ConvPool-CNN-C)**, **읽기 이해 모델(CNN/Daily Mail)**, **이미지–문장 순서 임베딩(OE)** 등 다채로운 아키텍처에 “LayerNorm 대신 RMSNorm”을 드롭인(drop-in) 적용.  

### 2.4 성능 향상  
- **훈련 속도**: 프레임워크·모델별로 7∼64% 훈련 시간 단축.  
- **정확도**:  
  - RNNSearch BLEU: LayerNorm 대비 동등[ +0.0∼+0.1 ] 향상  
  - Transformer BLEU: 학습 가능, LayerNorm과 동등  
  - CIFAR-10 테스트 오류율: Baseline 8.96→RMSNorm 8.83 (개선)  
- **일반화 성능**: CIFAR-10·이미지–문장 OE 등에서 LayerNorm보다 혹은 동등하게 더 나은 테스트 성능 달성.  

### 2.5 한계  
- **pRMSNorm 속도 이점 불안정**: 요소 슬라이싱 구현에 따라 오히려 느려질 수 있음.  
- **평균 비정규화로 인한 이론적 안정성 감소 우려**: 극단적 데이터 분포에 대한 영향은 추가 연구 필요.  

## 3. 일반화 성능 향상 가능성  
- RMSNorm은 **분산 정규화만 수행**하여도 내부 활성도 분포 폭을 안정화하고, 과적합 억제 효과를 유지한다.  
- CIFAR-10에서 LayerNorm이 테스트 오류율을 오히려 증가시킨 반면, RMSNorm은 **테스트 오류율 개선**을 보여 일반화 능력을 보다 잘 보존함.  
- pRMSNorm도 소수 요소만으로 RMS 근사를 해도 성능 저하가 거의 없어, 대규모 모델에 적용 시 메모리·연산 절감과 일반화 간 균형을 이룰 수 있다.  

## 4. 향후 연구에 미치는 영향 및 고려할 점  
- **영향**:  
  - 다양한 아키텍처에 간단한 교체만으로 학습 효율을 높이는 새로운 정규화 패러다임 제시  
  - 대형 언어 모델, 시계열·강화학습 등 확장 적용 가능성  
- **고려할 점**:  
  1. pRMSNorm 구현 최적화로 진정한 속도 이점 확보  
  2. 비정규 분포·이상치에 대한 안정성 분석  
  3. 다양한 노름($$\ell_1,\ell_p$$) 실험으로 최적 정규화 형태 탐색

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/051313a8-3c0a-4caf-9795-6db0983364e4/1910.07467v1.pdf)
