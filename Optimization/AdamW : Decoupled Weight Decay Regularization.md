# Decoupled Weight Decay Regularization 

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
- L2 정규화(L2 regularization)와 가중치 감쇠(weight decay)는 표준 SGD에서는 학습률에 비례하여 동등하지만, Adam 같은 적응적 최적화 알고리즘에서는 본질적으로 다르다.  
- 기존 딥러닝 프레임워크에서 “weight decay”로 구현된 것은 사실상 L2 정규화이며, 이로 인해 Adam의 일반화 성능이 저하되었다.  
- 가중치 감쇠를 그래디언트 업데이트와 분리(decoupled)하여 직접 적용하는 방식(AdamW)을 도입하면, Adam이 SGD 수준의 일반화 성능을 확보할 수 있다.

**주요 기여**  
1. 표준 SGD와 Adam에서 가중치 감쇠와 L2 정규화의 수학적·실험적 차이를 명확히 규명  
2. Adam에 decoupled weight decay(AdamW) 기법을 제안  
3. 다양한 학습률 스케줄(고정, step-drop, cosine annealing)과 훈련 예산(에폭 수)에 걸친 포괄적 실험을 통해 AdamW의 일반화 우수성 입증  
4. normalized weight decay 및 warm restarts(AdamWR) 확장으로 하이퍼파라미터 튜닝 용이성 및 anytime 성능 개선  

***

## 2. 해결하고자 하는 문제  
- **문제 인식**: Adam 등 적응적 최적화 알고리즘이 이미지 분류 등 여러 과제에서 SGD 대비 일반화 성능이 낮게 나타남  
- **원인 분석**: 흔히 쓰이는 L2 정규화는 파라미터별로 그래디언트 스케일에 따라 다른 강도로 작용하지만, 원래의 weight decay(모든 파라미터 동등 감쇠)와 달라 일반화 효율이 떨어짐  

***

## 3. 제안 방법  
### 3.1 Decoupled Weight Decay (AdamW)  
- **전통적 weight decay**:  

$$
    \theta_{t+1} = (1 - \lambda)\theta_t - \alpha \nabla f(\theta_t)
  $$  

- **L2 정규화를 포함한 업데이트**:  

$$
    \theta_{t+1} = \theta_t - \alpha \bigl(\nabla f(\theta_t) + \lambda'\theta_t\bigr)
  $$  

- **AdamW 업데이트** (Alg. 2 라인 12):  

$$
    \theta_t \leftarrow \theta_{t-1}
      - \eta_t\Bigl(\tfrac{\alpha\,\hat m_t}{\sqrt{\hat v_t}+\epsilon}\Bigr)
      - \eta_t\,\lambda\,\theta_{t-1}
  $$  
  
여기서  
  - $$\hat m_t, \hat v_t$$: 모멘텀과 분산 보정 그래디언트  
  - $$\eta_t$$: 학습률 스케줄러(예: cosine annealing)  
  - $$\lambda$$: weight decay 계수  

### 3.2 Normalized Weight Decay  
- 배치 크기 $$b$$, 전체 데이터 수 $$B$$, 총 에폭 수 $$T$$를 고려해  

$$
    \lambda = \lambda_{\rm norm}\sqrt{\frac{b}{B\,T}}
  $$  

- 에폭 수에 따른 최적 weight decay 변화 완화 및 하이퍼파라미터 독립성 강화  

### 3.3 Warm Restarts with Cosine Annealing (AdamWR)  
- **Cosine Annealing**:  

$$
    \eta_t = \tfrac12\bigl(1 + \cos(\tfrac{\pi\,T_{\rm cur}}{T_i})\bigr)
  $$  

- **Warm Restart**: 주기 $$T_i$$마다 학습률을 다시 최고치로 리셋하여 anytime 성능 개선  

***

## 4. 모델 구조 및 성능 향상  
- **모델**: Shake-Shake 정규화가 적용된 3-branch ResNet(26-2×64d, 26-2×96d)  
- **데이터셋**: CIFAR-10, ImageNet32×32  
- **주요 결과**  
  - AdamW는 고정·step-drop·cosine annealing 전 스케줄에서 모두 Adam 대비 최대 15% 상대 테스트 에러 감소  
  - 하이퍼파라미터(학습률·weight decay) 검색 공간이 더 분리(separable)되어 튜닝 용이  
  - AdamWR은 SGDR(SGD warm restarts) 대비 최대 10배 빠른 anytime 성능 달성  
- **외부 검증**:  
  - EEG 신호 분류, Face Detection, Transformer 언어 모델 등 다양한 과제에서 AdamW 도입 시 일관된 성능 개선  

***

## 5. 한계 및 고려사항  
- **과제 범위**: 이미지 분류 중심 실험, 자연어·시계열·강화학습 등 타 도메인 일반화 검증 필요  
- **스케줄 민감도**: cosine annealing과 warm restarts 하이퍼파라미터 선택이 성능에 영향  
- **이론적 확장**: AdaGrad, AMSGrad 등 다른 적응적 최적화 기법으로 일반화 가능성 탐색  

***

## 6. 향후 연구에의 영향 및 고려할 점  
- **영향**:  
  - AdamW는 딥러닝 프레임워크 표준 옵티마이저로 자리 잡아, 다양한 모델의 일반화 성능 상향 견인  
  - 하이퍼파라미터 튜닝 부담 경감  
- **연구 시 고려 사항**:  
  1. **적용 도메인 확장**: 자연어 처리, 강화학습, 그래프 신경망 등 다중 과제에서의 효용 검증  
  2. **하이퍼파라미터 자동화**: normalized weight decay·스케줄 파라미터 자동 최적화 기법 개발  
  3. **이론적 통합**: Bayesian filtering 프레임워크 기반 다른 adaptive methods와의 연계 이론 정교화  
  4. **효율성 개선**: 대규모 분산 환경에서의 학습 안정성 및 통신 비용 최적화 연구

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/73f8d242-4091-443d-8437-55afea2f5544/1711.05101v3.pdf

https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html
