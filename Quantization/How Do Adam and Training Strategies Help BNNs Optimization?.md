# How Do Adam and Training Strategies Help BNNs Optimization?

## 1. 핵심 주장 및 주요 기여  
이 논문은 **Binary Neural Networks(BNNs)** 최적화에 있어 Adam 옵티마이저와 다양한 학습 전략이 왜 SGD 대비 더 우수한 성능을 발휘하는지 그 **근본 원인**을 규명한다.  
- **주요 기여**  
  1. Adam과 SGD를 대규모 ImageNet 실험으로 공정 비교, BNN에서 Adam이 **죽은(dead) 가중치**를 재활성화하고 **더 우수한 일반화 능력**을 획득하는 이유를 시각화·분석  
  2. BNN 학습에서 활성화 포화(activation saturation)와 그로 인한 그래디언트 소실 문제를 수식 및 손실 지형(loss landscape) 시각화를 통해 설명  
  3. 실수값 잠재 가중치(latent weight)의 **신뢰도(confidence)** 관점과, weight decay가 BNN 안정성과 초기값 의존성에 미치는 영향을 정량화하는 두 가지 지표(FF ratio, C2I ratio) 제안  
  4. 기존의 다단계(two-step) 학습 전략에 기반해, 단계별 적절한 weight decay 스킴을 도입하여 최종적으로 ReActNet 대비 **1.1%p** 높은 70.5% top-1 정확도 달성  

## 2. 문제 정의, 제안 기법 및 성능  
### 2.1 해결하고자 하는 문제  
- BNN은 이진화로 인한 **극도로 울퉁불퉁한(rugged) 손실 지형**과 활성화 포화로 인한 **그래디언트 소실** 현상 때문에 SGD로 학습 시 “죽은” 가중치 채널이 많아 성능이 저하  
- Adam이 실제로는 **일반화가 떨어진다**는 기존 연구(Wilson et al., 2017)와 달리, BNN에선 왜 더 나은 성능을 내는지 설명 부재  

### 2.2 제안 방법  
1. **활성화 포화 및 그래디언트 소실**  
   - BNN 이진화(sign) 뒤 역전파 시 $$\frac{d}{dx}\mathrm{sign}(x)\approx \mathrm{clip}'(x)$$ 을 써서 $$|x|>1$$ 구간에서 그래디언트가 0이 되는 현상 시각화  
2. **Adam의 이점**  
- SGD: $$v_t = \gamma v_{t-1} + g_t, \ \Delta w_t = -\eta\,v_t$$  
- Adam:  

$$
       m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad
       v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,
     $$  

$$
       \hat m_t = \frac{m_t}{1-\beta_1^t},\quad
       \hat v_t = \frac{v_t}{1-\beta_2^t},\quad
       \Delta w_t = -\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
     $$  
   
- 두 번째 모멘텀($$\hat v_t$$) 정규화가 “죽은” 가중치에 비균등한 학습률 할당, **평활(flattened) 구간** 탈출에 유리  
3. **실수값 잠재 가중치의 의미**  
   - latent weight $$w_r$$의 절댓값은 이진화된 $$w_b=\mathrm{sign}(w_r)$$ 값에 대한 **신뢰도**로 해석  
   - Adam 학습 후 $$w_r$$ 분포가 3봉우리(peaks)를 형성해 “신뢰도 높은” 결정 경계 확보, SGD는 중앙 집중형 분포로 불안정  
4. **Weight Decay 분석 및 지표**  
   - **Flip-Flop Ratio (FF ratio)**: 학습 중 이진 가중치의 빈번한 부호 전환 비율  
   - **Correlation-to-Initialization Ratio (C2I ratio)**: 최종 이진 가중치와 초기값 부호의 일치도  
   - 실험 결과, weight decay 증가 시 FF↑·C2I↓ 상반 관계, 최적값 탐색 어려움  
5. **두 단계(two-step) 학습 전략**  
   - **Step1**: 활성화만 이진화, 실수 가중치에 적절한 weight decay 적용 → 초기값 의존성 억제 (높은 C2I)  
   - **Step2**: 양쪽 모두 이진화, weight decay 0 적용 → 안정성 확보 (낮은 FF)  
   - 최적 decay $$5\times10^{-6}$$ 적용으로 StrongBaseline +2.3%p, ReActNet +1.1%p 성능 상승  

### 2.3 성능 및 한계  
- **ImageNet** top-1 정확도: 기존 ReActNet-A 69.4% → 본 기법 70.5%  
- **한계**:  
  - 제안된 weight decay 스킴과 두 단계 전략은 **추가 실험 없이 다른 구조나 과제**에 그대로 일반화 가능성 검증 필요  
  - 활성화 포화와 latent weight 분석이 ResNet-18, MobileNet 계열에 국한  

## 3. 일반화 성능 향상 관점  
- Adam의 두 번째 모멘텀 기반 **개별 원소 정규화**는 BNN의 불규칙한 최적화 지형에서 **미약한 그래디언트 채널**에 보정된 학습률을 부여  
- 두 단계 학습으로 **초기화 의존성(과적합)과 안정성(과소적합)** 간 균형 달성  
- 이로 인해 **일반화 갭(validation–training)**이 축소되어 검증 성능 향상  

## 4. 향후 연구에 미치는 영향 및 고려사항  
- **영향**: BNN 최적화 연구에서 옵티마이저 선택과 학습률 스케줄, weight decay 설정의 중요성을 부각시켜 차세대 **이진 전용 옵티마이저 개발**로 연결  
- **고려점**:  
  - 제안 지표(FF, C2I) 및 두 단계 전략의 **다양한 네트워크·데이터셋·태스크** 적용성 실험  
  - BNN의 비이진 부분(예: real-valued shortcut) 구조 변화에 따른 **loss landscape** 특성 재분석  
  - Adam 변형 순간 최적화(예: RAdam, AdaBelief)와 weight decay 스킴 간 **상호작용** 연구

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/036e926e-b2d3-4f71-bacd-52209ee590ee/2106.11309v1.pdf)
