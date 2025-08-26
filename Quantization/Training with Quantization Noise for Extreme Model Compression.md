# Training with Quantization Noise for Extreme Model Compression

## 1. 핵심 주장과 주요 기여  
**핵심 주장**  
본 논문은 훈련 과정에서 가중치의 무작위 부분집합에 양자화 노이즈(Quant-Noise)를 적용함으로써, 매우 높은 압축률에서도 원본 모델의 성능을 유지하거나 향상시킬 수 있음을 주장한다.  

**주요 기여**  
- Quant-Noise라는 기법을 도입하여, 훈련 중 전체 가중치가 아닌 임의로 선택된 부분에만 양자화를 적용함으로써 편향 없는 그래디언트 흐름을 유지하고 일반화 성능을 개선.  
- Int4/Int8 고정소수점 양자화와 Product Quantization(PQ)을 포함한 다양한 양자화 기법에 Quant-Noise를 적용하여 기존 QAT 대비 높은 압축률에서 성능 저하를 크게 완화.  
- RoBERTa, Transformer, EfficientNet 등 SOTA 모델들에서 극단적 압축(×14∼×94)에도 원본 대비 유사한 정확도 및 퍼플렉서티를 달성.  
- 이미 학습된 모델에도 post-hoc 방식으로 Quant-Noise 기반 미세조정(fine-tuning)을 적용, 추가 학습 없이도 양자화 성능 개선 가능.

## 2. 문제 정의·제안 기법·모델 구조·성능 및 한계

### 2.1 해결하고자 하는 문제  
- 기존 Quantization Aware Training(QAT)은 양자화 범위가 클수록 STE로 인한 그래디언트 편향이 커져 모델 성능이 급락.  
- PQ와 같은 고밀도 벡터 양자화(iPQ)는 누적된 재구성 오차로 인해 네트워크 활성값이 drift되어 성능 저하.

### 2.2 Quant-Noise 기법  
- 가중치 행렬 $$W$$를 블록 $$\{b_{kl}\}$$으로 분할하고, 각 순전파마다 임의 블록 집합 $$J$$에만 distortion 함수 $$\phi$$를 적용:  

$$
    \psi(b_{kl}\mid J) =
    \begin{cases}
      \phi(b_{kl}), & (k,l)\in J,\\
      b_{kl}, & \text{otherwise}.
    \end{cases}
  $$

- **IntN 양자화 노이즈**:  

$$\phi_{\text{intN}}(w) = (\mathrm{round}(w/s+z) - z ) \times s$$,  

$$s=\frac{\max W-\min W}{2^N-1},\; z=\mathrm{round}(\min W/s)$$.  

- **Product Quantization 노이즈(프록시)**:  
  블록을 0으로 마스킹($$\phi_{\mathrm{proxy}}(b)=0$$)하여 서브벡터 간 상관관계를 학습하도록 유도.  
- 순전파 시 노이즈 적용, 역전파 시 STE로 원본 가중치 기준 그래디언트 전달.  
- 여러 노이즈를 합성하여 양자화 + 프루닝 + 공유 등 다양한 압축 기법 동시 학습 가능.

### 2.3 모델 구조 및 학습 설정  
- **언어 모델**: 16-layer Transformer, WikiText-103, Adaptive Input & Adaptive Softmax, LayerDrop(0.2)  
- **문장 표현**: RoBERTa-base, MNLI 파인튜닝, LayerDrop(0.2)  
- **컴퓨터 비전**: EfficientNet-B3, ImageNet-1K, Classy Vision 구현  
- Quant-Noise 비율 $$p$$는 실험적으로 0.05∼0.2에서 최적화  
- 훈련 속도 저하는 5% 이하

### 2.4 성능 향상  
- **언어 모델**: iPQ 압축(×25) 적용 시 QAT에서 PPL 41.2 → Quant-Noise에서 PPL 20.7로 대폭 개선  
- **문장 분류**: RoBERTa(480 MB) → iPQ + Quant-Noise(38 MB)로 압축(×12.6) 시 MNLI 정확도 82.5%→83.6%로 소폭 상승  
- **이미지 분류**: EfficientNet-B3(46.7 MB) → iPQ + Quant-Noise(3.3 MB, ×14)에서 Top-1 80.0% 유지  
- **추가 압축**: Weight sharing, LayerDrop pruning과 조합 시 최대 ×94 압축에서도 PPL 24.7, Top-1 77.8% 달성

### 2.5 한계  
- Quant-Noise 비율이 과도하면(예: $$p>0.5$$) 성능 저하 발생  
- PQ 정확 노이즈 대신 프록시 사용 시 상관관계 의존, 특수한 분포에서 한계 가능성  
- 하드웨어별 실제 속도 이득은 검증 필요

## 3. 일반화 성능 향상 가능성  
Quant-Noise는 **훈련 중 양자화 드리프트에 노출시킴으로써 모델이 양자화 노이즈에 견고해지도록 학습**시킨다.  
- **무작위 부분 적용**으로 편향 없는 그래디언트 흐름을 보장, 과도한 노이즈에도 학습 안정성 유지  
- 드롭아웃/DropConnect 유사한 정규화 효과로 과적합 방지  
- 다양한 양자화 조합(iPQ+Int8 등) 및 프루닝, 공유와 결합 시에도 일관된 성능 향상 관찰  
- post-hoc 미세조정 시에도 SOTA 모델 일반화 성능 저하 최소화

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **범용 압축 프레임워크**: Quant-Noise는 양자화·프루닝·공유 등 다양한 압축 기법 동시 적용 가능, 경량화 연구의 표준 기법으로 확산 예상  
- **하드웨어 최적화**: 실제 임베디드·모바일 환경에서 속도 및 전력 절감 효과 검증 필요  
- **노이즈 스케줄링**: 훈련 초기/후기 단계별 노이즈 세기 조절→더 나은 수렴·성능 예측  
- **비지도·자기지도 학습과 통합**: 대규모 사전학습 모델의 압축에도 Quant-Noise 적용 방안 연구  
- **이론적 분석**: Quant-Noise가 일반화 경계에 미치는 영향·수렴 이론 정립 필요  

이 논문은 **극단적 모델 압축**에서도 **원본 성능 보존** 및 **일반화 강건성**을 보장하는 새로운 정규화·압축 학습 기법으로, 향후 경량화·온디바이스 AI 분야에서 중요한 기반이 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/898b740d-52df-42aa-8e6e-fa9be415b220/2004.07320v3.pdf)
