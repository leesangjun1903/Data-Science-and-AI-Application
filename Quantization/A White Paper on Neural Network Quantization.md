# A White Paper on Neural Network Quantization

## 1. 핵심 주장 및 주요 기여
“**A White Paper on Neural Network Quantization**”은 딥러닝 모델을 에지 장치에서 저전력·저지연으로 실행하기 위해 **네트워크 가중치와 활성화를 저비트 정수로 표현**(quantization)하고, 이로 인한 성능 저하를 최소화하는 **최신 알고리즘과 워크플로우**를 체계적으로 제시한다.[1]

- **Post-Training Quantization (PTQ)**: 재학습 없이 사전 학습된 FP32 모델을 즉시 8비트까지 양자화하여 대부분의 경우 부동소수점 성능에 근접하도록 실험 기반 파이프라인 제안  
- **Quantization-Aware Training (QAT)**: 학습 과정에서 양자화 노이즈를 모사하고, 스케일·제로포인트 학습, STE(직통통과추정) 등을 통해 4비트 이하 저비트 양자화에서도 높은 정확도를 유지하는 훈련 기법 제안  
- **하드웨어 고려**: 균일(Uniform) 대칭·비대칭 양자화, 텐서 단위(Per-tensor) 대 채널 단위(Per-channel) 양자화, 배치정규화(BN) 접기(folding) 및 활성화 함수 융합(fusing) 방법 등 구체적 구현 세부사항 정리  

## 2. 문제 정의 및 제안 방법
### 2.1 해결 과제
- 에지 디바이스용 NN 추론 시 연산량·메모리·전력 과다  
- 저비트 양자화 시 발생하는 **양자화 노이즈**로 인한 정확도 저하  

### 2.2 제안 방법
#### 2.2.1 Quantization 기본 수식
- 연산 전·후 스케일링:  

  $$\hat x = q(x; s,z,b) = s\Big(\mathrm{clamp}(\lfloor x/s\rfloor + z;\,0,2^b-1)-z\Big) $$  

- MAC 단계에서 고정소수점 연산:  

  $$\hat A_n = \hat b_n + \sum_m \hat W_{n,m}\,\hat x_m = s_w s_x\sum_m W^{\rm int}\_{n,m} x^{\rm int}_{m} + \hat b_n $$  

#### 2.2.2 Post-Training Quantization (PTQ)
1. **Range Setting**: 가중치·활성화 클리핑 경계 결정  
   - Min–Max, MSE 최적화, 마지막 레이어 소프트맥스는 Cross-Entropy 기반[1]
2. **Cross-Layer Equalization**: 인접 레이어 스케일 균등화로 동적 범위 편차 축소  
3. **Bias Correction**: 예상 출력 분포 이동 보정  
4. **AdaRound**: 스케일러블 이진 최적화 기법으로 라운딩 방향(위/아래) 학습[1]
5. **최종 파이프라인**: CLE → 양자화 블록 삽입 → 범위 설정 →(AdaRound/바이어스 보정)→ 활성화 범위 설정[1]

#### 2.2.3 Quantization-Aware Training (QAT)
- **Simulated Quantization**: STE 이용해 라운딩 연산의 기울기 통과  
- **BN Folding**: 학습 중 정적 접기(단순 재매개변수화) 혹은 채널별 양자화 시 배치정규화 파라미터 스케일 통합  
- **학습 파이프라인**: CLE → 양자화 블록 삽입(대칭 가중치·비대칭 활성화·채널 단위 권장) → 범위 초기화(MSE) → 스케일·제로포인트 학습 → SGD/Adam 훈련  

## 3. 성능 향상 및 한계
- **8비트 양자화**: PTQ·QAT 모두에서 FP32 대비 <1% 성능 저하  
- **4비트 가중치**: PTQ로 ResNet/Inception 계열은 ~1% 이내, 경량 모델(MobileNet·EfficientNet)은 채널 단위 QAT 시 <1.5% 저하  
- **4비트 활성화**: MobileNet 계열 등 심층 경량화 모델에서 성능 저하 두드러짐(PTQ는 실사용 어려움, QAT로 4~5% 저하)  
- **제한점**: Mixed precision(가변 비트폭), 특수 활성화 함수·연산(Concat/Add) 양자화 시 추가 연구 필요  

## 4. 일반화 성능 향상 가능성
- **노이즈 견고화**: QAT 중 양자화 노이즈 주입이 정규화 효과를 내고, 작은 데이터셋 과적합 억제 및 일반화 성능 향상 잠재  
- **AdaRound** 최적화가 특정 층별 중요 특징 보존하여 도메인 전이 시 정확도 유지 가능성  
- **CLE 기반 스케일링**이 네트워크 전반의 분포 균일화로 **다양한 데이터 분포**에 대한 적응력 강화  

## 5. 향후 연구 영향 및 고려 사항
- **Mixed-Precision 적용**: 레이어별·채널별 가변 비트폭 전략과 하드웨어 친화적 구현  
- **비전 외 도메인**: 음성·자연어처리·그래프 신경망 등에서 PTQ·QAT 파이프라인 확장  
- **자동화 및 NAS 통합**: Neural Architecture Search와 양자화 최적화를 공동 설계하여 훈련·양자화 동시 최적화  
- **양자화 친화적 활성화 함수 개발**: Swish·GELU와 같은 비선형 활성화의 저비트 환경 적합성 연구  
- **실제 디바이스 검증**: 양자화 시뮬레이션과 실제 하드웨어 구현 간 성능 격차 최소화  

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/51468fc8-0c9d-4efe-84f0-aadaba12de97/2106.08295v1.pdf
