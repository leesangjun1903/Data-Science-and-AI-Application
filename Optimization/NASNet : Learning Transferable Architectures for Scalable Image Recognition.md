# NASNet : Learning Transferable Architectures for Scalable Image Recognition | Image classification, NAS

# 핵심 요약

**Learning Transferable Architectures for Scalable Image Recognition** 논문은 소규모 데이터셋(CIFAR-10)에서 최적의 컨볼루션 셀 구조를 탐색한 뒤, 이를 대규모 데이터셋(ImageNet)으로 전이해 높은 성능을 달성하는 **NASNet** 아키텍처를 제안한다.  
이 방법은 전체 네트워크가 아닌 **반복 가능한 모듈(셀)** 단위로 구조를 검색함으로써 탐색 비용을 크게 절감(7×)하고, 설계된 셀의 **이식성(transferability)**을 확보한다. 최종적으로 NASNet은 ImageNet에서 **Top-1 82.7%**, **Top-5 96.2%**를 달성해 당시 인위적 설계 모델을 능가했다.

# 1. 해결하고자 하는 문제

기존 심층 CNN 설계는 전문적 건축 공학(architecture engineering)에 의존하며, 대규모 데이터셋(예: ImageNet)에서 자동 구조 탐색(NAS)은 **연산 비용이 매우 큼**  
→ 소규모 데이터셋에서 반복 가능한 모듈(Cell) 하나만 탐색하여, 이를 쌓아 대규모 네트워크를 구성할 수 있으면 탐색 비용을 줄이면서도 성능을 유지·향상할 수 있다는 문제 설정

# 2. 제안 방법

## 2.1 NASNet 탐색 공간 설계
- **Cell**: 네트워크를 구성하는 최소 단위  
  - 두 종류:  
    - **Normal Cell**: 입력과 동일한 공간 해상도 유지  
    - **Reduction Cell**: 해상도를 절반으로 축소  
  - 각 셀은 **B개의 블록**으로 구성되며, 블록당 5가지 선택(step)을 controller RNN이 예측  
    1. 첫 번째 입력 히든 상태 선택  
    2. 두 번째 입력 히든 상태 선택  
    3. 첫 입력에 적용할 연산 선택  
    4. 두 입력에 적용할 연산 선택  
    5. 결과 병합 방법(덧셈 또는 concatenate) 선택  

- **가능 연산**  
  - identity, 1×1 conv, 3×3 conv, depthwise-separable conv (3×3, 5×5, 7×7), dilated conv, max/avg pooling 등  

## 2.2 강화학습 기반 탐색
- **Controller**: 한 층짜리 LSTM + 10B개 softmax(5B Normal + 5B Reduction)  
- **보상**: CIFAR-10 검증 정확도  
- **업데이트**: Proximal Policy Optimization (PPO)  
- **효율성**: 500 GPUs × 4일 → 약 2,000 GPU-hour (기존 800 GPUs × 28일 대비 7× 절감)

## 2.3 ScheduledDropPath
- DropPath을 학습 진행에 따라 선형 증가하는 확률로 적용해 regularization 강화  
- 기존 고정 확률 DropPath 대비 **일관된 성능 향상**

# 3. 모델 구조

- CIFAR-10에서 탐색된 최상위 셀 **NASNet-A**  
- 셀 반복 수 N과 필터 수 F를 조절해 다양한 크기의 네트워크 구성  
  - 예: NASNet-A (7@2304) → ImageNet 최상위 모델  
- 기본 네트워크:  
  1. Stem conv  
  2. 여러 개의 Normal/Reduction Cell 반복  
  3. 최종 풀링 + 분류기  

# 4. 성능 향상

| 데이터셋 | 모델               | Top-1 (%) | Top-5 (%) | Mult-Adds     |
|----------|--------------------|-----------|-----------|---------------|
| CIFAR-10 | NASNet-A (7@2304) + cutout | 97.60 (error 2.40%) | —         | —             |
| ImageNet | NASNet-A (6@4032)  | 82.7      | 96.2      | 23.8B         |
| Mobile   | NASNet-A (4@1056)  | 74.0      | 91.6      | 564M          |

- ImageNet에서 human-designed 모델 대비 **1.2%** Top-1 정확도 향상  
- Mobile 환경에서도 기존 MobileNet, ShuffleNet 대비 **3.1%** 이상 개선  

# 5. 일반화 성능 향상

- **Transferability**: CIFAR-10에서 학습된 셀 그대로 ImageNet으로 전이 가능  
- 다양한 계산 예산(모바일부터 대형)에서 셀 반복 수·필터 수 조절만으로 최적화  
- **객체 검출(Faster-RCNN + NASNet-A)**에서도 mAP 43.1% 기록, 기존 대비 4%p 이상 향상  

# 6. 한계

- **탐색 비용**: 2,000 GPU-hour는 여전히 큰 자원 소모  
- **탐색 공간 고정성**: 셀 구조 탐색 공간이 모델 종류나 연산 셋에 따라 한계  
- **랜덤 탐색 대비**: RL이 우수하나, random search 성능 차이가 크지 않음(≈1%p)  

# 7. 향후 연구에 미치는 영향 및 고려 사항

- **모듈화된 NAS** 개념 확장: 셀 단위 구조 탐색을 다양한 태스크(비전 외, NLP 등)로 전이  
- **탐색 효율화**: 하드웨어 비용 절감 위한 더 효율적 탐색 알고리즘(프루닝, 지식 증류 등)  
- **자동화된 정규화 기법**: ScheduledDropPath 외, 셀 구조에 맞춘 동적 regularization 연구  
- **탐색 공간 설계**: 연산 셋, 연결 규칙 확장으로 더 다양한 구조 발견 가능성 검토  

이 논문은 **자동화된 아키텍처 검색**과 **전이 학습**을 결합해 실질적 성능 향상을 증명했으며, 이후 NAS 연구의 방향성과 효율적 구조 설계의 기틀을 마련했다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/185341dc-a895-4dea-b808-7f8c3594d6e5/1707.07012v4.pdf
