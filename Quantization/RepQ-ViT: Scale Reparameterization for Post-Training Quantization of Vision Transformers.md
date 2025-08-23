# RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers

## 1. 핵심 주장 및 주요 기여 (간결 요약)
**RepQ-ViT**은 비전 트랜스포머(ViT)의 후처리 양자화(Post-Training Quantization, PTQ)에서 발생하는 저비트 환경(특히 4비트)에서의 정확도 저하 문제를 해결하기 위해,  
1) 복잡한 양자화 과정과 하드웨어 친화적 단순 양자화 과정을 **명시적 척도 재매개변수화(scale reparameterization)**로 분리하고,  
2) 각 단계에서 최적화된 양자화 기법(channel-wise, log√2)을 적용한 뒤 추론 시 layer-wise, log2 양자화로 변환함으로써  
– **높은 양자화 성능**과 **효율적 추론 처리**를 동시에 달성한다.  
이로써 4비트 PTQ에서도 실용적 수준의 정확도(예: DeiT-S 69.0%, Swin-S 79.5%)를 확보하며 기존 기법 대비 최대 27%p 이상의 개선을 보인다.

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제
– ViT는 LayerNorm 이후의 **심각한 채널 간 분포 차이**와 Softmax 이후의 **지수적(power-law) 분포**로 인해 하드웨어 친화적 단순 양자화(균일 layer-wise, log₂)로는 저비트에서 큰 정확도 손실이 발생  
– 기존 PTQ 기법들은 양자화 설계 시 HW 제약을 고려하여 간단한 양자기를 바로 사용하는데, 이는 분포 표현력을 희생  

### 2.2 RepQ-ViT의 양자화-추론 분리 패러다임
양자화 과정에서는 분포 보존을 위한 복잡한 양자기, 추론 과정에서는 HW 친화적 단순 양자기를 사용하고, 두 과정 사이를 척도 재매개변수화로 연결

#### 2.2.1 LayerNorm 후 활성화의 채널-와이즈 → 레이어-와이즈 변환
– 채널별 양자화 스케일 $$s \in \mathbb{R}^D,\ z \in \mathbb{Z}^D$$을 계산 후,  
- 평균 스케일 $$\tilde s = \frac{1}{D}\sum_{d}s_d,\ \tilde z = \frac{1}{D}\sum_{d}z_d$$로 레이어-와이즈 재매개변수화  
- 변환계수 $$r_1 = s / \tilde s,\ r_2 = z - \tilde z$$를 도출하여  

$$
    \begin{aligned}
      \tilde\gamma &= \gamma \,/\, r_1,\quad
      \tilde\beta = \beta + \frac{s\odot r_2}{r_1},
    \end{aligned}
  $$
  
  으로 LayerNorm affine 파라미터를 조정하고, 다음층 가중치 $$\{W_{qkv}, b_{qkv}\}$$도  

$$
    \tilde W_{qkv} = r_1 \odot W_{qkv},\quad
    \tilde b_{qkv} = b_{qkv} - (s\odot r_2)\,W_{qkv}
  $$
  
  로 보정하여 채널-와이즈 정확도를 유지하면서 layer-wise HW 양자화 형태로 변환  

#### 2.2.2 Softmax 후 활성화의 log√2 → log₂ 변환
– 분포 해상도를 높이는 **log√2 양자화**로 양자화 스케일 $$s$$를 구하고  
- 양자화 식:  

$$
    A(Z)=\text{clip}\bigl(-\log_{\sqrt2}\tfrac{A}{s},0,2^b-1\bigr)
    =\text{clip}\bigl(-2\log_2\tfrac{A}{s},0,2^b-1\bigr)
  $$

- 역양자화 시 $$\hat A = s\cdot2^{-A(Z)/2}$$의 지수 계수가 정수가 아닌 문제를  
  홀짝 인디케이터(parity indicator function) $$1(A(Z))$$를 이용해  

$$
    \tilde s
    = s\bigl[1(A(Z))(\sqrt2-1)+1\bigr]
  $$
  
  로 재매개변수화함으로써 **bit-shift만**으로 처리 가능한 log₂ 형태로 변환  

### 2.3 모델 구조 및 실험 설정
– ViT-B/S, DeiT-T/S/B, Swin-S/B 백본에 적용  
– PTQ 시 32장의 ImageNet 캘리브레이션 데이터, COCO 검출·분할엔 1장씩 사용  
– 하이퍼파라미터·재구성 과정 불필요  

## 3. 성능 향상 및 한계
– **ImageNet 4비트**: DeiT-S 79.85→69.03%(+25.5%p), Swin-S 83.23→79.45%(+2.3%p)  
– **COCO 검출·분할 4비트**: Mask R-CNN(Swin-T) box AP 46.0→36.1, mask AP 41.6→36.0  
– **6비트**: 일반화 성능 거의 유지(모델 크기 5.3배 축소 시 정확도 손실 ~0.5%p)  
– **한계**:  
  – LayerNorm 재매개변수화 후 다음층 재캘리브레이션으로 인한 약간의 오차  
  – log√2→log₂ 변환에서 parity 연산 오버헤드  

## 4. 일반화 성능 향상 가능성
RepQ-ViT는 **하이퍼파라미터와 재구성 과정 없이** 양자화 분포 보존력을 확보하므로,  
– 다양한 ViT 변형(예: 윈도 기반, 지역적 구조)에도 일관된 성능 개선 가능  
– 소량 데이터 캘리브레이션만으로 도메인 전환(예: 의료·위성 영상) 시에도 **강인한 일반화**  
– log√2 양자화가 극단적 분포를 더 잘 보존하므로, 비전 외 Transformer 구조(예: 음성, 언어)에도 적용 여지  

## 5. 향후 연구 영향 및 고려 사항
– **영향**: PTQ 분야에서 양자화-추론 분리 패러다임을 제시해, 복잡도와 HW 친화성 간 트레이드오프를 획기적으로 완화  
– **고려점**:  
  1. 더 많은 활성화 연산(e.g. GELU, GELU 입력)에도 재매개변수화 확장  
  2. parity 함수 등 추가 연산이 제한적 HW에 미치는 실제 오버헤드 분석  
  3. log√2와 log₂ 중간 체계를 도입해 분포 적합도와 HW 효율 간 균형 최적화  
  4. 도메인이 크게 다른 데이터(의료, 위성)로 일반화 실험으로 실전 활용성 검증

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/883ecc0c-55c4-4c54-b74a-4005f7c9d460/2212.08254v2.pdf)
