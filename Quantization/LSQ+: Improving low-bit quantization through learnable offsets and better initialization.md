# LSQ+: Improving low-bit quantization through learnable offsets and better initialization

**주요 주장 및 기여 요약**  
LSQ+는 최신 효율적 신경망(EfficientNet, MixNet)의 Swish 등 음수 활성화 함수를 처리하기 위해 학습 가능한 스케일(s)과 옵셋(β)을 도입한 비대칭 양자화 기법을 제안한다. 또한, MSE 기반 초기화로 양자화 파라미터의 학습 안정성을 크게 향상시킨다.

***

## 1. 해결하고자 하는 문제
현대 NAS 기반 아키텍처는 ReLU를 넘어 Swish, H-swish 등의 활성화를 사용하며, 음수 값도 표현한다.  
- 기존 LSQ, PACT 등은 활성화를 부호 없는(unsigned) 범위로 양자화하여 음수값을 0으로 클램핑해 정보 손실이 발생  
- 부호 있는(signed) 양자화는 별도 부호비트가 필요해 저비트(2–4bit)에서 비효율  

이로 인해 EfficientNet-B0 W4A4 양자화 시 최대 4.1% 성능 감소가 관찰됨.

***

## 2. 제안 방법
### 2.1 비대칭 학습 가능 양자화  
활성화 x를 scale s, offset β로 변환 후 클램핑:

$$
\bar{x} = \text{clamp}\Bigl(\tfrac{x - \beta}{s},\,n,\,p\Bigr),\quad
\hat{x} = \bar{x} \cdot s + \beta
$$

- n, p: 양자화 레인지 하·상한 (예: unsigned: 0, 2^b−1)  
- s, β 모두 학습  
- ReLU 계열 가중치는 대칭 양자화 사용  

### 2.2 MSE 기반 초기화  
기존 min–max, LSQ 초기화가 저비트에서 불안정함을 보완  

$$
(s_{\mathrm{init}},\beta_{\mathrm{init}})
=\arg\min_{s,\beta}\|\,\hat{x}(s,\beta)-x\,\|^2_F
$$

PyTorch 자동미분으로 몇 배치 forward를 통해 최적 s, β 추정

***

## 3. 모델 구조 및 실험 설정
- 백본: EfficientNet-B0, MixNet-S, ResNet-18  
- 양자화 비트: W2A2, W3A3, W4A4  
- 구성(Configuration):
  1. Unsigned + Symmetric (LSQ)
  2. Signed + Symmetric
  3. Signed + Asymmetric
  4. Unsigned + Asymmetric (LSQ+)
- 초기화 비교: Min–max, LSQ, LSQ+

***

## 4. 성능 향상 및 한계
### 4.1 성능 향상
- EfficientNet-B0 W4A4: +1.8%P (Config 4 vs LSQ)  
- EfficientNet-B0 W2A2: +5.6%P  
- MixNet-S W4A4: +1.3%P  
- ResNet-18 W4A4: floating-point 동등 성능 유지  
- 초기화 안정성: W2A2 표준편차 ∆acc: min-max±4.7%→LSQ+±1.9%

### 4.2 한계
- MSE 초기화에 추가 전처리 배치 필요  
- 활성화 분포 변화가 크면 재초기화 요구 가능  
- 부가 메모리·컴퓨팅 비용은 negligible하나, 프레임워크 지원 필수  

***

## 5. 일반화 성능 향상 관점
학습 가능한 옵셋 β는 스케일링 그리드가 음수 치우친 활성화를 효과적으로 포착하도록 학습돼,  
- 다양한 활성화 분포에 적응  
- 양자화 에러 분포를 평균 제곱 오차 기준으로 최소화  
→ 신경망이 저비트 조건에서도 표현력을 유지하며 과적합을 억제해 일반화 성능이 향상될 가능성

***

## 6. 향후 연구에 미치는 영향 및 고려 사항
- **프로그래머블 양자화 레이어**: 하드웨어별 옵셋·스케일 학습 지원 필요  
- **동적 초기화 전략**: 온라인 MSE 초기화로 분포 변화에 적응  
- **타 기법과의 결합**: 지식증류, NAS 기반 비트-레벨 검색과 통합 가능  
- **비균일 양자화 그리드**: 비대칭 그리드 확장 및 로그-스케일 적용 시 성능 분석  

향후 저지연 엣지 디바이스 및 대규모 언어·비전 모델의 저비트 압축 연구에 LSQ+의 비대칭·안정화 아이디어가 핵심적으로 활용될 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8db08e2f-7f58-4203-8c97-8846b5ddb3ce/2004.09576v1.pdf)
