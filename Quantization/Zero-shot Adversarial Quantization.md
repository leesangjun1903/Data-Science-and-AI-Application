# Zero-shot Adversarial Quantization

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
Zero-shot Adversarial Quantization(ZAQ)은 원본 학습 데이터에 접근하지 않고도, **적대적 학습(adversarial learning)**을 통해 양자화된 모델이 풀프리시전 모델 성능을 근사하도록 학습시키는 최초의 프레임워크이다.  

**주요 기여**  
- **두 단계 불일치(Discrepancy) 모델링**: 최종 출력 간 불일치 $$D_o$$와 중간 채널 간 관계 기반 불일치 $$D_f$$를 함께 사용해 양자화 손실을 정밀하게 측정.  
- **적대적 데이터 생성**: 제너레이터 $$G$$가 풀프리시전 모델 $$P$$와 양자화 모델 $$Q$$ 간 불일치를 최대화하는 샘플을 합성하여, 제한된 정보 환경에서 효과적 fine-tuning 데이터로 활용.  
- **활성화 정규화** $$\mathcal L_a$$: $$P$$의 활성화 맵이 높은 입력을 선호하도록 유도해, 실제 도메인에 가까운 샘플을 생성.  
- **세 가지 비전 과제**(분류·분할·검출)에서 ultra-low precision(2–6 bit)에도 최첨단 성능 달성 및 학습 효율 개선.

***

## 2. 문제 정의 및 제안 방법  
### 2.1 해결하고자 하는 문제  
- **데이터 프라이버시·보안**로 인해 원본 학습 데이터에 접근 불가.  
- 기존 post-training quantization(DFQ, ACIQ)·BNS 기반 샘플 생성(ZeroQ, GDFQ)은 ultra-low precision에서 성능 저하·샘플 다변성 부족 문제.

### 2.2 제안 방법  
1) **두 단계 불일치 모델링**  
- 출력 레벨 불일치:  

$$
       D_o(P,Q;G)=\mathbb E_{x_g\sim G(z)}\bigl\|\;P(x_g)-Q(x_g)\bigr\|_1
     $$
   
- 중간 레벨 불일치:  
     - 채널 관계 맵(Channel Relation Map, CRM) $$R^{(l)}\in\mathbb R^{C\times C}$$을 계산:  

$$
         R^{(l)}_{ij} = \frac{\langle f^{(l)}_i,\,f^{(l)}_j\rangle}{\|f^{(l)}_i\|_2\|f^{(l)}_j\|_2}
       $$
     
- 각 층 별 가중치 $$\omega^{(l)}$$를 적응적으로 계산하여:  

$$
       D_f(P,Q;G)=\sum_{l} \omega^{(l)} \frac{1}{C_l^2}\,\bigl\|\;R_P^{(l)}-R_Q^{(l)}\bigr\|_1
     $$

2) **적대적 학습 프레임워크**  
- 제너레이터 $$G$$ 업데이트 손실:  

$$
       \mathcal L_{DE} = -D_o(P,Q;G)-\alpha\,D_f(P,Q;G)\;+\;\beta\,\mathcal L_a
     $$

- 모델 $$Q$$ 업데이트 손실:  

$$
       \mathcal L_{KT} = D_o(P,Q;G)+\alpha\,D_f(P,Q;G)
     $$

3) **활성화 정규화**  

$$
     \mathcal L_a = -\frac{1}{M}\sum_{i=1}^M \bigl\|h^{P}_i\bigl\|_1
   $$

$$h^P_i$$: $$P$$ 최종 합성곱층 채널 활성화

### 2.3 모델 구조  
- **제너레이터**: DCGAN 변형, CIFAR는 채널 수 1/4로 경량화.  
- **양자화 함수**: 대칭 uniform quantization 적용, 스케일 $$S = \frac{2^{k-1}-1}{\max|x|}$$.  
- **학습 스케줄**: $$\alpha=0.1$$, $$\beta=0.05$$, CIFAR(200 epoch), ImageNet(300 epoch) 등.

***

## 3. 성능 향상 및 한계  
### 3.1 주요 성능 지표  
- **분류(ImageNet ResNet50 W4A4)**: ZAQ 70.06% vs ZeroQ 69.30% vs FT(원본 데이터) 76.13%  
- **분할(Cityscapes DeepLabv3 W4A4)**: ZAQ 55.12% vs ZeroQ 52.73% vs FT 55.98%  
- **검출(SSD MobileNetV2 W4A4)**: ZAQ 64.44% vs ZeroQ 62.72% vs FT 64.28%  
- **연산 효율**: 샘플 생성 GPU 시간 40–60% 절감

### 3.2 한계  
- **완전한 원본 도메인 복원 불가**: 생성 샘플이 사람 인식 가능 수준 아님.  
- **하이퍼파라미터 민감도**: $$\alpha,\beta$$ 값에 따라 성능 편차 존재.  
- **기존 고차원 구조(BERT 등) 적용 검증 미흡**

***

## 4. 일반화 성능 향상 가능성  
- CRM 기반 **중간 표현 학습**을 통해 단순한 출력 일치보다 더 풍부한 피처 관계를 전수하여, unseen 데이터에서도 **추론 일관성** 유지 가능성.  
- **활성화 정규화**가 모델의 입력 다양성을 증대시켜, 도메인 편중 없이 안정적 양자화 성능 제공 잠재력.  
- **적대적 샘플 생성**이 섀도우 데이터(shadow data)로서 다양한 경계 샘플 탐색을 유도, out-of-distribution 대응력 강화 기대.

***

## 5. 향후 연구 영향 및 고려 사항  
- **자동 혼합 정밀도**(automatic mixed precision)와 결합 시 효율+성능 동시 개선 기회.  
- **NLP·음성 모델** 등의 비전 외 도메인 확장: BERT ultra-low precision quantization 적용 연구.  
- **제너레이터 구조 개선**: 인간 인식 가능한 샘플 생성으로 interpretability 및 안전성 증대 필요.  
- **하이퍼파라미터 자동 탐색**: 최적 $$\alpha,\beta$$ 설정을 위한 메타러닝·베이지안 최적화 도입 검토.  
- **도메인 적응**: 타깃 도메인 샘플 일부 이용한 옵션적 fine-tuning과 결합하여 제너럴리티와 성능 균형 맞추기.  

***

**결론**: ZAQ는 데이터 프리 환경에서 양자화 모델의 성능 격차를 혁신적으로 줄이는 **첫 적대적 학습 기반** 접근으로, 향후 다양한 도메인·정밀도 설정에 적용 가능한 확장성과 안정적 일반화 성능을 제시한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e2307d1d-5f8e-4068-91f3-23d3e0c24b22/2103.15263v2.pdf)
