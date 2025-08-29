# DeepNet: Scaling Transformers to 1,000 Layers

**핵심 주장 및 주요 기여**  
DeepNet은 Residual-LayerNorm 구조를 새롭게 재정의한 **DeepNorm**을 도입하여 수백~천 개의 층을 가진 Transformer 모델을 안정적으로 학습할 수 있음을 보였다.  
- **DeepNorm**: 잔차 연결에 앞서 입력을 상수 α 배만큼 스케일링하고, 내부 서브레이어 가중치에 β를 곱해 초기화함으로써 모델 업데이트 크기를 상수 범위로 억제  
- **1,000층 학습 달성**: 기존 Post-LN/Pre-LN 구조로는 불가능했던 1,000층 이상의 Transformer 훈련을 성공적으로 수행  
- **성능 향상**: 200층·3.2B 모델이 48층·12B 상태-of-the-art 대비 BLEU +5 점, OPUS-100 다국어평가에서 평균 BLEU +4.4 점 달성  

***

## 1. 해결하고자 하는 문제  
Transformer가 **층 수**에 따라 학습 불안정성(모델 업데이트 폭발 및 그래디언트 소실)을 겪어 수백 개 층 이상의 확장에 실패한다는 문제를 다룬다.  

- Post-LN: 잔차 연결 후 LayerNorm → 초반 업데이트 폭발 후 소실  
- Pre-LN: 안정적이나 하위층 그래디언트가 과대하여 상위층 성능 저하  

## 2. 제안 방법: DeepNorm  
### 2.1 수식  
각 서브레이어 $$G_\ell$$(Self-Attn 또는 FFN) 뒤에 다음을 적용한다:  

$$
x_{\ell+1} = \mathrm{LayerNorm}(\alpha\,x_\ell + G_\ell(x_\ell; \theta_\ell)).
$$  

초기화 시 서브레이어 가중치 $$\theta_\ell$$에 대하여  

$$
\theta_\ell \leftarrow \beta\;\theta_\ell^{(\text{Xavier})},
$$  

여기서 $$\alpha,\beta$$는 전체 층수 $$N$$, 디코더 층수 $$M$$에 따라  

$$
\alpha_e = 0.81(N\,4M)^{\tfrac1{16}},\quad \beta_e = 0.87(N\,4M)^{-\tfrac1{16}},  
\quad
\alpha_d = (3M)^{\tfrac14},\quad \beta_d = (12M)^{-\tfrac14}.
$$

### 2.2 모델 구조  
기존 Transformer의 **Post-LN**을 DeepNorm으로 교체.  
- Encoder-only, Decoder-only, Encoder-Decoder 공통 적용  
- 잔차 연결 스케일 α, 초기화 스케일 β만 조정  

## 3. 성능 향상 및 한계  
### 3.1 성능  
- WMT-17 En→De: 100L-100L DeepNet 28.9 BLEU (Post-LN diverged)  
- IWSLT-14 De→En: 10L→100L까지 일관된 성능 향상[그림6]  
- OPUS-100 (100언어): 200L-863M 모델이 12L-133M 대비 +6.6 BLEU, 1,000L-3.8B 모델이 +7.6 BLEU[표2]  
- FLORES-101: 3.2B DeepNet vs 12B M2M-100 비교에서 모든 87언어쌍 평균 +2.1 BLEU[그림10 vs 그림9]  

### 3.2 한계  
- **계산 비용**: 수천 층 학습에 막대한 GPU 메모리·시간 요구  
- **일반화**: 대규모 다국어 번역 외 다른 작업에서의 일반화 연구 미비  
- **하이퍼파라미터 의존성**: α, β 튜닝 필요성  

## 4. 일반화 성능 향상 가능성  
- **업데이트 안정화**로 초기 학습 단계에서 과도한 로컬 옵티마 진입 방지→다양한 데이터셋에 적용 시 과적합 감소 기대  
- **깊이 증가의 표현력 향상**으로 소수 샘플 학습(few-shot) 및 파인튜닝 시 더 풍부한 특성 학습 가능  
- 언어 모델·비전·단백질 접힘 등 다른 도메인 Pre-train 및 Fine-tune 일반화  

## 5. 향후 연구 영향 및 고려 사항  
- **초깊은 구조**의 안정적 활용 가능성 제시: 수천 층 Transformer 연구 가속  
- **하이퍼파라미터 범용화**: α, β 자동 추정 또는 적응형 스케일링 방안 연구  
- **다중 모달 확장**: ViT, 단백질 모델, 음성·비전 멀티모달 학습에 적용  
- **효율성 개선**: 메모리 절감·모델 압축 기법과 결합하여 실용성 확보  

DeepNorm은 **잔차 연결과 초기화 설계**만으로도 극한 깊이의 Transformer를 안정화함으로써, 대규모 모델의 성능과 일반화 능력을 한 단계 끌어올리는 새로운 패러다임을 제시한다. 앞으로는 하이퍼파라미터 자동화, 다양한 도메인 적용, 비용 효율성 개선을 중점 고려해야 할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/05d6d24e-4ce3-45e4-a528-f9bfb211c09a/2203.00555v1.pdf)
