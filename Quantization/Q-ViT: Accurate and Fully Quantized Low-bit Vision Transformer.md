# Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer

**핵심 주장 및 주요 기여 요약**  
Q-ViT은 비트 수가 극히 낮은(2∼4비트) 완전 양자화(Vision Transformer; ViT) 모델에서도 정보 손실 없이 정확도를 유지하거나 향상시킬 수 있음을 보인다. 이를 위해 (1) 양자화된 어텐션 맵의 정보 왜곡을 최소화하는 **Information Rectification Module(IRM)**, (2) 어텐션 분포 차이를 정교하게 좁히는 **Distribution Guided Distillation(DGD)** 기법을 도입한다. 이 결과, 4비트 Q-ViT는 Full-precision 모델보다 ImageNet Top-1 정확도가 최대 1.0%p 향상되며, 6× 이상의 속도 가속을 달성한다.

***

## 1. 해결하고자 하는 문제  
- 대형 사전학습 ViT는 수백만~수십억 개 파라미터로 모바일·엣지 디바이스에 비효율적.  
- 기존 **Post-training Quantization(PTQ)** 방식은 ultra-low bit(≤4비트)에서 성능 급락.  
- **Quantization-Aware Training(QAT)** 기반 저비트 ViT 연구가 미비하며, 양자화된 self-attention에서 정보 왜곡이 심각하여 표현력이 떨어짐.

***

## 2. 제안 방법

### 2.1 Information Rectification Module (IRM)  
양자화 과정에서 쿼리·키 분포의 분산이 과도하게 축소되어 정보량이 크게 떨어지는 문제를 정보이론 관점(엔트로피 최대화)에 접근해 해결.  
- Full-precision 분포:  
  $$ q \sim \mathcal{N}(\mu_q,\,\sigma_q^2), \quad k \sim \mathcal{N}(\mu_k,\,\sigma_k^2) $$  
- IRM 변환:  

$$
    \tilde{q} = \frac{q - \mu_q + \beta_q}{\gamma_q}\,\sqrt{\sigma_q^2 + \varepsilon_q}, 
    \quad
    \tilde{k} = \frac{k - \mu_k + \beta_k}{\gamma_k}\,\sqrt{\sigma_k^2 + \varepsilon_k}
  $$  
  
  여기서 $$\gamma, \beta$$는 학습 가능한 파라미터, $$\varepsilon$$는 안정화 상수.  
- 양자화 후 엔트로피:  

$$
    H\bigl(Q(\tilde{q})\bigr)=\tfrac12\log\bigl(2\pi e\,\gamma_q^2(\sigma_q^2+\varepsilon_q)\bigr)
    \longrightarrow\text{최대화}
  $$

### 2.2 Distribution Guided Distillation (DGD)  
- 기존 클래스 토큰 기반 출력 로짓 비교만으로는 양자화된 어텐션 최적화에 한계.  
- **패치 간유사도 행렬**을 교사-학생 어텐션 활성화로부터 계산해 직접 지도:  

```math
    G^{(l)}_{q,h} = \frac{\tilde{q}_{l,h}\,\tilde{q}_{l,h}^\top}{\|\tilde{q}_{l,h}\,\tilde{q}_{l,h}^\top\|_2}, 
    \;\;
    G^{(l)}_{k,h} = \frac{\tilde{k}_{l,h}\,\tilde{k}_{l,h}^\top}{\|\tilde{k}_{l,h}\,\tilde{k}_{l,h}^\top\|_2\!},
``` 

- 손실함수:  

```math
    L_{\rm DGD}
    = \sum_{l,h}\Bigl\|G^{(l),T}_{q,h}-G^{(l)}_{q,h}\Bigr\|_2
    +\Bigl\|G^{(l),T}_{k,h}-G^{(l)}_{k,h}\Bigr\|_2,
    \quad
    L_{\rm total}=L_{\rm CE}+\lambda L_{\rm DGD}
```

***

## 3. 모델 구조  
- ViT-S, ViT-B, Swin-T, Swin-S 백본에 완전 양자화(가중치·활성화 2∼4비트) 적용  
- 첫 번째 패치 임베딩, 마지막 분류 헤드는 8비트 유지  
- IRM-변형된 Q-Attention 및 DGD 추가한 Q-ViT 블록 반복

***

## 4. 성능 향상 및 한계  
| 백본     | 비트(Weight-Activation) | Baseline Top-1 | Q-ViT Top-1 | 향상폭 |
|---------|-------------------------|--------------|------------|-------|
| DeiT-S  | 4-4                     | 79.7%        | **80.9%**  | +1.2% |
| DeiT-S  | 2-2                     | 68.2%        | **72.1%**  | +3.9% |
| Swin-T  | 4-4                     | 80.5%        | **82.5%**  | +2.0% |
| Swin-S  | 4-4                     | 82.9%        | **84.4%**  | +1.5% |

- **Ultra-low bit(2비트)**에서도 full-precision 대비 단 9%p 하락에 그치며, 6× 이상 연산 가속  
- *한계*: 적은 비트에서 분포가 극도로 단순해지면 IRM 파라미터 최적화가 어려워 불안정 발생 가능. 또한, 다른 태스크(물체 검출·분할) 일반화 검증 필요.

***

## 5. 일반화 성능 향상 가능성  
- IRM의 엔트로피 최대화는 다양한 입력 분포에 적응적 학습을 유도, unseen 도메인에서도 표현력 유지에 유리  
- DGD의 패치 수준 정교 지도는 객체 밀집 장면·도메인 이동 상황에서 의미론적 구조 보존에 기여  
- 따라서 *도메인 적응*, *자율 주행·의료 영상* 등 다양한 CV 태스크에서 일반화 성능 개선 기대

***

## 6. 향후 연구에 미치는 영향 및 고려 사항  
- **초저비트 트랜스포머** 연구 활성화: Q-ViT의 정보 이론적 접근은 새로운 양자화 설계 패러다임 제시  
- **하드웨어 협업**: 엔트로피 최대화 모듈이 실제 AI 칩에서 효율적으로 구현될 수 있는지 평가 필요  
- **다양한 태스크 확장**: 검출·분할·영상 생성 등으로 DGD/IRM 일반화 검증  
- **안정성 연구**: 1비트 초저비트 영역에서의 학습 안정성, 파라미터 초기화 전략 등 추가 고찰이 요구됨.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e414afa5-3b86-4ddc-82d7-73fc32faf323/2210.06707v1.pdf)
