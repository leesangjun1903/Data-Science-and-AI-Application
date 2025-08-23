# Post-Training Quantization for Vision Transformer

## 1. 핵심 주장 및 주요 기여  
이 논문은 **사전 학습된 Vision Transformer(ViT·DeiT 등)에** 추가적인 재학습 없이도 모바일·엣지 환경에서 효율적으로 구동할 수 있도록 하는 **포스트 트레이닝 양자화(Post-Training Quantization, PTQ)** 알고리즘을 제안한다.  
- **유사도 인식 양자화**: 양자화 후 출력 특성맵과 원본 간 상관관계를 극대화  
- **순위 인식 양자화**: 어텐션 맵의 상대적 순서를 유지하도록 순위 기반 손실(ranking loss) 도입  
- **바이어스 보정**: 양자화 오차에 의한 출력 편향을 보정  
- **혼합 정밀도 결정**: 각 레이어의 **어텐션 맵·출력 특징의 핵심 특이값 핵노름(nuclear norm)** 을 활용해 민감도가 높은 레이어에는 더 높은 비트폭 할당  

이 결과, 추가 데이터 없이도 8-bit 양자화된 ViT 모델이 원본 대비 성능 저하를 1–2% 이내로 억제하고, 일반화 성능 또한 유지·향상함을 보였다.

***

## 2. 문제 정의 및 제안 기법 상세  

### 2.1 해결하고자 하는 문제  
- Vision Transformer는 수백만~수억 개의 파라미터와 대규모 행렬 연산으로 실시간·저전력 환경에서 실행이 어려움  
- 기존 PTQ 기법은 CNN에 최적화되어 있어, **Transformer의 self-attention 특성**을 보존하지 못해 성능 저하가 크다는 한계  

### 2.2 제안 방법  

#### 2.2.1 유사도 인식 양자화 (Similarity-Aware Quantization)  
레이어 $$l$$의 출력  

$$
O_l = X_l W_l
$$  

를 양자화했을 때  

$$
\hat O_l = \Psi_{\Delta X_l}(X_l)\,\Psi_{\Delta W_l}(W_l)\,\Delta X_l\,\Delta W_l
$$  

로 근사하며, **Pearson 상관계수** $$\Gamma$$를 최대화:  

$$
\max_{\Delta X_l,\Delta W_l} \frac1N\sum_{i=1}^N \Gamma\bigl(O_l^{(i)},\,\hat O_l^{(i)}\bigr)
$$  

#### 2.2.2 순위 인식 양자화 (Ranking-Aware Quantization)  
어텐션 맵 $$A_l$$의 값들 순서가 바뀌면 성능이 급격히 떨어지므로, **힌지 함수 기반 순위 손실** 추가:  

$$
\min_{\Delta X_l,\Delta W_l} -\frac1N\sum_i \Gamma(O_l^{(i)},\hat O_l^{(i)}) + \gamma\,L_{\text{ranking}}
$$  

$$
L_{\text{ranking}}
=\sum_{k=1}^h \sum_{i < j}\max\bigl(0,\theta - (\hat A_{ki}-\hat A_{kj})\text{sign}(A_{ki}-A_{kj})\bigr)
$$  

#### 2.2.3 바이어스 보정 (Bias Correction)  
양자화 오차 $$\epsilon_X,\epsilon_W$$가 출력 평균을 이동시키는 편향을 교정:  

$$
E[\hat O] = E[O] + E[\epsilon_W X] + E[\epsilon_X W] + E[\epsilon_X\epsilon_W]
$$  

교정값을 레이어 바이어스에 빼주는 방식으로 분포 변화 억제  

#### 2.2.4 핵노름 기반 혼합 정밀도 (Mixed-Precision)  
각 레이어의 **어텐션 맵과 MLP 출력 특징** 행렬에 대한 **핵노름**(singular values 합)을 계산하여 민감도 측정,  
Pareto 전선 기반으로 총 모델 크기 제약 내 비트 조합 최적화  

### 2.3 모델 구조  
- 입력 패치 임베딩 → 다중 헤드 Self-Attention + MLP 레이어를 반복  
- Softmax·LayerNorm은 양자화 대상에서 제외  
- 양자화된 가중치·입력으로 행렬곱 실행  

***

## 3. 성능 향상 및 한계  

| 모델       | 데이터셋   | 비트폭 | Top-1 정확도 (원본) | 비고                                     |
|-----------|-----------|-------|---------------------|-----------------------------------------|
| DeiT-B    | ImageNet  | 8-bit MP | 81.8%               | 8-bit 혼합정밀도: 81.29% (−0.51% 감소)  |
| ViT-B     | CIFAR-10  | 8-bit MP | 98.13%              | 97.79% (−0.34%)                         |
| DETR      | COCO2017  | 8-bit MP | 42.0 mAP            | 41.7 mAP (−0.3 mAP)                     |

- **일반화 성능 보존**: 소규모(100–1,000장) 보정 데이터만으로도 다양한 데이터셋(CIFAR, ImageNet, COCO)에서 결국 원본 대비 1% 내외 성능 저하  
- **계산·메모리 절감**: 8-bit 양자화 시 모델 크기 약 75% 감소, 연산량 약 60% 절감  
- **한계**:  
  - 소규모 보정 데이터가 도메인 불일치 시 일반화 보장이 불분명  
  - 순위 손실·혼합정밀도 탐색에 추가 계산 비용  
  - Softmax·Norm 미양자화로 하드웨어 완전 정수화 미지원  

***

## 4. 향후 연구에의 영향 및 고려 사항  
이 논문은 **Transformer 양자화 연구**에 다음과 같은 발판을 제공한다.  
- **순위 손실** 개념을 통해 어텐션 기법 특성 보존 중요성 강조  
- **핵노름 기반 민감도 측정**으로 혼합정밀도 자동화 가능성 제시  

향후 연구 시 고려점:  
- 다양한 도메인·보정 데이터 규모 간 일반화 경계 분석  
- Softmax·Norm 정수화 기법과의 연계  
- 순위 손실 가중치·핵노름 민감도 지표 최적화  
- 탐색 비용 저감을 위한 효율적 비트폭 결정 알고리즘  

이로써 엣지·모바일 환경에서 비전 트랜스포머를 고성능으로 운영하기 위한 **양자화·압축 연구**에 중요한 방향성을 제시한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3df35b10-abea-4300-9444-6fd4bebfad2a/2106.14156v1.pdf)
