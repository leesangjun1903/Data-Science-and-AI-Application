# LoRA: Low-Rank Adaptation of Large Language Models | LLM, Fine-tuning, Mathematical Reasoning

## 1. 핵심 주장 및 주요 기여  
“LoRA(저계수 적응)”는 초대형 사전학습 언어 모델(예: GPT-3 175B)의 전체 파라미터를 미세조정(fine-tuning)하는 대신, 사전학습된 가중치는 고정하고 각 Transformer 계층의 일부 밀집층(weight matrices)에 **저계수(rank-r) 보조 행렬**만 학습해 downstream 과제에 효율적으로 적응하는 기법이다.  
- **파라미터 효율성**: 학습 가능한 파라미터 수를 최대 10,000배까지 감소.  
- **메모리 절감**: Adam 옵티마이저 상태 저장 비용을 줄여 GPU 메모리 요구량을 최대 3분의 1로 저감.  
- **추론 속도 유지**: 학습된 보조 행렬을 사전학습 가중치와 병합 가능하여, Adapter 대비 추가 추론 지연이 없음.  
- **동등 이상 성능**: RoBERTa, DeBERTa, GPT-2/3에서 full fine-tuning과 동등하거나 더 높은 성능 달성.  

## 2. 문제 정의 및 제안 방법  
### 2.1 문제 정의  
사전학습 모델 $$P_{\Phi_0}(y\mid x)$$를 downstream 데이터 $$\mathcal{Z}=\{(x_i,y_i)\}$$에 맞춰 적응할 때, 전통적 full fine-tuning은 파라미터 증분 $$\Delta\Phi\in\mathbb{R}^{|\Phi_0|}$$를 학습하여  

$$
\max_{\Phi}\sum_{(x,y)\in\mathcal{Z}}\sum_{t=1}^{|y|}\log P_{\Phi}(y_t\mid x,y_{ < t })
$$

을 해온다. 모델이 커질수록 $$|\Delta\Phi|\approx|\Phi_0|$$이 비용·저장·전환 측면에서 비실용적이다.  

### 2.2 LoRA 기법  
사전학습 가중치 $$W_0\in\mathbb{R}^{d\times k}$$는 고정하고, 그 변화를 저계수 행렬 $$A\in\mathbb{R}^{r\times k}, B\in\mathbb{R}^{d\times r}$$로 근사:  

$$
W = W_0 + \Delta W,\quad\Delta W = B\,A,\quad r\ll\min(d,k).
$$

입력 $$x$$에 대해  

$$
h = W_0 x + (B A)x = W_0 x + B(Ax).
$$

- **초기화**: $$A\sim\mathcal{N}(0,\sigma^2)$$, $$B=0$$ → 학습 초기 $$\Delta W=0$$.  
- **스케일링**: $$\Delta W x$$에 $$\alpha/r$$ 배수 적용으로 학습률 재조정 불필요.  
- **병합**: 추론 시 $$W_0 + BA$$로 병합해 추가 계층 없이 원래 연산으로 처리.  

### 2.3 모델 구조 적용  
- Transformer의 **Self-Attention** 내 $$W_q,W_k,W_v,W_o$$ 중 주로 $$W_q,W_v$$에 LoRA 적용.  
- MLP 및 LayerNorm 은 고정.  
- 전체 96-layer GPT-3 175B의 경우 $$r=4$$일 때 $$W_q,W_v$$ 두 행렬만 학습하여 35 MB 체크포인트(10,000× 절감).  

## 3. 성능 향상 및 한계  
### 3.1 성능 요약  
| 모델           | 파라미터 증가량 | downstream 과제 성능                        |
|---------------|---------------|-------------------------------------------|
| RoBERTa Large | +0.8 M       | GLUE 전체 + 0.1% 우위                     |
| DeBERTa XXL   | +4.7 M       | GLUE 전체 + 0.2% 우위                     |
| GPT-2 Medium  | +0.35 M      | E2E NLG 챌린지 BLEU +2.2점                  |
| GPT-3 175B    | +4.7 M       | WikiSQL, MNLI, SAMSum 모두 fine-tune 상회 성능 |

### 3.2 일반화 성능  
- LoRA 업데이트 $$\Delta W$$의 **“내재 랭크”**가 매우 낮음을 확인.  
  - $$r=1$$만으로도 GPT-3 WikiSQL 성능 70%+ 달성.  
  - 1차 상위 특이값 방향이 $$r=64$$에도 공통으로 학습됨 → 핵심 변화는 극히 저차원 공간에 집중.  
- $$\Delta W$$가 사전학습 가중치 $$W_0$$의 상위 특이 방향이 아닌 **미지지된(task-specific) 피처**를 증폭.  
  - 증폭 계수 $$\|\Delta W\|\_F / \|P_{ \mathrm{span}(\Delta W)}(W_0)\|_F\approx20$$ (r=4) → task에 특화된 신호 강화로 일반화.  
- 저계수 제약에도 **과적합 방지** 효과 → 적은 학습 샘플에서도 안정적 성능.  

### 3.3 한계 및 주의점  
- **배치 내 다과제 학습**: 서로 다른 $$A,B$$를 가진 샘플 동시 처리 어려움.  
- **MLP·LayerNorm 미적용**: 추후 연구로 확장 필요.  
- **랭크-과소 설정 위험**: 극단적 $$r$$ 축소 시 표현력 제한 가능성.  

## 4. 향후 연구 방향 및 고려 사항  
- **다중 효율적 적응 기법 결합**: prefix-tuning, 케프-기반 어댑터와 조합한 하이브리드 탐색  
- **적응 행렬 선택 전략**: 어떤 계층·어텐션 행렬에 LoRA를 적용할지 자동화  
- **정량적 설명가능성**: $$\Delta W$$ 내재 구조 분석으로 미세조정 메커니즘 이론적 규명  
- **사전학습 모델 랭크 제약**: 사전 가중치 $$W_0$$ 자체의 저랭크 성질 연구로 경량화 모델 설계  

LoRA는 초대형 언어 모델 미세조정의 **효율성**과 **일반화 가능성**을 크게 높이며, 향후 경량화·적응적 NLP 시스템 연구의 핵심 패러다임으로 자리매김할 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e0ce1087-964d-4457-ad37-6b9d226aac2c/2106.09685v2.pdf

LoRA(Low-Rank Adaptation)는 대규모 언어 모델(LLM)을 효율적으로 미세 조정(fine-tuning)하기 위한 혁신적인 기법입니다. 기존의 전체 파라미터 재조정 방식과 달리, 사전 훈련된 모델 가중치를 고정한 상태에서 **저순위 행렬 분해(low-rank decomposition)**를 활용해 적은 수의 파라미터만 업데이트합니다. 이로 인해 GPT-3 175B 같은 초대형 모델의 경우 **훈련 가능 파라미터를 10,000배 이상 감소**시키면서도 동등하거나 더 나은 성능을 달성합니다[1][2].  

#### 1. **핵심 작동 원리**  
- **저순위 행렬 쌍 적용**: Transformer 레이어의 기존 가중치 행렬 $$W$$에 $$W + A \times B$$ 형태로 업데이트합니다. 여기서 $$A$$(차원 $$d \times r$$)와 $$B$$(차원 $$r \times d$$)는 **초기값이 0인 저순위 행렬**($$r \ll d$$)로, 훈련 중에만 조정됩니다[1][4].  
- **파라미터 효율성**: 예를 들어 $$d=1,000$$, $$r=8$$일 경우 기존 1M 파라미터 대비 $$2 \times 1,000 \times 8 = 16,000$$개만 훈련하면 됩니다[2].  

#### 2. **주요 장점**  
- **저장 공간 절감**: GPT-3 175B의 체크포인트 크기가 1TB에서 **25MB로 감소**[2].  
- **추론 지연 제거**: 훈련 후 $$A \times B$$를 원본 가중치에 병합(merge)하여 **추론 속도 저하 없음**[1][4].  
- **다중 작업 전환**: 작업별 LoRA 모듈(예: $$A_{\text{프랑스어}} \times B_{\text{프랑스어}}$$)을 실시간 교체하며 **다양한 태스크 지원**[2][4].  

#### 3. **실험 결과**  
- **성능 검증**: RoBERTa, DeBERTa, GPT-2, GPT-3에서 **전체 미세 조정과 동등하거나 우수한 성능** 달성[1].  
- **훈련 효율성**: GPU 메모리 사용량 **3배 감소**, 훈련 처리량(throughput) 향상[1].  

#### 4. **적용 가이드**  
- **순위($$r$$) 선택**: 낮은 순위(예: $$r=8$$)로 시작해 필요 시 점진적으로 증가[2].  
- **전체 미세 조정 필요 시점**: 사전 훈련 데이터와 완전히 다른 도메인(예: 영어→화성어) 작업 시 권장[2].  
- **범용성**: **선형 변환을 사용하는 모든 모델 구조**(CNN, Transformer 등)에 적용 가능[2][4].  

#### 5. **공학적 활용 사례**  
- **RAM 캐싱**: 여러 LoRA 모듈을 메모리에 저장해 **실시간 작업 전환** 지원[2].  
- **병렬 훈련**: 서로 다른 배치 데이터로 **동시에 다중 LoRA 모듈 훈련**[2].  

> LoRA는 대규모 모델의 효율적 적용을 가능케 하는 핵심 기술로, 오픈소스 구현체는 [Microsoft의 GitHub 저장소](https://github.com/microsoft/LoRA)에서 확인할 수 있습니다[1].  

이 기법은 모델의 과적합(over-parametrization) 특성을 활용해 적은 자원으로도 높은 성능을 이끌어내며, LLM의 실용적 배포를 가속화합니다[1][2][4].

[1] https://openreview.net/forum?id=nZeVKeeFYf9
[2] https://weaviate.io/papers/lora
[3] https://openreview.net/pdf?id=nZeVKeeFYf9
[4] https://www.ibm.com/docs/en/watsonx/w-and-w/2.1.0?topic=tuning-lora-fine
[5] https://www.ibm.com/think/topics/lora
[6] https://arxiv.org/abs/2502.14816
[7] https://dl.acm.org/doi/10.1145/3727582.3728688
[8] https://www.semanticscholar.org/paper/LoRA:-Low-Rank-Adaptation-of-Large-Language-Models-Hu-Shen/a8ca46b171467ceb2d7652fbfb67fe701ad86092
[9] https://portkey.ai/blog/lora-low-rank-adaptation-of-large-language-models-summary/
[10] https://arxiv.org/abs/2309.14717
[11] https://arxiv.org/abs/2406.01775
[12] https://arxiv.org/abs/2409.02119
[13] https://ieeexplore.ieee.org/document/10711229/
[14] https://ieeexplore.ieee.org/document/10946960/
[15] https://arxiv.org/abs/2410.16801
[16] https://arxiv.org/abs/2502.19747
[17] https://arxiv.org/abs/2402.10462
[18] https://arxiv.org/abs/2106.09685
[19] https://github.com/microsoft/LoRA
[20] https://aclanthology.org/2024.lrec-main.206.pdf
