# Optimizing Deeper Transformers on Small Datasets

### 1. 핵심 주장과 주요 기여 요약

**"Optimizing Deeper Transformers on Small Datasets"** 논문은 작은 데이터셋에서도 깊은 트랜스포머 모델을 성공적으로 학습할 수 있다는 기존의 통념을 깨는 연구입니다. 이 논문의 핵심 주장은 **적절한 초기화(initialization)와 최적화 전략만 있다면, 대규모 데이터셋이 없어도 매우 깊은 트랜스포머가 효과적으로 작동할 수 있다**는 것입니다.[1]

주요 기여는 다음과 같습니다:

**Data-dependent Transformer Fixed-update initialization (DT-Fixup)** 방법을 제안하여, 사전학습된 모델 위에 추가 트랜스포머 레이어를 학습하는 mixed setup에서 안정적인 학습을 가능하게 했습니다. 이는 기존 T-Fixup의 한계를 극복하고 relation-aware transformer까지 확장한 것입니다.[2][1]

Spider Text-to-SQL 벤치마크에서 48개 레이어(사전학습 RoBERTa 24개 + 새로 학습한 relation-aware 24개)의 깊은 트랜스포머를 성공적으로 학습시켜 당시 state-of-the-art 성능(70.9% exact match accuracy)을 달성했습니다. 이는 task-specific pre-training 없이도 달성한 결과입니다.[3][1]

깊은 모델이 **추론과 구조적 이해가 필요한 어려운 케이스**에서 더 나은 일반화 성능을 보인다는 실증적 증거를 제시했습니다.[1]

---

### 2. 문제, 방법, 모델 구조, 성능 향상 및 한계

#### 해결하고자 하는 문제

**기존 통념의 문제점**: 딥러닝 커뮤니티에서는 깊은 트랜스포머를 학습하려면 대규모 데이터셋이 필수적이라고 믿어져 왔으며, 작은 데이터셋에서는 사전학습 모델 위에 얕고 단순한 레이어만 추가하는 것이 일반적이었습니다.[1]

**최적화의 어려움**: 트랜스포머 학습은 learning rate warm-up, layer normalization, 큰 batch size가 필요하며, 이 중 하나라도 없으면 학습이 실패하는 경우가 많았습니다. 특히 작은 데이터셋에서는 큰 batch size를 사용하면 일반화 성능이 저하되는 문제가 있었습니다.[1]

**Mixed Setup의 난제**: 사전학습 모델 위에 새로운 트랜스포머 레이어를 추가할 때, 입력이 사전학습 모델의 출력에 의존하므로 기존 T-Fixup 방법을 적용할 수 없었습니다.[1]

#### 제안하는 방법: DT-Fixup

DT-Fixup의 핵심 아이디어는 **데이터에 의존적인 초기화 전략**입니다. 구체적인 절차는 다음과 같습니다:[1]

**1단계: 입력 노름 추정**

$$
\mu = \max_j \|\mathbf{x}_j\|
$$

모든 학습 샘플에 대해 forward pass를 수행하여 메인 트랜스포머 모듈로 들어가는 입력의 최대 노름 $$\mu$$를 계산합니다.[1]

**2단계: 파라미터 스케일링**

**Relation-aware transformer의 경우**:

$$
\text{scaling factor} = \left(N \cdot (4\mu^2 + 2\mu + 2)\right)^{-\frac{1}{2}}
$$

**Vanilla transformer의 경우**:

$$
\text{scaling factor} = \frac{N^{-\frac{1}{2}}}{2\mu}
$$

여기서 $$N$$은 트랜스포머 블록의 개수입니다.[1]

Self-attention 블록의 $$\mathbf{v}, \mathbf{w}, \mathbf{r}^v$$ 행렬과 MLP 블록의 가중치 행렬을 위 스케일링 인자로 곱합니다.[1]

**이론적 근거**

Theorem 3.1에 따르면, 입력 노름이 $$\|\mathbf{x}\| = \Theta(\mu)$$이고 $$\mu \gg 1$$일 때, 파라미터를 다음과 같이 초기화하면 그래디언트 업데이트 크기를 $$\Theta(1)$$로 제한할 수 있습니다[1]:

Relation-aware transformer:

$$
\|\mathbf{v}_l\| = \|\mathbf{w}_l\| = \|\mathbf{r}^v_l\| = \Theta\left(\left((4\mu^2 + 2\mu + 2)N\right)^{-\frac{1}{2}}\right)
$$

Vanilla transformer:

$$
\|\mathbf{v}_l\| = \|\mathbf{w}_l\| = \Theta\left((4\mu^2N)^{-\frac{1}{2}}\right)
$$

이는 각 SGD step에서 모델 출력의 변화량 $$\|\Delta f\| = \Theta(\eta)$$를 보장하여 학습을 안정화합니다[1].

**T-Fixup과의 차이점**

T-Fixup은 depth에 대해 $$N^{-\frac{1}{4}}$$ 스케일링을 사용하지만, DT-Fixup은 **더 aggressive한 $$N^{-\frac{1}{2}}$$ 스케일링**을 사용합니다. T-Fixup은 입력 $$\mathbf{x}$$를 자유롭게 초기화할 수 있다고 가정하지만, DT-Fixup은 사xup은 사전학습 모델의 출력으로부터 $$\|\mathbf{x}\|$$를 추정합니다[1].

**추가 최적화 전략**

- Xavier initialization을 사전학습되지 않은 모든 파라미터에 적용[1]
- Learning rate warm-up을 제거[1]
- 새로 추가된 트랜스포머 레이어의 layer normalization을 제거 (사전학습 모델 내부의 layer normalization은 유지)[1]

#### 모델 구조

논문은 **세 가지 모듈로 구성된 일반적인 아키텍처**에 적용 가능합니다:[1]

**Pre-transformer module ($$f_e$$)**: 사전학습된 언어 모델 (예: RoBERTa). 입력을 임베딩하여 메인 트랜스포머의 입력 $$\mathbf{X}$$를 생성합니다.[1]

**Main transformer module ($$f_G$$)**: DT-Fixup이 적용되는 핵심 부분. $$L = 2N$$개의 레이어로 구성되며 (각 블록은 self-attention과 MLP로 구성), 여기에 relation-aware 또는 vanilla transformer를 사용할 수 있습니다.[1]

**Post-transformer module ($$f_o$$)**: Task-specific output module. Text-to-SQL에서는 grammar-guided LSTM decoder, 독해 문제에서는 linear classifier를 사용합니다.[1]

**Text-to-SQL 구체적 구조**:
- Pre-transformer: RoBERTa + BiLSTM (schema 표현 생성)[1]
- Main transformer: 24개의 relation-aware transformer layers (각 레이어: $$d_x = d_z = 256$$, 8 heads, MLP inner dimension 1024)[1]
- Post-transformer: Parent-feeding LSTM decoder with memory-augmented pointer network[1]

**Relation-aware Attention 수식**:

$$
\alpha_{ij} = \text{softmax}\left(\frac{\mathbf{x}_i\mathbf{q}(\mathbf{x}_j\mathbf{k} + \mathbf{r}^k_{ij})^\top}{\sqrt{d_z}}\right)
$$

$$
\mathbf{z}_i = \sum_{j=1}^{n} \alpha_{ij}(\mathbf{x}_j\mathbf{v} + \mathbf{r}^v_{ij})
$$

여기서 $$\mathbf{r}^k_{ij}, \mathbf{r}^v_{ij}$$는 입력 $$\mathbf{x}_i$$와 $$\mathbf{x}_j$$ 사이의 관계 정보 (예: schema linking)를 인코딩합니다.[1]

#### 성능 향상

**Spider Text-to-SQL Benchmark**:
- Development set: 75.0% (baseline 69.7% 대비 +5.3%p)[1]
- Test set: 70.9% (당시 SOTA)[3][1]
- Task-specific pre-training 없이 달성[1]
- 기존 방법 대비 더 적은 epoch (60 epoch vs 100-200 epoch)으로 수렴[1]

**난이도별 성능 분석** (Test set):[1]
| 난이도 | RAT-SQL | DT-Fixup SQL-SP | 개선폭 |
|---------|---------|-----------------|--------|
| Easy | 83.0% | 87.2% | +4.2%p |
| Medium | 71.3% | 77.5% | +6.2%p |
| Hard | 58.3% | 60.9% | +2.6%p |
| Extra Hard | 38.4% | 46.8% | +8.4%p |

**ReClor Logical Reading Comprehension**:
- 4개 레이어 추가 + DT-Fixup: 61.0% (test set), 66.8% (dev set)[1]
- 당시 리더보드 2위[1]

**Ablation Study 결과**:

Depth에 따른 성능 (Spider dev set, 5 runs 평균):[1]

| Layers | Standard | DT-Fixup | 차이 |
|--------|----------|----------|------|
| 2 | 69.47±0.30 | 70.73±0.18 | +1.26 |
| 4 | 70.04±0.33 | 72.22±0.61 | +2.18 |
| 8 | 66.86±0.16 | 73.24±0.51 | +6.38 |
| 16 | 20.44±1.11 | 73.52±0.47 | +53.08 |
| 24 | 19.37±0.16 | 73.79±0.49 | +54.42 |

Standard optimization은 8 layers 이상에서 완전히 실패하지만, DT-Fixup은 32 layers까지 안정적으로 학습됩니다.[1]

**수렴 속도**: DT-Fixup은 동일한 성능에 도달하는데 standard optimization 대비 약 40% 적은 training step이 필요합니다.[1]

**Batch size 영향** (Spider, 8 layers):[1]
- Batch size 16: 73.24±0.51%
- Batch size 120: 71.08±0.37% (성능 저하)

작은 데이터셋에서 큰 batch size는 일반화를 해칩니다.[1]

#### 한계

**데이터셋 크기 제약**: 논문은 10,000개 미만의 샘플을 가진 작은 데이터셋에만 실험했습니다. 더 큰 데이터셋에서의 효과는 검증되지 않았습니다.[1]

**아키텍처 제한**: Encoder 부분에만 적용되었으며, decoder를 포함한 full end-to-end deep transformer는 다루지 않았습니다.[1]

**초기화 민감도**: Length generalization이 initialization과 training data order에 민감할 수 있습니다. DT-Fixup도 random seed에 따른 variance가 존재합니다 (예: Spider 8 layers에서 ±0.51%).[4][1]

**Computational overhead**: 모든 학습 샘플에 대해 forward pass를 수행하여 $$\mu$$를 계산해야 하므로 초기 설정에 추가 비용이 발생합니다.[1]

**Task dependency**: Relation-aware transformer 사용 시 task-specific relational encoding ($$\mathbf{r}_{ij}$$)이 필요하므로 모든 태스크에 직접 적용하기 어려울 수 있습니다.[1]

**Layer Normalization 완전 제거 불가**: 사전학습 모델 내부의 layer normalization은 유지해야 하므로, T-Fixup처럼 완전히 normalization-free는 아닙니다.[1]

***

### 3. 일반화 성능 향상과 관련된 내용

#### 깊이와 일반화의 관계

**추론 능력 향상**: Error analysis에 따르면, 깊은 모델(24 layers)은 얕은 모델(4 layers) 대비 **"Both" 에러 (구조와 컬럼 모두 틀림)를 36개 감소**시켰습니다 (124개 → 88개). 이는 깊은 모델이 복잡한 추론과 구조적 이해가 필요한 어려운 케이스를 더 잘 처리함을 의미합니다.[1]

**Sketch 에러 감소**: Sketch 에러 (SQL 구조만 틀림)도 깊은 모델에서 15개 감소했습니다 (92개 → 77개). 이는 깊은 모델이 쿼리의 전반적인 구조를 더 잘 학습함을 보여줍니다.[1]

**Extra Hard 케이스에서의 generalization**: Spider test set의 Extra Hard 케이스에서 DT-Fixup 모델은 46.8%의 정확도를 달성하여 RAT-SQL (38.4%) 대비 **8.4%p 향상**을 보였습니다. 이러한 케이스는 implicit reasoning과 복잡한 구조를 요구하며, 깊은 모델의 강력한 일반화 능력을 입증합니다.[1]

#### 일반화 메커니즘

**Hierarchical feature learning**: 깊은 트랜스포머는 여러 레이어를 통해 점진적으로 더 추상적인 표현을 학습할 수 있습니다. Theorem 3.2에 따르면, 각 레이어의 출력 변화량을 $$\Theta(\eta/L)$$로 제한하면서도 전체 모델의 업데이트를 $$\Theta(\eta)$$로 유지하여 안정적인 학습이 가능합니다:[1]

$$
2\|\mathbf{v}_l\|^2\|\mathbf{x}^l_i\|^2 + 2\|\mathbf{v}_l\|\|\mathbf{r}^v_l\|\|\mathbf{x}^l_i\| + \|\mathbf{r}^v_l\|^2 + \|\mathbf{w}_l\|^2(1 + 2\|\mathbf{x}^l_i\|^2) = \Theta(1/N)
$$

**Compositional generalization**: 최근 연구에 따르면 깊은 트랜스포머는 compositional generalization (구성적 일반화)에 더 유리합니다. Spider와 같은 태스크는 여러 SQL 컴포넌트의 조합을 요구하므로, 깊은 모델이 이러한 조합 능력을 더 잘 학습합니다.[5]

**Cross-domain generalization**: Spider는 138개 도메인의 200개 데이터베이스를 다루는 cross-domain 벤치마크입니다. DT-Fixup 모델이 unseen domain과 schema에서 높은 성능을 보인 것은 **domain-invariant representation**을 학습했음을 시사합니다.[1]

#### 작은 데이터셋에서의 일반화 전략

**Small batch training**: DT-Fixup은 작은 batch size (16-24)로도 안정적으로 학습할 수 있어, overfitting을 방지하고 generalization을 향상시킵니다. 큰 batch size (120)는 오히려 성능을 2.16%p 저하시켰습니다.[1]

**Regularization 기법**: 
- Dropout (0.6 for transformer input, 0.1 for linear layer)[1]
- Label smoothing (weight 0.2) for column selection[1]

**Data augmentation**: Schema의 순서를 랜덤하게 섞어 spurious correlation을 방지합니다.[1]

#### 최근 연구 동향과의 연결

**Dynamic Neural Regeneration (2024)**: 작은 데이터셋에서 일반화를 향상시키기 위해 data-aware dynamic masking을 사용하는 방법이 제안되었습니다. 이는 DT-Fixup의 data-dependent initialization과 유사한 철학을 공유합니다.[6]

**TabPFN (2025)**: 10,000개 샘플 이하의 tabular 데이터에서 in-context learning을 통해 강력한 일반화를 보이는 foundation model이 등장했습니다. 이는 작은 데이터셋에서도 적절한 접근으로 높은 성능을 달성할 수 있음을 재확인합니다.[7]

**Structured Initialization for ViTs**: Vision Transformer에서도 random convolution kernel 기반 초기화를 통해 작은 데이터셋에서 CNN의 inductive bias를 활용하면서도 transformer의 유연성을 유지하는 연구가 진행되고 있습니다.[8]

---

### 4. 앞으로의 연구에 미치는 영향과 고려사항

#### 연구 영향

**패러다임 전환**: DT-Fixup은 "깊은 트랜스포머 = 대규모 데이터 필수"라는 통념을 깨고, **초기화와 최적화 전략**의 중요성을 부각시켰습니다. 이는 리소스가 제한된 환경에서도 강력한 모델을 개발할 수 있는 가능성을 열었습니다.[3][1]

**Mixed setup optimization**: 사전학습 모델과 새로운 레이어를 결합하는 mixed setup은 현대 NLP의 일반적인 패턴입니다. DT-Fixup은 이러한 setup에서 깊은 아키텍처를 효과적으로 학습하는 첫 번째 체계적 방법을 제시했습니다.[1]

**Domain-specific application**: Text-to-SQL과 logical reasoning 같은 구조적 이해가 필요한 태스크에서 깊은 모델의 우수성을 입증하여, 복잡한 추론 태스크에 대한 아키텍처 선택에 영향을 미쳤습니다.[1]

#### 앞으로 연구 시 고려사항 (최신 연구 기반)

**1. Length Generalization**

최근 연구에 따르면 트랜스포머의 length generalization은 data format과 position encoding에 크게 의존하며, initialization에도 민감합니다. **고려사항**:[4]
- DT-Fixup을 다양한 position encoding (Rotary PE, ALiBi 등)과 결합하여 length generalization을 개선할 수 있는지 연구 필요[4]
- RASP-Generalization Conjecture에 따라 태스크가 짧은 RASP 프로그램으로 표현 가능한지 분석하여 generalization 가능성 예측[9]

**2. Transformer Optimization의 통합**

**Transformers without Normalization (2025)**: Layer normalization 없이 학습하는 새로운 기법들이 등장하고 있습니다. **고려사항**:[10]
- DT-Fixup과 최신 normalization-free 기법을 결합하여 완전한 normalization-free mixed setup 구축
- DeepNet의 Post-LN 개선 기법과 DT-Fixup 통합 가능성 탐구[11]

**3. In-Context Learning과의 연결**

최근 연구는 트랜스포머의 in-context learning 능력과 generalization의 관계를 규명하고 있습니다. **고려사항**:[12][13]
- DT-Fixup으로 학습된 깊은 모델이 few-shot in-context learning에서 어떤 성능을 보이는지 평가
- Meta-learning과 DT-Fixup을 결합하여 더 범용적인 in-context learner 개발[14]

**4. 대규모 데이터셋으로의 확장**

논문은 10k 미만 샘플에만 실험했지만, 최근 연구는 큰 데이터셋에서도 dynamic masking 기법이 효과적임을 보였습니다. **고려사항**:[6]
- DT-Fixup을 중간 규모 (10k-100k) 및 대규모 데이터셋에 적용하여 scalability 검증
- 큰 데이터셋에서는 $$\mu$$ 추정을 mini-batch 단위로 수행하는 adaptive DT-Fixup 개발

**5. Multimodal 및 다양한 도메인 적용**

**Vision Transformer**: SegFormer3D 등 의료 영상 분야에서 lightweight transformer가 주목받고 있습니다. **고려사항**:[15]
- Vision Transformer의 patch embedding 후 DT-Fixup 적용 가능성 연구
- 작은 의료 영상 데이터셋에서 깊은 ViT 학습[8]

**Multimodal Learning**: **고려사항**:
- 텍스트-이미지 pair가 제한된 상황에서 CLIP-style model에 DT-Fixup 적용
- 각 modality별로 별도의 $$\mu$$ 추정 필요

**6. 효율성과 성능의 균형**

**Mini-Sequence Transformers**: Intermediate memory를 최적화하는 새로운 기법이 등장했습니다. **고려사항**:[16]
- DT-Fixup과 효율적인 attention mechanism (Flash Attention, Memory-efficient Attention) 결합
- 깊은 모델의 inference latency를 줄이기 위한 layer-wise early exit 전략 연구

**7. Theoretical Understanding 심화**

**Convergence Analysis**: 최근 연구는 transformer 학습의 non-asymptotic convergence를 분석하고 있습니다. **고려사항**:[17]
- DT-Fixup의 수렴 속도에 대한 엄밀한 이론적 분석
- Cross-entropy loss의 linear convergence rate와 DT-Fixup의 관계 규명[17]

**Simplicity Bias**: RASP framework에 따르면 transformer는 짧은 프로그램을 선호하는 경향이 있습니다. **고려사항**:[9]
- DT-Fixup이 model complexity에 미치는 영향 분석
- Sparse Rate Reduction과 같은 information-theoretic objective와 DT-Fixup 연결[18]

**8. 실무 적용 가이드라인**

**Hyperparameter Sensitivity**: **고려사항**:
- $$\mu$$ 추정 시 outlier 처리 방법 (max 대신 95th percentile 사용 등)
- Learning rate scheduling과 DT-Fixup의 상호작용 분석
- Dropout rate와 depth의 최적 조합 규명

**Deployment Considerations**: **고려사항**:
- 깊은 모델의 quantization 및 pruning 전략
- Edge device에서의 실시간 inference를 위한 경량화 기법

**9. 새로운 태스크 영역**

**Mathematical Reasoning**: 최근 연구는 transformer의 수학적 추론 능력을 평가하고 있습니다. **고려사항**:[19]
- 수학 문제 해결, 정리 증명 등에서 깊은 모델 + DT-Fixup 효과 검증

**Program Synthesis**: **고려사항**:
- 코드 생성 태스크에서 구조적 이해가 중요하므로 DT-Fixup의 이점 활용 가능

**Time Series Forecasting**: Transformer 기반 시계열 예측이 주목받고 있습니다. **고려사항**:[20]
- 제한된 시계열 데이터로 깊은 temporal transformer 학습

**10. Ethical and Practical Implications**

**Data Efficiency**: DT-Fixup은 데이터 수집이 어렵거나 비용이 높은 분야 (의료, 법률, 저자원 언어)에서 특히 중요합니다. **고려사항**:
- Privacy-preserving learning과 DT-Fixup 결합 (작은 데이터로 강력한 모델 → 데이터 공유 최소화)
- Fairness: 작은 minority 그룹 데이터로도 효과적인 모델 학습 가능

**Computational Sustainability**: 깊은 모델을 적은 epoch로 학습할 수 있어 에너지 효율적입니다. **고려사항**:[1]
- Carbon footprint 측정 및 비교 연구

#### 결론

DT-Fixup은 작은 데이터셋에서 깊은 트랜스포머를 성공적으로 학습할 수 있는 이론적으로 정당화된 방법을 제시했으며, 이는 딥러닝 연구의 democratization에 기여하고 있습니다. 앞으로의 연구는 (1) 더 다양한 도메인과 태스크로의 확장, (2) 최신 transformer optimization 기법과의 통합, (3) 이론적 이해의 심화, (4) 실무 적용 가이드라인 정립에 초점을 맞춰야 합니다. 특히 2024-2025년 최신 연구들은 작은 데이터셋에서의 학습, in-context learning, generalization 메커니즘에 대한 더 깊은 통찰을 제공하고 있으며, 이를 DT-Fixup과 결합하면 더욱 강력하고 범용적인 학습 프레임워크를 구축할 수 있을 것입니다.[13][7][12][6][8][3][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/869b7142-6c1a-4eaf-a232-1f86fa7a799d/2012.15355v4.pdf)
[2](https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf)
[3](https://aclanthology.org/2021.acl-long.163)
[4](https://arxiv.org/abs/2402.09371)
[5](https://aclanthology.org/2024.naacl-long.402.pdf)
[6](https://proceedings.neurips.cc/paper_files/paper/2024/file/779cb405b8b916f7db70e73d51650ed2-Paper-Conference.pdf)
[7](https://www.nature.com/articles/s41586-024-08328-6)
[8](https://arxiv.org/html/2505.19985v1)
[9](https://proceedings.iclr.cc/paper_files/paper/2024/file/45ed1a72597594c097152ef9cc187762-Paper-Conference.pdf)
[10](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Transformers_without_Normalization_CVPR_2025_paper.pdf)
[11](https://www.semanticscholar.org/paper/e0995bad59c8638ea8c319bb7220c0f0b1ed5dca)
[12](https://arxiv.org/html/2503.15579v1)
[13](https://openreview.net/forum?id=yOhNLIqTEF)
[14](https://arxiv.org/pdf/2212.04458.pdf)
[15](https://ieeexplore.ieee.org/document/10678245/)
[16](https://nips.cc/virtual/2024/poster/96824)
[17](http://arxiv.org/pdf/2409.17335.pdf)
[18](https://arxiv.org/pdf/2411.17182.pdf)
[19](https://arxiv.org/pdf/2305.12563.pdf)
[20](https://www.sciencedirect.com/science/article/abs/pii/S1364032125010299)
[21](https://www.mdpi.com/2073-4441/17/20/2994)
[22](https://arxiv.org/pdf/2302.14017.pdf)
[23](http://arxiv.org/pdf/2406.09804.pdf)
[24](http://arxiv.org/pdf/2309.16935.pdf)
[25](https://arxiv.org/pdf/2502.03605.pdf)
[26](https://arxiv.org/pdf/2410.10442.pdf)
[27](http://arxiv.org/pdf/2404.01646.pdf)
[28](https://arxiv.org/html/2410.23182)
[29](http://arxiv.org/pdf/2102.06336.pdf)
[30](https://aclanthology.org/2021.acl-long.163.pdf)
[31](https://icml.cc/media/icml-2020/Slides/6684.pdf)
[32](https://neurips.cc/virtual/2024/poster/95361)
[33](https://arxiv.org/pdf/2402.13380.pdf)
[34](https://arxiv.org/html/2503.13195v1)
[35](https://arxiv.org/html/2501.14176v1)
[36](https://www.sciencedirect.com/science/article/pii/S2352847824001552)
[37](https://openreview.net/forum?id=OX4yll3X53)
[38](https://www.semanticscholar.org/paper/Optimizing-Deeper-Transformers-on-Small-Datasets-Xu-Kumar/cd02e0a094953077217e2e62f3557b36a365acff)
[39](https://www.sciencedirect.com/science/article/pii/S277244252400042X)
[40](https://www.sciencedirect.com/science/article/abs/pii/S1568494624009360)
[41](https://www.nature.com/articles/s41524-022-00734-6)
[42](https://arxiv.org/html/2405.05409v4)
[43](https://proceedings.mlr.press/v119/huang20f.html)
[44](https://ojs.aaai.org/index.php/AAAI/article/view/30410)
[45](https://ojs.istp-press.com/dmd/article/view/594)
[46](https://pubs.acs.org/doi/10.1021/acssensors.3c02654)
[47](https://www.hindawi.com/journals/ijbi/2024/3022192/)
[48](https://ieeexplore.ieee.org/document/10900009/)
[49](https://ieeexplore.ieee.org/document/10698734/)
[50](https://ieeexplore.ieee.org/document/10761510/)
[51](https://journals.sagepub.com/doi/10.1177/03611981241258753)
[52](https://ieeexplore.ieee.org/document/10376174/)
[53](http://arxiv.org/pdf/2410.01774.pdf)
[54](https://arxiv.org/html/2408.09523)
[55](https://arxiv.org/pdf/2410.13981.pdf)
[56](https://www.sciencedirect.com/science/article/abs/pii/S1568494621007584)
[57](https://www.canwindg.com/a-news-forecasting-the-future-key-trends-in-the-transformer-industry-for-2024)
[58](https://ai.plainenglish.io/15-deep-learning-projects-to-advance-your-career-in-2024-2025-d7b51092c319)
[59](https://neurips.cc/virtual/2024/events/datasets-benchmarks-2024)
[60](https://dirox.com/post/deep-learning-best-applications)
[61](https://www.techrxiv.org/users/875202/articles/1255061-a-systematic-review-on-optimization-approaches-for-transformer-and-large-language-models)
[62](https://arxiv.org/abs/2310.08661)
[63](https://dl.acm.org/doi/10.1145/3428666)
[64](https://ieeexplore.ieee.org/document/10509679/)
[65](https://proceedings.neurips.cc/paper_files/paper/2024/file/9bfa0c155653e24120760a5ead819376-Paper-Conference.pdf)
[66](https://365datascience.com/trending/public-datasets-machine-learning/)
[67](https://arxiv.org/html/2511.00907v1)
[68](https://www.sciencedirect.com/science/article/abs/pii/S1566253524002690)
[69](https://www.kaggle.com/datasets)
