# DARTS: Differentiable Architecture Search | NAS

## 1. 핵심 주장 및 주요 기여 (간결 요약)
DARTS는 아키텍처 검색을 **연속적 최적화** 문제로 정형화하여, 기존의 강화학습 및 진화기반 이산 탐색 대비 **수천 배 빠른** 네트워크 구조 탐색을 가능하게 한다.  
주요 기여:  
- 탐색 공간을 **연속적**으로 이완하고, 아키텍처 매개변수 α와 네트워크 가중치 w를 **이중수준 최적화**(bilevel optimization)로 동시 학습  
- 1–4 GPU일 내외의 저비용으로 CIFAR-10, ImageNet, PTB, WikiText-2에서 **최신 성능**에 근접 혹은 능가  
- 컨볼루션 셀과 순환 셀 모두에 적용 가능하며, 구조 간 전이학습 가능성 입증  

## 2. 상세 설명
### 2.1 해결하려는 문제
자동 신경망 구조 검색(NAS)은 일반적으로 discrete 공간에서 강화학습 또는 진화를 사용하므로, 평가 횟수가 많아 **수천 GPU일**이 소요되는 높은 계산비용 문제를 안고 있다.

### 2.2 제안 방법
#### 2.2.1 연속적 이완
- 각 노드 간 엣지에 여러 연산 $$o\in O$$의 **softmax 혼합**을 도입하여  
  
```math
    \bar o^{(i,j)}(x) = \sum_{o\in O} \frac{\exp(\alpha^{(i,j)}_o)}{\sum_{o'}\exp(\alpha^{(i,j)}_{o'})}\,o(x)
```

- 이로써 이산적 구조 선택을 연속적 α 매개변수 최적화로 변환  

#### 2.2.2 이중수준 최적화
- 상위 문제: 검증 손실 최소화  
  $$\min_\alpha \,L_{val}(w^*(\alpha),\alpha)$$  
- 하위 문제: 학습 손실 최적화  
  $$w^*(\alpha)=\arg\min_w L_{train}(w,\alpha)$$  
- 근사 기법: 한 스텝 미분(unrolled step)으로 계산 비용 절감  

### 2.3 모델 구조
- **컨볼루션 셀**: 7개 노드, 8종 연산(3×3, 5×5 separable conv, dilated conv, pooling, identity, zero).  Normal/Reduction 셀 각각 α 공유  
- **순환 셀**: 12개 노드, tanh/relu/sigmoid 활성화, identity, zero. Highway bypass 사용, 출력은 모든 노드 평균  

### 2.4 성능 향상
- CIFAR-10: test error 2.76 ± 0.09% (3.3 M 파라미터, 탐색 비용 4 GPU일) — NASNet-A(2.65%, 2000 GPU일) 대비 유사 성능[1]
- PTB: test perplexity 55.7 (23 M 파라미터, 탐색 비용 1 GPU일) — 기존 NAS(64.0, 1e4 CPU일) 대비 대폭 개선[1]
- ImageNet 모바일: top-1 error 26.7% (574 M FLOPs, 4 GPU일) — NASNet-A(26.0%, 2000 GPU일) 근접[1]
- WikiText-2 전이: perplexity 69.6 — ENAS 대비 우수 전이 성능[1]

### 2.5 한계
- **연속→이산 변환 갭**: softmax 혼합에서 최종 discrete 셀 유도 시 정보 손실  
- **하이퍼파라미터 민감도**: ξ, learning rates 등 설정에 성능 편차  
- **검색-평가 불일치**: 탐색 시 작은 네트워크, 평가 시 대형 네트워크 간 구조/채널 수 불일치  

## 3. 일반화 성능 향상 관점
- **이중수준 최적화**로 아키텍처 α가 검증 손실을 직접 최소화하여 과적합 완화  
- **연속적 탐색** 동안 모든 후보 연산이 학습되므로, 초기 탐색 단계에서 다양한 구조를 고루 평가  
- 전이 실험(CIFAR→ImageNet, PTB→WikiText-2)에서 **강건한 일반화** 입증  

## 4. 향후 연구 영향 및 고려사항
- **연속→이산 매핑 개선**: softmax 온도 조절, 구조 성능 예측 기반 선택  
- **검색 공간 확장**: 모듈화된 블록, 그래프 이성질체 제거를 통한 효율성 제고  
- **동적/데이터 적응형 NAS**: 대상 데이터셋에 즉시 적응 가능한 구조 최적화  
- **하이퍼파라미터 자동조정**: ξ·ηα 등 메타매개변수 동시 최적화  

DARTS는 NAS 연구에 **연속 최적화** 패러다임을 도입하며, 빠른 탐색과 우수한 일반화 능력을 양립시켰다. 앞으로 연속적인 구조 매핑과 meta-learning 기법의 결합이 활발히 모색될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e601ce09-cce2-4ae0-84ac-987068566322/1806.09055v2.pdf
