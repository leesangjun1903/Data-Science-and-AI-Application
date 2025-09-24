# Incremental Learning Through Deep Adaptation | Incremental Learning

## 1. 핵심 주장 및 기여  
**Incremental Learning Through Deep Adaptation (Rosenfeld & Tsotsos, 2018)** 은 기존에 학습된 심층 신경망의 필터를 **선형 조합(linear combination)** 으로 재활용함으로써  
– 원래 도메인 성능을 **정확히 보존**  
– 새로운 도메인 추가 시 파라미터 증가율을 **약 13%**로 대폭 절감  
– **불필요한 공동 학습(joint training)** 없이 **단계적 증분 학습** 가능  
– 표준 양자화와 결합 시 파라미터 비용을 **약 3%** 수준으로 축소  
를 실현함으로서 **전통적 파인튜닝 대비 효율 및 수렴 속도**, **catastrophic forgetting** 문제 해결에 기여한다.

## 2. 해결 과제  
1. **Catastrophic Forgetting**: 새 도메인 학습 시 기존 도메인 성능 붕괴  
2. **파라미터 효율화**: 도메인별 네트워크 복제 없이 경량화  
3. **단계적 학습**: 이전 도메인 데이터가 없어도 새로운 도메인 추가  

## 3. 제안 방법  
### 3.1. 컨트롤러 모듈  
- 기존 네트워크 각 컨볼루션층 필터 $$F_l\in\mathbb{R}^{C_o\times C_i\times k\times k}$$ 을 **평탄화(flatten)** 후  
- **선형 변환 행렬 $$W_l\in\mathbb R^{C_o\times C_o}$$** 으로 재조합

$$
    \widetilde{F}^a_l = W_l\cdot\widetilde{F}_l,\quad
    F^a_l = \text{unflatten}(\widetilde{F}^a_l)
  $$
- 입력 $$x_l$$ 에 대해 스위칭 변수 $$\alpha\in\{0,1\}$$ 로 기존/적응 필터 전환:  

$$
    x_{l+1} = [\alpha(F^a_l)+(1-\alpha)F_l]\ast x_l + [\alpha b^a_l + (1-\alpha)b_l]
  $$
- $$\alpha$$ 를 **원-핫 벡터**로 확장해 다수 도메인 전환 지원  

### 3.2. 모델 구조  
1. **Base Net**: Task 1 학습 완료된 CNN (예: VGG-B, Wide ResNet)  
2. **Controller Modules**: 각 합성곱층 뒤에 부착, 파라미터 $$W_l,b^a_l$$ 만 학습  
3. **Task-specific Head**: 새 fully-connected 레이어를 Task마다 추가  
4. **스위처 α**: 수동 설정 혹은 “Dataset Decider” 서브네트워크로 자동 선택  

## 4. 성능 및 효율  
|방법|평균 정확도|파라미터 증가율|
|---|---:|---:|
|Fine-tuning 전체|≈87.7%|100%|
|Feature Extractor (마지막층만)|≈54%|1%|
|DAN (Linear)|≈87.3%|≈13%|
|DAN + Quantization (8-bit)|≒87.0%|≈3%|

- **Visual Decathlon Challenge**:  
  – DAN 방식은 각 도메인 독립 학습으로 **Decathlon 점수 ≈2851** 획득 (Residual Adapters ≈2643)  
  – 파라미터 증가는 **2.17×** (기존 10× 대비)  
- **수렴 속도**: Fine-tuning 대비 **더 빠르게** 최고 성능 도달  

## 5. 한계 및 일반화 성능  
- **표현력 제한**: 컨트롤러가 기반 필터의 선형 조합만 허용 → 기초 필터 공간이 새 도메인의 특징을 충분히 포괄해야 함  
- **효율성 저하**: 기저 필터가 타스크 특성에서 멀면 더 깊은 층에서 복구 필요  
- **Residual 구조**에서는 정보 보존 완화돼 한층 덜 민감  
- **일반화**: 다양한 이미지 도메인(자연/스케치/문자)에서 유의미한 전이 성능 입증 → 기저 필터의 **다양성**과 **선형 결합**이 과잉 학습 방지 및 규제 효과  

## 6. 향후 영향 및 고려 사항  
- **다중 도메인·다중 과제 통합**: α를 실수 벡터로 확장 시 도메인 간 **연속적 특성 조합** 가능 → Domain‐Mix 활용  
- **컨트롤러 구조 최적화**: 비선형/저차원 표현 학습을 위한 **텐서 분해** 또는 **스파스 제약** 연구  
- **기저 네트워크 선택**: 전이 학습 “Transferability” 지표에 기반한 **베이스넷 자동 선정** 전략 필요  
- **합성곱 이후 모듈 확장**: Attention, Transformer 블록 등 신축적 확장  

향후 연구에서는 **기저 필터 초기화** 및 **컨트롤러 정규화**를 조합한 **적응식 메타러닝** 기법으로 확장하고, 비전 외 모달리티(음성·텍스트)로의 적용 가능성도 검토해야 한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ff6f53b4-a09a-4141-af5c-33be30de7dec/1705.04228v2.pdf
