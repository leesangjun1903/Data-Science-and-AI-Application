# MetaFormer : MetaFormer Is Actually What You Need for Vision | Image classification, Object detection, Semantic segmentation

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
- Vision Transformer(ViT) 계열 모델의 성능은 **어텐션 기반 토큰 믹서(token mixer)**보다는, 어텐션·MLP·풀링 등 구체적 모듈에 무관한 **MetaFormer**라는 **일반화된 아키텍처**(Residual + Norm + TokenMixer + MLP 블록)가 주로 견인한다.  
- 극단적으로 단순한 풀링(pooling) 연산을 토큰 믹서로 사용한 PoolFormer도 ViT·MLP-like 모델과 비슷하거나 더 우수한 성능을 낸다.

**주요 기여**  
1. **MetaFormer 개념 제안**: Transformer를 ‘토큰 믹서’를 지정하지 않은 일반 아키텍처로 추상화.  
2. **PoolFormer 모델**: 풀링 연산만으로 토큰을 섞는 극단적 단순화 예시 제시. ImageNet-1K에서 ViT-B 대비 파라미터 35%↓, MACs 50%↓에도 Top-1 82.1% 달성.  
3. **다양한 태스크 검증**: 분류, 물체 검출‧인스턴스 분할(COCO), 의미 분할(ADE20K)에서 경쟁력 입증.  
4. **아키텍처 중심 연구 제안**: 토큰 믹서 연구를 넘어 MetaFormer 자체 개선 방향 제시.

## 2. 문제, 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- ViT와 MLP-like 모델 성능의 주 원인으로 **어텐션**이나 **공간 MLP** 모듈이 지목되지만, 실제로는 **전체 블록 구조(Residual + Norm + TokenMixer + MLP)**가 핵심인지 불명확.  
- 따라서 극단적으로 단순한 토큰 믹서를 써도 MetaFormer 구조만 지키면 충분히 높은 성능을 낼 수 있는지 검증 필요.

### 2.2 제안 방법  
1. **MetaFormer 블록**  
   - 입력 $$X\in\mathbb{R}^{N\times C}$$에 대해  

  $$
       Y = \mathrm{TokenMixer}(\mathrm{Norm}(X)) + X,\quad
       Z = \mathrm{MLP}(\mathrm{Norm}(Y)) + Y.
     $$

2. **풀링 기반 TokenMixer**  
   - $$K\times K$$ 평균풀링으로만 토큰 혼합:  

  $$
       T'\_{:,i,j}
       = \frac1{K^2}\sum_{p,q=1}^K T_{:,\,i+p-\frac{K+1}2,\,j+q-\frac{K+1}2}
         - T_{:,i,j}.
     $$
   - PyTorch로는 `AvgPool2d(pool_size, stride=1, padding=…) – identity` 구현.

3. **계층적 구조**  
   - CNN·PVT 유사한 4단계: 해상도 $$\frac H4$$, $$\frac H8$$, $$\frac H{16}$$, $$\frac H{32}$$ 토큰.  
   - 블록 수 비율 $$[L/6,\,L/6,\,L/2,\,L/6]$$, MLP 확장비율 4, Modified LayerNorm(토큰+채널 정규화).

### 2.3 모델 구조  
| 모델        | 파라미터(M) | MACs(G) | Top-1 Accuracy(%) |
|-------------|-------------|---------|-------------------|
| PoolFormer-S24 | 21.4        | 3.4     | 80.3              |
| DeiT-S    | 22.0        | 4.6     | 79.8              |
| ResMLP-S24| 30.0        | 6.0     | 79.4              |
| PoolFormer-M36| 56.1        | 8.8     | 82.1              |
| DeiT-B    | 86.0        | 17.5    | 81.8              |
| ResMLP-B24| 116.0       | 23.0    | 81.0              |

### 2.4 성능 향상  
- **경량화 대비 우수**: PoolFormer-S24는 DeiT-S 대비 MACs 26%↓, 파라미터 유사하면서 0.5%↑.  
- **범용성**: COCO 물체 검출·분할, ADE20K 의미 분할에서도 ResNet 대비 AP·mIoU 3-4점씩 상회.  
- **아블레이션**:  
  - TokenMixer 없이 Identity만 써도 74.3%.  
  - 풀링 → 랜덤 매트릭스, DW-Conv, 어텐션 혼합 등 다양한 믹서 대체 실험에서도 모두 합리적 성능 유지.  

### 2.5 한계  
- **풀링 단독**은 전역 정보 학습 능력 한계(특히 크고 복잡한 패턴).  
- **자연어·다른 도메인 검증 부족**: NLP나 비디오 등에서 MetaFormer 범용성 추가 검증 필요.  
- **하이브리드 설계 최적화**: 풀링+어텐션/MLP 조합에 대한 구조 탐색 미완.

## 3. 모델 일반화 성능 향상 관점  
- **Residual 연결과 Norm** 중심의 블록이 안정적 학습과 기울기 전달 보장.  
- **TokenMixer 다양화**(풀링, 어텐션, MLP)에도 높은 성능 유지→**오버피팅 방지** 및 **도메인 적응력** 기대.  
- **하이브리드 Stage 설계**: 저단계 풀링→고단계 어텐션 조합으로 전역·지역 표상 균형 확보 시 일반화 개선 가능.

## 4. 논문의 향후 영향 및 고려사항  
- **아키텍처 우선 접근**: 토큰 믹서 설계보다 **MetaFormer 블록 구조**(정규화·잔차·MLP) 최적화 연구 가치 강조.  
- **효율적 하이브리드 모델** 개발: 풀링·어텐션·MLP 조합으로 성능·효율 균형 달성.  
- **도메인·태스크 확장**: NLP, 비디오, 3D 등 다양한 입력에 MetaFormer 적용성·일반화 검증 필요.  
- **학습·정규화 기법**: Modified LayerNorm, LayerScale, Stochastic Depth 등 안정화 기법이 핵심—추가 개선 여지.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/36683c49-3e5a-4ed8-ab6f-1ba7d0ea51d1/2111.11418v3.pdf
