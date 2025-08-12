# Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking

# 요약 및 상세 설명

## 1. 핵심 주장 및 주요 기여 (간결 요약)
이 논문은 **Oxford 5k**와 **Paris 6k** 이미지 검색 벤치마크의 기존 한계를 지적하고, 새로운 **ROxford**와 **RParis** 데이터셋 및 **R1M**(1백만) 개의 까다로운 디스트랙터 세트를 제안한다.  
주요 기여는 다음과 같다.  
- 기존 어노테이션의 오류(거짓 양/음성), 작은 규모, 낮은 난이도를 해결하기 위해 **재주석(Revisited annotation)** 및 **난이도별 평가 프로토콜**(Easy, Medium, Hard)을 도입  
- 각 데이터셋에 15개의 **새로운 까다로운 쿼리** 추가(총 70개)  
- GPS 정보 기반으로 4.1M 후보에서 최적 난이도의 **1,001,001개 디스트랙터(R1M)** 구축  
- 로컬-피처 방식부터 최신 CNN 기반 전역 디스크립터까지 **광범위한 성능 비교 평가** 제공  

## 2. 논문이 해결하고자 하는 문제
1. **어노테이션 오류**  
   - 기존 Oxford/Paris에는 거짓 부정(긍정 이미지를 누락) 및 거짓 긍정(배경·유사 구조물 포함) 사례가 다수 존재  
   - 서로 다른 각도나 야간/주간 이미지가 같은 그룹으로 묶여 평가가 불공정함  
2. **데이터셋 규모 부족**  
   - 5k·6k 이미지로는 대규모 실제 검색 시나리오를 평가하기에 부족  
   - 기존 100k 디스트랙터에도 타깃 랜드마크 이미지가 섞여 노이즈 유발  
3. **난이도 포화**  
   - 최첨단 방법이 이미 near-perfect 성능을 보여 더 이상 발전 구분이 어려움  

## 3. 제안 방법
### 3.1. 재주석(Revisited annotation)
- **쿼리 그룹 분할**: 비대칭 건물면, 주·야간 쿼리 분리 → Oxford: 11→13 그룹, Paris: 11→12 그룹  
- **추가 쿼리**: 기존 55개에 각 데이터셋당 15개씩, 총 70개 쿼리  
- **라벨링 절차**  
  1) 잠재 양성 후보(Easy/Hard/Unclear) 수집  
  2) 다섯 명의 어노테이터가 각 이미지에 대해 {Easy, Hard, Unclear, Negative} 라벨 지정  
  3) 다수결 및 엔트로피 기반 정제 → 최종 라벨 확정  
- **난이도별 평가 프로토콜**  
  - Easy: Easy → 양성, Hard/Unclear → 무시  
  - Medium: Easy+Hard → 양성, Unclear → 무시  
  - Hard: Hard → 양성, Easy/Unclear → 무시  

### 3.2. R1M 디스트랙터 세트 구성
- Flickr YFCC100M에서 GPS 기반으로 5M 샘플 → 영국·프랑스·라스베가스 제외 → 4.1M 고해상도 이미지 확보  
- 세 가지 기본 검색 방법(AlexNet–MAC, ResNet–GeM, ASMK⋆)으로 각 쿼리별로 얼마나 **“혼동도(distracting)”** 높은지 점수화  
- 상위 1,001,001장 선택 → R1M 세트  

### 3.3. 평가 및 분석
- **로컬-피처 기반**(HesAff–RootSIFT–ASMK⋆, DELF–ASMK⋆)부터 **CNN 전역 디스크립터**(GeM, R-MAC, NetVLAD)까지 30여 종 기법 비교  
- **쿼리 확장(αQE, HQE)** 및 **확산(DFS)** 기법도 실험  
- Medium/Hard 설정과 대규모(R1M 포함)에서 기존 Easy 설정 대비 성능이 크게 하락함을 확인  
- **로컬-피처 + CNN 확산** 복합 기법이 상호 보완적 이점을 보임  

## 4. 제안 방법의 수식적 개요
- **mAP(mean Average Precision)**  

$$
    \mathrm{mAP} = \frac{1}{Q}\sum_{q=1}^{Q} \frac{1}{|P_q|}\sum_{k=1}^{N} P_q(k) \cdot \mathbb{I}[\text{rank}_q(k)\leq |P_q|]
  $$  
  
  $$P_q$$: 쿼리 $$q$$의 양성 이미지 집합, $$P_q(k)$$: 상위 $$k$$까지의 Precision  

- **ASF/Aggregated Selective Match Kernel(ASMK)⋆**  
  - 비트 바이너리 잔차량 기반 매칭  

$$
    K(\mathbf{x},\mathbf{y}) = \sum_{w=1}^W \alpha_w \, \kappa(\mathbf{r}_w(\mathbf{x}), \mathbf{r}_w(\mathbf{y}))
  $$  
  
  $$w$$: 비주얼 워드, $$\mathbf{r}_w$$: 해당 워드에 할당된 잔차 벡터, $$\kappa$$: 해밍 기반 커널  

- **GeM(Generalized Mean Pooling)**  

$$
    \mathrm{GeM}(\mathbf{X})\_c = \bigl(\frac{1}{|\Omega|}\sum_{i\in\Omega} X_{i,c}^p\bigr)^{1/p}
  $$  
  
  $$\mathbf{X}$$: 마지막 합성곱 맵, $$c$$: 채널, $$p$$: 학습 가능한 파라미터  

## 5. 한계 및 향후 과제
1. **반자동 라벨링 의존**: 대규모 데이터셋(1M+) 완전 수작업은 불가능, 일부 오류 여전히 존재할 수 있음  
2. **특정 메소드 편향 우려**: 디스트랙터 선정에 사용된 3개 방법에 따라 난이도 점수 매김이 편향될 가능성  
3. **정적 어노테이션**: 새로운 건물, 조명·날씨 변화, 구조 변경 등 동적 환경 반영 어려움  
4. **조합 기법의 계산 비용**: 최상위 성능 달성 시 로컬+CNN+확산 복합 전략은 실용적으로 무거움  

---  
이상으로, 기존 Benchmark의 **어노테이션 오류**, **규모 한계**, **난이도 포화** 문제를 해결하기 위해 제안된 **재주석**, **새 쿼리**, **대규모 디스트랙터(R1M)** 및 **난이도별 평가 프로토콜**, 그리고 이를 토대로 수행된 **광범위한 성능 비교**가 본 논문의 핵심 기여이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cc63afa1-5939-4ad7-894b-fde283cedb4e/1803.11285v1.pdf
