# Kolmogorov–Arnold Networks (KANs)

## 1. 핵심 주장 및 주요 기여  
Kolmogorov–Arnold Networks(KANs)는 다변수 연속함수를 저차원 단변수 함수들의 합으로 나타낼 수 있다는 **Kolmogorov–Arnold 표현 정리**를 모델 설계에 적용한 신경망 구조이다[1].  
- 기존 MLP의 고정된 활성화 함수를 노드에 배치한 것과 달리, KAN은 **엣지(edge)마다 학습 가능한 단변수 활성화 함수를** 배치하며, 모든 가중치를 B-스플라인 기반의 단변수 함수로 대체한다.  
- 이 단순한 구조 변경만으로도 작은 규모의 KAN이 훨씬 큰 MLP보다 유사 혹은 더 나은 정확도와 **뛰어난 해석 가능성**을 보인다[1].

## 2. 해결 문제 및 제안 방법  
### 2.1 해결하고자 하는 문제  
- MLP 기반 모델은 많은 파라미터와 불투명한 내부 구조로 인해 과잉 학습과 해석 불가능성(black-box)이 문제된다.  
- 대규모 과학·공학 문제에서 “어떤 피처가, 어떻게 결정에 기여하는가”를 직관적으로 이해하기 어렵다.

### 2.2 제안 방법  
- **KAN 레이어**: $$n_{\text{in}}\to n_{\text{out}}$$ 연결을 $$\Phi=\{\phi_{q,p}\}$$ 행렬 함수로 정의[1]  

$$
  x_{l+1,j} = \sum_{i=1}^{n_l} \phi^l_{j,i}(x_{l,i}),
  \quad
  \phi(x)=w_b\,\mathrm{silu}(x) + w_s\sum_{k}c_kB_k(x).
  $$  
  
- **심플리피케이션 & 상징식 회귀**: L1·엔트로피 규제 후 중요 노드·엣지를 프루닝하여 최소 네트워크 아키텍처를 찾고, 수치 함수→기호 함수로 자동·수동 스냅핑[1].  
- **그리드 확장(grid extension)**: 스플라인 그리드 세분화로 손실 계단식 감소, 빠른 신경 확장 법칙 실현[1].

## 3. 모델 구조 및 성능 향상  
| 비교 대상 | 구조 | 파라미터 수 | 테스트 RMSE | 신경 확장 지수 α |
|:----------|:-----|:-----------:|:-----------:|:---------------:|
| MLP(ReLU) depth 2–5 | [n,…,1] | 10²–10⁵ | 10⁻²–10⁻¹ | ≈1/d[2] |
| KAN depth 2–3 | [n, …,1] | 10²–10³ | 10⁻⁴–10⁻³ | ≈4 [이론·실험 일치][1] |

- **정량적 개선**: 다수 실험(함수근사·PDE해·특수함수 근사)에서 KAN이 MLP보다 **수십 배 작은** 파라미터로 동등 혹은 더 나은 정확도 달성[1][3][4].  
- **빠른 확장 법칙**: MLP가 $$\ell\propto N^{-(k+1)/d}$$로 COD(차원 저주)에 제약되는 반면, KAN은 $$\ell\propto N^{-(k+1)}$$로 차원 무관한 최적 지수 α=k+1=4 달성[1].  
- **PDE 응용**: Poisson 방정식 PINN에 KAN 도입 시 MLP 대비 $$10^2$$배 정확도·효율 개선[1].  
- **해석성**: 스플라인 활성화 직관적 시각화, 노드·엣지 중요도 · 기호적 상징식 도출 가능[1].

## 4. 일반화 성능 향상 전략  
- **이론적 일반화 경계**: KAN의 일반화 오차는 네트워크 너비·깊이 대신 스플라인 그리드 정수 차수에 의존하며, **노드 수 지수 의존 배제**[5].  
- **규제 기법**: L1+엔트로피 규제를 통해 중요 활성화만 남기고 불필요 함수 제거, 과적합 저감[1].  
- **스플라인 그리드 적응**: 학습 중 그리드 갱신으로 모델 용량을 점진 확장, 소규모 모델→대규모 모델 재학습 없이 정확도 개선 가능[1].

## 5. 한계 및 향후 연구 과제  
1. **학습 효율**: 현재 KAN은 MLP 대비 학습 속도 10× 느림[1][4]. GPU 병렬화 가능한 RBF 근사 등 구현 최적화 필요.  
2. **수학적 기저 확대**: 깊이-2 이상의 KAN 표현 정리 정식화, 스플라인 차수-깊이 간 상관관계 이론 증명 미흡.  
3. **하이브리드 구조**: MLP·KAN 하이브리드, LAN(learnable activation)·Multi-head KAN 등 다양한 구조 탐색 필요.  
4. **대규모 응용**: 언어·비전·생물정보 등 고차원 실제 데이터셋 적용성 검증 확대.

## 6. 향후 영향 및 고려 사항  
- KAN은 **AI + Science**에서 과학자와 AI 간 상징 언어(함수) 기반 협업 도구로 활용될 것.  
- 해석 가능 모델의 표준화를 촉진하며, 특히 물리 법칙 발견·심볼릭 회귀 분야에 활력 제공.  
- “빠른 신경 확장 법칙”을 통한 저차원 구조 학습 가능성 탐색, 대규모 언어 모델에도 응용 연구 기대.  
- 실용적 채택을 위해 **학습 가속**, **메타-러닝​(MetaKAN)**​ 등을 통한 메모리·연산 효율화가 필수적이다[6].

> **결론**: 느린 학습 속도가 걸림돌이지만, 정확도·해석성 개선 효과가 뚜렷해 작은 규모·과학 응용 과제에선 KAN 도입을 권장한다. 대규모 모델엔 MLP 대응 학습 최적화 연구가 선행되어야 한다.

[1] https://arxiv.org/abs/2404.19756
[2] https://www.edlitera.com/blog/posts/kolmogorov-arnold-networks
[3] https://www.jstage.jst.go.jp/article/transinf/E108.D/7/E108.D_2024EDL8083/_article
[4] https://arxiv.org/abs/2405.06721
[5] https://arxiv.org/html/2410.08026v1
[6] https://icml.cc/virtual/2025/poster/46218
[7] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6f482e1a-cd3b-44ab-b242-fc67e1c0ddca/2404.19756v5.pdf
[8] https://www.ssrn.com/abstract=4835325
[9] https://www.ssrn.com/abstract=4825654
[10] https://arxiv.org/abs/2408.10205
[11] https://arxiv.org/abs/2406.13155
[12] https://arxiv.org/abs/2405.07200
[13] https://arxiv.org/html/2407.11075v4
[14] https://openreview.net/forum?id=Ozo7qJ5vZi
[15] https://www.datacamp.com/tutorial/kolmogorov-arnold-networks
[16] https://openreview.net/forum?id=q5zMyAUhGx
[17] https://iclr.cc/virtual/2025/oral/31858
[18] https://openreview.net/pdf?id=yPE7S57uei
[19] https://openreview.net/notes/edits/attachment?id=7OZKU2c02J&name=pdf
[20] https://academic.oup.com/bib/article/26/2/bbaf129/8101506
[21] https://iclr.cc/virtual/2025/papers.html
[22] https://brunch.co.kr/@leadbreak/24
[23] https://arxiv.org/abs/2407.13044
[24] http://arxiv.org/pdf/2404.19756.pdf
[25] https://paperswithcode.com/task/kolmogorov-arnold-networks?page=4&q=
[26] https://arxiv.org/abs/2406.11173
[27] https://arxiv.org/abs/2405.08790
[28] https://arxiv.org/pdf/2410.08451.pdf
[29] http://arxiv.org/pdf/2501.07032.pdf
[30] http://arxiv.org/pdf/2405.11318.pdf
[31] http://arxiv.org/pdf/2407.11075.pdf
[32] http://arxiv.org/pdf/2411.00278.pdf
[33] http://arxiv.org/pdf/2502.14681.pdf
[34] http://arxiv.org/pdf/2405.07200.pdf
[35] https://arxiv.org/html/2503.15209v1
[36] http://arxiv.org/pdf/2502.16664.pdf
[37] https://arxiv.org/pdf/2502.06018.pdf
[38] https://icml.cc/virtual/2025/poster/45584
[39] https://dl.acm.org/doi/10.1145/3704137.3704166
[40] https://ojs.aaai.org/index.php/AAAI/article/view/33986/36141
