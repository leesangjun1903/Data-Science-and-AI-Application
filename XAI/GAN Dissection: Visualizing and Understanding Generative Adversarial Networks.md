# 핵심 주장 및 주요 기여  
GAN Dissection는 GAN이 내부적으로 시각 세계를 어떻게 표현하는지 탐색하기 위한 **유닛(unit)**, **객체(object)**, **장면(scene)** 수준의 해석적 프레임워크를 제안한다.  
첫째, **Segmentation 기반 네트워크 디섹션**을 통해 객체 개념과 높은 상관관계를 보이는 해석 가능한 유닛을 자동으로 식별한다.  
둘째, 네트워크 **개입(intervention)** 기법을 도입하여 특정 유닛 집합을 켜고 끔으로써 객체의 생성 여부를 인과적으로 제어하고, 그 **평균 인과 효과(ACE)**를 정량화한다.  
이로써 내부 표현의 비교·분석, 인위적 결함 유닛 제거를 통한 생성 품질 개선, 객체 수준의 상호작용적 조작 등 다양한 응용이 가능함을 보였다.[1]

# 해결하고자 하는 문제  
- GAN은 놀라운 생성 성능에도 불구하고 내부 표현이 **블랙박스**로 남아 있어,  
  - 객체 개념이 어떻게 인코딩되는지,  
  - 생성물의 아티팩트(비현실적 결과) 원인이 무엇인지,  
  - 아키텍처 선택이 학습에 미치는 영향 등을 알 수 없다는 한계가 있다.[1]

# 제안하는 방법  
## 1. Dissection  
각 유닛 $$u$$의 업샘플된 활성화 $$\,r_{u,P}^\uparrow$$ 와 이미지 내 개념 $$c$$의 분할 마스크 $$s_c(x)$$ 간의 일치도를 **Intersection-over-Union(IoU)**로 측정한다:  

$$
\mathrm{IoU}\_{u,c} =\frac{\mathbb{E}\_z\bigl[(r_{u,P}^\uparrow>t_{u,c})\land s_c(x)\bigr]}
      {\mathbb{E}\_z\bigl[(r_{u,P}^\uparrow>t_{u,c})\lor s_c(x)\bigr]},
$$

여기서 임계값 $$t_{u,c}$$는 상호정보 대비 엔트로피 비율을 최대로 하는 값으로 선택된다.[1]

## 2. Intervention  
특정 유닛 집합 $$U$$를 강제로 **off**($$r_{U,P}=0$$) 또는 **on**($$r_{U,P}=k$$) 함으로써 객체 $$c$$ 생성에 미치는 인과 효과를 측정한다.  
원본 생성물 $$x=G(z)$$, 유닛 제거 $$x_a=f(0,r_{U,P};r_{\bar U,P})$$, 유닛 삽입 $$x_i=f(k,r_{U,P};r_{\bar U,P})$$일 때,  
**평균 인과 효과(ACE)**는  

$$
\delta_{U\to c}
=\mathbb{E}\_{z,P}\bigl[s_c(x_i)\bigr]
-\mathbb{E}_{z,P}\bigl[s_c(x_a)\bigr]
$$

으로 정의된다[1].  
다중 유닛을 연속적 개입 계수 $$\alpha\in[0,^d$$로 확장하여  

$$
\alpha^*=\arg\min_\alpha\bigl(-\delta_{\alpha\to c}+\lambda\|\alpha\|_2^2\bigr)
$$

를 SGD로 최적화함으로써 최소 크기의 인과 유닛 집합을 찾아낸다.[1]

# 모델 구조  
- 생성기: Progressive GANs(14개 합성곱 층)  
- 분할기: ADE20K 데이터로 학습된 FCN 계열 세그멘테이션 네트워크(336개 객체·파트 클래스)[1]  
- 분석 대상: LSUN 장면 데이터셋(교회·거실·주방 등)

# 성능 향상 및 한계  
- **아티팩트 제거**: 인간이 식별한 20개 결함 유닛 제거 후 Fréchet Inception Distance(FID)가 43.16→27.14로 개선, AMT 선호도 72.4% 획득[1]  
- **객체 조작**: 사람·커튼·창문 등은 소수 유닛만 제거해도 거의 완전 삭제 가능하지만, 테이블·의자 등 일부 객체는 완전 제거 어렵고 형태만 축소됨[1]  
- **한계**:  
  1. 분할기 오류 전파(비현실 이미지에 오탐 가능성)  
  2. 객체 삽입 거부 사례(부적절한 문맥에서 삽입 무시)  
  3. 복잡도: 대형 모델·고해상도 생성기 적용 시 계산 비용 상승

# 일반화 성능 향상 가능성  
- **노이즈 유닛 제거**로 과적합 감소 및 샘플 다양성 증가  
- **단위별 인과 유닛**만 전이학습에 활용 시, 소규모 데이터셋에서도 빠른 수렴 가능  
- **데이터셋·모델 비교**를 통해 학습된 개념의 보편성과 특수성 파악 → 일반화 방안 설계

# 향후 연구 영향 및 고려사항  
- **표준 해석 도구**로서 GAN 내부 가시화·진단 지원  
- 분해·개입 기법을 **다른 생성 모델**(VAE, Normalizing Flow 등)로 확장  
- **인과 구조 학습** 연구: 개입 결과의 층간 전파 메커니즘 규명  
- 세분화 모델의 **데이터 바이어스** 검토 및 개선 필요[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/00cdd49d-fd8d-4d12-b794-e6886a802aa6/1811.10597v2.pdf
[2] https://www.sec.gov/Archives/edgar/data/1799332/000149315225010404/form10-k.htm
[3] https://www.sec.gov/Archives/edgar/data/1799332/000164117225012422/form8-k.htm
[4] https://www.sec.gov/Archives/edgar/data/1799332/000164117225003721/form8-k.htm
[5] https://www.sec.gov/Archives/edgar/data/1799332/000164117225011492/form8-k.htm
[6] https://www.sec.gov/Archives/edgar/data/1799332/000164117225009440/form8-k.htm
[7] https://www.sec.gov/Archives/edgar/data/1799332/000164117225009454/form10-q.htm
[8] https://www.sec.gov/Archives/edgar/data/1799332/000149315225010396/form8-k.htm
[9] https://www.semanticscholar.org/paper/08500ea9c55593efcabd3dfbf2eff44aaaa66689
[10] https://www.semanticscholar.org/paper/fc35a72375a8f8cfb7679bdf3e51e676618275a8
[11] https://dl.acm.org/doi/10.1145/3329781.3329783
[12] https://hess.copernicus.org/articles/28/917/2024/
[13] http://arxiv.org/pdf/1811.10597.pdf
[14] https://arxiv.org/pdf/1701.00160.pdf
[15] https://velog.io/@tobigs16gm/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-GAN-Dissection-Visualizing-and-Understanding-Generative-Adversarial-Networks
[16] https://arxiv.org/abs/1811.10597
[17] https://openreview.net/forum?id=Hyg_X2C5FX
[18] https://ysbsb.github.io/gan/2021/01/18/GAN-Dissection.html
[19] https://paperswithcode.com/paper/gan-dissection-visualizing-and-understanding
[20] https://openreview.net/pdf?id=Hyg_X2C5FX
[21] https://www.slideshare.net/slideshow/gan-dissection/239722190
[22] https://ar5iv.labs.arxiv.org/html/2201.07646
[23] https://developers.google.com/machine-learning/gan/gan_structure
[24] https://mitibmwatsonailab.mit.edu/research/blog/gan-dissection-visualizing-and-understanding-generative-adversarial-networks/
[25] https://www.semanticscholar.org/paper/df7ad8eeb595da5f7774e91dae06075be952acff
[26] https://www.semanticscholar.org/paper/ba656ad4bed88ea63148fb422ccb0108058270df
[27] https://www.semanticscholar.org/paper/6633dc4f3e091a49983b676c2eb61d7dfe8546e0
[28] https://www.semanticscholar.org/paper/4e222f31ba82a5ef5a615e0eff12fa045e62ada8
[29] https://gandissect.csail.mit.edu
