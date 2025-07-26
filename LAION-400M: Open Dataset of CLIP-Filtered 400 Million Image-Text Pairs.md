# LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs

## 핵심 주장과 주요 기여

LAION-400M 논문의 **핵심 주장**은 대규모 공개 데이터셋의 부재가 다중모달 언어-비전 모델 연구의 주요 장벽이었으며, 이를 해결하기 위해 **4억 개의 CLIP 필터링된 이미지-텍스트 쌍**을 공개적으로 제공한다는 것입니다[1].

주요 기여는 다음과 같습니다:

**데이터셋 구축**: Common Crawl에서 추출한 4억 개의 이미지-텍스트 쌍을 CLIP 기반 필터링을 통해 품질을 보장하고, CLIP 임베딩과 k-최근접 이웃 인덱스를 함께 제공[1][2]

**개방성과 접근성**: 기존의 사유 대규모 데이터셋과 달리 완전히 공개적으로 접근 가능한 자원을주화에 기여[1][3]

**기술적 도구 제공**: img2dataset 라이브러리 개발로 효율적인 이미지 다운로드와 처리를 지원하며, 웹 데모를 통한 검색 기능 제공[1]

## 해결하고자 하는 문제와 제안 방법

### 문제 정의

논문은 다음과 같은 핵심 문제를 해결하고자 합니다:

**데이터 규모의 한계**: CLIP, DALL-E와 같은 모델들이 수억 개의 이미지-텍스트 쌍으로 훈련되어 놀라운 zero-shot 및 few-shot 학습 능력을 보여주었으나, 이러한 규모의 공개 데이터셋이 전무했던 상황[1][2]

**연구 접근성 부족**: 대규모 모델 훈련에 필요한 데이터가 기업의 사유 자산으로만 존재하여 연구 커뮤니티의 접근이 제한되었던 문제[1]

### 제안 방법

**분산 처리 워크플로우**: Common Crawl의 WAT 파일을 분산 처리하여 alt-text가 있는 HTML IMG 태그를 추출하고, 비동기 요청을 통해 이미지를 다운로드[1]

**다단계 필터링 시스템**:
- 기본 품질 필터: 5자 미만의 alt-text나 5KB 미만의 이미지 제거
- 중복 제거: URL과 alt-text 기반 블룸 필터 적용
- **CLIP 유사도 필터링**: 이미지와 텍스트의 CLIP 임베딩 간 코사인 유사도가 0.3 이하인 샘플 제거[1]

CLIP 유사도 임계값 선택을 위한 수식적 접근:

$$ \text{similarity}(I, T) = \frac{\text{CLIP}\_{\text{image}}(I) \cdot \text{CLIP}\_{\text{text}}(T)}{|\text{CLIP}\_{\text{image}}(I)| \times |\text{CLIP}_{\text{text}}(T)|} \geq 0.3 $$

여기서 $$I$$는 이미지, $$T$$는 텍스트, 0.3은 인간 검수를 통해 결정된 임계값입니다[1].

## 성능 향상 및 검증

논문은 DALL-E 아키텍처를 사용한 **개념 증명(Proof of Concept)** 실험을 통해 데이터셋의 효용성을 입증했습니다:

**실험 설정**: 720만 개의 무작위 선택된 LAION-400M 샘플로 RTX 2070 Super에서 1 에포크 훈련[1]

**성능 비교**: Conceptual Captions 3M 및 12M과 비교하여 빠른 수렴과 충분한 품질의 생성 결과를 달성[1]

**다양성 분석**: 웹 데모를 통한 검색 결과는 높은 의미적 관련성과 다양성을 보여주며, 다양한 해상도의 이미지 분포 제공 (표 1: 전체 4억 1300만 샘플 중 1024×1024 이상 해상도가 960만 개)[1]

## 모델의 일반화 성능 향상 가능성

### 스케일링 법칙과 일반화

LAION-400M은 **스케일링 법칙(Scaling Laws)**의 중요성을 강조합니다. 연구에 따르면 데이터 규모만 증가시켜도 모델 성능이 향상되며, 모델과 컴퓨팅 예산을 함께 확장하면 데이터 규모에 병목되지 않는 한 일반화 및 전이 성능이 더욱 향상됩니다[1][4]

**Power Law 스케일링**: 후속 연구들은 LAION 데이터셋을 사용한 CLIP 훈련에서 zero-shot 분류, 검색, 선형 프로빙 등 다양한 하위 작업에서 전력 법칙 스케일링을 확인했습니다[5][6]

### Zero-shot 및 Few-shot 학습 능력

LAION-400M으로 훈련된 모델들은 뛰어난 **zero-shot 일반화 능력**을 보여줍니다:

**다양한 도메인 적용**: 세밀한 객체 분류, 지리적 위치 파악, 비디오 동작 인식, OCR 등 30개 이상의 다양한 데이터셋에서 zero-shot 성능 달성[7]

**Few-shot 적응**: 사전 훈련 데이터 양이 증가할수록 few-shot 일반화 성능이 전력 법칙을 따라 향상되며, 새로운 클래스에 대한 few-shot 성능이 기존 클래스보다 빠르게 수렴하는 현상 관찰[4]

## 한계점

### 데이터 품질과 편향성 문제

**편향성 증폭**: LAION-400M은 웹에서 수집된 필터링되지 않은 데이터로 인해 성별, 인종, 연령 등에 대한 편향을 포함하고 있습니다. 연구에 따르면 LGBTQ+ 커뮤니티, 고령 여성, 젊은 남성 등 특정 인구 집단과 관련된 데이터가 높은 비율로 제외되는 문제가 발견되었습니다[8]

**유해 콘텐츠**: 데이터셋 감사 결과, 혐오 발언, 공격적 콘텐츠, 타겟팅된 콘텐츠의 비율이 데이터셋 규모 증가와 함께 증가하는 것으로 나타났습니다. LAION-2B-en에서는 이러한 문제적 콘텐츠의 비율이 LAION-400M보다 12.26% 더 높게 측정되었습니다[9]

### 기술적 한계

**NSFW 필터링 한계**: 기존 NSFW 필터가 성적으로 노골적인 콘텐츠를 완전히 제거하지 못하는 문제[8]

**저작권 문제**: 대규모 웹 스크래핑으로 인한 저작권이 있는 콘텐츠의 높은 비율 포함[8]

## 미래 연구에 미치는 영향

### 연구 민주화와 개방성

LAION-400M은 **연구 접근성의 패러다임 전환**을 가져왔습니다. 이전에는 Google, OpenAI 등 대기업만이 접근할 수 있었던 대규모 다중모달 데이터셋을 공개함으로써, 전 세계 연구자들이 최첨단 모델을 처음부터 훈련할 수 있게 되었습니다[1][10]

**후속 데이터셋들**: LAION-400M의 성공은 LAION-5B, DataComp와 같은 더 큰 규모의 공개 데이터셋 개발로 이어졌습니다[3][11][12]

### 기술적 발전 촉진

**모델 아키텍처 혁신**: 공개 데이터셋의 존재로 연구자들은 데이터 수집보다는 모델 아키텍처와 훈련 기법 개발에 집중할 수 있게 되었습니다[13][14]

**스케일링 법칙 연구**: 재현 가능한 대규모 실험을 통해 다중모달 모델의 스케일링 법칙에 대한 체계적 연구가 가능해졌습니다[5][6]

### 응용 분야 확장

**생성형 AI**: Stable Diffusion, DALL-E 2 등 주요 텍스트-이미지 생성 모델들이 LAION 데이터셋으로 훈련되어 생성형 AI 분야의 급속한 발전을 이끌었습니다[3][15]

**의료 및 과학 분야**: 바이오메디컬 이미지 분석[16][17], 저선량 방사선 치료[17] 등 도메인 특화 응용으로 확장

## 향후 연구 시 고려사항

### 윤리적 데이터 큐레이션

**편향성 완화**: 미래 연구에서는 데이터 수집 단계에서부터 인구통계학적 균형을 고려하고, 훈련 과정에서 적대적 디바이어싱 기법을 적용해야 합니다[18][19][20]

**사용자 동의와 프라이버시**: 웹 스크래핑 데이터의 사용에 대한 명시적 동의 확보와 개인정보 보호 방안 마련이 필요합니다[3][21]

### 데이터 품질 관리

**필터링 기법 개선**: CLIP 기반 필터링을 넘어서 인간의 선호도 데이터를 활용한 보상 모델 기반 필터링 등 더 정교한 품질 관리 방법 개발이 요구됩니다[22]

**다양성과 품질의 균형**: 단순한 필터링보다는 데이터의 다양성을 유지하면서도 품질을 확보하는 방향으로 발전해야 합니다[23]

### 지속가능한 연구 생태계

**환경적 고려**: 대규모 모델 훈련의 환경적 비용을 고려하여 효율적인 훈련 방법과 모델 재사용 전략 개발이 필요합니다[24]

**국제적 협력**: 언어와 문화적 다양성을 반영한 글로벌 데이터셋 구축을 위한 국제적 협력 체계 구축이 중요합니다[25]

LAION-400M은 단순한 데이터셋을 넘어서 **AI 연구의 민주화와 개방성을 실현한 이정표**로 평가됩니다. 그러나 편향성과 윤리적 문제들은 향후 연구에서 반드시 해결해야 할 과제로 남아있으며, 이는 더욱 책임감 있고 포용적인 AI 시스템 개발을 위한 중요한 교훈을 제공합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/fa03ed7f-850f-4988-ad06-2e7a946d9ab8/2111.02114v1.pdf
[2] https://www.semanticscholar.org/paper/b668ce936cff0b0ca8b635cd5f25a62eaf4eb3df
[3] https://www.aiaaic.org/aiaaic-repository/ai-algorithmic-and-automation-incidents/laion-400m-dataset
[4] https://www.semanticscholar.org/paper/9c392c7d79a28f6b6825c0193f1d1695ad1e73b5
[5] https://ieeexplore.ieee.org/document/10205297/
[6] https://openaccess.thecvf.com/content/CVPR2023/papers/Cherti_Reproducible_Scaling_Laws_for_Contrastive_Language-Image_Learning_CVPR_2023_paper.pdf
[7] https://openai.com/index/clip/
[8] https://arxiv.org/html/2405.08209v1
[9] https://openreview.net/pdf?id=6URyQ9QhYv
[10] https://www.eleuther.ai/papers-blog/laion-400m-open-dataset-of-clip-filtered-400-million-image-text-pairs
[11] https://snorkel.ai/research-paper/datacomp-in-search-of-the-next-generation-of-multimodal-datasets/
[12] https://arxiv.org/abs/2304.14108
[13] https://arxiv.org/abs/2303.07226
[14] https://openreview.net/forum?id=IpJ5rAFLv7
[15] https://science.lpnu.ua/sa/all-volumes-and-issues/volume-6-number-1-2024/training-neural-network-image-styling
[16] https://www.themoonlight.io/en/review/scaling-large-vision-language-models-for-enhanced-multimodal-comprehension-in-biomedical-image-analysis
[17] https://arxiv.org/abs/2501.15370
[18] https://aclanthology.org/2023.findings-acl.403.pdf
[19] https://milvus.io/ai-quick-reference/how-do-visionlanguage-models-handle-bias-in-imagetext-datasets
[20] https://arxiv.org/html/2407.02814v1
[21] https://arxiv.org/html/2409.00252v1
[22] https://arxiv.org/html/2312.06726v4
[23] https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Too_Large_Data_Reduction_for_Vision-Language_Pre-Training_ICCV_2023_paper.pdf
[24] https://proceedings.neurips.cc/paper_files/paper/2024/file/f1a6a2cdc7e65dbb4579e78f97cd2665-Supplemental-Datasets_and_Benchmarks_Track.pdf
[25] https://aclanthology.org/2020.lrec-1.494.pdf
[26] https://link.springer.com/10.1007/978-3-031-73748-0_5
[27] https://www.mdpi.com/1999-4907/15/7/1171
[28] https://ieeexplore.ieee.org/document/10569478/
[29] https://ieeexplore.ieee.org/document/10755475/
[30] https://dl.acm.org/doi/10.1145/3606038.3616160
[31] https://ieeexplore.ieee.org/document/10209019/
[32] http://tarupublications.com/doi/10.47974/JDMSC-1777
[33] https://openaccess.thecvf.com/content/ICCV2023/papers/Kang_Noise-Aware_Learning_from_Web-Crawled_Image-Text_Data_for_Image_Captioning_ICCV_2023_paper.pdf
[34] https://arxiv.org/abs/2111.02114
[35] https://laion.ai/laion-400-open-dataset/
[36] https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_Unified-IO_2_Scaling_Autoregressive_Multimodal_Models_with_Vision_Language_Audio_CVPR_2024_paper.pdf
[37] https://arxiv.org/html/2307.03132v2
[38] https://www.vkit.ru/index.php/archive-rus/1371-020-025
[39] https://avantipublishers.com/index.php/ijpt/article/view/1458
[40] https://www.elibrary.ru/item.asp?id=54350167
[41] https://academic.oup.com/jas/article/102/Supplement_3/207/7757118
[42] https://iopscience.iop.org/article/10.1088/1741-4326/ac207e
[43] https://papers.neurips.cc/paper_files/paper/2023/file/996e2b446391fcb8bf32a3d1645cc799-Paper-Conference.pdf
[44] https://www.ultralytics.com/blog/understanding-few-shot-zero-shot-and-transfer-learning
[45] https://milvus.io/ai-quick-reference/what-role-does-transfer-learning-play-in-fewshot-and-zeroshot-learning
[46] https://openai.com/index/scaling-laws-for-neural-language-models/
[47] https://openaccess.thecvf.com/content/ACCV2020/papers/Fei_Few-Shot_Zero-Shot_Learning_Knowledge_Transfer_with_Less_Supervision_ACCV_2020_paper.pdf
[48] https://arxiv.org/html/2312.04567v1
[49] https://www.geeksforgeeks.org/deep-learning/few-shot-learning-vs-transfer-learning/
[50] https://openaccess.thecvf.com/content/CVPR2024/papers/Mahmoud_Sieve_Multimodal_Dataset_Pruning_using_Image_Captioning_Models_CVPR_2024_paper.pdf
[51] https://arxiv.org/abs/2110.06990
[52] https://velog.io/@yjkim0520/Few-Shot-Learning-One-shot-Zero-shot-
[53] https://github.com/rom1504/cc2dataset
[54] https://openreview.net/pdf?id=_uOnt-62ll
[55] https://account.jpr.winchesteruniversitypress.org/index.php/wu-j-jpr/article/view/130
[56] https://birjournal.com/index.php/bir/article/view/344
[57] https://jurnal.usk.ac.id/JAROE/article/view/36135
[58] http://www.emerald.com/medar/article/32/1/1-41/278827
[59] https://link.springer.com/10.1007/s11356-024-34535-9
[60] https://iptek.its.ac.id/index.php/ijmeir/article/view/21475
[61] https://link.springer.com/10.1007/s10479-023-05251-3
[62] https://irjems.org/irjems-v3i10p124.html
[63] https://www.secoda.co/glossary/open-source-datasets
[64] https://proceedings.neurips.cc/paper_files/paper/2024/file/17d25665bf6f46b7b3d32bd5cad3cbb2-Supplemental-Datasets_and_Benchmarks_Track.pdf
[65] https://www.anaconda.com/blog/useful-sites-for-finding-datasets
[66] https://qc-cuny.libguides.com/c.php?g=1209931&p=8861273
[67] https://ojs.aaai.org/index.php/AIES/article/view/31657
[68] https://blog.roboflow.com/free-research-datasets/
[69] https://openaccess.thecvf.com/content/CVPR2024/papers/Howard_SocialCounterfactuals_Probing_and_Mitigating_Intersectional_Social_Biases_in_Vision-Language_Models_CVPR_2024_paper.pdf
[70] https://encord.com/blog/open-source-datasets-ml/
[71] https://arxiv.org/html/2503.00020v1
[72] https://arxiv.org/abs/2309.14381
[73] https://github.com/awesomedata/awesome-public-datasets
[74] https://ieeexplore.ieee.org/document/10418923/
[75] https://ieeexplore.ieee.org/document/10462008/
[76] http://arxiv.org/pdf/2110.01963.pdf
[77] https://arxiv.org/pdf/2108.00114.pdf
[78] https://arxiv.org/pdf/2307.03132.pdf
[79] http://arxiv.org/pdf/2405.04623.pdf
[80] http://arxiv.org/pdf/2402.04841.pdf
[81] https://arxiv.org/html/2407.08303v2
[82] https://arxiv.org/pdf/1907.07174.pdf
[83] https://arxiv.org/pdf/2106.07411v1.pdf
[84] https://arxiv.org/pdf/1707.02968.pdf
[85] https://openreview.net/pdf?id=M3Y74vmsMcY
[86] https://voxel51.com/blog/a-history-of-clip-model-training-data-advances
[87] https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models
[88] https://github.com/kakaobrain/coyo-dataset
[89] https://github.com/rom1504/laion-prepro
[90] https://vestnik.utmn.ru/eng/energy/vypuski/2024-tom-10/-4-40/1255390/
[91] https://www.semanticscholar.org/paper/50ad17c11eeae5982f81e90385a2182f30330afa
[92] https://arxiv.org/pdf/2212.07143.pdf
[93] http://arxiv.org/pdf/2407.01456.pdf
[94] https://arxiv.org/pdf/2502.12051.pdf
[95] http://arxiv.org/pdf/2412.07942.pdf
[96] https://arxiv.org/pdf/2102.06701.pdf
[97] https://arxiv.org/pdf/2210.16859.pdf
[98] https://arxiv.org/pdf/2411.06646.pdf
[99] http://arxiv.org/pdf/2405.15074.pdf
[100] https://arxiv.org/pdf/2208.08489.pdf
[101] https://arxiv.org/pdf/2402.01092.pdf
[102] https://yai-yonsei.tistory.com/30
[103] https://arxiv.org/html/2406.11271v1
[104] https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Scaling_Laws_of_Synthetic_Images_for_Model_Training_..._for_CVPR_2024_paper.pdf
[105] https://deepdata.tistory.com/477
[106] http://www.scielo.cl/scielo.php?script=sci_arttext&pid=S0718-58392022000100177&lng=en&nrm=iso&tlng=en
[107] https://gspjournals.com/ijrebs/index.php/ijrebs/article/view/67/78
[108] https://arxiv.org/pdf/2311.03449.pdf
[109] https://arxiv.org/html/2412.08580v1
[110] https://arxiv.org/html/2401.12425v1
[111] https://pmc.ncbi.nlm.nih.gov/articles/PMC10406607/
[112] https://pmc.ncbi.nlm.nih.gov/articles/PMC7614633/
[113] https://pmc.ncbi.nlm.nih.gov/articles/PMC11922739/
[114] https://pmc.ncbi.nlm.nih.gov/articles/PMC10703687/
[115] https://openreview.net/forum?id=6URyQ9QhYv
[116] https://openreview.net/forum?id=FwdnG0xR02
