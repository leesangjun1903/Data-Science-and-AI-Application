# One-Class SVMs for Document Classification | PU-Learning(Positive-Unlabeled Learning)

## 핵심 주장과 주요 기여

이 논문은 **문서 분류에서 긍정적 예제만을 사용한 학습**이라는 새로운 패러다임을 제시합니다[1]. Schölkopf 등이 제안한 one-class SVM 알고리즘을 정보 검색 분야에 최초로 적용하여, 전통적인 two-class 분류 방법론의 한계를 극복하고자 했습니다[1][2].

주요 기여는 다음과 같습니다:

**1. 실용적 응용 가능성 확장**: 웹 사용자의 관심사이트 분류처럼 긍정적 예제만 얻을 수 있는 실제 상황에서 활용 가능한 분류 기법을 제시했습니다[1][2].

**2. 포괄적 비교 연구**: one-class SVM을 Rocchio 알고리즘, 최근접 이웃, naive Bayes, 압축 신경망 등 기존 one-class 방법들과 체계적으로 비교했습니다[1][2].

**3. 데이터 표현 방법의 중요성 발견**: 이진 표 Hadamard 표현 등 다양한 데이터 표현 방식이 성능에 미치는 영향을 실증적으로 분석했습니다[1][2].

## 해결하고자 하는 문제

**문제 정의**: 전통적인 문서 분류는 긍정적 예제와 부정적 예제 모두를 필요로 하지만, 실제 많은 상황에서는 긍정적 예제만 수집 가능한 경우가 있습니다[1]. 예를 들어, 웹 사용자의 관심 사이트를 추적하여 긍정적 예제는 식별할 수 있지만, 대표적인 부정적 예제를 식별하기는 어렵습니다[1][2].

**기존 방법의 한계**: 부정적 정보의 부재는 불가피한 성능 저하를 수반하며, 기존의 two-class SVM 방법론을 직접 적용할 수 없다는 근본적 문제가 있습니다[1][2].

## 제안하는 방법 및 수식

논문에서는 두 가지 one-class SVM 접근법을 제안합니다:

### 1. Schölkopf 방법론

이 방법은 **원점으로부터 데이터를 분리**하는 접근을 취합니다[1][2]. 

**수학적 정식화**:
데이터셋이 특성 공간에서 확률 분포 P를 가진다고 가정할 때, P로부터의 테스트 포인트가 영역 S 밖에 위치할 확률이 사전 지정된 값 ν로 제한되는 "단순한" 부분집합 S를 찾는 것이 목표입니다[1][2].

**최적화 문제**:

$$
\min_{w,\rho,\xi} \frac{1}{2}\|w\|^2 + \frac{1}{\nu l}\sum_{i=1}^l \xi_i - \rho
$$

제약조건:

$$
(w \cdot \Phi(x_i)) \geq \rho - \xi_i, \quad i = 1, 2, ..., l
$$

$$
\xi_i \geq 0
$$

여기서 Φ는 커널 맵, ν는 이상치 비율의 상한선, ξᵢ는 완화 변수입니다[1][2].

**결정 함수**:

$$
f(x) = \text{sign}((w \cdot \Phi(x)) - \rho)
$$

이 함수는 훈련 집합에 포함된 대부분의 예제에 대해 양수 값을 가집니다[1][2].

### 2. Outlier 방법론

이 방법은 **원점과 가까운 데이터 포인트들을 이상치로 식별**하는 접근을 취합니다[1][2]. 해밍 거리를 사용하여 비영 항목의 개수가 임계값보다 적은 벡터들을 이상치로 분류합니다[1][2].

## 모델 구조

### 데이터 표현 방식

논문에서는 네 가지 문서 표현 방식을 제안합니다[1][2]:

1. **이진 표현**: m차원 이진 벡터에서 i번째 항목은 i번째 키워드가 문서에 나타나면 1, 아니면 0
2. **빈도 표현**: i번째 항목은 특정 문서에서 i번째 키워드의 정규화된 빈도
3. **tf-idf 표현**: $$\text{tf-idf}(\text{keyword}) = \text{frequency}(\text{keyword}) \cdot [\log \frac{n}{N(\text{keyword})} + 1] $$
4. **Hadamard 표현**: i번째 항목은 문서 내 i번째 키워드 빈도와 전체 문서에서의 빈도의 곱

### 커널 선택

linear, sigmoid, polynomial, radial basis 커널을 실험하여 최적의 성능을 찾았습니다[1][2].

## 성능 향상 및 한계

### 성능 향상

**실험 결과**: Schölkopf의 one-class SVM은 압축 신경망을 제외한 모든 다른 방법들보다 우수한 성능을 보였습니다[1][2]. 특히:
- **최적 조건**: 이진 표현, 10개 특성, radial basis 커널 사용 시 F1 점수 0.519 달성[1][2]
- **신경망과 비교**: 압축 신경망과는 본질적으로 비교 가능한 성능을 보였으나, 구현이 더 간단합니다[1][2]

### 주요 한계

**1. 매개변수 민감성**: 가장 중요한 한계는 **표현 방식과 커널 선택에 대한 극도의 민감성**입니다[1][2]:
- 이진 표현에서만 우수한 성능을 보이며, tf-idf나 Hadamard 표현에서는 성능이 극도로 저조합니다[1][2]
- 커널별로 최적 특성 개수가 상이합니다 (polynomial: 20개, radial basis: 10개)[1][2]

**2. 투명성 부족**: 이러한 매개변수 선택이 성능에 미치는 영향의 메커니즘이 명확하지 않습니다[1][2].

**3. 견고성 부족**: 매개변수 변경에 따른 성능 변화가 극적이어서, 매개변수 선택에 대한 깊은 이해 없이는 안정적인 성능을 보장하기 어렵습니다[1][2].

## 일반화 성능 향상 가능성

### 현재 상황

논문의 실험 결과에 따르면, one-class SVM의 일반화 성능은 **매개변수 설정에 극도로 의존적**입니다[1][2]. 특히:

**카테고리별 특성 수**: 각 카테고리마다 카테고리별 특성을 사용하여, 전체 카테고리에 걸친 총 특성 수는 약 100개 정도입니다[1][2].

**성능 변화의 극단성**: 특성 수를 10개에서 20개로 증가시킬 때, radial basis 커널에서는 성능이 급격히 저하되지만 polynomial 커널에서는 오히려 향상됩니다[1][2].

### 향후 개선 방향

**1. 자동 매개변수 선택**: 교차 검증이나 메타 학습을 통한 자동 매개변수 최적화 기법 개발이 필요합니다[3].

**2. 견고한 표현 방법**: 다양한 도메인과 데이터 타입에서 안정적인 성능을 보이는 새로운 문서 표현 방법 연구가 요구됩니다[4].

**3. 앙상블 방법**: 여러 매개변수 설정의 결과를 결합하여 견고성을 향상시키는 방법을 고려할 수 있습니다[3].

## 앞으로의 연구에 미치는 영향과 고려사항

### 연구에 미치는 영향

**1. Positive-Unlabeled Learning의 발전**: 이 연구는 긍정-미분류 학습(PU learning) 분야의 기초를 마련했습니다[5][6][7]. 현재까지도 이 분야에서 활발한 연구가 진행되고 있습니다.

**2. 도메인 확장**: 의료 진단[8][9], 이상 탐지[10], 문서 스트림 처리[11][12] 등 다양한 응용 분야로 확산되었습니다.

**3. 방법론적 개선**: SVDD(Support Vector Data Description)와의 연결[13][14], 다중 one-class SVM[11], 특권 정보를 활용한 one-class SVM[15] 등으로 발전했습니다.

### 앞으로 연구 시 고려할 점

**1. 매개변수 선택 문제 해결**: 
- 자동 매개변수 튜닝 알고리즘 개발[8][3]
- 도메인 지식을 활용한 사전 설정 방법 연구
- 매개변수 민감성을 줄이는 새로운 목적 함수 설계

**2. 데이터 표현의 일반화**:
- 딥러닝 기반 특성 추출과의 결합[9]
- 다중 모달 데이터 처리 방법
- 도메인 적응형 표현 학습

**3. 확장성과 효율성**:
- 대규모 데이터셋에 대한 확장성 개선[16]
- 온라인 학습 및 점진적 학습 지원[11]
- 분산 처리 방법 개발

**4. 이론적 기반 강화**:
- 일반화 경계 이론 발전
- 최적성 보장 조건 분석
- 노이즈 견고성 이론 개발

**5. 실용적 응용 확대**:
- 불균형 데이터셋 처리 개선[4][6]
- 개념 드리프트 환경에서의 적응성[17]
- 설명 가능한 AI와의 결합

이 논문은 one-class 학습 패러다임을 정보 검색 분야에 도입한 선구적 연구로서, 현재까지도 관련 연구의 중요한 참조점이 되고 있습니다. 특히 매개변수 민감성 문제는 여전히 해결해야 할 핵심 과제로 남아있어, 향후 연구에서 지속적인 관심과 개선 노력이 필요합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9c29aaac-f039-4482-b673-d2ae6b50bcb4/manevitz01a.pdf
[2] https://www.jmlr.org/papers/volume2/manevitz01a/manevitz01a.pdf
[3] http://ieeexplore.ieee.org/document/7344789/
[4] https://www.tandfonline.com/doi/full/10.1080/02664763.2021.1919063
[5] http://www.inf.u-szeged.hu/~nistvan/data/TSD-2.pdf
[6] https://arxiv.org/html/2503.13562v1
[7] https://www.cs.uic.edu/~liub/publications/ijcai03-textClass.pdf
[8] https://dl.acm.org/doi/10.1145/3343147.3343152
[9] https://ieeexplore.ieee.org/document/9206955/
[10] https://pmc.ncbi.nlm.nih.gov/articles/PMC3041295/
[11] https://www.semanticscholar.org/paper/bff0902c0e2c1ad894d4eccad57891d9da00317c
[12] http://link.springer.com/10.1007/s10032-017-0286-6
[13] https://www.baeldung.com/cs/one-class-svm
[14] https://stats.stackexchange.com/questions/313857/why-one-class-svm-seperate-from-the-origin
[15] https://ieeexplore.ieee.org/document/10314493/
[16] https://www.sciencedirect.com/science/article/abs/pii/S0360835219301652
[17] http://ieeexplore.ieee.org/document/6628692/
[18] http://link.springer.com/10.1007/978-981-10-3153-3_11
[19] https://www.semanticscholar.org/paper/25c61beda55be5a1f665190384971980afef5386
[20] https://link.springer.com/10.1007/978-3-319-96133-0_4
[21] https://www.jmlr.org/papers/v2/manevitz01a.html
[22] https://www.xlstat.com/solutions/features/1-class-support-vector-machine
[23] https://github.com/RazMalka/One_Class_SVM_Document_Classification
[24] https://web.ece.ucsb.edu/~hespanha/published/one_class_svm_icpr.pdf
[25] https://dl.acm.org/doi/10.5555/944790.944808
[26] https://zephyrus1111.tistory.com/468
[27] https://dl.acm.org/doi/10.1145/1401890.1401920
[28] https://cris.haifa.ac.il/en/publications/one-class-svms-for-document-classification
[29] http://papers.neurips.cc/paper/1723-support-vector-method-for-novelty-detection.pdf
[30] https://stackoverflow.com/questions/13937720/how-to-train-a-classifier-with-only-positive-and-neutral-data
[31] https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
[32] https://ieeexplore.ieee.org/document/9738234/
[33] https://www.semanticscholar.org/paper/f4fe3c22e7f6cc86ee2fd44314d81c4be90efbee
[34] http://ieeexplore.ieee.org/document/5764798/
[35] https://www.semanticscholar.org/paper/a09375e7ec806e36010ccfb5b1a159806fbcb489
[36] https://www.semanticscholar.org/paper/aa5cce47cfa8f8ce5cdceb0765f740b39e7fcabf
[37] https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
[38] https://www.dfki.de/fileadmin/user_upload/import/6957_One-class-SVM_anomaly-detection.pdf
[39] https://flonelin.wordpress.com/2017/03/29/novelty%EC%99%80-outlier-detection/
[40] https://arxiv.org/pdf/1909.09862.pdf
[41] https://www.sciencedirect.com/science/article/abs/pii/S0167865514003584
[42] https://stackoverflow.com/questions/58394996/one-class-svm-and-isolation-forest-for-novelty-detection
[43] https://www.tandfonline.com/doi/full/10.1080/08839514.2013.785791
[44] https://scikit-learn.org/stable/modules/outlier_detection.html
[45] https://jayhey.github.io/novelty%20detection/2017/10/18/Novelty_detection_overview/
[46] https://www.numberanalytics.com/blog/mastering-one-class-svm-predictive-modeling
[47] https://velog.io/@moonjeongro/Novelty-and-Outlier-Detection
[48] https://scikit-learn.org/stable/modules/svm.html
[49] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002285550
[50] https://www.semanticscholar.org/paper/8f03eb2589370280be86a35765f36e382a865490
[51] https://arxiv.org/pdf/1802.03989.pdf
[52] https://figshare.com/articles/journal_contribution/Unsupervised_document_classification_integrating_web_scraping_one-class_SVM_and_LDA_topic_modelling/14494925/1/files/27766001.pdf
[53] http://www.isroset.org/pub_paper/IJSRCSE/1-ISROSET-IJSRCSE-09235.pdf
[54] https://pmc.ncbi.nlm.nih.gov/articles/PMC9930816/
[55] https://aclanthology.org/2021.eacl-main.296.pdf
[56] https://arxiv.org/pdf/1901.08688.pdf
[57] https://arxiv.org/pdf/1609.08039.pdf
[58] https://ccsenet.org/journal/index.php/cis/article/download/1058/1074
[59] http://arxiv.org/pdf/2403.07753.pdf
[60] https://www.mdpi.com/1424-8220/13/12/17130/pdf
[61] https://arxiv.org/ftp/arxiv/papers/1301/1301.2785.pdf
[62] https://sol.sbc.org.br/index.php/eniac/article/download/25738/25554
[63] https://limitsinx.tistory.com/147
[64] https://dataplatform.cloud.ibm.com/docs/content/wsd/nodes/ocsvm.html?locale=ko
[65] https://ieeexplore.ieee.org/document/1033211/
[66] http://link.springer.com/10.1007/978-3-319-46454-1_51
[67] http://arxiv.org/pdf/1909.09862.pdf
[68] http://arxiv.org/pdf/1804.04888.pdf
[69] http://arxiv.org/pdf/2107.06713.pdf
[70] http://arxiv.org/pdf/1706.09795.pdf
[71] http://arxiv.org/pdf/2011.03243.pdf
[72] https://arxiv.org/pdf/2203.00399.pdf
[73] http://arxiv.org/pdf/2312.14795.pdf
