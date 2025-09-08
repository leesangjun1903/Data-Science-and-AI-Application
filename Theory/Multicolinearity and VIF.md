# Multicolinearity and VIF(Variance Inflation Factors) 가이드

독립 변수 X는 종속 변수 Y 하고만 상관 관계가 있어야 하며, 독립 변수끼리 상관 관계가 있어서는 안 됩니다.  
독립 변수간 상관 관계를 보이는 것을 다중공선성(Multicollinearity)이라고 합니다. 다중공선성이 있으면 부정확한 회귀 결과가 도출됩니다. (X와 Y의 상관 관계가 반대로 나온다던가 검정 결과가 다르게 나온다던가 말이죠.)

다중공선성(Multicollinearity)은 회귀분석에서 독립변수들 간에 매우 높은 상관관계가 있을 때 발생하는 문제입니다. 즉, 독립변수들이 서로 거의 같은 정보를 가지고 있어 각각의 영향을 구분하기 어려운 상태를 말합니다.

이로 인해 회귀계수 추정이 불안정해지고, 독립변수 각각의 효과를 제대로 파악하기 어려워지며, 모델의 예측력도 떨어질 수 있습니다. 예를 들어, 소득과 소비 지출처럼 서로 밀접하게 연관된 변수를 동시에 사용할 때 나타날 수 있습니다.

다중공선성(Multicollinearity)에서 VIF(Variance Inflation Factors)는 회귀모델에서 예측력은 비슷한데 계수가 불안정하거나, 부호가 역전되는 문제를 진단·해결하는 핵심 도구입니다.[1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17][18][19][20][21][22][23][24]  
특히 VIF는 각 독립변수가 다른 독립변수들로 얼마나 잘 설명되는지를 수치로 보여주며, 일반적으로 **5 이상 주의, 10 이상 심각**으로 해석하는 실무 규칙이 널리 쓰입니다.[9][10][11][12][13][14][15][16][17][18][19][20][21][22][24][1]

## 개념 한눈에
- 정의: 다중공선성은 독립변수들 사이의 높은 상관·중복 정보 때문에 계수 분산이 커지고 추정이 불안정해지는 현상입니다.[13][16][21][22]
- 징후: 계수 부호 역전, p-값 급변, 표준오차 팽창, 변수 추가/제거 시 계수 급등락 등입니다.[16][19][21][24]
- 영향: 해석 난이도 증가와 추정 불확실성 확대가 핵심이며, 예측력(R²)이 그대로여도 계수 신뢰성은 크게 악화될 수 있습니다.[19][21][24][16]

### 주요 문제 :
- 순수 영향력의 왜곡: 다중공선성이 존재할 때 회귀 계수는 한 독립 변수가 다른 독립 변수들의 영향을 통제한 후 나머지 변수들에 미치는 순수 영향력을 나타내야 하지만, 변수 간 상호작용으로 인해 그 의미가 왜곡됩니다. 
- 불안정한 계수: 변수 간 상관관계가 강할수록 추정된 회귀 계수 값은 매우 불안정해집니다. 조금만 데이터가 바뀌어도 계수 값이 크게 변동할 수 있으며, 이는 해당 변수의 영향력을 믿기 어렵게 만듭니다. 
- 부호의 변화: 이론적으로 예상되는 부호와 다른 부호의 회귀 계수가 나타날 수 있습니다. 예를 들어, 양의 관계를 가질 것으로 기대되는 변수가 음의 계수를 보일 수 있습니다. 
- 신뢰구간의 확대: 높은 다중공선성은 회귀 계수의 표준 오차를 증가시켜 신뢰구간을 넓게 만듭니다. 이는 추정된 계수의 정확성이 낮다는 것을 의미합니다. 
- 해석의 어려움: 결국 다중공선성이 강할수록 특정 변수의 회귀 계수가 종속 변수에 미치는 정확한 영향을 해석하는 것이 매우 어려워집니다. 

## VIF(Variance Inflation Factors) 핵심 정리
- 정의/수식: 변수 $$X_j$$의 VIF는 $$\mathrm{VIF}_j = \frac{1}{1-R_j^2}$$로, $$R_j^2$$는 $$X_j$$를 다른 모든 독립변수들로 회귀분석했을 때의 결정계수입니다.[13][16]
- 해석: $$\mathrm{VIF}=1$$이면 독립, 값이 커질수록 공선성 심화, 상한은 무한대입니다.[16][13]
- 임계값: 문헌·실무에서 5 또는 10을 많이 사용하며, 표본 크기·맥락에 따라 2.5 같은 보수 기준도 제시됩니다.[24][19][13][16]

즉, $(R_j^2)$ 가 0이라면 즉, 다른 변수들과 상관관계가 없다면 VIF는 1이며, 분산이 전혀 증가하지 않은 상태를 의미합니다. 반면 $(R_j^2)$ 가 커지면(다른 변수로 해당 변수를 잘 설명할수록) 분모가 작아져 VIF가 커져 분산이 크게 '팽창(Inflate)'되었다고 판단합니다. 일반적으로 VIF가 4 이상이면 주의를 요하고 10 이상이면 심각한 다중공선성이 있다고 봅니다.

요약하면, VIF는 특정 독립변수가 다른 독립변수들과 얼마나 선형적으로 관련되어 있는지를 수치화하여 회귀계수 추정의 불확실성이 얼마나 커졌는지를 알려주는 지표입니다.

### 결정계수
결정계수는 회귀모형이 주어진 데이터에서 종속변수 변동을 얼마나 잘 설명하는지를 나타내는 척도로, 0부터 1 사이의 값을 가지며 값이 클수록 설명력이 높다는 뜻입니다. 주로 ($R^2$)로 표기합니다.

즉, 결정계수는 총변동(SST) 중에서 회귀모형으로 설명 가능한 변동(SSR)의 비율로 정의되며, 수식으로는

```math
[R^2 = 1 - \frac{SSE}{SST}
]
```

- ( $SST = \sum (y_i - \bar{y})^2 )$ : 데이터의 총 변동량으로, 종속 변수의 관측값($y_i$)과 표본의 평균($ȳ$)의 차이(편차)를 제곱하여 합한 값입니다. 
- ( $SSE = \sum (y_i - \hat{y}_i)^2 )$ : 모델이 설명하지 못한 오차 변동량으로, $ŷ$는 회귀 모형의 예측값입니다.

(여기서 SSE는 오차제곱합)로 표현됩니다.

하지만 설명 변수가 많아질수록 자연스럽게 상승하기 때문에 독립변수 수와 표본 크기를 고려한 수정된 결정계수가 더 정확한 모형 평가에 사용됩니다.

결정계수는 상관계수와 다르며, 결정계수는 모형의 설명력을, 상관계수는 변수 간 상관성의 방향과 크기를 나타냅니다.

## 실무 체크리스트
- 1차 진단: 산점도 행렬/상관행렬로 구조적 상관 파악하기(빠르고 직관적).[11][21][16]
- 정량 진단: 변수별 VIF 계산(회귀분석 모형에서의 상수항 포함)과 해석 기준 적용하기.[14][17][20]
- 교차 검증: 문제 변수 제거 전후 계수·표준오차·성능 변화를 비교합니다.[19][24][16]

## 파이썬 코드 템플릿(statsmodels)
- 목적: 산점도/상관행렬로 1차 점검 후, VIF로 정량 진단하고, 변수 제거·재학습으로 안정화합니다.[17][20][14]
- 핵심 포인트: VIF 계산 시 상수항을 반드시 포함하고, 모든 변수에 대해 순회 계산합니다.[20][14][17]

```python
# 0) 패키지
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1) 데이터 로드
df = pd.read_csv("house_prices.csv")  # 예: price, area, bedrooms, bathrooms 등 [22]

# 2) 1차 탐색: 산점도/상관행렬
sns.pairplot(df[['bedrooms','bathrooms','area']])  # 시각적 상관관계 점검 [26]
corr = df[['bedrooms','bathrooms','area']].corr()
print(corr.round(2))  # 높은 상관 확인 [16]

# 3) 기본 회귀 적합
X = add_constant(df[['area','bedrooms','bathrooms']])  # 상수항 포함이 중요 [19]
y = df['price']
ols_full = sm.OLS(y, X).fit()
print(ols_full.summary())  # 계수 부호/SE/p-값 확인 [15]

# 4) VIF 계산 함수
def compute_vif(design_df):
    X_ = add_constant(design_df, has_constant='add')  # 상수항 보장 [19]
    vifs = pd.Series(
        [variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])],
        index=X_.columns,
        name='VIF'
    )
    return vifs

vif_full = compute_vif(df[['area','bedrooms','bathrooms']])
print(vif_full)  # 5~10 이상 변수 주의 [18]

# 5) 개선안 1: 중복 변수 제거
X_red = add_constant(df[['area','bedrooms']])  # bathrooms 제거 예시 [26]
ols_red = sm.OLS(y, X_red).fit()
print(ols_red.summary())
print(compute_vif(df[['area','bedrooms']]))  # VIF 하락 확인 [26]

# 6) 개선안 2: 정규화 회귀(릿지)로 분산 안정화
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ridge = make_pipeline(StandardScaler(with_mean=True), RidgeCV(alphas=np.logspace(-3,3,25)))
ridge.fit(df[['area','bedrooms','bathrooms']], y)  # 계수 수축으로 공선성 완화 [21]
print("Ridge alpha:", ridge.named_steps['ridgecv'].alpha_)  # 선택된 규제강도 [21]

# 7) 개선안 3: PCA로 상관 구조 분해 후 회귀
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
X_std = scaler.fit_transform(df[['area','bedrooms','bathrooms']])
pca = PCA().fit(X_std)
X_pcs = pca.transform(X_std)[:, :2]  # 상위 주성분 2개 예시 [21]
lm = LinearRegression().fit(X_pcs, y)
cv_rmse = (-cross_val_score(LinearRegression(), X_pcs, y, scoring='neg_root_mean_squared_error', cv=5)).mean()
print("PCA-reg CV RMSE:", round(cv_rmse,2))  # 예측안정성 비교 [21]
```


## 딥러닝 관점에서 왜 중요할까?
- 표현학습 전전처리: 탭(Tabular)·수치형 변수 입력의 DNN에서 공선성은 가중치 식별성·해석 가능성을 저해할 수 있어, 입력 정제·차원축소가 유리합니다.[24][16][19]
- 정규화의 역할: 릿지/라쏘는 선형층의 가중치 분산을 수축하여 공선성 민감도를 낮추며, 심층모델에서도 가중치 감쇠나 스킵연결 설계가 비슷한 안정화 효과를 냅니다.[16][19][24]
- 특성 선택과 병행: 임계 VIF 초과 변수 제거, 임베딩 차원 축소, PCA/오토인코더 병행으로 데이터 효율성과 일반화를 함께 확보할 수 있습니다.[17][24][16]

## 진단→해결 워크플로우
- Step 1: 산점도/상관행렬로 구조적 상관을 빠르게 점검합니다(패턴·군집·선형성 확인).[21][11]
- Step 2: 변수별 VIF 산출 후 기준(5/10)을 적용해 후보 변수를 표기합니다(상수항 포함).[14][13][16]
- Step 3: 제거 실험(Forward/Backward/Stepwise), 릿지/라쏘, PCA 등 대안을 비교합니다.[19][24][16]
- Step 4: 전후 비교는 계수 안정성, 표준오차, 교차검증 성능으로 판단합니다(R² 유사해도 계수 신뢰도는 크게 개선 가능).[24][16][19]

## 해석 팁과 주의사항
- 임계값은 맥락적: 표본 크기·모형 목적에 따라 2.5/5/10 기준을 선택합니다(설명 중심일수록 보수적으로).[13][19][24]
- 상호작용·다항항: 생성 특성은 공선성을 급격히 키울 수 있어 표준화·정규화·정규화회귀와 병행합니다.[16][24]
- 상수항 포함: VIF 계산 전 add_constant를 꼭 확인하지 않으면 결과가 왜곡됩니다.[20][14][17]

## 예시(케이스 스터디) :
- 현상: bedrooms와 bathrooms가 강한 양의 상관 → 다중공선성으로 bedrooms의 회귀계수 부호가 음수로 역전.[10][21]
- 조치: bathrooms 제거 후 재학습 → bedrooms 계수 정상화, VIF<10로 안정화, R² 거의 동일 유지.[21][10]
- 교훈: 예측력은 유지되더라도, 해석 가능성과 추정 안정성은 크게 개선됩니다.[21][19][24]

## 참고 링크(학습·실전)
- VIF 임계값과 문헌 정리: 5/10 규칙과 보수 기준 2.5까지 정리된 개관 자료.[13]
- 파이썬 VIF 계산 실무 팁: 상수항 포함과 구현 패턴(Stack Overflow).[14]
- 교육 자료: VIF 정의·수식·해석과 임계값의 맥락성 요약.[16]

## 원문 가이드 참고
- 산점도→OLS→VIF→변수제거→재적합의 흐름과 예시 설명이 잘 정리되어 있습니다(“DATA-20” 글의 코드·설명 구조 참조).[10]
- 동일 R²에서 변수 제거로 계수 해석력이 회복되는 실제 사례가 제시됩니다.[10]

[1](https://www.mdpi.com/2078-2489/13/3/138/pdf)
[2](https://arxiv.org/pdf/2109.05227.pdf)
[3](https://arxiv.org/html/2407.21275v2)
[4](http://downloads.hindawi.com/archive/2013/671204.pdf)
[5](https://arxiv.org/pdf/1206.6430.pdf)
[6](https://arxiv.org/html/2312.13148v2)
[7](https://arxiv.org/pdf/2204.13089.pdf)
[8](https://arxiv.org/pdf/1805.11183.pdf)
[9](https://aliencoder.tistory.com/17)
[10](https://bkshin.tistory.com/entry/DATA-18)
[11](https://direction-f.tistory.com/46)
[12](https://wnsgud4553.tistory.com/150)
[13](https://quantifyinghealth.com/vif-threshold/)
[14](https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python)
[15](https://iamnotwhale.tistory.com/7)
[16](https://library.fiveable.me/introduction-econometrics/unit-7/variance-inflation-factor-vif/study-guide/hJe2cL9unRj6djuQ)
[17](https://www.datascienceconcepts.com/tutorials/python-programming-language/multicollinearity-in-python/)
[18](https://steadiness-193.tistory.com/271)
[19](https://pmc.ncbi.nlm.nih.gov/articles/PMC6713981/)
[20](https://wikinist.tistory.com/202)
[21](https://piscesue0317.tistory.com/37)
[22](https://www.investopedia.com/terms/v/variance-inflation-factor.asp)
[23](https://heeya-stupidbutstudying.tistory.com/entry/%ED%86%B5%EA%B3%84-%EB%8B%A4%EC%A4%91%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D-%EC%98%88%EC%A0%9C-Statsmodel%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EA%B3%A0%EC%9C%A0%EA%B0%92-vif-%ED%99%95%EC%9D%B8)
[24](https://blasbenito.com/post/variance-inflation-factor/)
[25](https://bkshin.tistory.com/entry/DATA-20-%EB%8B%A4%EC%A4%91%EA%B3%B5%EC%84%A0%EC%84%B1%EA%B3%BC-VIF)
[26](https://arxiv.org/pdf/2406.08776.pdf)
[27](http://arxiv.org/pdf/2405.04043.pdf)
[28](https://arxiv.org/pdf/1906.12123.pdf)
[29](https://arxiv.org/html/2406.02545v2)
[30](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)
[31](https://corporatefinanceinstitute.com/resources/data-science/variance-inflation-factor-vif/)
[32](https://www.mindscale.kr/course/basic-stat-python/13)

# Reference
https://bkshin.tistory.com/entry/DATA-20-%EB%8B%A4%EC%A4%91%EA%B3%B5%EC%84%A0%EC%84%B1%EA%B3%BC-VIF
