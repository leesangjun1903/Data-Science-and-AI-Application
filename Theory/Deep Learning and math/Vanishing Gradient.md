# Gradient Flow in Recurrent Nets: the Difficulty of Learning Long-Term Dependencies

아래는 Hochreiter et al.(1991)이 제시한 “Gradient Flow in Recurrent Nets: the Difficulty of Learning Long-Term Dependencies” 논문에서 사용된 핵심 수식들을 원문 수준으로 정리한 것입니다.

## 1. 순환 신경망의 기본 구조

- **은닉 유닛의 순전파**  

$$
  net_i(\tau) = \sum_j w_{ij} a_j(\tau-1)
  $$ 
  
$$
  a_i(\tau) = f_i(net_i(\tau))
  $$
  
  여기서  
  - $$net_i(\tau)$$: i번 유닛의 순입력  
  - $$a_j(\tau-1)$$: 이전 시점의 j번 유닛의 출력  
  - $$w_{ij}$$: 유닛 j → i로의 가중치  
  - $$f_i$$: i번 유닛의 활성화 함수

## 2. BPTT(Backpropagation Through Time)에서의 에러 신호 전파

- **출력 유닛 k에서의 에러 신호**  

$$
  \delta_k(t) = \frac{\partial E(t)}{\partial net_k(t)}
  $$  
  - $$E(t)$$: t시점의 에러 함수

- **은닉 유닛 j에서의 역전파 에러 신호**  

$$
 \delta_j(\tau) = f'\_j(net_j(\tau)) \sum_i w_{ij} \delta_i(\tau+1)
  $$

- **가중치 $$w_{jl}$$에 대한 업데이트**  

$$
  \Delta w_{jl} = \eta \, \delta_j(\tau) \, a_l(\tau-1)
  $$
  - $$\eta$$: 학습률

## 3. 에러 전파의 경로 적분 (Error Path Integral)

### (1) 재귀식 형태

$$
\frac{\partial \delta_v(s)}{\partial \delta_k(t)} =
\begin{cases}
f'\_v(net_v(t-1)) w_{kv} & t-s = 1 \\
f'\_v(net_v(s)) \sum_{l=1}^{n} \frac{\partial \delta_l(s+1)}{\partial \delta_k(t)} w_{lv} & t-s > 1
\end{cases}
$$

### (2) 시간 축을 따라 언롤링 했을 때의 공식

$$
\frac{\partial \delta_v(s)}{\partial \delta_k(t)} =
\sum_{l_{t-1}=1}^{n} \cdots \sum_{l_{s+1}=1}^{n}
\left[
w_{l_t l_{t-1}}
\prod_{\tau=t-1}^{s+1} f'\_{l_\tau}(net_{l_\tau}(\tau)) w_{l_\tau l_{\tau-1}}
\, f'\_{l_s}(net_{l_s}(s))
\right]
$$

여기서  
- $$l_t = k, \, l_s = v$$로 고정  
- $$\prod$$: 시간 역방향 곱

## 4. 기울기(Gradient)의 소실/폭발 조건

- **기울기 크기의 조건식**

$$
  |f'\_{l_\tau}(net_{l_\tau}(\tau)) w_{l_\tau l_{\tau-1}}|
  $$
  
  - 모든 $$\tau$$에 대해  
    - 위 값이 1보다 크면: 기울기 폭발 (Gradient Exploding)
    - 위 값이 1보다 작으면: 기울기 소실 (Vanishing Gradient)

- **시그모이드 함수의 미분 최대치**  

$$
  f'_{max} = 0.25
  $$

## 5. 기울기 경로의 상한 (썸네일 공식)

- **매트릭스 및 벡터 정의**

$$
  [W]_{ij} := w_{ij}
  $$
  
$$
  W_v = \text{v의 아웃고잉 가중치 벡터}
  $$
  
$$
  W_k^T = \text{k의 인커밍 가중치 벡터}
  $$
  
$$
  F'(t) = \text{대각행렬},\; [F'(t)]_{ij} = 
  \begin{cases}
  f'_i(net_i(t)), & \text{ if } i = j \\
  0, & \text{else}
  \end{cases}
  $$

- **상한 공식**  

$$
  \left|\frac{\partial \delta_v(s)}{\partial \delta_k(t)}\right|
  \leq n \, (f'\_{max} \|W\|_A)^{t-s}
  $$
  
  (여기서 $$\|W\|_A$$ : 가중치 행렬의 노름)

- **로그 식 사용시의 감쇠 조건**

$$\lambda = \frac {n w\_{max}}{4.0} 1 $$ : 상태가 불안정

- **체인 룰에 따른 전체 기울기**

$$
  \frac{\partial y(t)}{\partial y(s)} =
  \frac{\partial y(t)}{\partial y(t-1)}
  \frac{\partial y(t-1)}{\partial y(t-2)}
  \cdots
  \frac{\partial y(s+1)}{\partial y(s)}
  $$

- **손실 함수의 파라미터별 미분**

$$
  \frac{\partial E(t)}{\partial W}
  = \sum_{\tau \le t} \frac{\partial E(t)}{\partial y(\tau)}
  \frac{\partial y(\tau)}{\partial W}
  $$
  
$$
  = \sum_{\tau \le t} \frac{\partial E(t)}{\partial y(t)} \,
  \frac{\partial y(t)}{\partial y(\tau)} \,
  \frac{\partial y(\tau)}{\partial W}
  $$

이상의 수식들은 논문의 핵심 이론 분석을 뒷받침하는 기반입니다. 각 수식 옆의 개념들도 참고하여 전체 흐름을 이해할 수 있습니다[1].

[1] https://ml.jku.at/publications/older/ch7.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c06f5e07-18e4-4ce3-9954-a04cc4fc781c/Gradient_Flow_in_Recurrent_Nets_the_Difficulty_of.pdf
[3] https://researchonline.jcu.edu.au/57790/
[4] https://www.actahort.org/books/691/691_56.htm
[5] https://arxiv.org/pdf/1306.0514.pdf
[6] https://arxiv.org/pdf/2403.12887.pdf
[7] https://arxiv.org/pdf/2103.05487.pdf
[8] https://arxiv.org/pdf/2310.19870.pdf
[9] https://arxiv.org/pdf/1412.7753.pdf
[10] http://arxiv.org/pdf/2309.10002v1.pdf
[11] https://sferics.idsia.ch/pub/juergen/gradientflow.pdf
[12] https://arxiv.org/abs/1907.12545
[13] https://papers.ssrn.com/sol3/Delivery.cfm/5001243.pdf?abstractid=5001243&mirid=1
[14] https://srdas.github.io/DLBook/RNNs.html
[15] https://arxiv.org/pdf/1808.03314.pdf
[16] https://www.quarkml.com/2023/08/backpropagation-through-time-explained-with-derivations.html
[17] https://cs231n.stanford.edu/slides/2025/lecture_7.pdf
[18] https://velog.io/@syj1031/Cs231n-Lecture-10-Recurrent-Neural-Networks
[19] https://arxiv.org/html/2405.21064v1
[20] https://people.idsia.ch/~juergen/gradientflow/gradientflow.html
[21] https://abc.us.org/ojs/index.php/ei/article/view/570
[22] https://openreview.net/forum?id=vcJiPLeC48
[23] https://ai.stackexchange.com/questions/34840/what-does-it-mean-by-gradient-flow-in-the-context-of-neural-networks
[24] http://arxiv.org/pdf/2402.12241.pdf
[25] http://arxiv.org/pdf/2310.14982.pdf
[26] https://arxiv.org/html/2308.12075v2
[27] https://arxiv.org/pdf/2209.13394.pdf
[28] https://arxiv.org/html/2412.10094v1
[29] http://arxiv.org/pdf/2405.21064.pdf
[30] http://arxiv.org/pdf/1910.03471.pdf
[31] https://arxiv.org/pdf/2108.00051.pdf
[32] https://arxiv.org/pdf/2107.06608.pdf
[33] https://arxiv.org/pdf/1903.07120.pdf
[34] https://arxiv.org/pdf/1911.11033.pdf
[35] https://arxiv.org/pdf/2302.01687.pdf
