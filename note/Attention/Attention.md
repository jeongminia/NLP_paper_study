# Attention

---

<aside>
💡 **Attention**
`Decoder`에서 출력 단어를 예측하는 매 시점마다, `Encoder`에서의 전체 입력 문장을 다시 한번 참고
해당 시점에서 예측해야 할 단어와 연관이 있는 입력 단어 부분을 더 **attention**해서 보게 됨
→ 필요한 정보에만 **“Attention”**

</aside>

---

## 0. Abstract

---

> **Neural machine translation by jointly learning to align and translate**
> 
> - 정렬과 번역을 동시 학습에 의한 신경망을 사용한 기계번역(NMT)

- 신경망 기계 번역은 번역 성능을 극대화하기 위해 인코더와 디코더가 하나의 신경망으로 통합되어 학습
- 고정된 길이의 벡터 사용이 기본 encoder-decoder 아키텍처의 성능 향상에 있어 **병목현상**이라 추측
- ⭐ 제안
    - 소스 문장의 부분을 자동으로 검색할 수 있도록 모델을 확장하는 방법 제안
        
        → Attention Mechanism을 통해 디코더가 소스 문장의 특정 부분을 동적으로 참조하고 
        
            필요한 정보를 효과적으로 활용할 수 있게 하는 방법을 제안
        

## 1. Introduction

---

🦾 **신경망 기계 번역 NMT** 

- 각 언어에 대해 Encoder - Decoder 를 갖거나 각 문장에 적용된 언어별 인코더 출력을 비교하는 방식
- Attention 등장 전,  Encoder의 마지막 Hidden State만을 기반해 출력 문장 생성
    - **문제점**
        - 신경망이 원문 문장의 모든 필요한 정보를 고정된 길이의 벡터로 압축
        - 긴문장을 처리하는 데에 어려움이 있으며 입력 문장의 길이가 길어질수록 급격히 저하

**기존 NMT의 해결방안 ⇒ Attention**

- Encoder - Decoder 모델을 확장해 정렬과 번역을 공동으로 학습하는 방법 제안
    - 단어를 생성할 때마다 원문 문장의 가장 관련성 높은 정보가 집중된 위치 집합을 검색
    - context 위치와 이전 생성된 모든 타겟 단어와 관련된 context vector 기반으로 타겟 단어 예측
- ⭐ 전체 입력 문장을 단일 ‘고정 길이’ 벡터로 인코딩하려고 시도하지 않음
    - 입력 문장을 일련된 벡터로 encoding, 번역을 decoding하는 동안 입력 문장의 모든 벡터를 참고
        - Encoder 출력의 길이는 입력 문장의 길이에 따라 변경
    - 모델은 입력 문장의 모든 정보를 하나의 고정된 벡터에 압축할 필요 없음
    
    → Attention 등장 후, Hidden state만이 아닌 weighted combination of all input states 고려
    

## 2. Model Architecture

---

확률적 관점에서 **번역** $= armax_yp(y|x)$

원문 문장 $x$가 주어졌을 때 조건부 확률 $p(y|x)$을 최대화하는 타겟문장 $y$를 찾는 것 

**신경망 기계 번역**

병렬 훈련 corpus를 사용하면 **문장 쌍의 조건부 확률을 최대화**하도록 매개변수화된 모델을 맞춤

→ 주어진 원문 문장에 대해 조건부 확률을 최대화하는 문장을 검색해 대응되는 번역 생성

### **RNN Encoder-Decoder Architecture**

입력 문장을 인코딩하고 이를 디코딩해 번역을 생성하는 프레임워크

**Encoder**

- 인코딩 : 입력 문장은 시퀀스 형태로 RNN에 의해 처리되고 마지막 은닉 상태가 입력 문장의 요약인 c로 사용
- 입력 문장 $x = (x_1, … , x_{T_x})$를 고정된 길이 벡터 $c$ 로 변환
    - 각 단어 $x_t$는 RNN을 통해 처리되어 hidden state $h_t = f(x_t, h_{t-1})$ 갱신
    - encoder의 마지막 *hidden state* $h_{T_x}$는 전체 입력 문장을 요악한 context vector $c$로 사용될 수 있음
        
        → $c = h_{T_x}$
        

**Decoder**

- 디코딩 : c와 이전에 예측된 단어들을 사용해 다음 단어의 조건부 확률을 계산하고 이를 기반으로 번역 생성
- context vector $c$와 이전에 생성된 단어들을 기반으로 다음 단어 예측
    - decoder는 다음 단어 $y_{t’}$를 예측하기 위해 context vector c와 이전에 예측 된 단어들
        
        {${y_1, ..., y_{t’-1}}$}을 사용
        
    - 조건부 확률 $p(y_t|y_1, …, y_{t-1}, c)$는 디코더의 hidden state 를 기반으로 하는 비선형 함수 $g(y_{t-1},s,c)$에 의해 모델링 됨
        - 이때 비선형 함수는 LSTM이나 GRU와 같은 복잡한 구조를 가질 수 있음

### ↪️ Learning to Align and Translate

각 타겟 단어 $y_t$를 예측할 때 고정된 context vector $c$ 대신, 동적으로 변하는 context vector $c_i$ 사용

![스크린샷 2024-07-28 오전 8.27.20.png](Attention%209d2d24bb6f704d8c81f70531e90659df/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-28_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.27.20.png)

**조건부 확률 정의**

$$
p(y_i|y_1, …, y_{i-1}, x) = g(y_{i-1},s_i,c_i)
$$

- $s_i$ 는 시간 i에서의 디코더의 hidden state로 $s_i = f(s_{i-1}, y_{i-1}, c_i)$
- 각 타겟 단어 $y_i$에 대해 서로 다른 context vector $c_i$를 조건으로 함
- 새로 구한 Context Vector $C_t$와 디코더 이전 Hidden State Vector $S_{t-1}$와 이전 출력 단어 $Y_{t-1}$를 입력으로 받음 → $S_t$를 갱신하고 이를 이용해 새로운 출력 단어 $Y_t$ 결정

**context vector  $c_i$**

어텐션 가중치 $\alpha_{ij}$와 인코더의 은닉 상태 $h_j$의 가중합으로 계산 : $c_i = \Sigma\alpha_{ij}h_j$

- 디코더가 번역할 때 특정 타겟단어를 생성하기 위해 입력 문장의 어떤 부분에 집중할지 결정
- 가중치와 은닉 상태
    
    ![밑바닥부터 시작하는 딥러닝2](Attention%209d2d24bb6f704d8c81f70531e90659df/d8cb26ee-21fb-44a6-9c6a-c9454c6f50b0.png)
    
    밑바닥부터 시작하는 딥러닝2
    
    - hidden state $h_t = f(x_t, h_{t-1})$로 생성
    - attention weight $\alpha$
        
        ![스크린샷 2024-07-28 오전 8.24.46.png](Attention%209d2d24bb6f704d8c81f70531e90659df/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-28_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.24.46.png)
        
        ![스크린샷 2024-07-28 오전 8.25.00.png](Attention%209d2d24bb6f704d8c81f70531e90659df/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-28_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.25.00.png)
        

### Encoder : Bidirectional RNN for Annotating Sequences

양방향 RNN은 Attention 메커니즘의 필수적인 요소는 아니지만, 어노테이션의 품질을 높이고 디코더가 **더 정확한 정렬 가중치를 계산하는 데 도움**됨

- 일반적인 RNN의 확장으로 입력 시퀀스를 순방향과 역방향 모두 처리해 입력 시퀀스가 각 단어가 그 이전과 이후 단어들의 정보를 모두 요약할 수 있도록 함

![스크린샷 2024-07-28 오전 8.41.04.png](Attention%209d2d24bb6f704d8c81f70531e90659df/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-28_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.41.04.png)

- 해당 단어의 순방향 은닉 상태와 역방향 은닉 상태를 연결해 얻어짐
- 이전 단어와 이후 단어의 요약을 모두 포함하게 되어 입력 값의 주변 단어에 더욱 attention
    - ***forward RNN***과 ***backward RNN***으로 구성됨
        
        
        ***Forward RNN* $\overrightarrow{f}$** 
        
        ![스크린샷 2024-07-27 오후 2.47.59.png](Attention%209d2d24bb6f704d8c81f70531e90659df/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-27_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_2.47.59.png)
        
        - $x_1$부터 $x_{T_x}$순서로
        hidden state $(\overrightarrow{h_1},・・・,\overrightarrow{h_{T_x}})$ 계산
        
        ***Backward RNN $\overleftarrow{f}$***
        
        ![스크린샷 2024-07-27 오후 2.48.59.png](Attention%209d2d24bb6f704d8c81f70531e90659df/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-27_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_2.48.59.png)
        
        - $x_{T_x}$부터 $x_1$(reverse)순서로
        hidden state $(\overleftarrow{h_1},・・・,\overleftarrow{h_{T_x}})$ 계산
        

## 3.  Experiments

---

<aside>
🗣 **English-to-French translation** CL WMT ’14에서 제공하는 이중 언어 병렬 코퍼스를 사용
일반적인 RNN Enc-Dec 🆚 RNNresearch Attention

</aside>

- 모델 구조
    - 각 모델을 두 번 훈련하는데, 처음에는 최대 30단어 길이의 문장으로 (RNNencdec-30, RNNsearch-30), 최대 50단어 길이의 문장으로 (RNNencdec-50, RNNsearch-50) 훈련
    - RNNencdec의 인코더와 디코더는 각각 1000개의 은닉 유닛
    - RNNsearch의 인코더는 순방향 및 역방향 순환 신경망(RNN)으로 구성되며, 각각 1000개의 은닉 유닛
    - 다중 레이어 네트워크를 사용하며, 단일 maxout(Goodfellow et al., 2013) 은닉 레이어를 통해 각 타겟 단어의 조건부 확률을 계산
    - 각 모델을 훈련하기 위해 우리는 미니배치 확률적 경사 하강법(SGD) 알고리즘과 Adadelta(Zeiler, 2012)를 사용
    - 각 SGD 업데이트 방향은 80개의 문장으로 구성된 미니배치를 사용

### General Translation

![[표1] BLEU 점수로 측정된 번역 성능](Attention%209d2d24bb6f704d8c81f70531e90659df/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-28_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_9.08.45.png)

[표1] BLEU 점수로 측정된 번역 성능

![[그림2]](Attention%209d2d24bb6f704d8c81f70531e90659df/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-28_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_9.12.51.png)

[그림2]

- RNNsearch 모델은 알려진 단어로만 구성된 문장에서는 기존의 구문 기반 번역 시스템(Moses)과 동일한 수준의 성능을 발휘
- 문장의 길이에 더 강인한 모습을 보이며 성능저하 없음

![스크린샷 2024-07-28 오전 9.11.24.png](Attention%209d2d24bb6f704d8c81f70531e90659df/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-28_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_9.11.24.png)

**어탠션 매커니즘 시각화**

각각의 서브플롯은 특정 문장의 번역과정에서 어탠션이 어떻게 분배되는지 보여줌

- attention score $\alpha_{ij}$ 가 높을수록 밝음
- X축의 각 영어 단어가 번역될 때 Y축의 프랑스어 단어들에 어떻게 집중되는지 보여줌

**Alignment**

- 어노테이션 가중치 시각화를 통해 번역 과정에서 원본 문장의 어떤 부분이 중요한지 직관적으로 파악
- 문장 길이 차이를 자연스럽게 처리할 수 있어 보다 정확하고 유연한 번역 가능

### Long Sequence

RNNsearch가 긴 문장을 고정 길이 벡터로 완벽하게 인코딩할 필요 없이 특정 단어 주변의 입력 문장 부분만 정확하게 인코딩하기 때문에 긴 문장에서도 좋은 성능

**Ex**

`ENG`

An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centre to carry out a diagnosis or a procedure, based on his status as a health care worker at a hospital.

`RNNencdec-50`

Un privilege d’admission est le droit d’un m  edecin de reconna ´ ˆıtre un patient a l’hopital ou un centre m ˆ edical ´ d’un diagnostic ou de prendre un diagnostic en fonction de son etat ´ de sante.

⇒  [a medical center] 이후로 원문의 의미에서 벗어났기에 긴 문장에서 좋은 성능을 보이지 않음

`RNNsearch-50`

Un privilege d’admission est le droit d’un m edecin d’admettre un patient ´ a un hopital ou un centre m ˆ edical ´ pour effectuer un diagnostic ou une procedure, ´ selon son statut de travailleur des soins de sante´ a` l’hopital.

## 4. Conclusion

---

**새로운 아키텍쳐 제안 Attention**

- 인코더-디코더 모델을 확장해 각 타겟 단어를 생성할 때 입력 문장의 관련 부분을 **동적**으로 검색
- 효과
    - 모델이 전체 문장을 하나의 벡터로 압축할 필요 없음
    - 다음 타겟 단어를 생성하는 데 관련된 정보에만 집중
    - 긴 문장에서 좋은 성능을 보임

**실험 결과**

- RNNsearch는 문장 길이에 관계없이 기존의 인코더-디코더 모델 `RNN enc-dec` 보다 훨씬 우수한 성능을 보임
- 타겟 단어와 원문 단어의 정렬이 올바르게 이루어짐
- 기존의 구문 기반 통계 기계 번역 시스템과 견줄만한 성능 달성

## 🔎 궁금증 ..

---

Q. annotation 이라는 말이 잘 와닿지 않는데 각 단어를 잘 이해할 수 있도록 돕는 아이라고 생각해야 하나

“h는 입력 문장의 각 단어에 대한 주석(annotation)을 의미”

A. 입력 시퀀스의 각 단어에 대해 생성된 은닉 상태를 의미

## ➕ 참고

---

- 밑바닥부터 시작하는 딥러닝2
- 자연어 처리 입문