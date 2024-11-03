# Time Series Transformer

---

<aside>
💡 **Time Series Transformer**
Transformer기반 시계열 데이터 Time Series 예측 모델 → **LTSF-Linear**라는 간단한 선형 모델도입

</aside>

![[https://github.com/qingsongedu/time-series-transformers-review](https://github.com/qingsongedu/time-series-transformers-review)](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/image.png)

[https://github.com/qingsongedu/time-series-transformers-review](https://github.com/qingsongedu/time-series-transformers-review)

---

## 0. Abstract

---

> **Are Transformers Effective for Time Series Forecasting?**
> 
> - 장기 시계열 예측에 Transformer가 효과적인가?

- 시계열 데이터에서는 시간에 따른 **이전 데이터가 다음 데이터에 영향을 미치는** 연속적인 특성이 핵심
- 순서에 무관한(self-attention) 메커니즘의 특성상 시간 정보가 필연적으로 손실

⇒ 매우 간단한 하나의 층(layer)으로 구성된 선형 모델인 **LTSF-Linear**

## 1. Introduction

---

### 기존 연구

**TSF** Time Series Forecast 

- Solution은 전통적인 통계 방법(ARIMA), 기계 학습 기법(GBRT), 심층 학습 기반 솔루션(RNN Recurrent Neural Networks, TGN Temporal Convolutional Networks)로 발전

**Transformer**

- 긴 시퀀스 내 요소들 간의 의미적 상관관계를 추출하는 데 탁월한 능력
    - 구조 재확인
        - sequence를 한번에 처리함 ↔ 기존 RNN 모델의 경우, 순차적으로 데이터를 처리했음
            - 학습과 병렬화가 굉장히 쉽도록 도움
            - 여러 개로 분할해서 **병렬로 어텐션을 실행**하고 **결과값을 다시 합치는 방식** 진행
        
        > ⏬ **인코더 블럭 구조**
        > 
        > 
        > ![[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_12.06.28.png)
        > 
        > [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
        > 
        > ( `Input Sequence` → `Embedding Layer` → `Positional Encoding` )
        > → `Multi-head Self-Attention` → `Feed Forward Neural Network`
        > 
        > ---
        > 
        > - 단어를 벡터로 임베딩
        > - `Multi-head Self-Attention`
        >     - unmasked : 순차적으로 정보를 처리할 필요가 없어 한번에 다 가져옴
        >     - 문장에서 각 단어끼리 얼마나 관계가 있는 지를 계산해서 반영
        >     - 각각의 단어가 갖고 있는 위치(position)은 그대로 유지가 됨
        > - `Feed Forward Neural Network`
        >     - 각각 단어에 대해 Neural Network 적용
        
        ⇒ 매번 인코더 블럭의 가중치는 충분히 학습을 통해서 달라질 수 있음
        
        > 🔃 **디코더 블럭 구조**
        > 
        > 
        > ![[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_12.06.55.png)
        > 
        > [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
        > 
        > ( `Input Sequence` → `Embedding Layer` → `Positional Encoding` )
        > → `Masked Multi-Head Self-Attention` → `Multi-Head Attention (Encoder-Decoder)` 
        > → `Feed Forward Neural Network` 
        > 
        > ---
        > 
        > - `Masked Multi-Head Self-Attention`
        >     - 추측해야 하는 단어를 masked 처리한 상태에서 진행
        >     - masked : 문장을 만드는 과정에서 순서가 중요하기에 순서에 따라서 masking이 되어 있음
        >     - 문장에서 각 단어끼리 얼마나 관계가 있는 지를 계산해서 반영
        >         
        >         → 자기 자신끼리 self로 attention 진행
        >         
        > - `Multi-Head Attention (Encoder-Decoder)`
        >     - 인코더-디코더 층에서 인코더가 처리한 정보를 받아 어텐션 매커니즘을 수행 → 정보 반영
        >     - self 가 아님
        >         
        >         → Encoder에서 가져온 정보로 attention 수행 
        >         
        > - `Feed Forward Neural Network`
- BUT. self-attention은 순서에 무관한(permutation-invariant) 특성을 갖고 있음
    - 다양한 유형의 위치 인코딩 기법을 사용하여 순서 정보를 어느 정도 보존할 수는 있음
    - 그러나 그 위에 self-attention을 적용한 후에는 시간 정보 손실이 불가피
- 일부 논의된 방법들은 **오차 축적 효과**로 인해 심각한 성능 저하를 겪고 있음

🔄 순서 자체가 가장 중요한 역할을 하는 시계열 연구에서 Transformer가 과연 정말 좋은 역할을 할까?

<aside>
💡

**Goal**

---

1.  **DMS** direct multi-step 예측 전략을 통해 Transformer 기반 LTSF 솔루션의 실제 성능을 검증
    - LTSF이 비교적 명확한 경향성과 주기성을 가진 시계열에 대해서만 실현 가능하다고 가정

1. **LTSF-Linear**라는 매우 간단한 모델들을 새로운 비교 기준으로 도입
    - 하나의 층으로 구성된 선형 모델을 사용해 과거 시계열을 회귀 분석하고 미래 시계열을 예측
    - 교통, 에너지, 경제, 날씨, 질병 예측 등 다양한 실제 응용 분야를 다루는 9개의 널리 사용되는 벤치마크 데이터셋에서 실험을 수행
        
        ---
        
        **결과**
        
        - LTSF-Linear가 기존의 복잡한 Transformer 기반 모델들을 모든 경우에서 능가
        - 때로는 상당한 차이(20% ∼ 50%)로 더 우수한 성능
        - 대부분의 경우 긴 시퀀스로부터 시간적 관계를 제대로 추출하지 못한다는 것을 발견
        - TTSF 솔루션의 다양한 설계 요소들이 성능에 미치는 영향을 연구하기 위해 실험 수행
</aside>

### Preliminaries

> 다변량 시계열 데이터가 C개의 변수로 구성된다고 가정.
> 
> - 과거 데이터  $X = {{X_1^t, ...,X_C^t}}_{t=1}^L$
>     
>     ($L$ : 과거의 관찰 구간 Look-Back Window,  $X_1^t$ : $i$번째 변수의 $t$시점의 값)
>     
> - 미래 데이터 예측하기 위해 시계열 예측 작업 진행하며 이때 예측 과정에서 크게 두가지 방법 존재 🔽
>     
>     $\hat{X} = {{\hat{X}_1^t, ...,\hat{X}_C^t}}_{t=L+1}^{L+T}$
>     
- **IMS** Iterated Multi-Step
    - 한 번에 한 단계씩 예측하는 방식
    - 단점 : 앞 선 단계에서 예측이 틀리면 그 다음 예측도 틀릴 가능성이 높아짐
        
        ← 짧은 구간 예측에 강점
        
- **DMS** Direct Multi-Step
    - 여러 단계의 예측을 한꺼번에 하는 방식
    - 단점 : 오차가 쌓이는 문제는 없지만, 모델을 처음부터 더 정확하게 만드는 게 어려움
        
        ← 긴 구간 예측에 강점
        

## 2. Model Architecture

---

Transformer 모델의 장기 종속성(long-range dependencies)을 포착하는 능력을 고려해 

장기 예측(LTSF) 문제에 집중했으나 몇가지 **한계 직면**

- self-attention에서 오는 **시간/메모리 복잡성**이 매우 높다는 문제
- self regression decoder 설계로 인해 발생하는 **오차 축적** 문제

⏩ 해당 문제 해결과 복잡성을 줄이는 **DMS 전략을 사용하는 새로운 Transformer 아키텍처를 제안**

<aside>
💡

**다양한 시계열 특징을 모델에 도입**

![스크린샷 2024-09-21 오전 1.01.09.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.01.09.png)

---

### Time series decomposition 시계열 분해

시계열 분해는 시계열 데이터를 분석할 때 자주 사용하는 방법

시계열 데이터를 더 잘 예측하기 위해 데이터를 추세 Trend 와 계절성 Seasonality 으로 나눔

- **`Autoformer`**라는 모델이 신경망 안에서 이 분해 과정을 수행하여 데이터를 처리
- **`FEDformer`**
    - **`Autoformer`**에서 발전된 모델
    - 이동 평균(Moving Average)을 이용해 다양한 크기의 커널을 통해 추세를 더 잘 파악

### Input embedding strategies 입력 임베딩

- 시간적인 ****정보를 임베딩하여 입력 시퀀스에 추가를 위한 **Positional Encoding** 방법 사용
- `Pyraformer`, `Autoformer` → 이러한 임베딩 방법을 활용해 시간적 맥락을 잘 반영하려고 노력
    
    ⇒ (문제) 
    
    그래도,, 여전히 **시간 정보 손실**이 발생
    
    Transformer를 시계열 데이터에 적용하는 것이 실제로 효과적인지 **재검토**할 필요
    

### Self-attention schemes 자기 어텐션 방식

시계열 데이터는 시퀀스의 길이가 길수록 계산 복잡도가 커진다는 문제를 해결하기 위해 도입

`Sparsity Bias`

- 필요하지 않은 정보는 계산에서 제외하는 방식
    - LogTrans는 Logsparse 마스크를 사용해 계산 복잡도를 줄임
    - Pyraformer는 Pyramidal Attention을 사용해 시계열 데이터의 다양한 스케일에서의 시간적 패턴을 효율적으로 처리

`저차원 특성 사용` 

- Informer와 FEDformer는 자기-어텐션 행렬의 저차원 특성을 사용하여 계산을 간소화

### Decoder

이전에 예측한 결과를 바탕으로 다음 값을 예측하는 방식은 **오차 축적** 문제가 발생하기에 이를 해결하기 위해 다음과 같은 방안 도입 

- `Informer` ****
    - Generative Decoder를 사용
    - 한 번에 여러 값을 예측해 오차가 누적되는 문제를 줄임
- `Autoformer` ****
    - 추세와 계절적 구성 요소를 이용해 최종 예측
- `FEDformer`
    - 주파수 어텐션 블록을 사용해 예측
</aside>

### Simple Baseline

**LTSF-Linear**

- 기존의 Transformer 기반 모델들은 **주로 DMS방식을 사용**하여 성능이 좋았다고 가정
- 기본적으로 과거 데이터를 사용해 가중치를 곱한 합을 계산하여 미래 값을 예측하는 모델
- 시간 축을 따라 선형 계층(일종의 수학적 공식)을 적용해서 과거의 데이터를 바탕으로 미래 값을 예측

![스크린샷 2024-09-21 오전 1.01.27.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.01.27.png)

⇒ LTSF-Linear는 **다른 변수들 간의 상관관계**를 고려하지 않고, 시간에 따라서만 예측

⏬  LTSF-Linear를 개선한 모델

**DLinear**

- 시계열 데이터를 Trend와 Seasonality으로 나누는 방법을 사용하는 모델
- 각 부분에 **간단한 선형 모델**을 적용해서 각각의 변동을 따로 예측한 후, 두 결과를 합쳐서 최종 예측값 생성
    
    → 데이터에 명확한 추세가 있을 때 성능을 더 높임
    

**NLinear**

- 데이터의 분포가 변할 때(즉, 시간이 지나면서 데이터 패턴이 달라질 때) 성능을 개선하려는 모델
- 데이터를 먼저 정규화해 데이터의 변화나 분포가 달라질 때 더 안정적인 예측을 할 수 있도록 도움

## 3.  Experiments

---

> **Experimental Settings**
> 
> 
> ---
> 
> `Dataset`
> 
> - multivariate time series
> - ETT (ElectricityTransformer Temperature)(ETTh1, ETTh2, ETTm1,ETTm2),
>     
>     Traffic, Electricity, Weather, ILI, ExchangeRate 
>     
>     ![스크린샷 2024-09-21 오전 1.01.51.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.01.51.png)
>     
> 
> `Evaluation metric`
> 
> - MSE
> - MAE
> 
> `Compared methods`
> 
> - FEDformer, Autoformer, Informer, Pyraformer, LogTrans, Repeat
>     
>     ![스크린샷 2024-09-22 오전 1.44.27.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-22_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.44.27.png)
>     

### Comparison with Transformers

> Quantitative results
> 

![스크린샷 2024-09-21 오전 1.02.04.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.02.04.png)

- **LTSF-Linear**가 **대부분의 경우에서 Transformer 기반 모델들보다 20%~50% 더 나은 성능**
    - Transformer 기반 모델들이 복잡한 구조임에도 불구하고, 시간적 종속성을 제대로 모델링하지 못할 가능성이 있음
    - Transformer와 같은 복잡한 모델들이 항상 최선은 아니며, 단순한 선형 모델이 더 나은 선택

- 다변량 예측에서 LTSF-Linear는 여러 변수들 간의 상관관계를 모델링하지 않음에도 불구하고 좋은 성능을 보임

- 매우 단순한 Repeat 방법은 대부분의 Transformer 모델들보다 환율 예측에서 더 나은 성능을 보임
- Repeat 방법은 이런 오버피팅 문제가 없기 때문

> Qualitative results
> 

![GPT.ver 
**(a) 전력 데이터(Electricity)**: 빨간 선이 실제 데이터이고, 나머지 선들은 각 모델이 예측한 결과를 나타냅니다. 모델들이 실제 데이터와 얼마나 가까운지를 비교할 수 있습니다. 
**(b) 환율 데이터(Exchange-Rate)**: 환율 예측 결과를 보여주며, 특히 DLinear와 FEDformer가 다른 모델에 비해 더 나은 성능을 보이는 반면, Informer는 예측이 크게 벗어나는 것을 확인할 수 있습니다.
**(c) ETTh2 데이터**: 역시 모델들의 예측이 실제 데이터(GroundTruth)와 어떻게 다른지 비교할 수 있습니다. DLinear와 Autoformer가 상대적으로 안정적인 예측을 보여주는 반면, FEDformer는 주파수 처리로 인해 데이터의 주기성을 잘 포착하고 있습니다.](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.04.15.png)

GPT.ver 
**(a) 전력 데이터(Electricity)**: 빨간 선이 실제 데이터이고, 나머지 선들은 각 모델이 예측한 결과를 나타냅니다. 모델들이 실제 데이터와 얼마나 가까운지를 비교할 수 있습니다. 
**(b) 환율 데이터(Exchange-Rate)**: 환율 예측 결과를 보여주며, 특히 DLinear와 FEDformer가 다른 모델에 비해 더 나은 성능을 보이는 반면, Informer는 예측이 크게 벗어나는 것을 확인할 수 있습니다.
**(c) ETTh2 데이터**: 역시 모델들의 예측이 실제 데이터(GroundTruth)와 어떻게 다른지 비교할 수 있습니다. DLinear와 Autoformer가 상대적으로 안정적인 예측을 보여주는 반면, FEDformer는 주파수 처리로 인해 데이터의 주기성을 잘 포착하고 있습니다.

- 각 데이터셋은 서로 다른 시간적 패턴을 가지고 있고 어떤 데이터는 일정한 주기성, 어떤 데이터는 주기성이 명확하지 않거나 불규칙
- Transformer 모델들은 **전력 데이터**와 **ETTh2 데이터**에서 **미래 데이터의 규모**나 **편향**을 제대로 예측하지 못함
    
    → LTSF-Linear는 추세나 시간적 흐름을 더 잘 포착
    

### More Analyses on LTSF-Transformers

> Can existing LTSF-Transformers extract temporal relations well from longer input sequences?
> 

**Look-Back Window**

- 회귀 창으로 과거 데이터를 얼마나 길게 가져와서 미래를 예측하는지와 관련된 부분
    - 과거 데이터를 96단계까지 보느냐, 720단계까지 보느냐에 따라 모델이 학습할 수 있는 정보의 양이 달라짐
- 강력한 시계열 예측 모델은 긴 회귀 창을 통해 더 많은 과거 데이터를 참고할 수 있고, 이를 바탕으로 더 정확한 예측

![스크린샷 2024-09-21 오전 1.04.31.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.04.31.png)

- 입력 회귀 창 크기를 **{24, 48, 72, 96, 120, 144, 168, 192, 336, 504, 672, 720}** 단계로 설정해 실험을 진행

- Transformer 기반 모델들은 회귀 창 크기가 커져도 성능이 크게 나아지지 않거나 오히려 나빠짐
- Transformer 모델들은 입력 데이터가 길어지면 성능이 크게 향상되지 않으며, 오히려 시간적 잡음에 맞추어 예측을 잘못할 가능성이 높음
- LTSF-Linear는 회귀 창이 길어질수록 더 많은 정보를 활용하여 더 정확한 예측

> What can be learned for long-term forecasting?
> 

단기 예측에서는 과거 데이터를 얼마나 잘 활용하는지가 중요

장기 예측에서는 추세와 주기성을 잘 포착하는 게 더 중요

![스크린샷 2024-09-21 오전 1.04.43.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.04.43.png)

두 가지 종류의 과거 데이터를 사용해서 예측 실험

- **Close**: 더 최근(96번째~191번째 시간 데이터) 데이터를 사용.
- **Far**: 더 이전(0번째~95번째 시간 데이터) 데이터를 사용.

- 최근 데이터를 사용하든 더 이전 데이터를 사용하든 큰 차이가 없음
- Transformer 모델들이 비슷한 시간적 패턴만 포착
- 매개변수를 너무 많이 사용하면 오히려 과적합이 발생할 수 있고, 이것이 Transformer보다 **LTSF-Linear** 모델이 더 좋은 성능을 보임

> Are the self-attention scheme effective for LTSF?
> 

![스크린샷 2024-09-21 오전 1.04.53.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.04.53.png)

- self-attention 레이어를 **선형 레이어**로 바꾸고, FFN 같은 보조 모듈을 하나씩 제거해 진행

⏬

- 결과적으로 모델이 더 단순해질수록 오히려 성능이 좋음
- Transformer의 복잡한 설계가 실제로는 장기 시계열 예측에 필요하지 않음

> Can existing LTSF-Transformers preserve temporal order well?
> 
- Transformer는 그 순서를 잘 고려하지 않음 → inherently permutation invariant

![스크린샷 2024-09-21 오전 1.05.08.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.05.08.png)

- 입력 데이터를 일부 섞어서 실험
    - **Shuf.**: 전체 데이터를 완전히 무작위로 섞기.
    - **Half-Ex.**: 입력 데이터의 앞부분과 뒷부분을 서로 교환하기.

- Transformer 기반 모델들은 입력이 섞였을 때에도 성능이 거의 변하지 않음.
    
    →  Transformer가 시간적 순서를 제대로 활용하지 않음
    
- LTSF-Linear 모델은 순서가 섞일 경우 성능이 크게 떨어짐
    
    → LTSF-Linear가 시간적 순서를 더 잘 활용하고 있음을 보여줌
    

⏬ 특정 데이터셋에서 Transformer 성능 차이

- ETTh1 데이터셋에서는 FEDformer와 Autoformer가 상대적으로 성능 감소가 적음
    - 모델들이 주기적인 패턴을 더 잘 포착할 수 있었기 때문
    - 시계열 편향을 도입해 주기성 같은 시간적 정보를 더 잘 다루게 설계됨
- Exchange  데이터셋처럼 명확한 주기성이 없는 데이터에서는 Transformer들이 시간 정보를 잃어버리고 성능이 많이 떨어짐

> How effective are different embedding strategies?
> 

Positional Embedding, Timestamp Embedding이 시계열 예측에서 얼마나 중요한지를 평가

![스크린샷 2024-09-21 오전 1.05.21.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.05.21.png)

- **Informer**
    - 위치 임베딩을 제거하면 성능이 크게 떨어짐
    - 특히 타임스탬프 임베딩이 없으면 예측 길이가 길어질수록 성능이 점점 더 악화
    
    ⇒  Informer가 각 토큰에서 **단일 시간 단위**를 사용하기 때문에 시간정보를 임베딩으로 넣어주는 것이 필수적
    
- **FEDformer,** **Autoformer**
    - 타임스탬프 정보를 여러 개의 시퀀스로 입력에 포함하기 때문에 고정된 위치 임베딩이 없어도 괜찮은 성능을 유지
    - 타임스탬프 임베딩을 제거하면 Autoformer의 성능이 급격히 감소.

> Is training data size a limiting factor for existing LTSFTransformers?
> 

![스크린샷 2024-09-21 오전 1.05.33.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_1.05.33.png)

- **적은 양의 데이터로 훈련**시킨 모델이 더 낮은 예측 오류를 보임
- 짧지만 완전한 데이터가 긴 데이터셋보다 더 명확한 시간적 패턴을 포함하고 있기 때문
    
    **⇒** 데이터 크기가 Transformer 성능의 제한 요인이 아님을 보여줌
    

> Is efficiency really a top-level priority?
> 

![스크린샷 2024-09-21 오전 1.05.47.png](Time%20Series%20Transformer%20%E2%AD%90%205f190f4a95d64fa2ab69a2183259e7c3/5eaf7e0f-116a-4c4e-8b1b-2408734efb50.png)

- 실제로는 더 많은 설계 요소가 추가되면서 **실제 추론 시간**과 **메모리 비용**이 오히려 증가
- **DLinear**는 매우 낮은 메모리 비용과 빠른 추론 시간을 가지지만, Transformer 모델들은 더 많은 메모리와 계산 자원을 소모
- **Autoformer**와 **FEDformer**는 메모리와 추론 시간 측면에서 매우 비효율

## 4. Conclusion

---

- Transformer 기반 솔루션들이 장기 시계열 예측 문제에서 얼마나 효과적인지에 대해 의문을 제기
    
    → 기존 연구에서 주장한 만큼 효과적이지 않다는 점을 입증
    
- 매우 단순한 선형 모델을 DMS 예측의 기준 모델로 사용

## 🎹 코드 리뷰

---

## ➕ 참고

---

[Time Series Transformer](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)

[https://github.com/cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
