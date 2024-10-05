# BERT ⭐

---

![image.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image.png)

<aside>
💡 **BERT** Bidirectional Encoder Representations from Transformers

---

Transformer의 attention 기법을 이용한 embedding 모델로 트랜스포머 Encoder의 multi-head attention mechanism을 사용

- 2018년에 Google AI Language 팀에서 발표한 자연어 처리(NLP) 모델
</aside>

- 용어
    
    👐🏻 **서브워드 유닛** subword unit
    
    - 단어보다 더 작은 단위인 접두사, 접미사, 또는 심지어 글자(character)를 의미
    
    ![image.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%201.png)
    

---

## 0. Abstract

---

> **Pre-training of Deep Bidirectional Transformers for Language Understanding**
> 
> - 언어 이해를 위해 깊은 양방향 트랜스포머의 사전학습
> - [https://arxiv.org/pdf/1810.04805](https://arxiv.org/pdf/1810.04805)

BERT는 모든 레이어에서 왼쪽과 오른쪽 문맥을 동시에 고려하여 비지도 학습으로 심층적인 양방향 표현을 사전 학습하도록 설계

→ **Bidirectional**

그 결과, 사전 학습된 BERT 모델은 단지 하나의 추가적인 출력 레이어만으로도 질문 응답, 언어 추론 등 다양한 작업에서 최첨단 모델을 만들 수 있으며, 작업에 특화된 아키텍처 수정 없이도 가능

→ **Pre-training & Fine-tuning**

## 1. Introduction

---

- **모델이 토큰 수준에서 세밀한 출력을 생성**
- 사전 학습된 언어 표현을 다운스트림 작업에 적용하는 기존의 두 가지 전략 존재
    - **피처 기반(feature-based) 접근법**
        - 사전 학습된 표현을 추가적인 피처로 표현하는 작업별 아키텍쳐 사용
            - ex. ELMo
    - **미세 조정(fine-tuning) 접근법**
        - 작업별 파라미터를 최소화하고 모든 사전 학습된 파라미터를 단순히 미세 조정하여 다운스트림 작업에서 학습
            - ex. OpenAI GPT
        
        ![OpenAI GPT는 단방향(좌에서 우)으로 학습, ELMo는 좌-우 독립 LSTM을 결합하여 문맥을 표현](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-16_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_11.14.33.png)
        
        OpenAI GPT는 단방향(좌에서 우)으로 학습, ELMo는 좌-우 독립 LSTM을 결합하여 문맥을 표현
        
    
    ⇒ 사전 학습 동안 동일한 손실함수를 공유하며, 일반적인 언어 표현을 학습하기 위해 단방향 언어 모델을 사용
    

💡 **문제 상황** → 새로운 모델 구조가 필요한 이유

- **단방향 언어 모델의 한계**
    
    텍스트의 앞부분(좌측)에서 뒤쪽(우측)을 예측하는 형태로 이루어지며, 이로 인해 모델은 각 토큰을 예측할 때 해당 토큰의 이전에 있는 토큰들에만 접근
    
    - **OpenAI GPT**: 좌에서 우로 텍스트를 처리하는 단방향 모델
    - GPT는 트랜스포머 아키텍처를 사용하지만, 각 토큰이 이전 토큰들에만 주의를 기울일 수 있으며 모델이 입력 문장의 앞부분에서 정보를 가져와 뒷부분을 예측하도록 설계
- **아키텍처 선택의 제한**
    
    이 단방향 접근 방식은 사전 학습된 언어 모델의 아키텍처에 제한을 가함 
    
    - 트랜스포머 셀프 어텐션 메커니즘이 주어진 문맥의 양방향(좌측과 우측)을 모두 활용하지 못하게 되어 문장 전체의 의미를 더 정확하게 파악하는 데 있어서 제한이 될 수 있음
- 이전의 Embedding 방식에서는 **다의어나 동음이의어를 구분하지 못하는 문제점**

⏩ **해결방안**

### 양방향 문맥의 필요성

양방향 모델은 각 단어를 예측할 때 그 단어의 앞쪽과 뒤쪽 문맥을 모두 고려

- 단어의 앞뒤 문맥을 모두 활용하는 사전 학습을 수행하며 토큰 수준의 작업에서 더 높은 성능을 발휘

<aside>
💡 **BERT** Bidirectional Encoder Representations from Transformers **특징**

- **양방향성 (Bidirectional)**
    - BERT는 입력 텍스트의 모든 단어가 양방향으로 처리
    - 각 단어의 의미를 결정할 때 그 단어 앞뒤의 문맥을 모두 고려해 문장 전체 맥락을 이해하는 데에 도움

---

- **트랜스포머(Transformer) 기반**
    - BERT는 트랜스포머 모델의 인코더(Encoder) 부분을 사용
    - 트랜스포머는 병렬 처리가 가능하여 큰 데이터셋에서 효과적으로 학습할 수 있으며, 멀티헤드 어텐션(Multi-Head Attention) 메커니즘을 통해 문장의 모든 단어 간의 관계를 효율적으로 학습

---

- **사전 학습 & 미세 조정**
    - **사전 학습 (Pre-training)**
        - Masked Language Modeling (MLM)
            - 입력 텍스트의 일부 단어를 [MASK] 토큰으로 대체하고, 모델이 이 단어들을 예측하도록 학습 → 좌우 문맥을 융합할 수 있게 하여 심층 양방향 트랜스포머를 사전 학습
            - Masked Language Model을 통해 양방향성을 얻음
        - Next Sentence Prediction (NSP)
            - 두 문장이 주어졌을 때, 두 번째 문장이 첫 번째 문장 다음에 오는 문장인지 여부를 예측
    - **미세 조정 (Fine-tuning)**
        - 사전 학습된 BERT 모델을 특정 Task 에 맞게 추가 학습되며, 전체 모델이 해당 작업에 맞게 조정
</aside>

**Related Work**

일반적인 언어 표현의 사전 학습에는 오랜 역사가 있으며, 이 섹션에서는 가장 널리 사용되는 접근 방식 검토

**2.1 비지도 학습 기반의 피처 기반 접근 방식**

단어와 문장 임베딩(embedding) 기술의 발전

⇒ 사전 학습 방법 → 문장 및 단락 임베딩 → ELMo 모델 

**2.2 비지도 학습 기반의 미세 조정 접근 방식**

단어 임베딩 파라미터만 사전 학습 → 문장/문서 인코더

**2.3 지도된 데이터에서의 전이 학습**

컴퓨터 비전 연구에서도 대규모 사전 학습된 모델에서의 전이 학습의 중요성이 입증

→ ImageNet(Deng et al., 2009; Yosinski et al., 2014) 사전 학습된 모델을 미세 조정

## 2. Model Architecture

---

### 모델 아키텍처

BERT의 모델 아키텍처는 다층 양방향 트랜스포머 인코더

![image.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%202.png)

|  | BERT Base | BERT Large |
| --- | --- | --- |
| L (layer 수) | 12 | 24 |
| H (hidden size) | 768 | 1024 |
| A (self-attention head 수) | 12 | 16 |
| Total Parameters | 110M | 340M |
- BERT와 GPT의 차이점
    - BERT Base는 GPT와 동일한 크기로 설계되어 있음 (L, H, A, Total Parameters 모두 동일)
    - 차이는 **BERT**는 양방향 self-attention을 사용하여 문맥의 양쪽 모두 참조
    - **GPT**는 제약된 self-attention을 사용하여 각 토큰이 자신의 왼쪽에 있는 토큰만 참조

![image.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%203.png)

**Encoder layer** 

1. Multi-head Self-Attention
2. Add & Norm
3. FFNN
4. Add & Norm

### **Frame work**

크게 두 단계로 구성: 사전 학습(Pre-training)과 미세 조정(Fine-tuning)

- `Pre-training` 동안, 모델은 다양한 사전 학습 작업에서 레이블이 없는 데이터를 사용해 학습
- `Fine-tuning` 단계에서는, BERT 모델이 먼저 사전 학습된 파라미터로 초기화되고, 이후 모든 파라미터가 다운스트림 작업에서 레이블이 있는 데이터를 사용하여 파인튜닝
    - 초기에는 동일한 사전 학습된 파라미터를 사용하지만, 미세 조정 과정에서 작업별로 파라미터가 달라지기 때문에, 결국 각 작업에 최적화된 별도의 모델
    - ex. 감정분석에 특화된 BERT, 질문응답에 특화된 BERT

![image.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%204.png)

![image.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%205.png)

### 2.1 Input Representation

---

> **input_embedding**
> 
> 
> **= word_embedding + positional_embedding + segment_embedding** 
> 

![스크린샷 2024-08-16 오후 11.12.02.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-16_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_11.12.02.png)

![image.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%206.png)

![1_twO-mzNHhwYCBIJsOSueBw.gif](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/1_twO-mzNHhwYCBIJsOSueBw.gif)

**Token Embedding**

일반적인 Token Embedding 입니다. Word Piece 임베딩 방식을 사용

길이가 긴 문장이나, 빈출도가 낮은 문장은 분절하여 sub-word 처리

**Position Embedding**

Position Encoding 방식은 위치 정보를 포함하는데에 탁월하지만, 다른 토큰들의 위치 정보들과 반응하지 못함

**Segment Embedding**

위의 사진에서는 my dog is cute 와 he likes play ing 이 두 문장이 한 쌍이 되어 입력으로 제공됩니다.

이를 구분할 수 있는 벡터로서, Segment Embedding 벡터를 생성한 후 적용합니다.

**➕ Attention Mask**

![image.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%207.png)

BERT가 어텐션 연산을 할 때, 불필요하게 패딩 토큰에 대해서 어텐션을 하지 않도록 실제 단어와 패딩 토큰을 구분할 수 있도록 알려주는 입력

### 2.2 Pre-training

---

BERT는 `마스크드 언어 모델(MLM)`을 통해 문장 내에서 양방향 문맥을 학습하고, 다음 `문장 예측(NSP)`을 통해 문장 간의 관계를 학습

1️⃣ **마스크드 LM (Masked LM)**

BERT는 일부 입력 토큰을 무작위로 마스킹하고, 마스킹된 토큰을 예측하는 작업을 통해 심층 양방향 표현을 학습

- 80%의 단어들은 [MASK]로 변경한다.
    - ex. my dog is hairy → my dog is [MASK]
- 10%의 단어들은 랜덤으로 단어가 변경된다.
    - ex. my dog is hairy → my dog is apple
- 10%의 단어들은 동일하게 둔다.
    - ex. my dog is hairy → my dog is hairy

⇒ [MASK]만 사용할 경우에는 [MASK] 토큰이 파인 튜닝 단계에서는 나타나지 않으므로 사전 학습 단계와 파인 튜닝 단계에서의 불일치가 발생

2️⃣ **다음 문장 예측 (Next Sentence Prediction, NSP)**

두 개의 문장을 준 후에 이 문장이 이어지는 문장인지 아닌지를 맞추는 방식으로 훈련하며 50:50 비율로 실제 이어지는 두 개의 문장과 랜덤으로 이어붙인 두 개의 문장을 주고 훈련

![image.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%208.png)

### 2.3 BERT Fine-tuing

---

트랜스포머의 셀프 어텐션 메커니즘을 통해 BERT는 단일 텍스트나 텍스트 쌍이 포함된 많은 다운스트림 작업을 모델링

![**Single Text Classification**](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%209.png)

**Single Text Classification**

![Tagging](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%2010.png)

Tagging

![**Text Pair Classification or Regression**](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%2011.png)

**Text Pair Classification or Regression**

![**Question Answering**](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/image%2012.png)

**Question Answering**

## 3.  Experiments

---

BERT의 11개 NLP 작업에 대한 미세 조정 결과를 제시

![스크린샷 2024-08-16 오후 11.14.56.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-16_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_11.14.56.png)

### 3.1 GLUE

![스크린샷 2024-08-16 오후 11.12.20.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-16_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_11.12.20.png)

**General Language Understanding Evaluation(GLUE) 벤치마크(Wang et al., 2018a)** 

: 다양한 자연어 이해 작업으로 구성된 컬렉션

- **미세 조정 과정**
    - 입력 시퀀스(문장 또는 문장 쌍)를 BERT에 입력하고, 첫 번째 토큰인 `[CLS]`의 최종 히든 벡터를 집계된 표현으로 사용
    - 미세 조정 중에는 분류 레이어의 가중치만 새로 도입되며, 이 가중치를 사용해 표준 분류 손실을 계산
    - 모든 GLUE 작업에 대해 batch_size=32, epoch=3 로 미세 조정을 수행
- **결과**
    - BERTBASE와 BERTLARGE는 모든 GLUE 작업에서 이전 최첨단 성능을 능가하며, 특히 BERTLARGE는 소규모 데이터셋에서 더 큰 성능 향상을 보여줌
    - BERTLARGE는 GLUE 리더보드에서 80.5점을 획득하며, OpenAI GPT의 72.8점 돌파
    

⏩ BERT의 미세 조정 과정과 GLUE 벤치마크에서의 뛰어난 성능을 강조

### 3.2 SQuAD v1.1 Stanford Question Answering Dataset

![스크린샷 2024-08-16 오후 11.12.39.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-16_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_11.12.39.png)

**SQuAD v1.1** Stanford Question Answering Dataset

- 10만 개의 크라우드소싱된 질문/답변 쌍으로 이루어진 데이터셋
- 주어진 단락에서 정답 텍스트의 범위를 예측하는 작업을 수행
- 질문 응답(QA) Task

- **모델 아키텍처 및 미세 조정**
    - 입력 질문과 단락을 단일 시퀀스로 결합하여 처리하며, 질문에는 A 임베딩, 단락에는 B 임베딩을 사용
    - 미세 조정 시, 답변 범위의 시작과 종료 위치를 예측하기 위한 시작 벡터(S)와 종료 벡터(E)를 도입
    - learning_rate=5e-5, batch_size=32로 epoch=3 동안 fine-tuing
- **성능 결과**
    - BERT는 리더보드 상위 시스템을 F1 점수 기준으로 앙상블 시스템보다 +1.5, 단일 시스템보다 +1.3 능가하는 성과를 보여줌
    - TriviaQA로 사전 미세 조정한 후 SQuAD에 대해 미세 조정하면 성능이 향상되지만, TriviaQA 데이터를 사용하지 않더라도 여전히 기존 시스템을 큰 차이로 능가

⏩ BERT가 SQuAD v1.1에서 뛰어난 성능을 발휘했음을 강조

### 3.3 SQuAD v2.0 Stanford Question Answering Dataset

![스크린샷 2024-08-16 오후 11.12.51.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/2412cefe-006e-43f0-a461-348fb56c3a4c.png)

**SQuAD v2.0** Stanford Question Answering Dataset

- SQuAD 2.0은 주어진 단락에 답변이 없는 경우도 포함

- **모델 아키텍처 및 미세 조정**
    - BERT 모델을 확장하여 답변이 없는 질문에 대해 [CLS] 토큰에서 시작과 종료 위치를 가지는 답변 범위로 처리
    - 답변이 없는 범위의 점수(`s_null = S·C + E·C`)를 가장 좋은 비-null 범위의 점수와 비교하여, 답변을 예측
- **성능 결과**
    - BERT 모델은 이전 최상위 시스템 대비 +5.1 F1 점수 향상을 달성

⏩ BERT가  SQuAD 2.0에서 이전보다 훨씬 더 나은 성능을 보여줌

### 3.4 SWAG Situations With Adversarial Generations

![스크린샷 2024-08-16 오후 11.13.04.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-16_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_11.13.04.png)

 **SWAG** Situations With Adversarial Generations

- 113,000개의 문장 쌍 완성 예제로 구성
- 주어진 문장에 대해 가장 그럴듯한 연속 문장을 선택하는 상식적인 추론 작업을 평가

- **모델 아키텍처 및 미세 조정**
    - BERT 모델은 문장 A와 네 가지 가능한 연속 문장 B를 입력으로 받아, [CLS] 토큰의 표현과의 내적을 통해 각 선택지의 점수를 계산한 후 소프트맥스를 사용해 정규화
    - BERTLARGE 모델은 학습률 2e-5, 배치 크기 16으로 3 에포크 동안 미세 조정
- **성능 결과**
    - ESIM+ELMo 시스템을 27.1%, OpenAI GPT를 8.3% 능가

### 부록.1 사전 학습 작업의 효과

BERT의 깊이 있는 양방향성의 중요성을 두 가지 사전 학습 목표를 평가하여 보여줌

![스크린샷 2024-08-16 오후 11.13.17.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/5adec00d-d0bd-452b-b637-64af31ad4683.png)

- **No NSP**
    - "Masked LM"(MLM)만을 사용하여 훈련된 양방향 모델
    - "다음 문장 예측"(NSP) 작업을 포함하지 않음
- **LTR & No NSP**
    - 좌에서 우(LTR) 언어 모델을 사용하여 훈련된 좌측 문맥 전용 모델로, MLM 대신 사용
    - NSP 작업 없이 사전 학습

**결과**

- NSP 작업을 제거하면 QNLI, MNLI, SQuAD 1.1 작업에서 성능이 크게 저하
- 양방향 모델과 좌측-우측(LTR) 모델을 비교한 결과, LTR 모델이 모든 작업에서 성능이 떨어지며, 특히 MRPC와 SQuAD에서 큰 성능 저하
- **BiLSTM 추가의 한계**
    - LTR 모델을 강화하기 위해 BiLSTM을 추가했으나, 이는 SQuAD에서만 성능을 향상시키고, 다른 작업에서는 여전히 양방향 모델보다 성능낮음

### 부록.2 모델 크기의 효과

![스크린샷 2024-08-16 오후 11.13.33.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-16_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_11.13.33.png)

- **모델 크기와 정확도**
    - L, H, A 크기를 늘리면 모든 데이터셋에서 정확도가 향상
- **모델 크기 비교**
    - BERTBASE와 BERTLARGE 모델은 기존 트랜스포머 모델보다 훨씬 큰 파라미터 수를 가지고 있으며, 이로 인해 더 높은 성능을 발휘.
    - BERTBASE는 110M 파라미터, BERTLARGE는 340M 파라미터를 갖고 있음
- **작은 규모의 작업에서도 성능 향상**
    - 매우 큰 모델 크기로 확장할 때 작은 규모의 작업에서도 큰 성능 향상을 달성할 수 있음

⏩ BERT 모델의 크기를 늘리면 다양한 자연어 처리 작업에서 성능이 향상되며, 이는 작은 데이터셋에서도 적용될 수 있음

### 부록.3 BERT를 활용한 피처 기반 접근 방식

![스크린샷 2024-08-16 오후 11.13.44.png](BERT%20%E2%AD%90%20d7fb38591d9a417186f17562adc007f3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-16_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_11.13.44.png)

- **미세 조정 접근 방식**
    - BERT 모델에 간단한 분류 레이어를 추가하고 모든 파라미터를 다운스트림 작업에서 공동으로 미세 조정합니다. 이 방식은 모든 작업에 대해 동일한 아키텍처를 사용할 수 있지만, 각 작업별로 모델을 미세 조정해야 합니다.
- **피처 기반 접근 방식**
    - 사전 학습된 BERT 모델에서 고정된 피처를 추출하여 이를 다른 작업에 활용합니다. 이 방식의 장점은 특정 작업에 대해 트랜스포머 인코더 아키텍처가 적합하지 않을 때 작업별 모델 아키텍처를 추가할 수 있고, 훈련 데이터의 표현을 사전 계산하여 계산 비용을 절감할 수 있다는 점입니다.
- **CoNLL-2003 개체명 인식(NER) 작업 적용**
    - 피처 기반 접근 방식에서는 BERT의 파라미터를 미세 조정하지 않음
    - 사전 학습된 트랜스포머의 상위 레이어에서 활성화를 추출하여 BiLSTM 모델에 입력으로 사용
- **결과**
    - BERTLARGE 모델은 최첨단 방법과 유사한 성능을 보였으며,  피처 기반 접근 방식이 미세 조정 접근 방식에 비해 약간 낮은 성능을 보였지만 여전히 효과적임

⏩ BERT는 미세 조정과 피처 기반 접근 방식 모두에서 강력한 성능을 발휘

## 4. Conclusion

---

- Pre-training 중요
    
    → 리소스가 부족한 작업에서도 심층 단방향 아키텍처의 이점을 누릴 수 있음 
    
- 심층 양방향 아키텍처
    
    → 동일한 사전 훈련된 모델이 광범위한 NLP 작업을 성공적으로 처리
    

## 🔎 궁금증 ..

---

⏬ BERT 파생 모델

- **ALBERT** : A Lite BERT for Self-supervised Learning of Language Representations
    - BERT의 파라미터 수를 줄여 효율성을 높인 모델
- **RoBERTa** : A Robustly Optimized BERT Pretraining Approach
    - BERT의 학습 방식을 최적화하여 성능을 더욱 향상시킨 모델
- **DistilBERT**
    - BERT의 경량화 버전으로, 더 빠르고 가벼운 모델
- **ELECTRA** : Efficiently Learning an Encoder that Classifies Token Replacements Accurately
    - replaced token detection(교체한 토큰 탐지) task를 통해 사전 학습을 진행한 모델

## 🎹 코드 리뷰

---

[Google Colab](https://colab.research.google.com/drive/1m2eDsR77I1uHNf6nXHo4tGYjX_rolXhk?usp=sharing)

## ➕ 참고

---

[https://github.com/google-research/bert](https://github.com/google-research/bert)

[17-02 버트(Bidirectional Encoder Representations from Transformers, BERT)](https://wikidocs.net/115055)

[BERT Architecture (Transformer Encoder)](https://eatchu.tistory.com/entry/BERT-Architecture-Transformer-Encoder)

[Breakdown The BERT In Pieces…](https://medium.com/subex-ai-labs/breakdown-the-bert-in-pieces-df46f60b65d8)

[Figure 1: The two stages of employing GREEK-BERT: (a) pre-traning BERT...](https://www.researchgate.net/figure/The-two-stages-of-employing-GREEK-BERT-a-pre-traning-BERT-with-the-MLM-and-NSP_fig1_344034047)