# ELMo ⭐

---

![BERT 논문에서 가장 많이 등장하고 관련된 논문인 ELMo](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/image.png)

BERT 논문에서 가장 많이 등장하고 관련된 논문인 ELMo

<aside>
💡 **ELMo** Embeddings from Language Model
****언어 모델로 하는 임베딩으로 사전 훈련된 언어 모델 Pre-trained language model을 사용하며 embedding에 sentence의 전체 context를 담음

---

- 2018년 논문임에도 불구하고 당시 인기를 끌었던 Transformer를 완전히 배제한 모형
- 기존 분산 표상 방식의 Representation들을 (Word2vec, Glove, …) Static Representation으로 정의하고, Language Model을 이용한 Representation을 Contextual Representation으로 정의
</aside>

- 용어
    
    👐🏻 **서브워드 유닛** subword unit
    
    - 단어보다 더 작은 단위인 접두사, 접미사, 또는 심지어 글자(character)를 의미
    
    ![image.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/image%201.png)
    

---

## 0. Abstract

---

> **Deep contextualized word representations**
> 
> - 문맥과 깊게 연관된 단어 표현

- 단어 사용의 복잡한 특성과 언어적 문맥에 따라 어떻게 달라지는지 모델링하는 방법으로 ELMo 도입
- word vector는 큰 말뭉치에서 학습된 deep bidirectional language model(**biLM**)의 내부 상태로부터 학습
    - 모델이 더 복잡한 문맥 정보를 활용할 수 있도록
- NLP 내의 6가지 주요 task에서 SOTA를 달성
    - Question answering = 질의응답
    - Textual entailment = 주어진 전제에서 가설이 참인지 판단하는 문제
    - Semantic role labeling = 의미역 결정
    - Coreference resolution = 상호참조
    - Named entity extraction = 개체명 인식
    - Sentiment analysis = 감성 분석

## 1. Introduction

---

- **사전 훈련된 워드 임베딩 Pre-trained Word Embedding** 은 중요한 요소이나 고품질의 표현을 학습하는 것은 어려움
    
    <aside>
    💡 **고품질의 표현 학습 조건**
    
    1. 단어 사용의 복잡한 특성 모델링 필요    ex. 구문 및 의미론
    2. 언어적 문맥에 따라 어떻게 달라지는지 모델링 필요  ex. 다의어를 모델링하기 위함
    </aside>
    

🔽 이 두가지 학습 조건을 해결한 

### ELMo(Embeddings from Language Models)

![image.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/image%202.png)

- 각 토큰이 전체 입력 문장의 함수로 표현
- 대규모 텍스트 코퍼스에서 결합된 언어 모델(LM) 목표와 함께 훈련된 양방향 LSTM에서 파생된 벡터를 사용

**특징**

- **문맥 기반 임베딩 Contextualized Embeddings**
    - 단어가 사용된 문맥에 따라 임베딩 다르게 생성
    - 다의어를 고려하지 않고 같은 단어는 항상 같은 벡터로 표현하는 문제를 해결
        - ex. bank라는 단어가 문장에서 금융기관이든, 강둑이든 동일한 벡터 표현 X
- **BiLSTM 구조**
    - 주어진 문장의 왼쪽에서 오른쪽으로, 오른쪽에서 왼쪽으로 문맥을 이해할 수 있어 단어의 문맥을 더 깊이 이해
    - 양방향 LSTM(BiLSTM) 네트워크를 사용하며 내부 레이어의 함수
        - BiLSTM이 여러 층(layer)으로 구성되어 있고, 이 층들이 각각 다른 문맥 정보를 학습해 최상위 LSTM 레이어만 사용하는 것보다 효과적
        - higer-level LSTM : 단어의 문맥에 따른 의미 변화를 포착하는 데 유용
        - lower-level LSTM : 문법적 특성을 모델링하는 데 유용
- **사전 학습된 언어 모델 Pre-trained Language Model**
    - 대규모 텍스트 데이터에서 미리 학습된 언어 모델
    - 사전학습 모델은 특정 작업에 맞게 fine-tuning 가능
- **문장 전체를 고려한 임베딩**
    - ELMo는 문장의 모든 단어를 고려하여 각 단어의 임베딩을 생성
    - 단어 사이의 상호 작용과 문맥을 더 잘 반영

- 문맥의존 표현을 학습하는 다른 연구
    - 양방향 LSTM을 사용하는 context2vec(Melamud et al., 2016)
    - 표현 안에 pivot word 자체를 포함하는 CoVe(McCann et al., 2017)
- 실험 요약
    
    광범위한 실험을 통해 ELMo 표현이 실제로 매우 잘 작동함을 입증
    
    1. ELMo 표현을 텍스트 간 추론, 질의응답 및 감정 분석을 포함한 여섯 가지 다양한 어려운 언어 이해 문제에 대한 기존 모델에 쉽게 추가할 수 있음
    2. ELMo 표현의 추가만으로도 모든 경우에서 최첨단 성능이 크게 향상되며, 최대 20%에 이르는 상대적 오류 감소
    3. ELMo가 신경 기계 번역 인코더를 사용해 문맥화된 표현을 계산하는 CoVe(McCann et al., 2017)를 능가
    4. ELMo와 CoVe의 분석은 심층 표현이 LSTM의 최상위 레이어에서 파생된 표현보다 뛰어남

## 2. Model Architecture

---

### ELMo의 구조

![68747470733a2f2f63646e2e616e616c79746963737669646879612e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031392f30332f6f75747075745f59794a6338452e676966.gif](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/68747470733a2f2f63646e2e616e616c79746963737669646879612e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031392f30332f6f75747075745f59794a6338452e676966.gif)

문장 전체를 고려한 단어 임베딩을 생성

- **문자 단위의 컨볼루션 Character-level CNN**
    - 단어를 문자 수준에서 표현하고, 이를 CNN을 사용해 임베딩으로 변환
    - 서브워드 유닛의 이점을 얻어 ****그 단어를 구성하는 문자의 패턴을 분석하여 단어의 구조적 특징을 학습
- **BiLSTM layers**
    - 두 개의 LSTM 레이어가 있으며, 하나는 앞에서 뒤로, 다른 하나는 뒤에서 앞으로 문장을 처리
- **지도학습 NLP 작업**
    - 사전 학습된 BiLSTM과 목표 NLP 작업을 위한 지도 학습 모델이 주어지면, BiLSTM을 사용해 작업 모델의 성능을 향상
    - **Task-specific weighting** 각 LSTM 레이어의 출력을 가중 평균하여 최종 임베딩을 만들며 가중치는 특정 NLP 작업에 따라 조정

### 양방향 언어모델 구조

- 토큰 임베딩 또는 문자를 **Character-level CNN**을 통해 문맥 독립적인 토큰 표현을 계산
- N개의 토큰$(t_1, t_2, ..., t_N)$으로 이루어진 시퀀스가 주어졌을 때, 순방향 언어 모델은 과거$(t_1, ..., t_{k−1})$를 기반으로 $t_k$ 토큰의 확률을 모델링하여 시퀀스의 확률을 계산
    
    $p(t_1,t_2,...,t_N)=∏_{k=1}^N p(t_k∣t_1,t_2,...,t_{k−1})$
    
- 최상위 LSTM 계층의 출력은 Softmax 계층을 사용하여 다음 토큰 tk+1을 예측하는 데 사용

![image.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/image%203.png)

![image.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/image%204.png)

⏩ `process` 

- 각 층의 출력값을 연결 → 각 층의 출력값별로 가중치 부여 → 각 층의 출력값 더함 → 벡터의 크기를 결정하는 스칼라 매개변수의 곱
    - 2) 가중치 설정 방법은 task에 따라 달라짐
        - 문법 정보를 모델링하는 경우 →  입력과 가까운 층의 가중치를 키움
        - 문맥 정보를 모델링하는 경우 → 출력과 가까운 층의 가중치를 키움
    - 2)번과 3)번의 단계를 요약하여 가중합(Weighted Sum)
    - 4) 벡터의 크기를 결정하는 스칼라 매개변수 $𝛾$ 를 곱함
        - $𝛾$  파라미터는 과제 모델이 전체 ELMo 벡터의 크기를 조정할 수 있게 함
        
        ![image.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/cf39d6c9-e046-448b-8628-6d99b70855fb.png)
        
        - 각 토큰 $t_k$에 대해 L층 BiLSTM은 2L + 1개의 표현 세트를 계산하며, 이를 다운스트림 모델에 포함하기 위해 모든 계층을 하나의 벡터로 축소
            - ELMo representation + Embedding Vector → NLP task 적용
            
            ![image.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/899a9125-7ca9-4f13-b0d1-7812962af698.png)
            

## 3.  Experiments

---

> 다양한 6개의 벤치마크 NLP 과제에서 **ELMo**의 성능 보여줌
> 
> - 각 과제에서 ELMo를 단순히 추가하는 것만으로도 새로운 최첨단 결과를 달성
> - 강력한 기본 모델에 비해 상대적 오류율이 6%에서 20%까지 감소

![**Table1** 6개의 벤치마크 NLP 과제에서 보이는 ELMo 성능](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.05.28.png)

**Table1** 6개의 벤치마크 NLP 과제에서 보이는 ELMo 성능

**질문 응답** Stanford Question Answering Dataset (SQuAD) 

- 데이터셋 구조
    - 정답이 포함된 질문-응답 쌍 100K+를 포함
- Baseline
    - Bidirectional Attention Flow 모델(BiDAF; 2017)을 개선한 버전
- 해석
    - 기본 모델에 ELMo를 추가한 후, 테스트 세트 F1 점수가 81.1%에서 85.8%로 4.7% 향상
    - 기본 모델 대비 24.9% 상대적 오류율이 감소하고, 전체 단일 모델에서 최첨단 결과를 1.4% 개선

**텍스트 함의(Entailment)** Stanford Natural Language Inference (SNLI) 

- 데이터셋 구조
    - "가설"이 "전제"를 기반으로 참인지 판단하는 과제
- Baseline
    - ESIM 시퀀스 모델
- 해석
    - ELMo를 ESIM 모델에 추가해 평균 0.7% 정확도가 향상

**의미 역할 부여(Semantic Role Labeling, SRL)**

- 데이터셋 구조
    - 문장의 술어-논항 구조를 모델링하며, 종종 "누가 누구에게 무엇을 했는가"라는 질문에 답하는 것
- Baseline
    - BIO 태깅 문제로 모델링하고, 8층 깊이의 biLSTM을 사용하여 순방향과 역방향을 교차
- 해석
    - ELMo를 추가했을 때 단일 모델 테스트 세트 F1 점수가 81.4%에서 84.6%로 3.2% 상승

**공동 참조 해결(Coreference Resolution)**

- 데이터셋 구조
    - 텍스트에서 동일한 실제 세계 엔티티를 참조하는 언급들을 클러스터링하는 작업
- Baseline
    - 스팬 기반 신경망 모델
- 해석
    - ELMo를 추가했을 때 평균 F1이 67.2에서 70.4로 3.2% 상승하여 새로운 최첨단을 설정

**개체명 인식(Named Entity Recognition, NER)**

- 데이터셋 구조
    - CoNLL 2003 NER 과제(Sang와 Meulder, 2003)는 Reuters RCV1 말뭉치에서 뉴스와이어 데이터를 가져와 4가지 다른 개체 유형(PER, LOC, ORG, MISC)으로 태그를 부여
- Baseline
    - 사전 학습된 단어 임베딩, 문자 기반 CNN 표현, 2개의 biLSTM 계층 및 조건부 확률장(CRF) 손실을 사용
- 해석
    - ELMo 강화 biLSTM-CRF는 5회 실행 평균 F1 92.22%를 달성

**감정 분석(Sentiment Analysis)**

- 데이터셋 구조
    - Stanford Sentiment Treebank에서의 세밀한 감정 분류 작업은 영화 리뷰에서 문장을 설명하는 다섯 가지 라벨(매우 부정적에서 매우 긍정적)을 선택하는 과제
- Baseline
    - biattentive classification network (BCN)
- 해석
    - BCN 모델에서 CoVe를 ELMo로 교체하면 최첨단 대비 정확도가 1.0% 절대적으로 향상

⏬ 세부적인 Task를 통해 ELMo 성능 파악

### 3.1 Alternate layer weighting schemes

- **데이터 크기와 가중치**
    - 학습 데이터의 크기와 ELMo가 학습한 가중치가 어떻게 변하는지, 그리고 그 결과가 어떻게 나타나는지도 살펴봄

![Table2](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.36.14.png)

Table2

⏩ 대부분의 경우 λ 값이 작을수록 ELMo의 성능이 좋음

⏩ NER과 같은 작은 데이터셋의 경우 λ 값의 변화에 덜 민감

### 3.2  Where to include ELMo

- **위치의 중요성**
    - ELMo를 모델의 어느 부분에 포함시키는지에 따라 성능이 달라질 수 있음

![Table3](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.36.37.png)

Table3

⏩ 단어 임베딩을 BiRNN의 출력에 포함시키는 것이 일부 작업에서는 전체 결과를 향상시킬 수 있음을 발견

### 3.3 What information is captured by the biLM’s representations?

- **ELMo의 장점**
    - ELMo는 단어의 문맥을 깊이 반영한 표현을 제공
    - 기존 모델들이 단어의 최상위 정보만 사용했던 것과 달리, ELMo는 여러 층의 정보를 결합해서 더 나은 성능을 보임

![스크린샷 2024-08-11 오전 8.41.02.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.41.02.png)

ELMo를 추가하면 단어 벡터만 사용할 때보다 성능이 향상되므로, biLM의 문맥 표현은 일반적으로 NLP 작업에 유용한 정보를 담고 있어야 함

⏩ 단어 의미 구별 및 품사 태깅 작업을 위한 내재적 평가를 사용

![스크린샷 2024-08-11 오전 8.41.16.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.41.16.png)

![스크린샷 2024-08-11 오전 8.41.26.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.41.26.png)

⏩ 단어 의미 명확화와 품사 태깅에서 에서 충분히 괜찮은 성능

### 3.4 Sample efficiency

- **ELMo의 장점**
    - ELMo는 단어의 문맥을 깊이 반영한 표현을 제공
    - 기존 모델들이 단어의 최상위 정보만 사용했던 것과 달리, ELMo는 여러 층의 정보를 결합해서 더 나은 성능을 보임

ELMo를 모델에 추가하면 샘플 효율성이 크게 증가

![스크린샷 2024-08-11 오전 8.51.11.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.51.11.png)

⏩ ELMo를 추가했을 때는 그렇지 않을 때보다 학습속도도 빠르며(최대 49배 정도) 학습데이터가 적을 때도 훨씬 효율적으로 학습

### 3.5 Visualization of learned weights

소프트맥스 정규화된 학습된 계층 가중치를 시각화

- **문법과 의미의 층**
    - ELMo가 문법적인 정보는 모델의 하위 층에서 잘 잡아내고, 의미적인 정보는 상위 층에서 잘 포착

![image.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/image%205.png)

⏩  ELMo의 학습된 계층 가중치를 시각화하여 입력층에서는 첫 번째 BiLSTM 계층이 선호되며, 출력층에서는 낮은 계층이 약간 선호

## 4. Conclusion

---

**ELMo**  biLMs로부터 문맥기반 표현을 모델링하기 위한 일반적인 접근 방식

- 다양한 NLP 작업에 적용할 때 큰 성능 향상이 있음
- biLM 레이어
    - 문맥 속의 단어에 대한 다양한 구문적 및 의미적 정보를 효율적으로 인코딩
    - 모든 레이어를 사용하는 것이 전체 작업 성능을 향상시킴

## 🔎 궁금증 ..

---

![image.png](ELMo%20%E2%AD%90%2035099adfd0d64e36ab2932a564382c43/image%206.png)

→ 다운 스트림

Q. L층 BiLSTM은 2L + 1개의 표현 세트를 계산

A. 각 층마다 순방향 출력, 역방향 출력, 그리고 기본 입력 표현까지 포함하여 총 2L + 1개의 표현이 생성

이렇게 생성된 2L + 1개의 표현들은 각 층마다 서로 다른 정보를 담고 있음 

예를 들어, 하위 층의 출력은 문법적 정보를 잘 포착할 수 있고, 상위 층의 출력은 의미적 정보를 더 잘 포착

## 🎹 코드 리뷰

---

ELMo는 아직 텐서플로우 2.x 버전에서는 사용이 불가능하며 1.x대 버전으로 낮추고 코드를 실행

Google Colab에서는 TensorFlow 1.x 버전이 더 이상 지원되지 않으며, TensorFlow 2.x 버전만 사용

## ➕ 참고

---

[The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/)

[A Review of Deep Contextualized Word Representations (Peters+, 2018)](https://www.slideshare.net/slideshow/a-review-of-deep-contextualized-word-representations-peters-2018/102809428#6)

[https://github.com/dreji18/Semantic-Search-using-Elmo](https://github.com/dreji18/Semantic-Search-using-Elmo)