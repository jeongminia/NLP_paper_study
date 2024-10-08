# GRU

---

<aside>
💡 **GRU(Gated Recurrent Unit)**
LSTM의 장기 의존성 문제에 대한 해결책을 유지하며, Reset gate, Update gate를 이용해 계산을 효율적으로 진행하며 LSTM보다 단순한 구조를 갖고 있으며, cell을 사용

</aside>

---

## 0. Abstract

---

> **Learning Phrase Representations using `RNN Encoder-Decoder` for Statistical Machine Translation**
> 
> - 통계적 기계 번역을 위해 제안된 아키텍처인 **RNN 인코더-디코더**를 사용해 구문표현 학습
> - 인코더-디코더 구조는 입력문장을 인코딩한 후, 이를 디코딩해 **번역**된 문장을 생성하는 방식

- RNN 인코더-디코더라는 새로운 신경망 모델로 두개의 RNN을 사용
    - `encoder` :  입력 시퀀스를 고정 길이 벡터표현으로 인코딩
    - `decoder` : encoder에서 생성된 벡터 표현을 다른 기호 시퀀스로 디코딩
- 모델 학습 방식
    - 인코더와 디코더는 **공동**으로 학습
    - 학습의 목표는 train_sequence에 대해 target_sequece가 나올 조건부 확률을 최대화함
- 성능향상
    - RNN 인코더-디코더에 의해 계산된 구문 쌍의 **조건부 확률을 추가 특징으로 사용**함으로써 향상
    - 메모리 용량과 훈련 향상을 위해 hidden unit 사용
- 모델의 표현 학습
    - 언어구문의 의미적/구문적으로 의미있는 표현을 학습할 수 있음
        
        → 언어의 의미와 구조를 이해하고 이를 바탕으로 번역
        

## 1. Introduction

---

### **1.1 등장배경**

- LSTM 은 RNN의 장기 기억 손실 문제를 해결하면서 긴 시퀀스를 가진 데이터에서도 좋은 성능을 내는 모델
    - 단점으로는 복잡한 구조때문에 RNN에 비해 파라미터가 많이 필요 → 데이터 불충분의 경우 과대적합
- LSTM의 over fitting 문제를 해결하기 위해 LSTM 의 변형인 GRU 등장
- 종종 비슷한 성능을 달성하지만 계산 속도가 더 빠르다는 장점이 있는 간소화된 버전의 LSTM 메모리 셀을 제공
    - LSTM : forget gate, get gate, update gate , cell
    - GRU : reset gate, update gate

### **1.2 아키텍처**

- 인코더와 디코더 역할을 하는 두 개의 순환 신경망 (RNN)으로 구성

### 1.3 실험 요약

**영어→프랑스어 번역 실험**

- 학습
    - 모델을 훈련시켜 영어 구문을 해당하는 프랑스어 구문으로 번역할 확률을 학습
- 과정
    - 모델은 phrase table의 각 구문 쌍을 scoring해 표준 구문 기반 SMT 시스템의 일부로 사용
- 결과
    - 기존 번역 모델과 GRU모델의 구문 점수를 비교해 언어적 규칙성을 더 잘 포착함을 깨달음
    - RNN 인코더와 디코더를 사용한 구문 쌍 점수화 접근 방식이 번역 성능 향상

## 2. RNN Encoder–Decoder

---

**단계 요약**

- 인코더는 입력 문장을 요약된 벡터로 변환
- 디코더는 이 요약된 벡터를 사용하여 출력 문장을 생성
- 각 단계에서 hiden state와 이전 출력 단어를 사용하여 다음 단어를 예측

![Figure 1: An illustration of the proposed RNN Encoder–Decoder](GRU%20dd619188951c4b039b057d09e3f7e0f2/Untitled.png)

Figure 1: An illustration of the proposed RNN Encoder–Decoder

**인코더 (Encoder)**

- 이 시퀀스를 순차적으로 읽고, 각 시점에서 hidden state 업데이트
- 마지막 입력 기호 $x_T$를 읽은 후, 인코더 RNN의 숨겨진 상태는 전체 입력 시퀀스의 요약인 $c$로 변환
    - **c : context vector**
        - 입력 시퀀스의 모든 중요한 정보를 압축하여 담고 있는 벡터
        - 디코더가 이를 기반으로 번역 수행

**디코더 (Decoder)**

- 주어진 요약 c와 이전 출력 기호 $y_{t−1}$를 기반으로 다음 출력 기호 $y_t$를 예측
- 각 시점에서 이전 상태 $h_{t−1}$, 이전 출력 $y_{t−1}$ 및 요약 c에 조건부로 새로운 숨겨진 상태 $h_t$를 계산
    - hidden state 계산 방법
        
        $h_t=f(h_{t−1},y_{t−1},c)$
        

⏩  조건부 로그 가능도를 최대화하도록 인코더와 디코더가 훈련됨

![스크린샷 2024-07-21 오전 2.06.32.png](GRU%20dd619188951c4b039b057d09e3f7e0f2/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_2.06.32.png)

- 주어진 입력 시퀀스에 대해 출력 시퀀스가 나올 확률을 최대화하는 것
- 각 (xn,yn)는 훈련 데이터셋의 (입력 시퀀스, 출력 시퀀스) 쌍

![Untitled](GRU%20dd619188951c4b039b057d09e3f7e0f2/Untitled%201.png)

**Reset gate**

![Untitled](GRU%20dd619188951c4b039b057d09e3f7e0f2/Untitled%202.png)

- 0에 가가울 때 hidden state는 이전 상태를 무시하고 현재 input인 x만으로 리셋
    - sigmoid 함수를 이용해 과거의 정보를 적당히 날리고 은닉층에 곱함
- 미래에 불필요하다고 판단되는 정보를 버릴 수 있도록 해 더 간결한 표현 가능

**Update gate**

![Untitled](GRU%20dd619188951c4b039b057d09e3f7e0f2/Untitled%203.png)

- 이전 은닉층 정보가 현재 상태로 얼마나 많이 전달될지 제어
- 장기 정보를 기억할 수 있도록 도움

## 3.  Statistical Machine Translation

---

- **SMT 시스템 목표**: 원문 문장 e에 대해 가장 적절한 번역문 f를 찾아 $p(f∣e)$를 최대화
- **로그 선형 모델**: 추가적인 특징과 가중치를 사용하여 $log_p(f∣e)$를 모델링
- **구문 기반 SMT**: 번역 모델을 구문 단위로 분해하여 처리
    - **구문 테이블 (Phrase Table)**:
        - SMT 시스템에서는 소스 언어와 타겟 언어의 구문 쌍(phrase pairs)을 포함한 테이블을 사용
- **신경망 언어 모델**: 신경망을 사용하여 번역 가설을 재평가하거나 원문 문장의 표현을 사용하여 번역 품질을 평가
    - **점수화 (Scoring)**:
        - 제안된 RNN 인코더-디코더 모델은 각 구문 쌍에 대해 점수를 부여하며 소스 언어 구문이 주어졌을 때 해당 타겟 언어 구문이 나타날 조건부 확률을 나타냄

⏩ 해당 논문에서,  구문 테이블에 있는 구문 쌍의 재평가만을 고려

### 제안된 RNN Encoder-Decoder의 특징

- **가변 길이 입력과 출력 처리 가능**
    - 원문 및 번역 구문의 단어 순서를 고려하여 타당한 번역과 타당하지 않은 번역을 구별할 수 있음.
    - 구문의 길이가 길어지거나 신경망을 다른 가변 길이의 시퀀스 데이터에 적용할 때 중요
- **기존 방법과의 차별성**
    - 단어 순서를 고려하여 구문을 평가할 수 있음.

## 4.  Experiments

---

<aside>
💡 **RNN Encoder-Decoder** 🆚 **CSLM (Context-Sensitive Language Model)
← English/French translation task** of the WMT’14 workshop

</aside>

### 4.1 실험 전 준비

**data**

- 언어 코퍼스에는 Europarl (6100만 단어), 뉴스 논평 (550만 단어), UN (4억 2100만 단어), 각각 9000만 단어와 7억 8000만 단어의 두 개의 크롤링된 코퍼스가 포함
- 프랑스어 언어 모델을 훈련하기 위해 크롤링된 신문 자료 약 7억 1200만 단어와 이중 언어 코퍼스의 타겟 부분이 추가로 제공
    - 모든 단어 수는 토큰화 후 프랑스어 단어를 기준
- GRU 훈련을 위해 영어와 프랑스어의 소스 및 타겟 어휘를 가장 빈번하게 사용되는 1만 5000단어로 제한
    - 데이터셋의 약 93%를 커버
    - 어휘 외 단어들은 특별 토큰 ([UNK])으로 매핑

**RNN encoder-decoder**

- 구조
    - **은닉 유닛**: 1000개, 인코더와 디코더에 제안된 게이트 포함.
    - **입출력 행렬 근사화**: 각 단어에 대해 차원 100의 임베딩 학습에 해당하는 랭크 100 행렬 사용.
    - **활성화 함수**: 쌍곡탄젠트 함수.
    - **디코더 출력 계산**: 500개의 맥스아웃 유닛을 가진 심층 신경망 사용
        - 파라미터
            - **가중치 초기화**
                - **일반 가중치**: 표준 편차가 0.01인 가우시안 분포에서 샘플링.
                - **재귀 가중치**: 가우시안 분포에서 샘플링 후 왼쪽 특이 벡터 행렬 사용.
            - **훈련**
                - **기법**: Adadelta와 확률적 경사 하강법.
                - **하이퍼파라미터**: ϵ=10−6, ρ=0.95.
                    
                    ϵ=10−6\epsilon = 10^{-6}
                    
                    ρ=0.95\rho = 0.95
                    
                - **데이터**: 구문 테이블에서 무작위로 선택된 64개의 구문 쌍 사용.
                - **훈련 시간**: 약 3일.

**연구 목적**

- RNN 디코더-인코더 vs CSLM
    - 구문 쌍을 점수화하기 위해 RNN Encoder-Decoder를 훈련
    - 7그램 모델을 사용하여 CSLM을 훈련
    - 해당 모델이 SMT 시스템의 구문 쌍 점수화에서 얼마나 효과적인지 평가

### 4.2 Quantitative Analysis

1. Baseline configuration
2. Baseline + RNN
3. Baseline + CSLM + RNN
4. Baseline + CSLM + RNN + Word penalty
    
    ![Untitled](GRU%20dd619188951c4b039b057d09e3f7e0f2/Untitled%204.png)
    
- CSLM과 RNN이 크게 상관관계가 있지는 않음
- 독립적으로 개선시키는 것으로도 성능을 높일 수 있음을 확인
- 모르는 단어 개수, 즉 리스트에 없는 단어들에 페널티를 부여했고, 선형로그 모델에 모르는 단어 개수만큼 피쳐를 추가하는 방식으로 구현

### 4.3 Qualitative Analysis

**성능 분석**

- RNN Encoder-Decoder와 기존 번역 모델의 구문 쌍 점수를 비교함으로써 성능 향상의 원인을 분석
- 기존 번역 모델은 빈도 기반으로 점수를 추정하는 반면, RNN Encoder-Decoder는 언어적 규칙에 기반하여 구문 쌍을 점수화

**주요 발견**

- RNN Encoder-Decoder가 선택한 타겟 구문은 일반적으로 실제 번역이나 문자 그대로의 번역에 더 가까움
- RNN Encoder-Decoder는 짧은 구문을 선호
- **추가 실험**:
    - RNN Encoder-Decoder가 실제 구문 테이블을 보지 않고도 잘 형성된 타겟 구문을 생성할 수 있음
    - 생성된 구문이 구문 테이블의 타겟 구문과 완전히 겹치지 않음.

### 4.4 Word and Phrase Representations

**RNN Encoder–Decoder가 단어와 구문에 대한 의미론적 및 문법적 구조를 효과적으로 학습**

1️⃣ **언어적 임베딩**

- **기대 효과**
    - RNN Encoder–Decoder는 연속 공간 벡터로 단어 및 구문을 매핑하므로, 의미론적으로 유사한 단어들이 클러스터링될 것으로 예상됨.
- **결과**
    - 2차원 임베딩 시각화 결과, **의미적으로 유사한 단어들이 서로 클러스터링**됨.
    - 이 과정은 Barnes-Hut-SNE 기법을 사용하여 수행됨.
    
    ![스크린샷 2024-07-21 오전 8.47.58.png](GRU%20dd619188951c4b039b057d09e3f7e0f2/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.47.58.png)
    

2️⃣ **구문 표현**

- **구문 벡터**: RNN Encoder–Decoder는 구문을 1000차원 벡터로 연속 공간에서 표현함.
- **시각화**: 4단어 이상의 구문을 Barnes-Hut-SNE로 시각화한 결과, RNN Encoder–Decoder가 **구문과 문법적 구조를 잘 포착**하고 있음을 확인함.
    
    ![스크린샷 2024-07-21 오전 8.51.06.png](GRU%20dd619188951c4b039b057d09e3f7e0f2/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-07-21_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.51.06.png)
    
    - 하단 왼쪽 **시간 관련 구문**
    - 하단 오른쪽 **의미론적 유사성**
    - 상단 오른쪽 **문법적 유사성**

## 5. Conclusion

---

1. RNN 인코더-디코더
    - 임의의 길이의 sequence를 다른 임의의 길이의 sequence로 매핑하는 학습 진행
    - 새로운 유닛
        - Reset Gate와 Update Gate를 포함해 각 히든 유닛이 시퀀스를 읽거나 생성할 떄 기억하거나 잊는 양 조절
2. 모델의 평가
    - 통계적 기계 번역 작업에서 모델을 평가하며 RNN 인코더-디코더를 사용해 구문 테이블의 각 구문 쌍을 평가
    - RNN 인코더-디코더 기여가 다른 신경망 모델과 함께 사용할 경우 성능이 더욱 향상
3. 향후 연구
    - 번역 태스크에서 타겟 구문을 잘 생성할 것으로 기대됨

## ➕ 참고

---

[10.2. Gated Recurrent Units (GRU) — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_recurrent-modern/gru.html)

- tistory
    
    [[논문 읽기] PyTorch 구현 코드로 살펴보는 GRU(2014), Learning Phrase Representation using RNN Encoder-Decoder for Statistical Machine Translation](https://deep-learning-study.tistory.com/691)
    
    [07-3. 순환 신경망 LSTM, GRU - (3)](https://excelsior-cjh.tistory.com/185)
    
- Youtube
    
    [[Paper Review] Gated RNN](https://www.youtube.com/watch?v=5Ar1aN9gceg)
    
- 용어
    
    👐🏻 **BLEU Score** **B**ilingual Evaluation Understudy Score
    
    - PPL 한계 : 번역의 성능을 직접적으로 반영하는 수치라 보기는 어려움
    - **“기계 번역의 성능이 얼마나 뛰어난가”**를 측정하기 위해 사용되는 대표적인 방법
    
    👐🏻 **LSTM과 비교** 
    
    ![Untitled](GRU%20dd619188951c4b039b057d09e3f7e0f2/Untitled%205.png)
    
    👐🏻 **CSLM** Context-Sensitive Language Model
    
    - 문맥에 따라 언어 모델링을 수행하는 신경망 기반 모델
    
    👐🏻 **WP** Word Penalty