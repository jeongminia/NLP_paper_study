# Word2Vec

---

<aside>
💡 **Word2Vec**
단어를 연속적인 벡터로 변환하여 단어의 의미적 유사성을 수치적으로 표현하는 기계 학습 모델
”**비슷한 분포를 가진 단어라면 비슷한 의미를 가질 것이다."**로, 자주 같이 등장할수록 두 단어는 비슷한 의미를 가진다는 것

</aside>

---

## 0. Abstract

---

> **Efficient Estimation of Word Representations in Vector Space**
> 
> - 벡터 공간에서 단어 표현의 효율적인 예측

- 대규모 데이터 집합에서 단어의 연속 벡터 표현을 계산하기 위한 두 가지 새로운 모델 아키텍처를 제안
    - CBOW, Skip-gram
- 기존 기술보다 더 높은 정확도와 낮은 계산 비용으로 우수한 성능을 보임
- 문법적. 의미적 단어 유사성 측정에서 최첨단 성능 기록

## 1. Introduction

---

**원-핫 인코딩(One-Hot Encoding)**

- 단어 간 유사성 개념을 다루기 전, 사용된 방법으로 단어 간 유사성을 다루지 않음
    - Vocabulary 구축해 단어들의 등장 빈도 순으로 순열을 부여
    - 단어집 V = {cat, fat, mat, sat, the, on} → V가 크기가 매우 커 매우 큰 차원의 희소벡터 필요하고 단어의 의미를 파악하기 어려움
- 한계
    - 자동 음성 인식이나 기계번역처럼 데이터가 제한된 작업에서 한계를 보임
    - 각 단어를 독립적인 단위로 취급하는 방식으로 단어 간의 의미적 유사성을 반영하지 못하고, 벡터 차원이 커지는 문제를 동반

### 1.1 목표

- 수십억 단어와 수백만 단어의 어휘를 가진 **대규모 데이터에서 고품질의 단어 벡터를 학습할 수 있는 기술**을 소개
    - 기존 아키텍처는 수억 단어를 초과해 훈련된 적이 없음
    - 단어 벡터의 차원은 50-100 사이로 제한적

- 비슷한 단어들이 가까운 벡터로 표현되어 단어의 여러 유사성 정도를 반영
- 훈련 시간과 정확도가 벡터 차원과 데이터 양에 어떻게 의존하는지도 논의
- **단어 벡터의 선형 규칙성을 보존**하며 **벡터 연산의 정확성을 극대화**하고 **구문적 및 의미적 규칙성을 측정**하기 위한 새로운 테스트 설계
    1. **단어 벡터의 선형 규칙성을 보존 :** "King" - "Man" + "Woman" = "Queen"과 같은 벡터 연산에서, 이러한 선형 규칙성을 통해 "Queen"이라는 결과 벡터가 제대로 얻어질 수 있도록 함 
    2. **벡터 연산의 정확성을 극대화** : 실제 단어의 의미적 관계를 잘 반영하도록 하여, 벡터 연산의 정확성을 높이는 것
        
        ![자연어처리 바이블](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_7.26.28.png)
        
        자연어처리 바이블
        
    3. **구문적 및 의미적 규칙성을 측정 :** 단어 벡터가 문법적 규칙이나 의미적 유사성을 얼마나 잘 반영하는지 측정할 수 있는 기준을 마련하

⏩ 학습 기술의 발전으로 더 큰 데이터 집합에서 복잡한 모델을 훈련할 수 있게 되었으며, 분산 표현을 사용하는 모델들이 간단한 모델보다 우수한 성능을 보임

### 1.2 이전 연구

단어를 연속 벡터로 표현하는 연구는 오래된 분야로, 신경망 언어 모델(NNLM)이 주요 발전

- **선형 프로젝션 레이어**  a linear projection layer : 단어를 벡터로 변환
- **비선형 은닉 레이어** a non-linear hidden layer ****: 더 복잡한 언어적 패턴을 학습

⏩ 발전

- 다양한 NLP 응용 프로그램에서 성능을 크게 향상시켰고, 모델을 단순화하는 데 기여
- 여러 모델 아키텍처가 개발되었고, 일부 단어 벡터는 향후 연구를 위해 공개
- but. 많은 아키텍처가 높은 **훈련 비용** 초래

## 2. Model Architecture

---

Word2Vec과 비교해 복잡한 구조와 계산 비용을 갖는 모델

1️⃣ **피드포워드 신경망 언어 모델 (NNLM)**

- 입력, 프로젝션, 은닉, 출력 레이어로 구성
- $N × D + N × D × H + H × V$

2️⃣ **순환 신경망 언어 모델 (RNNLM)**

- 입력, 은닉, 출력 레이어 구성
- $H × H + H × V$

🔽  두 가지 새로운 **단어 분산 표현 학습 모델 아키텍처**를 제안하며 계산 복잡성 최소화

> **Word2Vec**
> 
> - 딥러닝 기술로 분류되기도 하지만, 학습에 사용되는 모델 자체는 두개 계층을사용하는 얕은 것
> - 계산량이 적어 학습에 대량 데이터를 활용할 수 있고 모델의 복잡도보다 데이터 양이 더 중요함
> - EX. CBOW Continusous Bag of Words, Skip-gram

![스크린샷 2024-08-04 오전 7.09.49.png](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_7.09.49.png)

⏬ 모두 확률적으로 더 먼 거리의 단어들을 선택함으로써 넓은 범위의 문맥을 단어 벡터 학습에 활용

> **CBOW** Continuous Bag of Words
> 
- 주변에 있는 단어들을 입력으로 중간에 있는 단어들을 예측하는 방법
- 특정 단어를 중심으로 이전 n개의 단어와 이후 n개의 단어가 주어졌을 때 중심 단어를 예측하는 것을 목표로 학습해 정확히 분류하는 로그-선형 분류기를 구축
- NNLM과 유사하지만 **비선형 은닉층을 모두 제거**하고, **모든 단어에 대해 투영층이 공유**되는 모델
- 모델의 훈련 복잡성
    
    $Q = N × D + D × log_2(V)$
    
    - $N$ : 입력 단어의 수, $D$ : 단어 벡터의 차원, $V$ : 어휘의 크기
    
    ![Untitled](Word2Vec%2046a1367541824327b185350d273b73e5/Untitled.png)
    
    `작동 방식`
    
    1. **단어 벡터 평균화**
        - 모든 단어는 **동일한 위치로 투영** (벡터가 평균화 됨)
            - 모든 단어에 대해 동일한 투영층을 사용하여 같은 방식으로 처리한다는 뜻
        - 단어의 순서를 고려하지 않고, 단어들을 평균내어 **하나의 벡터**로 만듦
    2. **입력과 미래 단어 사용**
        - 현재 단어를 예측하기 위해 네 개의 이전 단어 + 네 개의 미래 단어를 사용
    3. **로그-선형 분류기**
        - 단어 벡터들의 평균값을 사용하여 현재 단어를 예측하는 분류기 생성
    
    ---
    
    - `Input`
        - 학습시킬 문장의 모든 단어들을 **one-hot encoding**방식으로 벡터화
    - `Hidden` 대신 `projection layer`
        - 각 단어에 파라미터 W를 곱하면 각 단어들은 one hot encoding 벡터이기 때문에 각 단어에 대응하는 W의 행 추출하며, 그 단어에 대응하는 embedding vector
        - 해당 임베딩 벡터는 문맥을 나타내는 벡터를 의미
    - `Output`
        - 해당 임베딩 벡터는 가중치 행렬을 통해 출력 벡터로 변환되며 softmax 함수를 통해 각 단어가 목표 단어일 확률 계산

> **Skip-Gram**
> 
- 중간에 있는 단어들을 입력으로 주변 단어들을 예측하는 방법
    - CBOW에서는 **문맥을 기반으로 현재 단어를 예측**
- 중심 단어가 주어졌을 때 이전 n개의 단어와 이후 n개의 단어를 예측하는 것을 목표로 학습해 현재 단어를 사용해 같은 문장 내의 다른 단어 분류
- 같은 문장에서 한 단어를 기반으로 **다른 단어를 최대한 정확하게 분류**하는 것을 목표
- 모델의 훈련 복잡성
    
    $Q = C × (D + D × log_2(V))$
    
    - $C$  : 단어의 최대 거리, $N$ : 입력 단어의 수, $D$ : 단어 벡터의 차원, $V$ : 어휘의 크기
    
    ![Untitled](Word2Vec%2046a1367541824327b185350d273b73e5/Untitled%201.png)
    
    `작동 방식`
    
    1. **현재 단어를 입력으로 사용**
        - 현재 단어를 연속적인 투영층이 있는 **로그-선형 분류기의 입력**으로 사용
    2. **주변 단어 예측**
        - 현재 단어를 기반으로 주변 단어들을 예측
        - 예측 범위는 과거 단어와 미래 단어로 구성
    
    ---
    
    - CBOW와 마찬가지로 `input`은 one-hot encoding 방식으로 들어감
    - `Input`
        - 목표 단어의 one-hot vector가 입력으로 주어지며 가중치 행렬을 통해 임베딩 벡터로 변환
    - `Hidden` 대신 `projection layer`
        - 임베딩 벡터는 여기서 사용되며 가중치 행렬을 통해 여러 출력 벡터로 변환
    - `Output`
        - 출력 벡터들은 각각 주변 단어일 확률 계산

## 3.  Experiments

---

### 3.1 Task Description

---

![8869개의 의미적 질의 / 10675개의 문법적인 질의](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.06.38.png)

8869개의 의미적 질의 / 10675개의 문법적인 질의

 다양한 단어 벡터 버전의 품질을 비교하기 위해 일반적으로 **예제 단어와 가장 유사한 단어들을 보여주는 표** 사용

- 벡터 공간에서 가장 가까운 단어를 찾으면 정답
- 'biggest' - 'big' + 'small’ = smaller

💡**실험결과**

- **의미적 질문**
    - **단어 벡터**가 의미적 관계를 잘 포착하면, 예를 들어 "France"와 "Paris"의 관계를 "Germany"와 "Berlin"의 관계로 유추할 수 있는 능력.
    - 단어 벡터가 높은 품질로 훈련되면, 세밀한 의미적 관계를 인식할 수 있음.
- **구문적 질문**
    - **구문적 관계**를 정확하게 인식하는 벡터의 능력.
    - 벡터의 정확도가 높아짐에 따라, 구문적 질문에서도 높은 성능을 보여줌.

### 3.2 **Maximization of Accuracy**

![CBOW 구조의 서로 다른 word vector 차원과 훈련 data set의 크기에 따른 결과](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.17.23.png)

CBOW 구조의 서로 다른 word vector 차원과 훈련 data set의 크기에 따른 결과

- 참고
    - 현재 많은 데이터에서 단어 벡터를 훈련하더라도 벡터 크기가 충분하지 않은 경우가 많음.
    - 훈련 데이터 양을 두 배로 늘리면 벡터 크기를 두 배로 늘리는 것과 유사한 계산 복잡도 증가.
    - **단어 벡터 차원** 및 **훈련 데이터 양**에 따른 CBOW 아키텍처의 결과를 평가.
    - 일정 지점을 지나면 차원이나 훈련 데이터를 추가해도 개선이 점차 줄어듦.
    - **벡터 차원과 훈련 데이터 양**을 함께 증가시켜야 최적의 결과를 얻을 수 있음.

**Google News Corpus**

- **train : 어휘 크기**는 가장 빈번한 **100만 단어**로 제한하여, 이 단어들만을 사용하여 모델을 훈련
- **test :** 특정 하위 집합에서 모델 아키텍처를 평가하기 위해 **어휘를 가장 빈번한 3만 단어**로 제한
- **목표** : 단어 벡터의 정확도를 높이기 위함

💡**실험결과**

- 많은 단어가 학습된다면 이에 대한 정보들을 담을 수 있는 충분한 dimension이 확보되어야 한다는 사실
- 벡터의 차원과 훈련데이터를 점점 키우면 점점 정확도가 향상
    - 벡터 차원 : 각 단어를 고차원의 벡터로 변환할 때 사용하는 벡터의 크기로 차원이 많을수록 더 많은 의미적 정보

### 3.3 Comparsion of Model Architecture

3️⃣ 서로 다른 모델의 성능을 비교하기 위해 똑같은 데이터와 모든 모델을 640차원의 워드 벡터로 맞춰 실행

![스크린샷 2024-08-04 오전 8.17.55.png](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.17.55.png)

💡**실험결과**

- RNNLM < NNLM < CBOW < Skip-gram

4️⃣ 오직 하나의 CPU를 사용하여 공적으로 사용 가능한 word vector들과 비교하여 평가

dimensionality/train data로 비교

![스크린샷 2024-08-04 오전 8.25.53.png](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.25.53.png)

💡**실험결과**

- 다른 여러 NNLM과 비교했을 때에도 CBOW, Skip-gram은 훨씬 더 좋은 성능을 보임

5️⃣ epoch으로 비교한 표

![스크린샷 2024-08-04 오전 8.26.44.png](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.26.44.png)

💡**실험결과**

- 많이 학습한 Skip-gram이 좋은 성능을 보임
- epoch 수보다도 데이터 양을 증가시키는 것이 중요

### 3.4 **Large Scale Parallel Training of Models**

6️⃣ 분산 연산 framework 인 DistBelief 사용

![스크린샷 2024-08-04 오전 8.27.17.png](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.27.17.png)

💡**실험결과**

- 대규모 모델과 데이터 세트를 훈련할 때, 분산 학습이 가능하며, Skip-gram이 좋은 성능을 보임

### 3.5 **Microsoft Research Sententce Completion Challenge**

7️⃣ Microsoft Research Sentence Completion Challenge는 1040개의 sentence가 주어지는게, 각 sentence는 1개의 단어가 빠져있고 이를 예측하는 task

![스크린샷 2024-08-04 오전 8.27.00.png](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.27.00.png)

💡**실험결과**

- skip-gram 단독으로는 기존의 model들에 비해 다소 낮은 수치를 보임
- RNNLM과 결합한 뒤에는 SOTA를 달성

### 🔽 **Examples of the Learned Relationships**

![스크린샷 2024-08-04 오전 8.18.56.png](Word2Vec%2046a1367541824327b185350d273b73e5/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_8.18.56.png)

- 더 높은 정확도를 달성하기 위해서는 더 많은 dataset을 사용
- 각 단어 사이의 상관관계 vector를 여러 단어쌍 사이의 subtract vector의 평균으로 만들어내면 정확도 향상

## 4. Conclusion

---

- **간단한 모델 아키텍쳐**와 **저렴한 계산 비용**
    - 매우 간단한 모델 아키텍처를 사용하여도 높은 품질의 단어 벡터를 학습 가능
        
        → 일반적인 신경망 모델보다 훨씬 낮은 계산 복잡도로 가능하며 훨씬 더 큰 데이터 집합에서 매우 정확한 고차원 단어 벡터 계산 가능
        
- **대규모 데이터 처리**
    - DistBelief 분산 프레임워크를 사용하면 CBOW 및 Skip-gram 모델을 단어 수가 1조에 달하는 코퍼스에서 훈련 가능
- **성능 향상**
    - SemEval-2012 Task 2에서 이전의 최고 성능을 크게 능가하는 결과를 달성
    - 공개된 RNN 벡터와 다른 기술을 활용하여 Spearman의 순위 상관계수를 50% 이상 향상
- **NLP 응용 가능성**, **지식 기반 활용**, **기계번역 성과**
- 미래 연구 방향
    - LRA 등 다른 방법들과 비교를 제안하며 포괄적인 테스트를 통해 단어 벡터 추청 기술의 개선을 목표

## 🔎 궁금증 ..

---

임베딩하고 난 뒤 모델링을 적용한다.

→ 임베딩을 통해 차원을 축소하여 RNN의 계산 효율성을 높이고, 학습 속도를 개선

```python
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# 예제 텍스트 데이터
texts = ["I love machine learning", "Deep learning is amazing"]

# 1. 텍스트 데이터 전처리 및 Word2Vec 학습
# 각 텍스트를 단어 리스트로 변환
tokenized_texts = [text.lower().split() for text in texts]

# Word2Vec 모델 학습
word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=50, window=5, min_count=1, workers=4)

# 단어 인덱스 생성
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# 시퀀스 데이터 생성
sequences = tokenizer.texts_to_sequences(texts)
max_len = max(len(seq) for seq in sequences)
X_train = pad_sequences(sequences, maxlen=max_len)

# 2. Word2Vec 임베딩 벡터를 Keras Embedding 레이어에 적용
embedding_dim = 50
vocab_size = len(word_index) + 1

# 임베딩 매트릭스 생성
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# 3. RNN 모델 구성 및 학습
model = Sequential()
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=False)
model.add(embedding_layer)
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 예제 레이블 데이터
y_train = np.array([1, 0])

# 모델 학습
model.fit(X_train, y_train, epochs=10)
```

## ➕ 참고

---

[Learning Word Embedding](https://lilianweng.github.io/posts/2017-10-15-word-embedding/)

[Word2vec 알고리즘 리뷰 1 : CBOW 와 Skip-gram](https://simonezz.tistory.com/35)

- 용어
    
    👐🏻 **단어 임베딩** Word Embedding
    
    - 문자를 숫자들의 배열인 벡터로 변환하는 방법으로 만들어진 단어 벡터 의미
        
        
        |  | 원-핫 벡터 | 임베딩 벡터 |
        | --- | --- | --- |
        | 차원 | 고차원(단어 집합의 크기) | 저차원 |
        | 다른 표현 | 희소 벡터의 일종 | 밀집 벡터의 일종 |
        | 표현 방법 | 수동 | 훈련 데이터로부터 학습함 |
        | 값의 타입 | 1과 0 | 실수 |
    
    👐🏻 **희소 표현** sparse representation
    
    - 벡터 또는 행렬의 값이 대부분이 0으로 표현되는 방법
    
    👐🏻 **분산 표현** distributed representation
    
    - 단어의 의미를 다차원 공간에 벡터화하는 방법을 사용
    - 분산 표현을 이용하여 단어 간 의미적 유사성을 벡터화하는 작업을 워드 임베딩(embedding)