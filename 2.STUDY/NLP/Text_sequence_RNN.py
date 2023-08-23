## 지도학습을 통해 홀수나 짝수로 구성된 각 문장을 분류하는 것을 목적으로 함

import numpy as np
import tensorflow as tf

batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32
times_steps = 6
element_size = 1

# 입력 문장들을 하나의 텐서에 넣으려면 어떻게든 같은 크기로 맞춰야 한다.
# 따라서 6보다 작은 길이의 문장은 0(또는 PAD 스트링)으로 채워 모든 문장의 길이를 맞춘다.
# 이러한 전처리 단계를 제로 패딩이라고 부른다

digit_to_word_map = {1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven", 8:"eight", 9:"nine"}
digit_to_word_map[0] = "PAD"

even_sentences = []
odd_sentences = []
seqlens = []

for i in range(10000):
    rand_seq_len = np.random.choice(range(3,7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1,10,2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2,10,2), rand_seq_len)

    if rand_seq_len < 6:
        rand_odd_ints = np.append(rand_odd_ints, [0]*(6-rand_seq_len))
        rand_even_ints = np.append(rand_even_ints, [0]*(6-rand_seq_len))

    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))

data = even_sentences+odd_sentences
# 홀수, 짝수 시퀀스의 seq 길이
seqlens *= 2

print(even_sentences[0:6])
print(odd_sentences[0:6])

# 원래 문장의 길이를 저장하는 이유는 RNN에 넣을 때 다 넣으면 PAD까지 고려하기 때문에 원래 문장의 길이에서 잘라야함

# 단어를 인덱스에 매핑
word2index_map = {}
index = 0
for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

# 지도 학습을 통한 분류 작업이므로 다른 예제와 마찬가지로 원-핫 포맷의 레이블의 배열, 학습과 테스트 데이터, 데이터 인스턴스의 일괄 작업을 생성하는 함수 및 플레이스홀더가 필요
# 먼저 레이블을 만들고 데이터를 학습 데이터와 테스트 데이터로 나눈다.
labels = [1] * 10000 + [0] * 10000
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding
