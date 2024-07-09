# word2index.py
import os
from ckip_transformers.nlp import CkipWordSegmenter
import numpy as np
import torch
import pickle

# 初始化CKIP分詞器
ws_driver = CkipWordSegmenter(model="albert-base", device=0)

# 資料路徑
data_path = 'data/PTT_Gossiping_Sex.txt'

# 字典保存路徑
data_word_index = 'data/QA_word_index_PTT_Gossiping_Sex.pkl'
data_index_word = 'data/QA_index_word_PTT_Gossiping_Sex.pkl'

# 初始化變數
data = []
data_Q = []
data_A = []

Q_len = 0
A_len = 0
total_lines = 0
valid_lines = 0

# 資料準備
with open(data_path, 'r', encoding='utf-8') as fp:
    all = fp.readlines()
    total_lines = len(all)
    for per in all:
        per = per.strip('\n')
        per_split = per.split()
        if len(per_split) < 2:
            print(f"Skipping line due to insufficient data: {per}")
            continue
        Q = per_split[0]
        A = per_split[1]
       
        data_Q.append(Q)
        data_A.append(A)
        data.append(Q)
        data.append(A)

        Q_len += len(Q)
        A_len += len(A)
        valid_lines += 1

# 將文本轉換為數字序列
input_texts = data_Q
target_texts = data_A
input_characters = sorted(list(set(''.join(input_texts))))
target_characters = sorted(list(set(''.join(target_texts))))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

data_texts = data
data_characters = sorted(list(set(''.join(data_texts))))
max_data_seq_length = max([len(txt) for txt in data_texts])
data_token_index = {'<pad>': 0}  # 將 <pad> 標記的索引設為 0
# 建立字典對照表
for i, char in enumerate(data_characters):
    data_token_index[char] = i + 1  
data_token_index['<sos>'] = len(data_token_index) 
data_token_index['<unk>'] = len(data_token_index)

data_texts_inverse = data
data_characters_inverse = sorted(list(set(''.join(data_texts_inverse))))
data_token_index_inverse = {0: '<pad>'}  # 將 <pad> 標記的索引設為 0
# 建立字典對照表
for i, char in enumerate(data_characters_inverse):
    data_token_index_inverse[i + 1] = char  
data_token_index_inverse[len(data_token_index_inverse)] = '<sos>'
data_token_index_inverse[len(data_token_index_inverse)] = '<unk>'

print(data_token_index_inverse)
print('max_data_seq_length:', max_data_seq_length)

# 儲存字典
with open(data_word_index, "wb") as fp:
    pickle.dump(data_token_index, fp)

with open(data_index_word, "wb") as fp:
    pickle.dump(data_token_index_inverse, fp)

# 計算可用資料比例
usable_data_ratio = valid_lines / total_lines * 100
print(f'Usable data : {valid_lines}')
print(f'Usable data ratio: {usable_data_ratio:.2f}%')
