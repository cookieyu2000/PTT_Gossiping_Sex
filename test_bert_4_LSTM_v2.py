# test_bert_4_LSTM_v2.py
import torch
import os
from transformers import BertTokenizerFast, BertModel
from model.model_bert_4_LSTM_v2 import Seq2Seq
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

# 預處理輸入文本的函數
def preprocess_input(input_text, bert_tokenizer, max_len):
    encoded_inputs = bert_tokenizer(
        input_text,
        add_special_tokens=False,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return encoded_inputs

# 測試模型的函數
def test(model, bert_tokenizer, input_text, index_word, word_index, max_len):
    input_text = preprocess_input(input_text, bert_tokenizer, max_len).to(device)
    target_text = [0] * max_len
    target_text[0] = word_index['<sos>']
    target_text = torch.tensor(target_text).to(torch.long).unsqueeze(0)

    with torch.no_grad():  # 不計算梯度
        input_text_input_ids = input_text['input_ids'].to(device)
        input_text_token_type_ids = input_text['token_type_ids'].to(device)
        input_text_attention_mask = input_text['attention_mask'].to(device)

        inputs1 = [input_text_input_ids, input_text_token_type_ids, input_text_attention_mask]
        inputs2 = target_text.to(device)
        origin_input2 = inputs2.to(device)

        # 迭代生成回答
        for num_word in range(max_len):
            outputs = model(inputs1, inputs2)
            outputs_argmax = torch.argmax(outputs, dim=-1)
            outputs_index = outputs_argmax[0][num_word].detach().cpu().numpy()

            if check_end(outputs_index) or num_word == max_len - 1:
                generate_ans = origin_input2[0].detach().cpu().numpy()
                generate_ans = generate_ans[1:]
                generate_ans = np.append(generate_ans, int(outputs_index))
                break
            else:
                origin_input2[0][num_word + 1] = torch.tensor(outputs_index)
                inputs2 = origin_input2

    generate_ans = convert_word(index_word, generate_ans)
    return generate_ans

# 檢查是否為結束標記的函數
def check_end(index):
    if index_word[int(index)] in ['<end>', '<unk>'] or int(index) == 0:
        return True
    return False

# 將索引轉換為詞語的函數
def convert_word(index_word, string):
    return_string = ''
    for index in string:
        if index_word[index] in ['<end>', '<unk>']:
            return return_string
        elif index == 0:
            return return_string
        else:
            return_string += index_word[index]
    return return_string

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_NAME = 'bert_4_LSTM.pth'
    WEIGHTS = f'weights/{WEIGHTS_NAME}'
    print('Loading weight:', WEIGHTS)

    MAX_LEN = 56
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)
    
    vocab_size = 7357
    data_index_word = f'data/QA_index_word_PTT_Gossiping.pkl'
    with open(data_index_word, 'rb') as fp:
        index_word = pickle.load(fp)
    data_word_index = f'data/QA_word_index_PTT_Gossiping.pkl'
    with open(data_word_index, 'rb') as fp:
        word_index = pickle.load(fp)

    model = Seq2Seq(
        bert_model=bert_model,
        output_size=vocab_size,
        embedding_dim=200
    ).to(device)

    model.load_state_dict(torch.load(WEIGHTS))
    model.eval()

    while True:
        input_text = str(input('請輸入對話:'))
        # 獲取模型預測的回答
        predicted_text = test(
            model=model,
            bert_tokenizer=bert_tokenizer,
            input_text=input_text,
            index_word=index_word,
            word_index=word_index,
            max_len=MAX_LEN
        )
        print("回答:", predicted_text)
