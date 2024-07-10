# test_bert_gpt2.py
import torch
import torch.nn as nn
import os, shutil
from transformers import BertTokenizerFast, AutoModel, BertModel
from dataset_bert_gpt2 import ChatDataset
from torch.utils.data import Dataset, DataLoader
from model.model_bert_gpt2 import Seq2Seq
import numpy as np
import tqdm
from utils.early_stop import early_stop
import matplotlib.pyplot as plt
from transformers import BertConfig
from torchsummaryX import summary
from transformers import (
   BertTokenizerFast,
   AutoModelForMaskedLM,
   AutoModelForCausalLM,
   AutoModelForTokenClassification,
)



def preprocess_input(input_text, bert_tokenizer, max_len):
    encoded_inputs = bert_tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded_inputs['input_ids']
    return input_ids.to(device)


def predict(input_text, model, bert_tokenizer, max_len):
    model.eval()
    # 生成回復
    with torch.no_grad():
        input_ids = preprocess_input(input_text, bert_tokenizer, max_len)
        logits = model(input_ids, input_ids, attention_mask=None)  # 使用相同的輸入作為問題和答案
        
        # gpt2_res = model.gpt2_model.generate(input_ids, 
        #                            max_length=MAX_LEN, 
        #                            num_beams=1,  
        #                            do_sample=False,
        #                            no_repeat_ngram_size = 2)
        
    predicted_ids = torch.argmax(logits, dim=-1).squeeze()  # 獲取預測的token索引
    predicted_text = bert_tokenizer.decode(predicted_ids)  # 將預測的token索引解碼為文本
    print(predicted_ids)
    # print(gpt2_res[0])
    # predicted_text2 = bert_tokenizer.decode(gpt2_res[0])  # 將預測的token索引解碼為文本
    # print(predicted_text2)
    return predicted_text
    





if __name__ =="__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    CURRENT_PATH = os.path.dirname(__file__)
    WEIGHTS_NAME = 'model_weights.pth'
    WEIGHTS = f'{CURRENT_PATH}/weights/model_bert_GPT2/{WEIGHTS_NAME}'


    MAX_LEN = 56
    
    # 載入繁體中文BERT模型作為Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name) 
    gpt2_model_name = 'ckiplab/albert-tiny-chinese'
    gpt2_model = AutoModelForMaskedLM.from_pretrained(gpt2_model_name).to(device)
    

    # 初始化模型
    config = BertConfig.from_pretrained(bert_model_name)
    vocab_size = bert_tokenizer.vocab_size
    hidden_size = 768  # BERT模型的隐藏層大小

    

    model = Seq2Seq(
        bert_model=bert_model,
        gpt2_model=gpt2_model
    ).to(device)

    # x = torch.LongTensor([[0]*MAX_LEN]).to(device)
    # x = x.to(torch.int)
    # print(summary(model,x,x))

    model.load_state_dict(torch.load(WEIGHTS))
    model.eval()

    input_text = str(input('請輸入對話:'))
    # 回應

    predicted_text = predict(input_text=input_text,
                             model=model,
                             bert_tokenizer=bert_tokenizer,
                             max_len=MAX_LEN)
    
    print("回答:", predicted_text)
