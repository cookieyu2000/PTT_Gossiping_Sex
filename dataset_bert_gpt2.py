# dataset_bert_gpt2.py
from transformers import BertTokenizerFast,BertModel
import torch
from torch.utils.data import Dataset,DataLoader
import os 
from transformers import BertConfig

class ChatDataset(Dataset):
    def __init__(self, data, bert_tokenizer,max_len):
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question, answer = self.data[index]
        q_input = self.bert_tokenizer.encode(question, add_special_tokens=False,return_tensors='pt',max_length=self.max_len,truncation=True,padding='max_length',)  # 问题（Q）不包含<SOS>和<END>
        a_input = self.bert_tokenizer.encode(answer, add_special_tokens=True,return_tensors='pt' ,max_length=self.max_len,truncation=True,padding='max_length',)  # 答案（A）包含<SOS>和<END>
        
        target_ids = a_input[0][1:] # target_ids 是 a_input 去掉<SOS>
        padding_length = self.max_len - target_ids.size(0)
        target_ids = torch.cat([target_ids, torch.zeros(padding_length, dtype=torch.long)], dim=0)  # #捕到一樣長度
        target_ids = target_ids.unsqueeze(0)  # 添加 batch 维度
        attention_mask = self.bert_tokenizer.batch_encode_plus([answer], add_special_tokens=True, return_tensors='pt',max_length=self.max_len, padding='max_length', truncation=True,)['attention_mask']

        return {'q_input': q_input, 'a_input': a_input, 'target': target_ids, 'attention_mask': attention_mask}
    

if __name__ == "__main__":
    CURRENT_PATH = os.path.dirname(__file__)
    # 載入繁體中文BERT模型作為Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    config = BertConfig.from_pretrained(bert_model_name)
    vocab_size = config.vocab_size
    print("Vocabulary Size:", vocab_size)

    data_path = f'{CURRENT_PATH}/data/PTT_Gossiping_Sex.txt'
    data = []
    with open(data_path,'r') as fp:
        all = fp.readlines()
        for per in all:
            per = per.strip('\n')
            per_split = per.split()
            Q = per_split[0]
            A = per_split[1]
            data.append((Q,A))

    max_len = 41
    dataset = ChatDataset(data,bert_tokenizer,max_len)
    dataloader = DataLoader(dataset,batch_size=1)
    d = next(iter(dataloader))
    ans = d
    print('In Q:',ans['q_input'].size())
    print('In A:',ans['a_input'].size())
    print('Target:',ans['target'].size())
    print('attention_mask:',ans['attention_mask'].size())

