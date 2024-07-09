# dataset_bert_4_v2.py
from transformers import BertTokenizerFast, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import os
from transformers import BertConfig
import pickle
import re

# 定义聊天数据集类
class ChatDataset(Dataset):
    def __init__(self, data, bert_tokenizer, max_len):
        data_word_index = f'data/QA_word_index_PTT_Gossiping_Sex.pkl'
        # 加载词汇索引文件
        with open(data_word_index, 'rb') as fp:
            self.word_index = pickle.load(fp)
        
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len

    # 返回数据集的长度
    def __len__(self):
        return len(self.data)

    # 返回数据集中指定索引的元素
    def __getitem__(self, index):
        question, answer = self.data[index]
        # 预处理问题和答案
        question = self.preprocess_text(question)
        answer = self.preprocess_text(answer)
        
        try:
            # 对问题进行分词和编码，并返回tensor
            q_input = self.bert_tokenizer(question, add_special_tokens=True, return_tensors='pt', max_length=self.max_len, truncation=True, padding='max_length')
        except Exception as e:
            print(f"Error in tokenizing question: {question}")
            raise e

        # 将答案转换为索引并补齐长度
        a_input = [self.sentence_pad_input2(answer)]
        a_input = torch.tensor(a_input).to(torch.long)

        # 生成目标ID，去掉 <sos> 标记，并补齐到最大长度
        target_ids = a_input[0][1:]  # target_ids 是 a_input 去掉 <sos>
        padding_length = self.max_len - target_ids.size(0)
        # 用0填充target_ids以确保长度一致
        target_ids = torch.cat([target_ids, torch.zeros(padding_length, dtype=torch.long)], dim=0)
        # 添加 batch 维度
        target_ids = target_ids.unsqueeze(0)

        return {'q_input': q_input, 'a_input': a_input, 'target': target_ids}
    
    # 将句子补齐并转换为索引
    def sentence_pad_input2(self, sentence):
        # 将句子转换为字符列表
        chars = list(sentence)
        # 添加开始和结束标记
        chars = ['<sos>'] + chars + ['<end>']
        # 将字符列表补齐到指定的最大长度
        padded_chars = chars[:self.max_len] + ['<pad>'] * (self.max_len - len(chars))
        # 将字符转换为索引，如果字符不在词汇索引中，使用 <unk> 索引
        indexed_chars = [self.word_index.get(char, self.word_index['<unk>']) for char in padded_chars]
        return indexed_chars
    
    # 预处理文本，移除特殊字符
    def preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', text)  # 移除多余的空白字符
        text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号和特殊字符
        return text

if __name__ == "__main__":
    # 载入繁体中文BERT模型作为Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    config = BertConfig.from_pretrained(bert_model_name)
    # 获取词汇表大小
    vocab_size = config.vocab_size
    print("Vocabulary Size:", vocab_size)

    # 载入聊天数据
    data_path = f'data/PTT_Gossiping_Sex.txt'
    data = []
    with open(data_path, 'r') as fp:
        all = fp.readlines()
        for per in all:
            per = per.strip('\n')  # 去掉换行符
            per_split = per.split()  # 分割问题和答案
            if len(per_split) >= 2:
                Q = per_split[0]
                A = per_split[1]
                data.append((Q, A))

    max_len = 56  # 设置最大长度
    # 创建聊天数据集对象
    dataset = ChatDataset(data, bert_tokenizer, max_len)
    # 创建数据加载器对象
    dataloader = DataLoader(dataset, batch_size=1)
    # 获取一个数据样本
    d = next(iter(dataloader))
    ans = d
    # 输出样本的问题、答案和目标
    print('In Q:', ans['q_input'])
    print('In A:', ans['a_input'])
    print('Target:', ans['target'])
