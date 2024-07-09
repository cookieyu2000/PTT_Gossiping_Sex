# train_bert_4_GPT2.py
import torch
import torch.nn as nn
import os
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModelForCausalLM
from dataset_bert_4_v2 import ChatDataset
from torch.utils.data import DataLoader
from model.model_bert_4_GPT2_v2 import Seq2Seq
import numpy as np
import tqdm
from utils.early_stop import early_stop
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# 计算词汇大小的函数
def calculate_vocab_size(data):
    counter = Counter()  # 初始化计数器
    for q, a in data:
        counter.update(q)  # 更新问题中的词汇计数
        counter.update(a)  # 更新答案中的词汇计数
    vocab_size = len(counter) + 4  # +4 是因为需要添加 <sos>, <end>, <pad>, <unk> 四个特殊标记
    return vocab_size

# 训练函数
def train(num_epochs, train_dataloader, model, criterion, optimizer):
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_acc = 0.0
        model.train()  # 设置模型为训练模式
        for batch in tqdm.tqdm(train_dataloader):
            input1 = batch['q_input']
            input1_input_ids = input1['input_ids'].squeeze(1).to(device)
            input1_token_type_ids = input1['token_type_ids'].squeeze(1).to(device)
            input1_attention_mask = input1['attention_mask'].squeeze(1).to(device)
            input1 = [input1_input_ids, input1_token_type_ids, input1_attention_mask]
            
            input2 = batch['a_input'].squeeze(1).to(device)
            target = batch['target'].squeeze(1).to(device)

            loss, logits = model(input1, input2=input2, labels=target)  # 模型前向传播
            
            total_loss += loss.item()
            acc = calculate_accuracy(logits, target)  # 计算准确度
            total_acc += acc

            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        average_loss = total_loss / len(train_dataloader)  # 计算平均损失
        accuracy = total_acc / len(train_dataloader)  # 计算平均准确度

        avg_loss = round(average_loss, 6)
        accuracy = round(accuracy * 100, 4)

        print(f'Epoch: {epoch + 1} | train_loss: {avg_loss} | train_acc: {accuracy}%')
        
        performance_value = [epoch, avg_loss, accuracy]

        EARLY_STOP(avg_loss, model=model, performance_value=performance_value)  # 早停
        if EARLY_STOP.early_stop:
            print('Early stopping')
            break

# 计算准确度的函数
def calculate_accuracy(logits, targets):
    _, predicted = torch.max(logits, dim=-1)
    mask = targets != 0
    correct = torch.sum((predicted == targets) * mask).item()
    total = torch.sum(mask).item()
    accuracy = correct / total
    return accuracy

# 计算每一层的参数量并保存到文件
def save_model_parameters(model, file_path):
    with open(file_path, 'w') as f:
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_params = param.numel()
                f.write(f'{name}: {layer_params} parameters\n')
                total_params += layer_params
        f.write(f'\nTotal parameters: {total_params}\n')

# 主程序
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WEIGHTS_NAME = 'bert_4_GPT2.pth'
    WEIGHTS = f'weights/{WEIGHTS_NAME}'

    BATCH_SIZE = 128
    SAVE_MODELS_PATH = f'weights/'
    EPOCHS = 5000
    MAX_LEN = 56
    SEED = 42
    EMBEDDING_DIM = 400
    DROPOUT = 0.1
    LR = 0.0007
    PATIENCE = 50
    MODE = 'min'
    MONITOR = 'train_loss'

    # 初始化BERT模型和分词器
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name).to(device)

    # 初始化GPT-2模型和分词器
    gpt2_model_name = 'ckiplab/gpt2-base-chinese'
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(device)

    data_path = f'data/PTT_Gossiping_Sex.txt'
    data = []
    with open(data_path, 'r') as fp:
        all = fp.readlines()
        for per in all:
            per = per.strip('\n')
            per_split = per.split()
            if len(per_split) >= 2:
                Q = per_split[0]
                A = per_split[1]
                data.append((Q, A))

    data = np.array(data)
    np.random.seed(SEED)
    np.random.shuffle(data)

    # 创建数据集和数据加载器
    train_data = ChatDataset(data, bert_tokenizer, MAX_LEN)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

    vocab_size = calculate_vocab_size(data)
    print('Vocabulary Size:', vocab_size)
    
    # 初始化模型
    model = Seq2Seq(
        bert_model=bert_model,
        gpt2_model=gpt2_model,
        dropout=DROPOUT  # 增加dropout以防止过拟合
    ).to(device)

    if os.path.exists(WEIGHTS):
        print("model loading SUCCESSFULLY !")
        state_dict = torch.load(WEIGHTS, map_location=device)
        model.gpt2.load_state_dict(state_dict, strict=False)
    else:
        print("No weights file found, starting training from scratch.")
    
    # 只包含 GPT-2 的参数
    gpt2_parameters = list(model.gpt2.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gpt2_parameters, lr=LR)

    EARLY_STOP = early_stop(save_path=SAVE_MODELS_PATH, mode=MODE, monitor=MONITOR, patience=PATIENCE, weight_name=WEIGHTS_NAME)

    # 保存模型参数信息到文件
    save_model_parameters(model, r'model/model_parameters.txt')

    # 开始训练
    train(num_epochs=EPOCHS, train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer)
