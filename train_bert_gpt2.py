# train_bert_gpt2.py
import torch
import torch.nn as nn
import os, shutil
import subprocess
from transformers import BertTokenizerFast, BertModel, GPT2Config, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from model.model_bert_gpt2 import Seq2Seq
import numpy as np
import tqdm
from utils.early_stop import early_stop
import matplotlib.pyplot as plt
from dataset_bert_gpt2 import ChatDataset
import warnings
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def train(num_epochs, train_dataloader, valid_dataloader, model, criterion):
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_acc = 0.0
        model.train()
        for batch in tqdm.tqdm(train_dataloader):
            q_input = batch['q_input'].squeeze(1).to(device)
            a_input = batch['a_input'].squeeze(1).to(device)
            target = batch['target'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)

            logits = model(q_input, a_input, attention_mask)
            logits = logits.view(-1, logits.shape[-1])
            target = target.view(-1)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * q_input.size(0)
            acc = calculate_accuracy(logits, target)
            total_acc += acc * q_input.size(0)
            
        validation_loss, validation_accuracy = valid(valid_dataloader, model, criterion)
        
        average_loss = total_loss / len(train_dataloader.dataset)
        accuracy = total_acc / len(train_dataloader.dataset)

        avg_loss = round(average_loss, 6)
        accuracy = round(accuracy * 100, 4)
        validation_loss = round(validation_loss, 6)
        validation_accuracy = round(validation_accuracy * 100, 4)

        print(f'Epoch: {epoch} | train_loss: {avg_loss} | train_acc: {accuracy}% | val_loss: {validation_loss} | val_acc: {validation_accuracy}%')

        performance_value = [epoch, avg_loss, accuracy, validation_loss, validation_accuracy]

        EARLY_STOP(avg_loss, model=model, performance_value=performance_value)
            
        if EARLY_STOP.early_stop:
            print('Early stopping')
            break   

def valid(valid_dataloader, model, criterion):
    total_loss = 0.0
    total_acc = 0.0

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(valid_dataloader):
            q_input = batch['q_input'].squeeze(1).to(device)
            a_input = batch['a_input'].squeeze(1).to(device)
            target = batch['target'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)

            logits = model(q_input, a_input, attention_mask)
            logits = logits.view(-1, logits.shape[-1])
            target = target.view(-1)

            loss = criterion(logits, target)

            total_loss += loss.item() * q_input.size(0)
            acc = calculate_accuracy(logits, target)
            total_acc += acc * q_input.size(0)

    avg_loss = total_loss / len(valid_dataloader.dataset)
    avg_acc = total_acc / len(valid_dataloader.dataset)
    return avg_loss, avg_acc

def calculate_accuracy(logits, targets):
    _, predicted = torch.max(logits, dim=1)
    mask = targets != 0
    correct = torch.sum(predicted[mask] == targets[mask]).item()
    total = torch.sum(mask).item()
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MAX_LEN = 56
    EPOCHS = 5000
    BATCH_SIZE = 128
    WIGHTS_NAME = 'model_bert_GPT2.pth'
    SAVE_MODELS_PATH = 'weights/'
    WEIGHTS_PATH = os.path.join(SAVE_MODELS_PATH, WIGHTS_NAME)

    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)
    bert_model.requires_grad_(False)

    gpt2_model_name = 'ckiplab/gpt2-base-chinese'
    gpt2_config = GPT2Config.from_pretrained(gpt2_model_name)
    gpt2_config.add_cross_attention = True
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name, config=gpt2_config).to(device)

    data_path = 'data/PTT_Gossiping_Sex.txt'
    data = []
    with open(data_path, 'r') as fp:
        all = fp.readlines()
        for per in all:
            per = per.strip('\n')
            per_split = per.split()
            Q = per_split[0]
            A = per_split[1]
            data.append((Q, A))
    valid_ratio = 0.2
    data = np.array(data)
    np.random.seed(6670)
    indices = list(range(len(data)))
    t_split = int(np.floor(valid_ratio * len(data)))
    
    train_indices, valid_indices = indices[t_split:], indices[:t_split]
    train_d = data[train_indices]
    valid_d = data[valid_indices]

    train_data = ChatDataset(train_d, bert_tokenizer, MAX_LEN)
    valid_data = ChatDataset(valid_d, bert_tokenizer, MAX_LEN)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE)

    model = Seq2Seq(bert_model=bert_model, gpt2_model=gpt2_model).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    EARLY_STOP = early_stop(save_path=SAVE_MODELS_PATH,
                            weight_name=WIGHTS_NAME,
                            mode='min',
                            monitor='train_loss',
                            patience=50)
    
    if os.path.exists(WEIGHTS_PATH):
        print("Loading existing weights...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    else:
        print("No existing weights found. Training from scratch...")
    train(num_epochs=EPOCHS,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            model=model,
            criterion=criterion
            )
