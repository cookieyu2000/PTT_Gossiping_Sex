import torch
import os
from transformers import BertTokenizerFast, BertModel, AutoModelForCausalLM
from model.model_bert_gpt2 import Seq2Seq
import warnings

warnings.filterwarnings("ignore")

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
    attention_mask = encoded_inputs['attention_mask']
    return input_ids.to(device), attention_mask.to(device)

def predict(input_text, model, bert_tokenizer, max_len):
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = preprocess_input(input_text, bert_tokenizer, max_len)
        logits = model(input_ids, input_ids, attention_mask=attention_mask)
        
    predicted_ids = torch.argmax(logits, dim=-1).squeeze()
    predicted_text = bert_tokenizer.decode(predicted_ids.tolist(), skip_special_tokens=True)
    return predicted_text


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    WEIGHTS_PATH = 'weights/model_bert_GPT2.pth'
    if not os.path.exists(WEIGHTS_PATH):
        print('No weights found')
        exit()

    MAX_LEN = 512
    
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)
    gpt2_model_name = 'ckiplab/gpt2-base-chinese'
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(device)
    
    model = Seq2Seq(bert_model=bert_model, gpt2_model=gpt2_model).to(device)
    
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    while True:
        input_text = str(input('請輸入對話:'))
        if input_text.lower() in ['exit', '離開', '結束', 'bye', '掰掰', '再見']:
            break
        predicted_text = predict(input_text=input_text, model=model, bert_tokenizer=bert_tokenizer, max_len=MAX_LEN)
        print("回答:", predicted_text)
