# test_bert_4_GPT2_v2.py
import torch
from transformers import BertTokenizerFast, GPT2Tokenizer, AutoModelForCausalLM, BertModel
from model.model_bert_4_GPT2_v2 import Seq2Seq
import os

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 參數設定
BERT_MODEL_NAME = 'bert-base-chinese'
GPT2_MODEL_NAME = 'ckiplab/gpt2-base-chinese'
WEIGHTS_PATH = 'weights/bert_4_GPT2.pth'

# 初始化BERT和GPT-2的分詞器和模型
bert_tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
gpt2_model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME).to(device)

# 初始化Seq2Seq模型
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
model = Seq2Seq(bert_model=bert_model, gpt2_model=gpt2_model).to(device)

# 加載訓練好的權重
if os.path.exists(WEIGHTS_PATH):
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.gpt2.load_state_dict(state_dict, strict=False)
    print("模型權重加載成功")
else:
    print("找不到權重文件，請檢查路徑")
    exit(1)

# 設置模型為評估模式
model.eval()

def generate_response(question, max_length=56):
    with torch.no_grad():
        # BERT分詞
        inputs = bert_tokenizer(question, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 生成回應
        response_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        response = gpt2_tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    while True:
        question = input("輸入問題 ('exit' 退出): ")
        if question.lower() == 'exit':
            break
        response = generate_response(question)
        print(f"回應: {response}")