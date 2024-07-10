# model/model_bert_gpt2.py
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel
from transformers import BertConfig
from transformers import BertTokenizerFast, BertModel
from transformers import (
   BertTokenizerFast,
   AutoModelForMaskedLM,
   AutoModelForCausalLM,
   AutoModelForTokenClassification,
)


class Seq2Seq(nn.Module):
    def __init__(self, bert_model, gpt2_model):
        super(Seq2Seq, self).__init__()
        self.bert_model = bert_model
        self.gpt2_model = gpt2_model
        self.linear = nn.Linear(bert_model.config.hidden_size, gpt2_model.config.n_embd)

    def forward(self, q_input, a_input, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert_model(input_ids=q_input, attention_mask=attention_mask)
        bert_embeddings = bert_outputs.last_hidden_state
        bert_embeddings = self.linear(bert_embeddings)

        gpt2_inputs = {
            'input_ids': a_input,
            'attention_mask': attention_mask,
            'encoder_hidden_states': bert_embeddings,
            'encoder_attention_mask': attention_mask
        }

        gpt2_outputs = self.gpt2_model(**gpt2_inputs)
        logits = gpt2_outputs.logits

        return logits

    
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 建立模型實例
    
    seq_len = 41

    # 載入繁體中文BERT模型作為Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)
    gpt2_model_name = 'ckiplab/gpt2-base-chinese'
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)

    config = BertConfig.from_pretrained(bert_model_name)
    vocab_size = config.vocab_size

    print(bert_model.config.hidden_size)
    print("Vocabulary Size:", vocab_size)
    
    model = Seq2Seq(bert_model=bert_model,
                    gpt2_model=gpt2_model,
                    ).to(device)

    # batch_size = 1
    
    #製造一個
    x = torch.LongTensor([[412]*seq_len]).to(device)
    x = x.to(torch.int)

    x_att = torch.LongTensor([[1]*seq_len]).to(device)
    x_att = x_att.to(torch.int)
    
    out = model(x,x,x_att) # works  
    print(out.size())