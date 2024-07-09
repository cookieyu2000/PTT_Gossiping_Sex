# model/model_bert_4_GPT2_v2.py
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, bert_model, gpt2_model, dropout=0.5):
        super(Seq2Seq, self).__init__()
        self.encoder = bert_model
        self.gpt2 = gpt2_model
        self.dropout = nn.Dropout(dropout)

        # Freeze BERT parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input1, input2=None, labels=None):
        input1_input_ids = input1[0]
        input1_token_type_ids = input1[1]
        input1_attention_mask = input1[2]
        
        encoder_hidden_state = self.encoder(input_ids=input1_input_ids,
                                            token_type_ids=input1_token_type_ids,
                                            attention_mask=input1_attention_mask).last_hidden_state
        encoder_hidden_state = self.dropout(encoder_hidden_state)
        
        if input2 is not None:
            gpt2_inputs = self.gpt2(input_ids=input2, attention_mask=input1_attention_mask, labels=labels)
            loss = gpt2_inputs.loss
            logits = gpt2_inputs.logits
            return loss, logits
        else:
            gpt2_inputs = self.gpt2.generate(input_ids=input1_input_ids, attention_mask=input1_attention_mask)
            return gpt2_inputs