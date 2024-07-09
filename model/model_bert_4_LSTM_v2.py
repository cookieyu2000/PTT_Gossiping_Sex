# model/model_bert_4_LSTM_v2.py
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, bert_model, output_size, embedding_dim, lstm_hidden_size=1024, num_layers=6, dropout=0.5):
        super(Seq2Seq, self).__init__()
        self.encoder = bert_model
        self.encoder.requires_grad_(False)  # 设置BERT模型的requires_grad为False
        self.encoder.eval()
        
        self.encoder_LSTM = nn.LSTM(768, lstm_hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        self.decoder_embedding = nn.Embedding(output_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim, lstm_hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(lstm_hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        input1_input_ids = input1[0]
        input1_token_type_ids = input1[1]
        input1_attention_mask = input1[2]
        
        encoder_hidden_state = self.encoder(input_ids=input1_input_ids,
                                            token_type_ids=input1_token_type_ids,
                                            attention_mask=input1_attention_mask).last_hidden_state
        
        encoder_LSTM_out, (encoder_LSTM_hidden, encoder_LSTM_cell) = self.encoder_LSTM(encoder_hidden_state)
        
        decoder_embedded = self.decoder_embedding(input2)
        decoder_output, _ = self.decoder(decoder_embedded, (encoder_LSTM_hidden, encoder_LSTM_cell))
        
        decoder_output = self.dropout(decoder_output)
        logits = self.linear(decoder_output)

        return logits
