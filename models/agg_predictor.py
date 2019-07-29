import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention
from models.base_predictor import BasePredictor
import math

class AggPredictor(BasePredictor):
    def __init__(self, num=6, *args, **kwargs):
        self.num = num
        super(AggPredictor, self).__init__(*args, **kwargs)

    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):
        self.q_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                num_layers=num_layers, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                num_layers=num_layers, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                num_layers=num_layers, batch_first=True,
                dropout=0.3, bidirectional=True)

        # Aggregators
        self.q_cs = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=True)
        self.hs_cs = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=True)
        self.W_cs = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.op_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, self.num)) # for 5 aggregators

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx):
        """
        Args:
            q_emb_var [batch_size, question_seq_len, embedding_dim] : embedding of question
            q_len [batch_size] : lengths of questions
            hs_emb_var [batch_size, history_seq_len, embedding_dim] : embedding of history
            hs_len [batch_size] : lengths of history
            col_emb_var [batch_size*num_cols_in_db, col_name_len, embedding_dim] : embedding of history
            col_len [batch_size] : number of columns for each query
            col_name_len [batch_size] : number of tokens for each column name. 
                                        Each column has infomation about [type, table, name_token1, name_token2,...]
            col_idx int: Index of the column which we are predicting the op for 
        Returns:
            agg [batch_size, self.num] : probability distribution over {max, min, avg, sum, count, none}
        """        
        batch_size = len(col_len)

        q_enc,_ = self.q_lstm(q_emb_var, q_len)  # [batch_size, question_seq_len, hidden_dim]
        hs_enc,_ = self.hs_lstm(hs_emb_var, hs_len)  # [batch_size, history_seq_len, hidden_dim]
        _, col_enc = self.col_lstm(col_emb_var, col_name_len) # [batch_size*num_cols_in_db, hidden_dim]
        col_enc = col_enc.reshape(batch_size, col_len.max(), self.hidden_dim) # [batch_size, num_cols_in_db, hidden_dim]

        # Get encoding of the column we are prediting the op for
        col_emb = col_enc[np.arange(batch_size),col_idx].unsqueeze(1) # [batch_size, 1, hidden_dim]

        # Predict aggregators
        H_q_cs = self.q_cs(q_enc, col_emb, q_len) # [batch_size, hidden_dim]
        H_hs_cs = self.hs_cs(hs_enc, col_emb, hs_len) # [batch_size, hidden_dim]
        H_cs = self.W_cs(col_emb).squeeze(1) # [batch_size, hidden_dim]
        
        return self.op_out(H_q_cs + int(self.use_hs)*H_hs_cs + H_cs) # [batch_size, self.num]

    def predict(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx, force_agg = False):
        output = self.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx)

        # Some models predict both the values and number of values
        if isinstance(output, tuple):
            numbers, values = output
            
            numbers = torch.argmax(numbers, dim=-1).detach().cpu().numpy()
            predicted_values = []
            predicted_numbers = []

            # Loop over the predictions in the batch
            for number,value in zip(numbers, values):
                # Pick the n largest values
                # Make sure we actually predict something
                if number>0:
                    predicted_values += [torch.argsort(-value)[:number].cpu().numpy()]
                predicted_numbers += [number]

            return (predicted_numbers, predicted_values)

        if force_agg:
            output[:,-1] = -math.inf

        return torch.argmax(output, dim=1).detach().cpu().numpy()

    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])
        col_emb_var, col_len, col_name_len = embedding.get_columns_emb(batch['columns_all'])
        batch_size, num_cols_in_db, col_name_lens, embedding_dim = col_emb_var.shape

        # Combine batch_size and num_cols_in_db into the first dimension, since this is what out model expects 
        col_emb_var = col_emb_var.reshape(batch_size*num_cols_in_db, col_name_lens, embedding_dim) 
        col_name_len = col_name_len.reshape(-1)

        return self(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, batch['column_idx'])
