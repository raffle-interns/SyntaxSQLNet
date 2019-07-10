import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import BagOfWord
from models.base_predictor import BasePredictor

class AndOrPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super(AndOrPredictor, self).__init__(*args, **kwargs)

    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):
        self.q_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                num_layers=num_layers, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                num_layers=num_layers, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.bag_of_word = BagOfWord()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_hs = nn.Linear(hidden_dim, hidden_dim)
        self.ao_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 2)) 

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len):
        """
        Args:
            q_emb_var [batch_size, question_seq_len, embedding_dim] : embedding of question
            q_len [batch_size] : lengths of questions
            hs_emb_var [batch_size, history_seq_len, embedding_dim] : embedding of history
            hs_len [batch_size] : lengths of history
        Returns:
            p_andor [batch_size, 2] : probabilities of AND, OR
        """
        q_enc, _ = self.q_lstm(q_emb_var, q_len) #[batch_size, question_seq_len, hidden_dim]
        hs_enc, _ = self.hs_lstm(hs_emb_var, hs_len) #[batch_size, history_seq_len, hidden_dim]

        # Calculate H_Q using Bag Of Word trick
        H_Q = self.bag_of_word(q_enc, q_len) #[batch_size, hidden_dim]

        # Project H_Q
        H_Q = self.W_q(H_Q) #[batch_size, hidden_dim]

        # Do the same for the history
        H_HS = self.bag_of_word(hs_enc, hs_len) # [batch_size, 1, hidden_dim]
        H_HS = self.W_hs(H_HS) # [batch_size, hidden_dim]

        return self.ao_out(H_Q + int(self.use_hs)*H_HS) # [batch_size, 2]

    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])
        
        return self(q_emb_var, q_len, hs_emb_var, hs_len)
