import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_utils import run_lstm, col_name_encode
from attention import BagOfWord, GeneralAttention, ConditionalAttention
from utils import length_to_mask
from lstm import PackedLSTM
from models.base_predictor import BasePredictor

class ValuePredictor(BasePredictor):
    def __init__(self, max_num_cols=6, *args, **kwargs):
        self.num = max_num_cols
        super(ValuePredictor, self).__init__(*args, *kwargs)

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

        # Value
        # TODO: the model is just adaptation of the ColumnPredictor. Could one use pointer network instead?
        self.col_q = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=False)
        self.hs_q = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=False)
        self.W_q = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.value_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))
   
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
            col_idx [batch_size]: Index of the column which we are predicting the op for 
        Returns:
            value [batch_size, question_seq_len] : probability distribution over the tokens in the question
        """
        batch_size = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = col_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)

        # Get encoding of the column we are prediting the value for
        col_emb = col_enc[np.arange(batch_size),col_idx].unsqueeze(1) # [batch_size, 1, hidden_dim]

        # Even though the col encoding is just a sequence of length 1, we need to lengths for masking
        col_len = [1]*batch_size

        # Run conditional encoding for column|question, and history|question 
        H_col_q = self.col_q(col_emb, q_enc, col_len, q_len) #[batch_size, question_seq_len, hidden_dim]
        H_hs_q = self.hs_q(hs_enc,q_enc, hs_len, q_len) #[batch_size, question_seq_len, hidden_dim]
        H_q = self.W_q(q_enc) #[batch_size, question_seq_len, hidden_dim]

        value = self.value_out(H_col_q + int(self.use_hs)*H_hs_q +H_q) # [batch_size, question_seq_len, 1]

        q_mask = length_to_mask(q_len)

        # Number of values might be different for each question, so we need to mask some of them
        value = value.masked_fill_(q_mask, 0)

        return value.squeeze()

    def loss(self, score, truth):
        # Here suppose truth looks like [[[1, 4], 3], [], ...]
        loss = 0
        B = len(truth)
        col_num_score, col_score = score

        # Loss for the column number
        truth_num = [len(t) - 1 for t in truth] # double check truth format and for test cases
        data = torch.from_numpy(np.array(truth_num))
        truth_num_var = Variable(data.cuda())
        loss += self.CE(col_num_score, truth_num_var)

        # Loss for the key words
        T = len(col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            gold_l = []
            for t in truth[b]:
                if isinstance(t, list):
                    gold_l.extend(t)
                else:
                    gold_l.append(t)
            truth_prob[b][gold_l] = 1
        data = torch.from_numpy(truth_prob)
        truth_var = Variable(data.cuda())
        pred_prob = self.sigm(col_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss
