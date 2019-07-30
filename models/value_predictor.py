import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention
from utils.utils import length_to_mask
from utils.dataloader import SpiderDataset, try_tensor_collate_fn
from embedding.embeddings import GloveEmbedding
from torch.utils.data import DataLoader
from models.base_predictor import BasePredictor
import math

class ValuePredictor(BasePredictor):
    """
    Similar to ColPredictor, but instead predict the start index of the value token, 
    and how many of proceeding tokens to include.
    """
    def __init__(self, max_num_tokens=10, *args, **kwargs):
        self.max_num_tokens = max_num_tokens
        super(ValuePredictor, self).__init__(*args, **kwargs)

    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):  
        self.value_pad_token =-10000

        self.q_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                              num_layers=num_layers, batch_first=True,
                              dropout=0.3, bidirectional=True)

        self.hs_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                               num_layers=num_layers, batch_first=True,
                               dropout=0.3, bidirectional=True)

        self.col_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                                num_layers=num_layers, batch_first=True,
                                dropout=0.3, bidirectional=True)

        # Number of tokens
        self.col_q_num = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=True)
        self.hs_q_num = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=True)
        self.tokens_num_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, self.max_num_tokens)) # num of tokens: 1-6

        # tokens
        self.col_q = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=False)
        self.hs_q = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=False)
        self.W_value = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.value_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))

        pos_weight = torch.tensor(3).double()
        if gpu: pos_weight = pos_weight.cuda()
        self.bce_logit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len):
        """
        Args:
            q_emb_var [batch_size, question_seq_len, embedding_dim] : embedding of question
            q_len [batch_size] : lengths of questions
            hs_emb_var [batch_size, history_seq_len, embedding_dim] : embedding of history
            hs_len [batch_size] : lengths of history
            col_emb_var [batch_size*num_cols_in_db, col_name_len, embedding_dim] : embedding of history
            col_len [batch_size] : number of columns for each query
            col_name_len [batch_size*num_cols_in_db] : number of tokens for each column name. 
                                        Each column has infomation about [type, table, name_token1, name_token2,...]
        Returns:
            num_cols = [batch_size, max_num_cols] : probability distribution over how many columns should be predicted
            p_col [batch_size, num_columns_in_db] : probability distribution over the columns given
        """
        batch_size = len(col_len)

        q_enc,_ = self.q_lstm(q_emb_var, q_len)  # [batch_size, question_seq_len, hidden_dim]
        hs_enc,_ = self.hs_lstm(hs_emb_var, hs_len)  # [batch_size, history_seq_len, hidden_dim]
        _, col_enc = self.col_lstm(col_emb_var, col_name_len) # [batch_size*num_cols_in_db, hidden_dim]
        col_enc = col_enc.reshape(batch_size, col_len.max(), self.hidden_dim) # [batch_size, num_cols_in_db, hidden_dim]

        #############################
        # Predict number of tokens  #
        #############################

        # Run conditional encoding for column|question, and history|question
        H_col_q = self.col_q_num(col_enc, q_enc, col_len, q_len)  # [batch_size, hidden_dim]
        H_hs_q = self.hs_q_num(hs_enc, q_enc, hs_len, q_len)  # [batch_size, hidden_dim]

        num_tokens = self.tokens_num_out(H_col_q + int(self.use_hs)*H_hs_q)

        ################################
        # Predict start index of token #
        ################################

        # Run conditional encoding for question|column, and history|column
        H_col_q = self.col_q(col_enc, q_enc, col_len, q_len)  # [batch_size, question_seq_len, hidden_dim]
        H_hs_q = self.hs_q(hs_enc, q_enc, hs_len, q_len)  # [batch_size, question_seq_len, hidden_dim]
        H_value = self.W_value(q_enc)  # [batch_size, question_seq_len, hidden_dim]

        values = self.value_out(H_col_q + int(self.use_hs)*H_hs_q + H_value).squeeze(2)  # [batch_size, question_seq_len]
        values_mask = length_to_mask(q_len).squeeze(2).to(values.device)

        # Number of tokens might be different for each question, so we need to mask some of them
        values = values.masked_fill_(values_mask, self.value_pad_token)

        return (num_tokens, values)

    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])
        batch_size = len(q_len)
        col_emb_var, col_len, col_name_len = embedding.get_columns_emb(batch['columns_all'])
        batch_size, num_cols_in_db, col_name_lens, embedding_dim = col_emb_var.shape

        # Combine batch_size and num_cols_in_db into the first dimension, since this is what out model expects
        col_emb_var = col_emb_var.reshape(batch_size*num_cols_in_db, col_name_lens, embedding_dim) 
        col_name_len = col_name_len.reshape(-1)

        return self(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len)

    def loss(self, prediction, batch):
        loss = 0
        tokens_num_score, value_score = prediction
        tokens_num_truth, value_truth = batch['num_tokens'], batch['value']

        # These are mainly if the data hasn't been batched and "tensorified" yet
        if not isinstance(tokens_num_truth, torch.Tensor):
            tokens_num_truth = torch.tensor(tokens_num_truth).reshape(-1)

        if not isinstance(value_truth, torch.Tensor):
            value_truth = torch.tensor(value_truth).reshape(-1, 3)

        if len(tokens_num_score.shape) < 2:
            tokens_num_score = tokens_num_score.reshape(-1, tokens_num_score.size(0))
            value_score = value_score.reshape(-1, value_score.size(0))

        # TODO: lstm doesn't support float64, but bce_logit only supports float64, so we have to convert back and forth
        if value_score.dtype != torch.float64:
            value_score = value_score.double()
            tokens_num_score = tokens_num_score.double()
        tokens_num_truth = tokens_num_truth.to(tokens_num_score.device)-1
        value_truth = value_truth.to(value_score.device)

        mask = value_score != self.value_pad_token

        # Add cross entropy loss over the number of keywords
        loss += self.cross_entropy(tokens_num_score, tokens_num_truth.squeeze(1))

        # And binary cross entropy over the keywords predicted
        loss += self.bce_logit(value_score[mask], value_truth[mask])

        return loss

    def accuracy(self, prediction, batch):

        tokens_num_score, value_score = prediction
        tokens_num_truth, value_truth = batch['num_tokens'], batch['value']
        batch_size = len(value_truth)
        # These are mainly if the data hasn't been batched and "tensorified" yet
        if not isinstance(tokens_num_truth, torch.Tensor):
            tokens_num_truth = torch.tensor(tokens_num_truth).reshape(-1)

        if not isinstance(value_truth, torch.Tensor):
            value_truth = torch.tensor(value_truth).reshape(-1, 3)

        if len(tokens_num_score.shape) < 2:
            tokens_num_score = tokens_num_score.reshape(-1, tokens_num_score.size(0))
            value_score = value_score.reshape(-1, value_score.size(0))

        # TODO: lstm doesn't support float64, but bce_logit only supports float64, so we have to convert back and forth
        if value_score.dtype != torch.float64:
            value_score = value_score.double()
            tokens_num_score = tokens_num_score.double()
        tokens_num_truth = tokens_num_truth.to(tokens_num_score.device)
        value_truth = value_truth.to(value_score.device)

        # Predict the number of columns as the argmax of the scores
        tokens_num_prediction = torch.argmax(tokens_num_score, dim=1)
        accuracy_num = (tokens_num_prediction+1 == tokens_num_truth.squeeze(1)).sum().float()/batch_size

        accuracy_value = (torch.argmax(value_score, dim=1) == torch.argmax(value_truth, dim=1)).sum().float()/batch_size
        
        return accuracy_num.detach().cpu().numpy(), accuracy_value.detach().cpu().numpy()


    def predict(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, ban_prediction = None, int_mask = None):
        output = self.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len)

        num_tokens, values = output

        if ban_prediction is not None:
            num, val = ban_prediction
            values[:,val] = -math.inf

        if int_mask is not None:
            for i, b in enumerate(int_mask):
                if not b: values[:,i] = -math.inf
        
        num_tokens = torch.argmax(num_tokens, dim=1).detach().cpu().numpy() + 1
        token_start_idx = torch.argmax(values, dim=1).detach().cpu().numpy()
        
        return (num_tokens, token_start_idx)
