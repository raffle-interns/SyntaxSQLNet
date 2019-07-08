import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention
from utils.utils import length_to_mask

class ColPredictor(nn.Module):
    def __init__(self, N_word, hidden_dim, num_layers, gpu=True, use_hs=True, max_num_cols=6):
        super(ColPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.gpu = gpu
        self.use_hs = use_hs

        self.q_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                              num_layers=num_layers, batch_first=True,
                              dropout=0.3, bidirectional=True)

        self.hs_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                               num_layers=num_layers, batch_first=True,
                               dropout=0.3, bidirectional=True)

        self.col_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                                num_layers=num_layers, batch_first=True,
                                dropout=0.3, bidirectional=True)

        #### Number of columns ####

        self.q_col_num = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=True)
        self.hs_col_num = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=True)

        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, max_num_cols))  # num of cols: 1-6

        #### Columns ####
        self.q_col = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=False)
        self.hs_col = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=False)
        self.W_col = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))

        self.cross_entropy = nn.CrossEntropyLoss()
        pos_weight=300*torch.tensor(1).double()
        if gpu:
            self.cuda()
            pos_weight = pos_weight.cuda()
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
        # Predict number of columns #
        #############################

        # Run conditional encoding for question|column, and history|column
        H_q_col = self.q_col_num(q_enc, col_enc, q_len, col_len)  # [batch_size, hidden_dim]
        H_hs_col = self.hs_col_num(hs_enc, col_enc, hs_len, col_len)  # [batch_size, hidden_dim]

        num_cols = self.col_num_out(H_q_col + int(self.use_hs)*H_hs_col)

        ###################
        # Predict columns #
        ###################

        # Run conditional encoding for question|column, and history|column
        H_q_col = self.q_col(q_enc, col_enc, q_len, col_len)  # [batch_size, num_cols_in_db, hidden_dim]
        H_hs_col = self.hs_col(hs_enc, col_enc, hs_len, col_len)  # [batch_size, num_cols_in_db, hidden_dim]
        H_col = self.W_col(col_enc)  # [batch_size, num_cols_in_db, hidden_dim]

        cols = self.col_out(H_q_col + int(self.use_hs)*H_hs_col + H_col).squeeze()  # [batch_size, num_cols_in_db]

        col_mask = length_to_mask(col_len).squeeze().to(cols.device)
        # number of columns might be different for each db, so we need to mask some of them
        cols = cols.masked_fill_(col_mask, 0)

        return (num_cols, cols)


    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])
        batch_size = len(q_len)
        #kw_emb_var, kw_len = embedding.get_history_emb(batch_size*[['where', 'order by', 'group by']])
        col_emb_var, col_len, col_name_len = embedding.get_columns_emb(batch['columns_all'])

        batch_size, num_cols_in_db, col_name_lens, embedding_dim = col_emb_var.shape
        #Combine batch_size and num_cols_in_db into the first dimension, since this is what out model expects 
        col_emb_var = col_emb_var.reshape(batch_size*num_cols_in_db, col_name_lens, embedding_dim) 
        col_name_len = col_name_len.reshape(-1)

        num_cols, cols = self(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len )

        return (num_cols, cols)    

    def loss(self, prediction, batch):
        loss = 0
        col_num_score, col_score = prediction
        col_num_truth, col_truth = batch['num_columns'], batch['columns']

        # These are mainly if the data hasn't been batched and "tensorified" yet
        if not isinstance(col_num_truth, torch.Tensor):
            col_num_truth = torch.tensor(col_num_truth).reshape(-1)

        if not isinstance(col_truth, torch.Tensor):
            col_truth = torch.tensor(col_truth).reshape(-1, 3)

        if len(col_num_score.shape) < 2:
            col_num_score = col_num_score.reshape(-1, 4)
            col_score = col_score.reshape(-1, 3)

        # TODO:lstm doesn't support float64, but bce_logit only supports float64, so we have to convert back and forth
        if col_score.dtype != torch.float64:
            col_score = col_score.double()
            col_num_score = col_num_score.double()
        col_num_truth = col_num_truth.to(col_num_score.device)
        col_truth = col_truth.to(col_score.device)
        # Add cross entropy loss over the number of keywords
        loss += self.cross_entropy(col_num_score, col_num_truth)
        # And binary cross entropy over the keywords predicted
        loss += self.bce_logit(col_score, col_truth)

        return loss

    def accuracy(self, prediction, batch):

        col_num_score, col_score = prediction
        col_num_truth, col_truth = batch['num_columns'], batch['columns']
        batch_size = len(col_truth)
        # These are mainly if the data hasn't been batched and "tensorified" yet
        if not isinstance(col_num_truth, torch.Tensor):
            col_num_truth = torch.tensor(col_num_truth).reshape(-1)

        if not isinstance(col_truth, torch.Tensor):
            col_truth = torch.tensor(col_truth).reshape(-1, 3)

        if len(col_num_score.shape) < 2:
            col_num_score = col_num_score.reshape(-1, 4)
            col_score = col_score.reshape(-1, 3)

        # TODO:lstm doesn't support float64, but bce_logit only supports float64, so we have to convert back and forth
        if col_score.dtype != torch.float64:
            col_score = col_score.double()
            col_num_score = col_num_score.double()
        col_num_truth = col_num_truth.to(col_num_score.device)
        col_truth = col_truth.to(col_score.device)

        # predict the number of columns as the argmax of the scores
        kw_num_prediction = torch.argmax(col_num_score, dim=1)
        accuracy_num = (kw_num_prediction+1 == col_num_truth).sum().float()/batch_size

        correct_keywords = 0
        for i in range(batch_size):
            num_kw = col_num_truth[i]
            # Compare the set of predicted keywords with target.
            # This should eliminate any ordering issues
            correct_keywords += set(torch.argsort(-col_truth[i, :])[:num_kw].cpu().numpy()) == set(torch.argsort(-col_score[i, :])[:num_kw].cpu().numpy())
        accuracy_kw = correct_keywords/batch_size

        return accuracy_num, accuracy_kw


if __name__ == '__main__':

    q_emb_var = torch.rand(3, 10, 30)
    q_len = np.array([8, 7, 10])
    hs_emb_var = torch.rand(3, 5, 30)
    hs_len = np.array([4, 2, 5])
    col_emb_var = torch.rand(3*20, 4, 30)
    col_len = np.array([7, 5, 6])
    col_name_len = np.array([2, 2, 3, 2, 2, 4, 3, 2, 1, 2, 2, 3, 1, 2, 3, 4, 4, 4, 2, 3, 2, 2, 4, 3, 2, 3, 2, 2, 4, 3,2, 2, 3, 2, 2, 4, 3, 2, 1, 2, 2, 3, 1, 2, 3, 4, 4, 4, 2, 3, 2, 2, 4, 3, 2, 3, 2, 2, 4, 3])

    colpred = ColPredictor(N_word=30, hidden_dim=30, num_layers=2, gpu=False)
    colpred(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len)
