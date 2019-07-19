import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention
from utils.dataloader import SpiderDataset
from models.base_predictor import BasePredictor

class KeyWordPredictor(BasePredictor):
    """
    Predict if the next token is (SQL key words):
    WHERE, GROUP BY, ORDER BY. excluding SELECT (it is a must)
    """
    def __init__(self, *args, **kwargs):
        super(KeyWordPredictor, self).__init__(*args, **kwargs)

    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):
        self.q_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                num_layers=num_layers, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                num_layers=num_layers, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.kw_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2,
                num_layers=num_layers, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_kw_num = ConditionalAttention(hidden_dim, use_bag_of_word=True)
        self.hs_kw_num = ConditionalAttention(hidden_dim, use_bag_of_word=True)
        self.kw_num_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 4)) # num of key words: 0-3

        self.q_kw = ConditionalAttention(hidden_dim, use_bag_of_word=False)
        self.hs_kw = ConditionalAttention(hidden_dim, use_bag_of_word=False)
        self.W_kw = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.kw_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))

        # TODO: Where does this 3 number come from? number of classes? 
        # Answer: pos_weight is a number that indicates how to balance positive to negative examples of a class
        # eg. for 1 class with 1 postive and 3 negative, set pos_weight to 3 such that the loss acts as if there where 3 positive examples
        self.bce_logit = nn.BCEWithLogitsLoss(pos_weight=3*torch.tensor(3).cuda().double())

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, kw_len):
        """
        Args:
            q_emb_var [batch_size, question_seq_len, embedding_dim] : embedding of question
            q_len [batch_size] : lengths of questions
            hs_emb_var [batch_size, history_seq_len, embedding_dim] : embedding of history
            hs_len [batch_size] : lengths of history
            kw_emb_var [batch_size, kw_name_len, embedding_dim] : embedding of the keywords {where, groupby, orderby}
            kw_len [batch_size] : lengths of the kw embeddings
        Returns:
            num_kw [batch_size, 4]
            kw [batch_size, 3] : probability distribution over {WHERE, GROUPBY, ORDERBY}
        """
        q_enc,_ = self.q_lstm(q_emb_var, q_len)
        hs_enc,_ = self.hs_lstm(hs_emb_var, hs_len)
        kw_enc,_ = self.kw_lstm(kw_emb_var, kw_len)

        H_q_kw_num = self.q_kw_num(q_enc, kw_enc, q_len, kw_len) # [batch_size, hidden_dim]
        H_hs_kw_num = self.hs_kw_num(hs_enc, kw_enc, hs_len, kw_len) # [batch_size, hidden_dim]
        num_kw = self.kw_num_out(H_q_kw_num + int(self.use_hs)*H_hs_kw_num) # [batch_size, hidden_dim]

        H_q_kw = self.q_kw(q_enc, kw_enc, q_len, kw_len) #[batch_size, num_keywords, hidden_dim]
        H_hs_kw = self.hs_kw(hs_enc, kw_enc, hs_len, kw_len) #[batch_size, num_keywords, hidden_dim]
        H_kw = self.W_kw(kw_enc) #[batch_size, num_keywords, hidden_dim]
        kw = self.kw_out(H_q_kw + int(self.use_hs)*H_hs_kw + H_kw).squeeze(2) # [batch_size, num_keywords]

        return (num_kw, kw)

    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])
        batch_size = len(q_len)
        kw_emb_var, kw_len = embedding.get_history_emb(batch_size*[['where', 'order by', 'group by']])

        return self(q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, kw_len)

    def loss(self, prediction, batch):
        loss = 0
        kw_num_score, kw_score = prediction
        kw_num_truth, kw_truth = batch['num_keywords'], batch['keywords']

        # These are mainly if the data hasn't been batched and "tensorified" yet
        if not isinstance(kw_num_truth, torch.Tensor):
            kw_num_truth = torch.tensor(kw_num_truth).reshape(-1)
        
        if not isinstance(kw_truth, torch.Tensor):
            kw_truth = torch.tensor(kw_truth).reshape(-1,3) 

        if len(kw_num_score.shape)<2:
            kw_num_score = kw_num_score.reshape(-1,4)
            kw_score = kw_score.reshape(-1,3)
        
        # TODO: lstm doesn't support float64, but bce_logit only supports float64, so we have to convert back and forth
        if kw_score.dtype != torch.float64:
            kw_score = kw_score.double()
            kw_num_score = kw_num_score.double()
        kw_num_truth = kw_num_truth.to(kw_num_score.device) 
        kw_truth = kw_truth.to(kw_score.device)

        # Add cross entropy loss over the number of keywords
        loss += self.cross_entropy(kw_num_score, kw_num_truth)

        # And binary cross entropy over the keywords predicted
        loss += self.bce_logit(kw_score, kw_truth)
        
        return loss

    def accuracy(self, prediction, batch):     
        kw_num_score, kw_score = prediction
        kw_num_truth, kw_truth =  batch['num_keywords'], batch['keywords']
        batch_size = len(kw_truth)
        # These are mainly if the data hasn't been batched and "tensorified" yet
        if not isinstance(kw_num_truth, torch.Tensor):
            kw_num_truth = torch.tensor(kw_num_truth).reshape(-1)
        
        if not isinstance(kw_truth, torch.Tensor):
            kw_truth = torch.tensor(kw_truth).reshape(-1,3) 

        if len(kw_num_score.shape)<2:
            kw_num_score = kw_num_score.reshape(-1,4)
            kw_score = kw_score.reshape(-1,3)
        
        # TODO: lstm doesn't support float64, but bce_logit only supports float64, so we have to convert back and forth
        if kw_score.dtype != torch.float64:
            kw_score = kw_score.double()
            kw_num_score = kw_num_score.double()
        kw_num_truth = kw_num_truth.to(kw_num_score.device) 
        kw_truth = kw_truth.to(kw_score.device) 

        # Predict the number of columns as the argmax of the scores
        kw_num_prediction = torch.argmax(kw_num_score, dim=1)
        accuracy_num = (kw_num_prediction==kw_num_truth).sum().float()/batch_size

        correct_keywords = 0
        for i in range(batch_size):
            num_kw = kw_num_truth[i]

            # Compare the set of predicted keywords with target. 
            # This should eliminate any ordering issues
            correct_keywords += set(torch.argsort(-kw_truth[i,:])[:num_kw].cpu().numpy()) == set(torch.argsort(-kw_score[i,:])[:num_kw].cpu().numpy())
        accuracy_kw = correct_keywords/batch_size

        return accuracy_num.detach().cpu().numpy(), accuracy_kw
