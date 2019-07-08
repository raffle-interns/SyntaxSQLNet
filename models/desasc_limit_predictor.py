import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention

class DesAscLimitPredictor(nn.Module):
    def __init__(self, N_word, hidden_dim, num_layers, gpu=True, use_hs=True):
        super(DesAscLimitPredictor, self).__init__()
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


        self.q_cs = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=True)
        self.hs_cs = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=True)
        self.W_cs = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)

        self.dal_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 6)) #for 5 desc/asc limit/none combinations

        self.cross_entropy = nn.CrossEntropyLoss()
        if gpu:
            self.cuda()

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
            dal [batch_size, 5] : probability distribution over {none, asc, desc, asc limit, desc limit}
        """
        
        batch_size = len(col_len)

        q_enc,_ = self.q_lstm(q_emb_var, q_len)  # [batch_size, question_seq_len, hidden_dim]
        hs_enc,_ = self.hs_lstm(hs_emb_var, hs_len)  # [batch_size, history_seq_len, hidden_dim]
        _, col_enc = self.col_lstm(col_emb_var, col_name_len) # [batch_size*num_cols_in_db, hidden_dim]
        col_enc = col_enc.reshape(batch_size, col_len.max(), self.hidden_dim) # [batch_size, num_cols_in_db, hidden_dim]

        #Get encoding of the column we are prediting for
        col_emb = col_enc[np.arange(batch_size),col_idx].unsqueeze(1) # [batch_size, 1, hidden_dim]

        H_q_cs = self.q_cs(q_enc, col_emb, q_len) # [batch_size, hidden_dim]
        H_hs_cs = self.hs_cs(hs_enc, col_emb, hs_len) # [batch_size, hidden_dim]
        H_cs = self.W_cs(col_emb).squeeze() # [batch_size, hidden_dim]
        dal = self.dal_out(H_q_cs + int(self.use_hs)*H_hs_cs + H_cs) # [batch_size, 4]

        return dal

    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])
        #get the index of the column we are predicting for
        col_idx = batch['column_idx']

        col_emb_var, col_len, col_name_len = embedding.get_columns_emb(batch['columns_all'])

        batch_size, num_cols_in_db, col_name_lens, embedding_dim = col_emb_var.shape
        #Combine batch_size and num_cols_in_db into the first dimension, since this is what out model expects 
        col_emb_var = col_emb_var.reshape(batch_size*num_cols_in_db, col_name_lens, embedding_dim) 
        col_name_len = col_name_len.reshape(-1)

        dal = self(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx )

        return dal  

    def loss(self, prediction, batch):

        truth = batch['desasc']

        truth = truth.to(prediction.device)

        # Add cross entropy loss over the number of keywords
        loss = self.cross_entropy(prediction, truth.long().squeeze())

        return loss



    def accuracy(self, prediction, batch):

        truth =  batch['desasc']
        batch_size = len(truth)
        
        truth = truth.to(prediction.device).squeeze().long() 

        #predict the number of columns as the argmax of the scores
        prediction = torch.argmax(prediction, dim=1)
        
        
        accuracy = (prediction==truth).sum().float()/batch_size
        
        return accuracy.detach().cpu().numpy()




if __name__ == '__main__':

    q_emb_var = torch.rand(3,10,30)
    q_len = np.array([8,7,10])
    hs_emb_var = torch.rand(3,5,30)
    hs_len = np.array([4,2,5])
    col_emb_var = torch.rand(3*20,4,30)
    col_len = np.array([7,5,6])
    col_name_len = np.array([2,2,3,2,2,4,3,2,1,2,2,3,1,2,3,4,4,4])
    col_idx = np.array([4,2,3])

    pred = DesAscLimitPredictor(N_word=30, hidden_dim=30, num_layers=2, gpu=False)
    print(pred(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx))
