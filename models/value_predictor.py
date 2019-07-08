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

class ValuePredictor(nn.Module):
    def __init__(self, N_word, hidden_dim, num_layers, gpu=True, use_hs=True, max_num_cols=6):
        super(ValuePredictor, self).__init__()
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

        #### value ####
        #TODO: the model is just adaptation of the ColumnPredictor. Could one use pointer network instead?
        self.col_q = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=False)
        self.hs_q = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=False)
        self.W_q = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.value_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))

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
            col_idx [batch_size]: Index of the column which we are predicting the op for 
        Returns:
            value [batch_size, question_seq_len] : probability distribution over the tokens in the question
        """
        batch_size = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = col_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)

        #Get encoding of the column we are prediting the value for
        col_emb = col_enc[np.arange(batch_size),col_idx].unsqueeze(1) # [batch_size, 1, hidden_dim]
        #even though the col encoding is just a sequence of length 1, we need to lengths for masking
        col_len = [1]*batch_size

        # Run conditional encoding for column|question, and history|question 
        H_col_q = self.col_q(col_emb, q_enc, col_len, q_len) #[batch_size, question_seq_len, hidden_dim]
        H_hs_q = self.hs_q(hs_enc,q_enc, hs_len, q_len) #[batch_size, question_seq_len, hidden_dim]
        H_q = self.W_q(q_enc) #[batch_size, question_seq_len, hidden_dim]

        value = self.value_out(H_col_q + int(self.use_hs)*H_hs_q +H_q) # [batch_size, question_seq_len, 1]


        q_mask = length_to_mask(q_len)
        #number of values might be different for each question, so we need to mask some of them
        value = value.masked_fill_(q_mask, 0)
        value = value.squeeze()
        return value
        

    def loss(self, score, truth):
        #here suppose truth looks like [[[1, 4], 3], [], ...]
        loss = 0
        B = len(truth)
        col_num_score, col_score = score
        #loss for the column number
        truth_num = [len(t) - 1 for t in truth] # double check truth format and for test cases
        data = torch.from_numpy(np.array(truth_num))
        truth_num_var = Variable(data.cuda())
        loss += self.CE(col_num_score, truth_num_var)
        #loss for the key words
        T = len(col_score[0])
        # print("T {}".format(T))
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
        # print("data {}".format(data))
        # print("data {}".format(data.cuda()))
        truth_var = Variable(data.cuda())
        #loss += self.mlsml(col_score, truth_var)
        #loss += self.bce_logit(col_score, truth_var) # double check no sigmoid
        pred_prob = self.sigm(col_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss


    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        col_num_score, col_score = [x.data.cpu().numpy() for x in score]
        for b in range(B):
            cur_pred = {}
            col_num = np.argmax(col_num_score[b]) + 1 #double check
            cur_pred['col_num'] = col_num
            cur_pred['col'] = np.argsort(-col_score[b])[:col_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            col_num, col = p['col_num'], p['col']
            flag = True
            if col_num != len(t): # double check truth format and for test cases
                num_err += 1
                flag = False
            #to eval col predicts, if the gold sql has JOIN and foreign key col, then both fks are acceptable
            fk_list = []
            regular = []
            for l in t:
                if isinstance(l, list):
                    fk_list.append(l)
                else:
                    regular.append(l)

            if flag: #double check
                for c in col:
                    for fk in fk_list:
                        if c in fk:
                            fk_list.remove(fk)
                    for r in regular:
                        if c == r:
                            regular.remove(r)

                if len(fk_list) != 0 or len(regular) != 0:
                    err += 1
                    flag = False

            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))



if __name__ == '__main__':

    q_emb_var = torch.rand(3,10,30)
    q_len = np.array([8,7,10])
    hs_emb_var = torch.rand(3,5,30)
    hs_len = np.array([4,2,5])
    col_emb_var = torch.rand(3*20,4,30)
    col_len = np.array([7,5,6])
    col_name_len = np.array([2,2,3,2,2,4,3,2,1,2,2,3,1,2,3,4,4,4])
    col_idx = np.array([4,2,3])

    pred = ValuePredictor(N_word=30, hidden_dim=30, num_layers=2, gpu=False)
    print(pred(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx))