import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import BagOfWord

class AndOrPredictor(nn.Module):
    def __init__(self, N_word, hidden_dim, num_layers, gpu=True, use_hs=True):
        super(AndOrPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.gpu = gpu
        self.use_hs = use_hs

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

        self.cross_entropy = nn.CrossEntropyLoss()
        if gpu:
            self.cuda()

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

        #Calculate H_Q using Bag Of Word trick
        H_Q = self.bag_of_word(q_enc, q_len) #[batch_size, hidden_dim]
        #project H_Q
        H_Q = self.W_q(H_Q) #[batch_size, hidden_dim]

        #Do the same for the history
        H_HS = self.bag_of_word(hs_enc, hs_len) # [batch_size, 1, hidden_dim]
        H_HS = self.W_hs(H_HS) # [batch_size, hidden_dim]

        p_andor = self.ao_out(H_Q + int(self.use_hs)*H_HS) # [batch_size, 2]

        return p_andor


    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])


        andor = self(q_emb_var, q_len, hs_emb_var, hs_len)
        return andor

    def loss(self, andor_prediction, batch):
        
        andor_truth = batch['andor']

        andor_truth = andor_truth.to(andor_prediction.device) 

        #Add cross entropy loss over the number of keywords
        loss = self.cross_entropy(andor_prediction, andor_truth.long().squeeze())
        
        return loss



    def accuracy(self, andor_prediction, batch):

        andor_truth =  batch['andor']
        batch_size = len(andor_truth)
        
        andor_truth = andor_truth.to(andor_prediction.device).squeeze().long() 

        #predict the number of columns as the argmax of the scores
        andor_prediction = torch.argmax(andor_prediction, dim=1)
        
        
        accuracy = (andor_prediction==andor_truth).sum().float()/batch_size
        
        return accuracy.detach().cpu().numpy()

        
    def check_acc(self, score, truth):
        err = 0
        B = len(score)
        pred = []
        for b in range(B):
            pred.append(np.argmax(score[b].data.cpu().numpy()))
        for b, (p, t) in enumerate(zip(pred, truth)):
            if p != t:
                err += 1

        return err


if __name__ == '__main__':
    q_emb = torch.rand(3,10,30)
    q_len = np.asarray([8,6,10])
    hs_emb = torch.rand(3,5,30)
    hs_len = np.asarray([4,3,5])

    andor = AndOrPredictor(N_word=30, hidden_dim=30, num_layers=2, gpu=False)
    andor(q_emb, q_len, hs_emb, hs_len)