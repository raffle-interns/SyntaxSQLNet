import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention
from models import model_list

class BasePredictor(nn.Module):
    def __init__(self, N_word, hidden_dim, num_layers, gpu=True, use_hs=True):
        super(BasePredictor, self).__init__()
        self.N_word = N_word
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gpu = gpu
        self.use_hs = use_hs
        self.name = model_list.models_inverse[self.__class__.__name__]
        self.cross_entropy = nn.CrossEntropyLoss()
        self.construct(N_word, hidden_dim, num_layers, gpu, use_hs)
        if gpu: self.cuda()

    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):
        pass
    
    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx):
        pass

    def process_batch(self, batch, embedding):
        pass

    def loss(self, prediction, batch):
        print(self.__class__.__name__)
        truth = batch[self.name].to(prediction.device)
        return self.cross_entropy(prediction, truth.long().squeeze())

    def accuracy(self, prediction, batch):
        truth =  batch[self.name]
        batch_size = len(truth)
        truth = truth.to(prediction.device).squeeze().long()

        # Predict number of columns as the argmax of the scores
        prediction = torch.argmax(prediction, dim=1)
        
        # Compute accuracy
        accuracy = (prediction==truth).sum().float()/batch_size
        
        return accuracy.detach().cpu().numpy()
