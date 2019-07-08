#from torch.utils.tensorboard import SummaryWriter
import pickle
import torch
from torch.optim import Adam
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataloader import SpiderDataset, try_tensor_collate_fn
from utils.utils import make_dirs, save_to_dirs, plot_from_dirs
from embedding.embeddings import GloveEmbedding
directory=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/logs/'

def train(model, train_dataloader, validation_dataloader, embedding, name="", num_epochs=1, lr=0.001):

    optimizer = Adam(model.parameters(), lr=lr)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create directory or remove and create if exists
    make_dirs(directory,name)

    embedding = embedding.to(device)
    train_loss_pickle, train_acc_pickle, train_num_train_pickle,iter_epoch =[],[],[],[]
    val_loss_pickle, val_acc_pickle, val_num_val_pickle =[],[],[]
    if device == torch.device('cuda'):
        model.cuda()

    for epoch in range(num_epochs):
        model.train()
        iter_epoch.append(epoch)
        train_loss = []
        accuracy_num_train = []
        accuracy_train = []
        predictions_train = []
        for i, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            prediction = model.process_batch(batch, embedding)

            loss = model.loss(prediction, batch)
            loss.backward()

            accuracy = model.accuracy(prediction, batch)

            optimizer.step()

            train_loss += [loss.detach().cpu().numpy()]
            #some of the models returns two accuracies
            if isinstance(accuracy, tuple):
                accuracy_num, accuracy = accuracy
                prediction_num, prediction = prediction
                accuracy_num_train += [accuracy_num.detach().cpu().numpy()]

            accuracy_train += [accuracy]
            predictions_train += [prediction.detach().cpu().numpy()]

        train_loss_pickle.append(np.mean(train_loss))
        train_acc_pickle.append(np.mean(accuracy_train)), train_num_train_pickle

        if len(accuracy_num_train)>0:
            train_num_train_pickle.append(np.mean(accuracy_num_train))


        model.eval()
        val_loss = []
        accuracy_num_val = []
        accuracy_val = []
        predictions_val = []
        for batch in iter(validation_dataloader):

            prediction = model.process_batch(batch, embedding)

            accuracy = model.accuracy(prediction, batch)

            val_loss += [loss.detach().cpu().numpy()]

            if isinstance(accuracy, tuple):
                accuracy_num, accuracy = accuracy
                predictions_num, prediction = prediction
                accuracy_num_val += [accuracy_num.detach().cpu().numpy()]

            accuracy_val += [accuracy]
            predictions_val += [prediction.detach().cpu().numpy()]


        val_loss_pickle.append(np.mean(val_loss))
        val_acc_pickle.append(np.mean(accuracy_val))

        if len(accuracy_num_train)>0:
            val_num_val_pickle.append(np.mean(accuracy_num_val))

        print(f"EPOCH {epoch}")

    #Save training and validation informaition to directory + {name} 
    info = [iter_epoch, train_loss_pickle, train_acc_pickle, train_num_train_pickle, val_loss_pickle, val_acc_pickle, val_num_val_pickle]
    save_to_dirs(directory,name,info) 
    #plot_from_dirs(directory,name,info)

     

if __name__ == '__main__':
    from keyword_predictor import KeyWordPredictor
    from col_predictor import ColPredictor
    from andor_predictor import AndOrPredictor
    from agg_predictor import AggPredictor
    from op_predictor import OpPredictor
    from having_predictor import HavingPredictor
    from desasc_limit_predictor import DesAscLimitPredictor
    from embedding.embeddings import GloveEmbedding        
    from utils.dataloader import  SpiderDataset, try_tensor_collate_fn
    from torch.utils.data import DataLoader
    import argparse
   
    emb = GloveEmbedding(path='/embedding/'+'glove.6B.300d.txt')
    spider_train = SpiderDataset(data_path='/tables/'+'train.json', tables_path='/tables/'+'tables.json', exclude_keywords=["between", "distinct", '-', ' / ', ' + '])
    spider_dev = SpiderDataset(data_path='/tables/'+'dev.json', tables_path='/tables/'+'tables.json', exclude_keywords=["between", "distinct", '-', ' / ', ' + '])
    
    #models= {'column':ColPredictor,'keyword':KeyWordPredictor,'andor':AndOrPredictor,'agg':AggPredictor,'op':OpPredictor,'having':HavingPredictor,'desasc':DesAscLimitPredictor}

	
	
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs',  default=3, type=int)
    parser.add_argument('--batch_size', default=248, type=int)
    parser.add_argument('--name_postfix',default='', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--N_word', default=300, type=int)
    parser.add_argument('--hidden_dim', default=30, type=int)
    parser.add_argument('--model', choices=['column','keyword','andor','agg','op','having','desasc'], default='having')
    args = parser.parse_args()
	
    #if args.model in models.keys():
    #doesn't work unfortunately
    #    model=models[args.model](N_word=args.N_word, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
    #    train_set = spider_train.generate_column_dataset()
    #    validation_set = spider_dev.generate_column_dataset()

    if args.model == 'column':
        model = ColPredictor(N_word=args.N_word, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_column_dataset()
        validation_set = spider_dev.generate_column_dataset()
    
    elif args.model == 'keyword':
        model = KeyWordPredictor(N_word=args.N_word, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_keyword_dataset()
        validation_set = spider_dev.generate_keyword_dataset()
        
    elif args.model == 'andor':
        model = AndOrPredictor(N_word=args.N_word, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_andor_dataset()
        validation_set = spider_dev.generate_andor_dataset()

    elif args.model == 'agg':
        model = AggPredictor(N_word=args.N_word, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_agg_dataset()
        validation_set = spider_dev.generate_agg_dataset()
    
    elif args.model == 'op':
        model = OpPredictor(N_word=args.N_word, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_op_dataset()
        validation_set = spider_dev.generate_op_dataset()
                        
    elif args.model == 'having':
        model = HavingPredictor(N_word=args.N_word, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_having_dataset()
        validation_set = spider_dev.generate_having_dataset()

    elif args.model == 'desasc':
        model = DesAscLimitPredictor(N_word=args.N_word, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_desasc_dataset()
        validation_set = spider_dev.generate_desasc_dataset()
        
        
    dl_train = DataLoader(train_set, batch_size=args.batch_size, collate_fn=try_tensor_collate_fn)
    dl_validation = DataLoader(validation_set, batch_size=len(validation_set), collate_fn=try_tensor_collate_fn)

    train(model, dl_train, dl_validation, emb, 
            name=f'{args.model}__num_layers={args.num_layers}__lr={args.lr}__epoch={args.num_epochs}__{args.name_postfix}', 
            num_epochs=args.num_epochs,
            lr=args.lr)
