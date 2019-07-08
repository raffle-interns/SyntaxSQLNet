from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from utils.dataloader import SpiderDataset, try_tensor_collate_fn
from embedding.embeddings import GloveEmbedding


def train(model, train_dataloader, validation_dataloader, embedding, name="", num_epochs=1, lr=0.001):

    train_writer = SummaryWriter(log_dir=f'logs/{name}_train')
    val_writer = SummaryWriter(log_dir=f'logs/{name}_val')
    optimizer = Adam(model.parameters(), lr=lr)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    embedding = embedding.to(device)

    if device == torch.device('cuda'):
        model.cuda()

    for epoch in tqdm(range(num_epochs)):
        model.train()
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

        train_writer.add_scalar('loss', np.mean(train_loss), epoch)
        train_writer.add_scalar('accuracy', np.mean(accuracy_train), epoch)
        #train_writer.add_histogram('predictions',predictions_train, epoch)

        if len(accuracy_num_train)>0:
            train_writer.add_scalar('accuracy_num', np.mean(accuracy_num_train), epoch)


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


        val_writer.add_scalar('loss', np.mean(val_loss), epoch)

        val_writer.add_scalar('accuracy', np.mean(accuracy_val), epoch)
        
        if len(accuracy_num_train)>0:
            val_writer.add_scalar('accuracy_num', np.mean(accuracy_num_val), epoch)


if __name__ == '__main__':
    from models.keyword_predictor import KeyWordPredictor
    from models.col_predictor import ColPredictor
    from models.andor_predictor import AndOrPredictor
    from models.agg_predictor import AggPredictor
    from models.op_predictor import OpPredictor
    from models.having_predictor import HavingPredictor
    from models.desasc_limit_predictor import DesAscLimitPredictor
    from embedding.embeddings import GloveEmbedding        
    from utils.dataloader import  SpiderDataset, try_tensor_collate_fn
    from torch.utils.data import DataLoader
    import argparse
   
    emb = GloveEmbedding(path='data/'+'glove.6B.50d.txt')
    spider_train = SpiderDataset(data_path='data/'+'train.json', tables_path='/data/'+'tables.json', exclude_keywords=["between", "distinct", '-', ' / ', ' + '])
    spider_dev = SpiderDataset(data_path='data/'+'dev.json', tables_path='/data/'+'tables.json', exclude_keywords=["between", "distinct", '-', ' / ', ' + '])
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs',  default=3, type=int)
    parser.add_argument('--batch_size', default=248, type=int)
    parser.add_argument('--name_postfix',default='', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--hidden_dim', default=30, type=int)
    parser.add_argument('--model', choices=['column','keyword','andor','agg','op','having','desasc'], default='having')
    args = parser.parse_args()
	
    #if args.model in models.keys():
    #doesn't work unfortunately
    #    model=models[args.model](N_word=emb.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
    #    train_set = spider_train.generate_column_dataset()
    #    validation_set = spider_dev.generate_column_dataset()

    if args.model == 'column':
        model = ColPredictor(N_word=emb.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_column_dataset()
        validation_set = spider_dev.generate_column_dataset()
    
    elif args.model == 'keyword':
        model = KeyWordPredictor(N_word=emb.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_keyword_dataset()
        validation_set = spider_dev.generate_keyword_dataset()
        
    elif args.model == 'andor':
        model = AndOrPredictor(N_word=emb.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_andor_dataset()
        validation_set = spider_dev.generate_andor_dataset()

    elif args.model == 'agg':
        model = AggPredictor(N_word=emb.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_agg_dataset()
        validation_set = spider_dev.generate_agg_dataset()
    
    elif args.model == 'op':
        model = OpPredictor(N_word=emb.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_op_dataset()
        validation_set = spider_dev.generate_op_dataset()
                        
    elif args.model == 'having':
        model = HavingPredictor(N_word=emb.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_having_dataset()
        validation_set = spider_dev.generate_having_dataset()

    elif args.model == 'desasc':
        model = DesAscLimitPredictor(N_word=emb.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_desasc_dataset()
        validation_set = spider_dev.generate_desasc_dataset()
        
        
    dl_train = DataLoader(train_set, batch_size=args.batch_size, collate_fn=try_tensor_collate_fn)
    dl_validation = DataLoader(validation_set, batch_size=len(validation_set), collate_fn=try_tensor_collate_fn)

    train(model, dl_train, dl_validation, emb, 
            name=f'{args.model}__num_layers={args.num_layers}__lr={args.lr}__epoch={args.num_epochs}__{args.name_postfix}', 
            num_epochs=args.num_epochs,
            lr=args.lr)
