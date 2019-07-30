import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import SpiderDataset, try_tensor_collate_fn
from utils.data_augmentation import AugmentedSpiderDataset
from embedding.embeddings import GloveEmbedding#, FastTextEmbedding
from models import model_list
import argparse

def train(model, train_dataloader, validation_dataloader, embedding, name="", num_epochs=1, lr=0.001, save=False):
    train_writer = SummaryWriter(log_dir=f'logs/{name}_train')
    val_writer = SummaryWriter(log_dir=f'logs/{name}_val')
    optimizer = Adam(model.parameters(), lr=lr)

    best_loss = float('inf')    

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = []
        accuracy_num_train = []
        accuracy_rep_train = []
        accuracy_train = []
        predictions_train = []
        for _, batch in enumerate(train_dataloader):

            # Backpropagate and compute accuracy
            optimizer.zero_grad()
            prediction = model.process_batch(batch, embedding)
            loss = model.loss(prediction, batch)
            loss.backward()
            accuracy = model.accuracy(prediction, batch)
            optimizer.step()
            train_loss += [loss.detach().cpu().numpy()]

            # Some models return two accuracies
            if isinstance(accuracy, tuple):
                try:
                    accuracy_num, accuracy_rep, accuracy = accuracy
                    _, _, prediction = prediction
                    accuracy_rep_train += [accuracy_rep]
                except:
                    accuracy_num, accuracy = accuracy
                    _, prediction = prediction
                accuracy_num_train += [accuracy_num]

            accuracy_train += [accuracy]

            preds = torch.argsort(-prediction)
            if len(preds.shape) > 1: preds = preds[:,0]
            predictions_train += list(preds.detach().cpu().numpy())

        train_writer.add_scalar('loss', np.mean(train_loss), epoch)
        train_writer.add_scalar('accuracy', np.mean(accuracy_train), epoch)
        train_writer.add_histogram('predictions',np.asarray(predictions_train), epoch)

        if len(accuracy_num_train)>0:
            train_writer.add_scalar('accuracy_num', np.mean(accuracy_num_train), epoch)
        if len(accuracy_rep_train)>0:
            train_writer.add_scalar('accuracy_rep', np.mean(accuracy_rep_train), epoch)

        # Compute validation accuracy and loss
        model.eval()
        val_loss = []
        accuracy_num_val = []
        accuracy_rep_val = []
        accuracy_val = []
        predictions_val = []
        for batch in iter(validation_dataloader):
            with torch.no_grad():    
                prediction = model.process_batch(batch, embedding)
                accuracy = model.accuracy(prediction, batch)
                val_loss += [loss.detach().cpu().numpy()]

                if isinstance(accuracy, tuple):
                    try:
                        accuracy_num, accuracy_rep, accuracy = accuracy
                        _, _, prediction = prediction
                        accuracy_rep_val += [accuracy_rep]
                    except:
                        accuracy_num, accuracy = accuracy
                        _, prediction = prediction
                    accuracy_num_val += [accuracy_num]

                accuracy_val += [accuracy]

                preds = torch.argsort(-prediction)
                if len(preds.shape) > 1: preds = preds[:,0]
                predictions_val += list(preds.detach().cpu().numpy())

        val_writer.add_scalar('loss', np.mean(val_loss), epoch)
        val_writer.add_scalar('accuracy', np.mean(accuracy_val), epoch)
        val_writer.add_histogram('predictions',np.asarray(predictions_val), epoch)
        if len(accuracy_num_val)>0:
            val_writer.add_scalar('accuracy_num', np.mean(accuracy_num_val), epoch)
        if len(accuracy_rep_val)>0:
            val_writer.add_scalar('accuracy_rep', np.mean(accuracy_rep_val), epoch)

        if save and np.mean(val_loss)<best_loss:
            torch.save(model.state_dict(), f"saved_models/{name}.pt")
            best_loss = np.mean(val_loss)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=2, type=int, help='Number of layers in the LSTMs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learnign rate')
    parser.add_argument('--num_epochs',  default=300, type=int, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--name_postfix',default='', type=str, help='Optional postfix of the model name')
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--save', default=True, type=bool,help='Save the model during training')
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--embedding_dim',default=300, type=int, help='Dimension of the embeddings')
    parser.add_argument('--num_augmentation', default=10000, type=int, help='Number of additional augmented questions to generate')
    parser.add_argument('--N_word',default=6, type=int, help='Number of trained tokens for the embedding, this just corresponds to the name')
    parser.add_argument('--model', choices=list(model_list.models.keys()), default='distinct')
    args = parser.parse_args()

    # Models with 100% validation accuracy:
    # andor, having

    # Load training and validation sets
    spider_train = AugmentedSpiderDataset(data_path='data/train.json', tables_path='/data/tables.json', aug_data_path='/data/train_augment.json', aug_tables_path='/data/wikisql_tables.json', exclude_keywords=[ '-', ' / ', ' + '], max_count=args.num_augmentation)
    spider_dev = SpiderDataset(data_path='data/dev.json', tables_path='/data/tables.json', exclude_keywords=[ '-', ' / ', ' + '])

    # Load pre-trained embeddings and dataset
    emb = GloveEmbedding(path='data/'+f'glove.{args.N_word}B.{args.embedding_dim}d.txt', gpu=args.gpu, embedding_dim=args.embedding_dim)

    # Select appropriate model to train
    model = model_list.models[args.model](N_word=args.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.gpu)

    # Generate appropriate datasets
    func_name = 'generate_' + args.model + '_dataset'
    train_set = getattr(spider_train, func_name)()
    validation_set = getattr(spider_dev, func_name)()

    # Initialize data loaders
    dl_train = DataLoader(train_set, batch_size=args.batch_size, collate_fn=try_tensor_collate_fn, shuffle=True)
    dl_validation = DataLoader(validation_set, batch_size=args.batch_size, collate_fn=try_tensor_collate_fn, shuffle=True)

    # Train our model
    train(model, dl_train, dl_validation, emb, 
            name=f'{args.model}__num_layers={args.num_layers}__lr={args.lr}__dropout={args.dropout}__batch_size={args.batch_size}__embedding_dim={args.embedding_dim}__hidden_dim={args.hidden_dim}__epoch={args.num_epochs}__num_augmentation={args.num_augmentation}__{args.name_postfix}', 
            num_epochs=args.num_epochs,
            lr=args.lr,
            save=args.save)
