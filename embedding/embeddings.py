import numpy as np
import torch
from torch.nn import Embedding, Module
from nltk.tokenize import word_tokenize
import os
from something.datascience.transforms.embeddings.sentence_embeddings.laser.laser import LaserSentenceEmbeddings

class PretrainedEmbedding(Module):
    """
    Wrapper for pretrained embeddings. 
    The embedding can be contructed with any pretrained embedding, 
    by given the embedding vectors, and corresponding word to index mappins
    
    Args:
        num_embeddings (int): vocabulary size of the embedding
        embedding_dim  (int): dimension of the resulting word embedding
        word2idx:     (dict): Dictionary that maps input word to index of the embedding vector
        vectors (numpy array): Matrix with all the embedding vectors
        trainable (bool): 
    """
    def __init__(self, num_embeddings, embedding_dim, word2idx, vectors, trainable=False, use_column_cache=True):
        super(PretrainedEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.word2idx = word2idx
        self.vectors = vectors
        self.column_cache={}
        self.use_column_cache = use_column_cache
        self.embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(vectors))

        if not trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, sentences, mean_sequence=False):
        """
        Args:
            sentences list[str] or str: list of sentences, or one sentence
            mean_sequence bool: Flag if we should mean over the sequence dimension
        Returns:
            embedding [batch_size, seq_len, embedding_dim] or [batch_size, 1, embedding_dim]
            lenghts [batch_size]
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        
        # Convert to lowercase words
        sentences = [str.lower(sentence) for sentence in sentences]

        batch_size = len(sentences)

        # Convert list of sentences to list of list of tokens
        # TODO: should we use shlex to split, to have words in quotes stay as one word? 
        # maybe these would just be unkown words though
        # Replace () since these might occur in column names for some reason
        sentences_words = [word_tokenize(sentence) for sentence in sentences]
        lenghts = [len(sentence) for sentence in sentences_words]
        max_len = max(lenghts)

        # Use 0 as padding token
        indicies = torch.zeros(batch_size, max_len).long().to(self.embedding.weight.device)
        
        # Convert tokens to indicies
        # TODO: chose more sensible unknown token instead of just using the first (".") token
        for i, sentence in enumerate(sentences_words):
            for j, word in enumerate(sentence):
                indicies[i,j] = self.word2idx.get(word,0)

        # TODO: pad tensors using pytorch instead of numpy?
        word_embeddings = self.embedding(indicies)

        if mean_sequence:
            word_embeddings = torch.sum(word_embeddings,dim=1)/torch.tensor(lenghts).float().to(self.embedding.weight.device)

        return word_embeddings, np.asarray(lenghts)

    def embed_token(self, token):
        """
        Embeds a token that may or may not contain whitespaces and underscores.
        Used to embed history in the get_history_emb function.
        """
        embs, words = [], token.split()
        for word in words:
            emb_list=[]
            for element in word.split('_'):
                # if we have a trailing _ we don't want to embed an empty string
                if element:
                    emb,_ = self(element, mean_sequence=True)
                    emb_list.append(emb)
            embs.append(torch.mean(torch.stack(emb_list), dim=0))

        return torch.mean(torch.stack(embs), dim=0)

    def get_history_emb(self, histories):
        """
        Args:
            histories list(list(str)): list of histories. format like [['select','col1 text db','min'], ['select','col2 text db','max']]
                                     each of the strings with multiple words should be meaned
        Returns:
            embedding [batch_size, history_len, embedding_dim]
            lengths [batch_size]
        """
        batch_size = len(histories)
        lengths = [len(history) for history in histories]
        max_len = max(lengths)

        # Create tensor to store the resulting embeddings in
        embeddings = torch.zeros(batch_size, max_len, self.embedding_dim).to(self.embedding.weight.device)
        for i,history in enumerate(histories):

            for j, token in enumerate(history):
                emb = self.embed_token(token)
                embeddings[i,j,:] = emb

        return embeddings, np.asarray(lengths)

    def get_columns_emb(self, columns):
        """
        Args:
            columns list(list(str)): nested list, where indicies corresponds to [i][j][k], i=batch, j=column, k=word  
        
        """
        batch_size = len(columns)

        # Get the number of columns in each database
        lengths = [len(column) for column in columns]


        # Get the number of tokens for each column, eg ['tablename','text','column','with','long','name']
        col_name_lengths = [[len(word) for word in column] for column in columns]
        max_len = max(lengths)

        # Join the column tokens to align the way we split them        
        columns_joined = [[' '.join(column) for column in columns_batch] for columns_batch in columns]
        # Get the number of tokens in each column
        col_name_lengths = [[len(word_tokenize(column)) for column in columns_batch] for columns_batch in columns_joined]
        # Get the maximum number of tokens for all columns
        max_col_name_len = max([max(col_name_len) for col_name_len in col_name_lengths])

        # Embeddings will have shape (batch size, # of columns, # of words in col name, embedding dim)
        embeddings = torch.zeros(batch_size, max_len, max_col_name_len, self.embedding_dim).to(self.embedding.weight.device)
        col_name_lengths = np.zeros((batch_size, max_len))

        for i, db in enumerate(columns_joined):
            if str(db) in self.column_cache:
                cached_emb, cached_lengths = self.column_cache[str(db)]

                # Different batches might have different padding, so pick the minumum needed
                min_size1 = min(cached_emb.size(0), max_len)
                min_size2 = min(cached_emb.size(1), max_col_name_len)

                embeddings[i,:min_size1,:min_size2,:] = cached_emb[:min_size1,:min_size2,:]
                col_name_lengths[i,:min_size1] = np.minimum(cached_lengths, max_col_name_len)[:min_size1]
                continue

            for j, column in enumerate(db):
                # Embedding takes in a sentence, to concat the words of the column into a sentence
                #column = ' '.join(column)
                emb,col_name_len = self(column)

                # Embeddings: (N, # columns, # words, # features)
                embeddings[i,j,:int(col_name_len),:] = emb
                col_name_lengths[i,j] = int(col_name_len)

            # Try and cache the embeddings for the columns in the db
            if self.use_column_cache:
                self.column_cache[str(db)] = (embeddings[i,:,:], col_name_lengths[i,:])

        return embeddings, np.asarray(lengths),col_name_lengths


class GloveEmbedding(PretrainedEmbedding):
    """
    Class responsible for GloVe embeddings.
    https://nlp.stanford.edu/pubs/glove.pdf
    """
    def __init__(self, path='data/glove.6B.50d.txt'):
        word2idx, vectors = {}, []

        # Load vectors and build dictionary over word-index pairs
        with open(path,'r', encoding ="utf8") as f:
            for idx, line in enumerate(f,1):
                line = line.split()     
                word2idx[line[0]] = idx
                vectors += [np.asarray(line[1:],dtype=np.float)]
            
            # Insert zero-embedding for unknown tokens at first index
            word2idx['<unknown>'] = 0
            vectors.insert(0, np.zeros(len(vectors[0])))
        
        # Convert to numpy
        vectors = np.asarray(vectors, dtype=np.float)
        super(GloveEmbedding, self).__init__(num_embeddings=len(word2idx), embedding_dim=len(vectors[0]), word2idx=word2idx, vectors=vectors)


class LaserEmbedding(PretrainedEmbedding):
    """
    Wrapper for pretrained LASER embeddings provided by raffle.ai.
    https://arxiv.org/abs/1812.10464
    """
    #word2idx={}, vectors=np.ones((1,1024),
    def __init__(self):
        super(LaserEmbedding, self).__init__(num_embeddings=1, embedding_dim=1024,word2idx={},, vectors=[] dtype=float)
        # Initialize the raffle.ai implementation of LASER
        self.embedder = LaserSentenceEmbeddings()

    def forward(self, sentences, mean_sequence=True):
        """
        Args:
            sentences list[str] or str: list of sentences, or one sentence
            mean_sequence bool: Flag if we should mean over the sequence dimension
        Returns:
            embedding [batch_size, seq_len, embedding_dim] or [batch_size, 1, embedding_dim]
            lenghts [batch_size]
        """
        if not isinstance(sentences, list):
            sentences = [sentences]

        # Convert words to lowercase
        sentences = [str.lower(sentence) for sentence in sentences]

        # Convert list of sentences to list of list of tokens
        # TODO: should we use shlex to split, to have words in quotes stay as one word? 
        #      maybe these would just be unkown words though
        sentences_words = [word_tokenize(sentence) for sentence in sentences]

        # Define sequence length as max length sentence in batch
        lengths = [len(sentence) for sentence in sentences_words]
        max_len = max(lengths)
        
        
        word_embeddings=[]
        
        if mean_sequence:
            # Embed full sentence by taking mean over sequence of words
            for sentence in sentences:
                word_embeddings.append(self.embedder(sentence, method="sentence", language='en'))
        
        else:
            # Otherwise embed words seperately (discouraged)
            for sentence, length in zip(sentences_words,lengths):
                sentences_to_matrix=[]
                for word in sentence:
                    if word not in word2idx:
                        word2idx{word} = len(word2idx)
                        vectors += [self.embedder(word, method="sentence", language='en')]
                    sentences_to_matrix.append(self.embedder(word, method="sentence", language='en'))
                while length<max_len:
                    sentences_to_matrix.append(np.zeros((1,self.embedding_dim), dtype=float))
                    length= length+1
                word_embeddings.append(sentences_to_matrix)
        
        # Convert embeddings and fix dimensions
        word_embeddings = torch.tensor(word_embeddings).squeeze().unsqueeze(0)
                
        return word_embeddings, np.asarray(lengths)
        

if __name__ == "__main__":
    for embedder in [LaserEmbedding()]:
        print('\nTesting functionality of', embedder.__class__.__name__ + '...')

        # Verify that sentence embedding works
        print(embedder(['test is a good thing'])[0].shape)

        # Verify that history is embedded as expected
        print(embedder.get_history_emb([['select', 'col1 text db', 'min'],['select', 'col2 text db max']])[0].shape)
