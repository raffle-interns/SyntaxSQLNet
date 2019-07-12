from sql.syntaxsql import SyntaxSQL
from utils.dataloader import SpiderDataset
from models.embeddings import GloveEmbedding
emb = GloveEmbedding(path='data/'+'glove.6B.300d.txt')
spider = SpiderDataset(data_path='data/dev.json', tables_path='data/tables.json',exclude_keywords=["between", "distinct", '-', ' / ', ' + ']) 
sample = spider[0]

syntax_sql = SyntaxSQL(embeddings=emb, N_word=emb.embedding_dim, hidden_dim=100, num_layers=2, gpu=True)
syntax_sql.GetSQL(sample['question'], sample['db'])