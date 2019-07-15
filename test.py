from tqdm import tqdm
from sql.syntaxsql import SyntaxSQL
from utils.dataloader import SpiderDataset
from embedding.embeddings import GloveEmbedding
emb = GloveEmbedding(path='data/'+'glove.6B.300d.txt')
spider = SpiderDataset(data_path='data/dev.json', tables_path='data/tables.json',exclude_keywords=['-', ' / ', ' + ']) 
syntax_sql = SyntaxSQL(embeddings=emb, N_word=emb.embedding_dim, hidden_dim=100, num_layers=2, gpu=True)

corrects = 0
for i in tqdm(range(len(spider))):
    sample = spider[i]

    predicted_sql = syntax_sql.GetSQL(sample['question'], sample['db'])

    if sample['sql'] == predicted_sql:
        corrects += 1

print(f"accuracy = {corrects/len(spider)}")
