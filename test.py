from tqdm import tqdm
from sql.syntaxsql import SyntaxSQL
from utils.dataloader import SpiderDataset
from embedding.embeddings import GloveEmbedding
import numpy as np
emb = GloveEmbedding(path='data/glove.6B.300d.txt')
spider = SpiderDataset(data_path='data/dev.json', tables_path='data/tables.json', exclude_keywords=['-', ' / ', ' + ']) 
syntax_sql = SyntaxSQL(embeddings=emb, N_word=emb.embedding_dim, hidden_dim=100, num_layers=2, gpu=True, num_augmentation=0)

corrects_components = {
        'select':[],
        'where':[],
        'groupby':[],
        'orderby':[],
        'having':[],
        'limit_value':[],
        'keywords':[]
}
corrects = 0
for i in tqdm(range(len(spider))):
    sample = spider[i]

    predicted_sql = syntax_sql.GetSQL(sample['question'], sample['db'])
    results = predicted_sql.component_match(sample['sql'])

    # Uncomment to print SQL queries
    # print(sample['question'])
    # print(predicted_sql)
    # print(sample['sql'])
    # print('\n')

    for result, component in zip(results, corrects_components):
        if result is not None:
            corrects_components[component] += [int(result)]
    if predicted_sql == sample['sql']:
        corrects += 1

print("\n# Components #")
for component in corrects_components:
    print(f"{component:12} accuracy = {np.mean(corrects_components[component]):0.3f}   global misrate = {(np.asarray(corrects_components[component])==0).sum()/len(spider):0.3f}")
print("\n#    Total   #")
print(f"total        accuracy = {corrects/len(spider):0.3f}")
