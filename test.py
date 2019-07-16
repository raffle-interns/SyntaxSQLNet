from tqdm import tqdm
from sql.syntaxsql import SyntaxSQL
from utils.dataloader import SpiderDataset
from embedding.embeddings import GloveEmbedding
emb = GloveEmbedding(path='data/'+'glove.6B.300d.txt')
spider = SpiderDataset(data_path='data/dev.json', tables_path='data/tables.json',exclude_keywords=['-', ' / ', ' + ']) 
syntax_sql = SyntaxSQL(embeddings=emb, N_word=emb.embedding_dim, hidden_dim=100, num_layers=2, gpu=True)

corrects_components = {
        'select':0,
        'where':0,
        'groupby':0,
        'orderby':0,
        'having':0,
        'limit_value':0
}
corrects = 0
for i in tqdm(range(len(spider))):
    sample = spider[i]

    predicted_sql = syntax_sql.GetSQL(sample['question'], sample['db'])

    results = predicted_sql.component_match(sample['sql'])

    for result, component in zip(results, corrects_components):
        corrects_components[component] += int(result)
    if sample['sql'] == predicted_sql:
        corrects += 1

print("\n# Components #")
for component in corrects_components:
    print(f"{component:12} accuracy = {corrects_components[component]/len(spider):0.3f}")
print("\n#    Total   #")
print(f"total        accuracy = {corrects/len(spider):0.3f}")
