import json
from torch.utils.data import Dataset  
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sql.sql import SQLStatement, DataBase, SQL_KEYWORDS, SQL_COND_OPS, SQL_AGG, SQL_OPS, SQL_ORDERBY_OPS, SQL_DISTINCT_OP
import numpy as np
from utils.dataloader import SpiderDataset
import random

class AugmentedSpiderDataset(SpiderDataset):
    """
    Extension of a wrapper around the Spider dataset that allows for easy data augmentation.
    """
    def __init__(self, data_path, tables_path, aug_data_path, aug_tables_path, exclude_keywords=[], debug=True, language='en', max_count=10000):
        """
        Args:
            data_path (string): file path of the data json file with questions
            tables_path (string): file path of the tables json file with db scheme
            aug_data_path (string): file path of the augmentation data json file
        """
        directory=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(directory)
		
        self.exclude_keywords = exclude_keywords
        self.data = []

        data = json.load(open(directory + '/' + data_path, 'r', encoding="utf8"))

        # Handle excluded keywords, by removing them, and logging how many of each type was found
        exclude_keywords_counts = {key: 0 for key in exclude_keywords}
        for d in data:
            keywords = [keyword for keyword in exclude_keywords if str.upper(keyword) in str.upper(d['query'])]
            if keywords:
                for keyword in keywords:
                    exclude_keywords_counts[keyword] += 1
            else:
                self.data += [d]
        if debug:
            for keyword in exclude_keywords_counts:
                print(f"Found {exclude_keywords_counts[keyword]} queries with excluded keyword {keyword}")
            print(f"Total number of removed queries = {len(data) - len(self.data)} / {len(data)}")

        self.tables = {}

        def add_tables_to_dict(path):
            tables = json.load(open(directory + '/' + path, 'r'))
            for table in tables:
                db_id = table['db_id']
                self.tables[db_id] = table

        # Load tables into dictionary
        add_tables_to_dict(aug_tables_path)
        num_aug_dbs = len(self.tables)
        add_tables_to_dict(tables_path)
        num_reg_dbs = len(self.tables) - num_aug_dbs
        print('Databases in augmentation set:', num_aug_dbs)
        print('Databases in regular training set:', num_reg_dbs)

        # Generate augmented samples
        self.samples = []
        self.generate_augmented_samples(aug_data_path, max_count=max_count)

        # Process regular training set
        failed = 0
        for i in range(len(self.data)):
            try:
                example = self.data[i]
                db_id = example['db_id']
                db = DataBase(self.tables[db_id])

                sql = SQLStatement(query=example['query'], database=db)
                # TODO: include other languages
                question = example['question'][language]
                history = sql.generate_history()

                sample = {'sql': sql, 'question': question, 'db': db, 'history': history}
                self.samples += [sample]
            except:
                failed += 1
        if failed > 0:
            print(f"{failed}/{len(self.data)} queries could not be loaded")

        # Shuffle to mix augmented samples with training set
        random.Random(1300135).shuffle(self.samples)

    def generate_augmented_samples(self, aug_data_path, max_count=10000):
        directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data = json.load(open(directory + aug_data_path, 'r', encoding="utf8"))
        count = 0

        if max_count == 0:
            print(f'Data augmentation has been disabled. Proceeding...')
            return

        # Get database name
        for db_idx, db_name in enumerate(self.tables):
            
            # Generate database object
            db = DataBase(self.tables[db_name])

            # Get table names
            table_names = self.tables[db_name]['table_names']
            table_names_original = self.tables[db_name]['table_names_original']

            # Get column names
            col_idx = 0
            for table_idx, column_name in self.tables[db_name]['column_names']:

                # Get original column name
                column_name_original = self.tables[db_name]['column_names_original'][col_idx][1]
                col_idx += 1

                # Generate hash value
                hashval = hash(table_names[table_idx] + column_name_original + str(count))

                # Get secondary column names
                j = (hashval + hash(col_idx)) % (len(self.tables[db_name]['column_names']) - 1) + 1
                column_name_2 = self.tables[db_name]['column_names'][j][1]
                column_name_original_2 = self.tables[db_name]['column_names_original'][j][1]

                # Use 1/10 of all columns
                if table_idx == -1 or hashval % 100 > 10: continue

                # Iterate over entries in data augmentation file
                for entry in data:

                    # Generate SQL query
                    query = entry['query'].replace('{COLUMN}', column_name_original).replace('{COLUMN2}', column_name_original_2).replace('{TABLE}', table_names_original[table_idx])

                    # Use approx. 1/20 of all queries
                    if (hashval + hash(query) + hash(count)) % 200 > 10: continue

                    # Generate SQL statement and history
                    try:
                        sql = SQLStatement(query=query, database=db)
                        history = sql.generate_history()

                        for question in entry['question']:

                            # Use approx. 1/2 of all questions
                            if (hashval + hash(question) + count) % 20 > 10: continue
                            
                            # Generate question and add to sample list
                            question = question.replace('{COLUMN}', column_name).replace('{COLUMN2}', column_name_2).replace('{TABLE}', table_names[table_idx])
                            sample = {'sql': sql, 'question': question, 'db': db, 'history': history}
                            self.samples += [sample]
                            count += 1
                            if count == max_count:
                                print(f'Generated {count} samples using {db_idx}/{len(self.tables)} databases. Proceeding...')
                                return
                    
                    except:
                        continue

        print(f'Generated {count} samples using all databases. Proceeding...')


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    spider = AugmentedSpiderDataset(data_path='data/train.json', tables_path='/data/tables.json', aug_data_path='/data/train_augment.json', aug_tables_path='/data/wikisql_tables.json', exclude_keywords=['-', ' / ', ' + '])
    dat = spider.generate_limitvalue_dataset()