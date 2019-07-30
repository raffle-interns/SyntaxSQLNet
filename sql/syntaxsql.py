from models.model_list import models
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.andor_predictor import AndOrPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.op_predictor import OpPredictor
from models.col_predictor import ColPredictor
from models.agg_predictor import AggPredictor
from models.limit_value_predictor import LimitValuePredictor
from models.distinct_predictor import DistinctPredictor
from models.value_predictor import ValuePredictor
from sql.sql import SQLStatement, Condition, ColumnSelect, SQL_OPS, SQL_AGG, SQL_COND_OPS, SQL_KEYWORDS, SQL_DISTINCT_OP, SQL_ORDERBY_OPS
from nltk.tokenize import word_tokenize
from utils.utils import text2int

class SyntaxSQL():
    """
    Main class for the SyntaxSQL model. 
    This takes all the sub modules, and uses them to run a question through the syntax tree
    """
    def __init__(self, embeddings, N_word, hidden_dim, num_layers, gpu, num_augmentation=10000):
        self.embeddings = embeddings
        self.having_predictor = HavingPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        self.keyword_predictor = KeyWordPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        self.andor_predictor = AndOrPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        self.desasc_predictor = DesAscLimitPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        self.op_predictor = OpPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        self.col_predictor = ColPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        self.agg_predictor = AggPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        self.limit_value_predictor = LimitValuePredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        self.distinct_predictor = DistinctPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        self.value_predictor = ValuePredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu).eval()
        
        def get_model_path(model='having', batch_size=64, epoch=50, num_augmentation=num_augmentation, name_postfix=''):
            return f'saved_models/{model}__num_layers={num_layers}__lr=0.001__dropout=0.3__batch_size={batch_size}__embedding_dim={N_word}__hidden_dim={hidden_dim}__epoch={epoch}__num_augmentation={num_augmentation}__{name_postfix}.pt'

        try:
            self.having_predictor.load(get_model_path('having'))
            self.keyword_predictor.load(get_model_path('keyword', epoch=300, num_augmentation=10000, name_postfix='kw2'))
            self.andor_predictor.load(get_model_path('andor', batch_size=256, num_augmentation=0))
            self.desasc_predictor.load(get_model_path('desasc'))
            self.op_predictor.load(get_model_path('op', num_augmentation=10000))
            self.col_predictor.load(get_model_path('column', epoch=300, num_augmentation=30000, name_postfix='rep2aug'))
            self.distinct_predictor.load(get_model_path('distinct', epoch=300, num_augmentation=0, name_postfix='dist2'))
            self.agg_predictor.load(get_model_path('agg', num_augmentation=0))
            self.limit_value_predictor.load(get_model_path('limitvalue'))
            self.value_predictor.load(get_model_path('value', epoch=300, num_augmentation=10000, name_postfix='val2'))

        except FileNotFoundError as ex:
            print(ex)

        self.current_keyword = ''
        self.sql = None
        self.gpu = gpu

        if gpu:
            self.embeddings = self.embeddings.cuda()

    def generate_select(self):
        # All statements should start with a select statement
        self.current_keyword = 'select'
        self.generate_columns()

    def generate_where(self):
        self.current_keyword = 'where'
        self.generate_columns()

    def generate_ascdesc(self, column):
        # Get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb([history['having'][-1]])
        
        col_idx = self.sql.database.get_idx_from_column(column)

        ascdesc = self.desasc_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)

        ascdesc = SQL_ORDERBY_OPS[int(ascdesc)]

        self.sql.ORDERBY_OP += [ascdesc]

        if 'LIMIT' in ascdesc:
            limit_value = self.limit_value_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)[0]
            self.sql.LIMIT_VALUE = limit_value

    def generate_orderby(self):
        self.current_keyword = 'orderby'
        self.generate_columns()

    def generate_groupby(self):
        self.current_keyword = 'groupby'
        self.generate_columns()

    def generate_having(self, column):
        # Get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb([history['having'][-1]])
        
        col_idx = self.sql.database.get_idx_from_column(column)

        having = self.having_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)
        if having:
            self.current_keyword = 'having'
            self.generate_columns()
    
    def generate_keywords(self):       
        self.generate_select()

        KEYWORDS =[self.generate_where, self.generate_groupby, self.generate_orderby]
        
        # Get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb(history['keyword'])
       
        num_kw, kws = self.keyword_predictor.predict(self.q_emb_var,self.q_len, hs_emb_var, hs_len, self.kw_emb_var, self.kw_len)

        if num_kw[0] == 0:
            return

        # We want the keywords in the same order as much as possible
        # Keywords are added FIFO queue, so sort it
        key_words = sorted(kws[0]) 

        # Add other states to the list
        for key_word in key_words:
            KEYWORDS[int(key_word)]()
        
    def generate_andor(self, column):
        # Get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb([history['andor'][-1]])
        
        andor = self.andor_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len)
        andor = SQL_COND_OPS[int(andor)]

        if self.current_keyword == 'where':
            self.sql.WHERE[-1].cond_op = andor
        elif self.current_keyword == 'having':
            self.sql.HAVING[-1].cond_op = andor
        
    def generate_op(self, column):
        # Get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb([history['op'][-1]])
        
        col_idx = self.sql.database.get_idx_from_column(column)

        op = self.op_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)
        op = SQL_OPS[int(op)]

        # Pick the current clause from the current keyword
        if self.current_keyword == 'where':
            self.sql.WHERE[-1].op = op
        else:
            self.sql.HAVING[-1].op = op

        return op

    def generate_distrinct(self, column):
        # Get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb([history['distinct'][-1]])

        col_idx = self.sql.database.get_idx_from_column(column)

        distinct = self.distinct_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)

        distinct = SQL_DISTINCT_OP[int(distinct)]
        
        if self.current_keyword == 'select':
            self.sql.COLS[-1].distinct = distinct
        elif self.current_keyword == 'orderby':
            self.sql.ORDERBY[-1].distinct = ''
        elif self.current_keyword == 'having':
            self.sql.HAVING[-1].distinct = distinct

    def generate_agg(self, column, early_return = False, force_agg = False):

        # Get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb([history['agg'][-1]])

        col_idx = self.sql.database.get_idx_from_column(column)

        agg = self.agg_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx, force_agg=force_agg)

        agg = SQL_AGG[int(agg)]

        if early_return is True:
            return agg
    
        if self.current_keyword == 'select':
            self.sql.COLS[-1].agg = agg
        elif self.current_keyword == 'orderby':
            self.sql.ORDERBY[-1].agg = agg
        elif self.current_keyword == 'having':
            self.sql.HAVING[-1].agg = agg
        
    def generate_between(self, column):
        ban_prediction = None

        # Make two predictions
        for i in range(2):
        
            # Get the history, from the current sql
            history = self.sql.generate_history()
            hs_emb_var, hs_len = self.embeddings.get_history_emb([history['value'][-1]])
            tokens=word_tokenize(str.lower(self.question))

            # Create mask for integer tokens
            int_tokens = [text2int(token.replace('-','').replace('.','')).isdigit() for token in tokens]

            num_tokens, start_index = self.value_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, ban_prediction, int_tokens)
            num_tokens, start_index = int(num_tokens[0]), int(start_index[0])

            try:
                value = ' '.join(tokens[start_index:start_index+num_tokens])

                if self.current_keyword == 'where':
                    if i == 0:
                        self.sql.WHERE[-1].value = value
                        ban_prediction = (num_tokens, start_index)
                    else:
                        self.sql.WHERE[-1].valueless = value

                elif self.current_keyword == 'having':
                    if i == 0:
                        self.sql.HAVING[-1].value = value
                        ban_prediction = (num_tokens, start_index)
                    else:
                        self.sql.HAVING[-1].valueless = value

            # The value might not exist in the question, so just ignore it
            except Exception as e:
                print(e)

    def generate_value(self, column):
        
        # Get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb([history['value'][-1]])

        num_tokens, start_index = self.value_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len)

        num_tokens, start_index = int(num_tokens[0]), int(start_index[0])
        tokens=word_tokenize(str.lower(self.question))

        try:
            value = ' '.join(tokens[start_index:start_index+num_tokens])
            value = text2int(value)
            
            if self.current_keyword == 'where':
                self.sql.WHERE[-1].value = value
            elif self.current_keyword == 'having':
                self.sql.HAVING[-1].value = value

        # The value might not exist in the question, so just ignore it
        except:
            pass

    def generate_columns(self):

        # Get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb([history['col'][-1]])
        
        num_cols, cols = self.col_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len)
        
        # Predictions are returned as lists, but it only has one element
        num_cols, cols = num_cols[0], cols[0]

        def exclude_all_from_columns():
            # Do not permit * as valid column in where/having clauses
            excluded_idx=[len(table.columns) for table in self.sql.database.tables]

            _, cols_new = self.col_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, 
                self.col_emb_var, self.col_len, self.col_name_len, exclude_idx=excluded_idx)
        
            return self.sql.database.get_column_from_idx(cols_new[0][0])

        for i, col in enumerate(cols):
            column = self.sql.database.get_column_from_idx(col)

            if self.current_keyword in ('where','having'):

                # Add the column to the corresponding clause
                if self.current_keyword == 'where':
                    if column.column_name == '*':
                        column = exclude_all_from_columns()

                    self.sql.WHERE += [Condition(column)]
                else:
                    self.sql.HAVING += [Condition(column)]

                # We need the value and comparison operation in where/having clauses
                op = self.generate_op(column)

                if op == 'BETWEEN':
                    self.generate_between(column)
                else:
                    self.generate_value(column)
                
                # If we predict multiple columns in where or having, we need to also predict and/or
                if num_cols>1 and i<(num_cols-1):
                    self.generate_andor(column)

            if self.current_keyword in ('orderby','select','having'):
                force_agg = False
                if self.current_keyword == 'orderby':
                    self.sql.ORDERBY += [ColumnSelect(column)]
                    if column.column_name == '*' and self.generate_agg(column, early_return=True) == '':
                        column = exclude_all_from_columns()
                        self.sql.ORDERBY[-1] = ColumnSelect(column)

                elif self.current_keyword == 'select':
                    force_agg = len(set(cols)) < len(cols)
                    self.sql.COLS += [ColumnSelect(column)] 

                # Each column should have an aggregator
                self.generate_agg(column, force_agg=force_agg)
                self.generate_distrinct(column)

            if self.current_keyword == 'groupby':
                if column.column_name == '*':
                    column = exclude_all_from_columns()

                self.sql.GROUPBY += [ColumnSelect(column)]

        if self.current_keyword == 'groupby' and len(cols)>0:
            self.generate_having(column)
        if self.current_keyword == 'orderby':
            self.generate_ascdesc(column)    
            
    def GetSQL(self, question, database):
        # Generate representation of the database in form of SQL clauses
        self.sql = SQLStatement(query=None, database=database)
        self.question = question
        
        self.q_emb_var, self.q_len = self.embeddings(question)

        columns = self.sql.database.to_list()

        # Get all columns from the database and split them
        columns_all_splitted = []
        for i, column in enumerate(columns):
            columns_tmp = []
            for word in column:
                columns_tmp.extend(word.split('_'))
            columns_all_splitted += [columns_tmp]

        # Get embedding for the columns and keywords
        self.col_emb_var, self.col_len, self.col_name_len = self.embeddings.get_columns_emb([columns_all_splitted])
        _, num_cols_in_db, col_name_lens, embedding_dim = self.col_emb_var.shape
        
        self.col_emb_var = self.col_emb_var.reshape(num_cols_in_db, col_name_lens, embedding_dim) 
        self.col_name_len = self.col_name_len.reshape(-1)

        self.kw_emb_var, self.kw_len = self.embeddings.get_history_emb([['where', 'order by', 'group by']])

        # Start recursively generating the sql history starting with the keywords, select and so on.
        self.generate_keywords()

        return self.sql
