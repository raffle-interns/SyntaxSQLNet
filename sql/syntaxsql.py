from models.model_list import models
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.andor_predictor import AndOrPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.op_predictor import OpPredictor
from models.col_predictor import ColPredictor
from models.agg_predictor import AggPredictor
from sql.sql import SQLStatement, Condition, ColumnSelect, SQL_OPS, SQL_AGG, SQL_COND_OPS, SQL_KEYWORDS, SQL_DISTINCT_OP, SQL_ORDERBY_OPS

class SyntaxSQL():
    """
    Main class for the SyntaxSQL model. 
    This takes all the sub modules, and uses them to run a question through the syntax tree
    """
    def __init__(self, embeddings, N_word, hidden_dim, num_layers, gpu):
        
        self.embeddings = embeddings
        self.having_predictor = HavingPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu)
        self.keyword_predictor = KeyWordPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu)
        self.andor_predictor = AndOrPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu)
        self.desasc_predictor = DesAscLimitPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu)
        self.op_predictor = OpPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu)
        self.col_predictor = ColPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu)
        self.agg_predictor = AggPredictor(N_word=N_word, hidden_dim=hidden_dim, num_layers=num_layers, gpu=gpu)

        self.current_keyword = ''
        self.sql = None
        self.gpu = gpu

    def generate_select(self):
        #All statements should start with a select statement
        self.current_keyword = 'select'
        self.generate_columns()

    def generate_where(self):
        self.current_keyword = 'where'
        self.generate_columns()

    def generate_ascdesc(self, column):
        # get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb(history['having'])
        
        if self.gpu:
            hs_emb_var = hs_emb_var.cuda()
        
        col_idx = self.sql.database.get_idx_from_column(column)

        ascdesc = self.desasc_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)

        ascdesc = SQL_ORDERBY_OPS[ascdesc]

        self.sql.ORDERBY_OP = ascdesc

    def generate_orderby(self):
        self.current_keyword = 'orderby'
        self.generate_columns()

    def generate_groupby(self):
        self.current_keyword = 'groupby'
        self.generate_columns()

    def generate_having(self, column):
        # get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb(history['having'])
        
        if self.gpu:
            hs_emb_var = hs_emb_var.cuda()
        
        col_idx = self.sql.database.get_idx_from_column(column)

        having = self.having_predictor.forward(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)
        if having:
            self.current_keyword = 'having'
            self.generate_columns()
    
    def generate_keywords(self):
        
        self.generate_select()

        KEYWORDS =[self.generate_where, self.generate_groupby, self.generate_orderby]
        
        # get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb(history['keyword'])
        
        if self.gpu:
            hs_emb_var = hs_emb_var.cuda()

        num_kw, kws = self.keyword_predictor.forward(self.q_emb_var,self.q_len, hs_emb_var, hs_len, self.kw_emb_var, self.kw_len)

        #We want the keywords in the same order as much as possible
        #Keywords are added FIFO queue, so sort it
        key_words = sorted(kws) 

        #Add other states to the list
        for key_word in key_words:
            KEYWORDS[key_word]()
        #First state should be a select state
        

    def generate_andor(self, column):

        # get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb(history['andor'])
        
        if self.gpu:
            hs_emb_var = hs_emb_var.cuda()

        col_idx = self.sql.database.get_idx_from_column(column)
        
        andor = self.andor_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)
        andor = SQL_COND_OPS[andor]

        if self.current_keyword == 'where':
            self.sql.WHERE[-1].cond_op = andor
        elif self.current_keyword == 'having':
            self.sql.HAVING[-1].cond_op = andor
        
    
    def generate_op(self, column):

        # get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb(history['op'])
        
        if self.gpu:
            hs_emb_var = hs_emb_var.cuda()
        
        col_idx = self.sql.database.get_idx_from_column(column)

        op = self.op_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)
        op = SQL_OPS[op]

        # Pick the current clause from the current keyword
        if self.current_keyword == 'where':
            self.sql.WHERE[-1].op = op
        else:
            self.sql.HAVING[-1].op = op

    def generate_agg(self, column):

        # get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb(history['agg'])

        if self.gpu:
            hs_emb_var = hs_emb_var.cuda()
        
        col_idx = self.sql.database.get_idx_from_column(column)

        agg = self.agg_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len, col_idx)
        agg = SQL_AGG[agg]
        
        if self.current_keyword == 'select':
                self.sql.COLS[-1].agg = agg
        elif self.current_keyword == 'orderby':
            self.sql.ORDERBY[-1].agg = agg

    
    def generate_value(self, column):
        pass

    def generate_columns(self):

        # get the history, from the current sql
        history = self.sql.generate_history()
        hs_emb_var, hs_len = self.embeddings.get_history_emb(history['col'])
        
        if self.gpu:
            hs_emb_var = hs_emb_var.cuda()

        num_cols, cols = self.col_predictor.predict(self.q_emb_var, self.q_len, hs_emb_var, hs_len, self.col_emb_var, self.col_len, self.col_name_len)
        
        #predictions are returned as lists, but it only has one element
        num_cols, cols = num_cols[0], cols[0]



        for i, col in enumerate(cols):
            column = self.sql.database.get_column_from_idx(col)

            if self.current_keyword in ('where','having'):
                
                # Add the column to the corresponding clause
                if self.current_keyword == 'where':
                    self.sql.WHERE += [Condition(column)]
                else:
                    self.sql.HAVING += [Condition(column)]

                #We need the value and comparison operation in where/having clauses
                self.generate_op(column)
                self.generate_value(column)
                
                #If we predict multiple columns in where or having, we need to also predict and/or
                if num_cols>1 and i>0 :
                    self.generate_andor(column)

            if self.current_keyword in ('orderby','select','having'):
                #Each column should have an aggregator
                self.generate_agg(column)

            

        if self.current_keyword == 'groupby' and len(cols)>0:
            self.generate_having(column)
        if self.current_keyword == 'orderby':
            self.generate_ascdesc(column)    
            
    def GetSQL(self, question, database):
        self.sql = SQLStatement(query=None, database=database)
        self.question = question

        
        self.q_emb_var, self.q_len = self.embeddings(question)

        columns = self.sql.database.to_list()
        self.col_emb_var, self.col_len, self.col_name_len = self.embeddings.get_columns_emb([columns])
        batch_size, num_cols_in_db, col_name_lens, embedding_dim = self.col_emb_var.shape
        
        self.col_emb_var = self.col_emb_var.reshape(num_cols_in_db, col_name_lens, embedding_dim) 
        self.col_name_len = self.col_name_len.reshape(-1)

        self.kw_emb_var, self.kw_len = self.embeddings.get_history_emb([['where', 'order by', 'group by']])

        if self.gpu:
            self.q_emb_var = self.q_emb_var.cuda()
            self.col_emb_var = self.col_emb_var.cuda()
            self.kw_emb_var = self.kw_emb_var.cuda()
        self.generate_keywords()