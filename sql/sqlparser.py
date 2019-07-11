from sql import SQLStatement, Condition, ColumnSelect
from states import KeyWordState, StateDatabase
from models import model_list

class SQLParser():
    
    def __init__(self, embeddings, N_word, hidden_dim, num_layers, gpu):
        self.stack = [KeyWordState()]
        self.history = []
        self.sql = SQLStatement()
        self.kw =""

        self.q_emb_var = q_emb_var
        self.q_len = q_len
        self.hs_emb_var = hs_emb_var
        self.hs_len = hs_len 

        self.kw_emb_var = kw_emb_var
        self.kw_len = kw_len

        self.col_emb_var = col_emb_var 
        self.col_len = col_len 
        self.col_name_len = col_name_len
        self.col_idx = col_idx

        self.having_predictor = model_list['having'](N_word=N_word, hidden_dim, num_layers, gpu)
        self.keyword_predictor = model_list['keyword'](N_word=N_word, hidden_dim, num_layers, gpu)
        self.andor_predictor = model_list['andor'](N_word=N_word, hidden_dim, num_layers, gpu)
        self.desasc_predictor = model_list['desasc'](N_word=N_word, hidden_dim, num_layers, gpu)
        self.op_predictor = model_list['op'](N_word=N_word, hidden_dim, num_layers, gpu)
        self.col_predictor = model_list['column'](N_word=N_word, hidden_dim, num_layers, gpu)
        self.agg_predictor = model_list['agg'](N_word=N_word, hidden_dim, num_layers, gpu)

    def SelectState(self):
        self.history.append(["select"])
        self.ColumnState()

    def WhereState(self):
        self.history.append(["where"])
        self.ColumnState()

    def AscDescState(self):
        ascdesc = self.desasc_predictor.forward(self.q_emb_var, self.q_len, self.hs_emb_var, self.hs_len, self.col_emb_var, self.col_len, self.col_name_len, self.col_idx)

        ascdesc = SQL_ORDERBY_OPS[torch.argmax(ascdesc)]
        self.history.append([ascdesc])

        self.sql.ORDERBY_OP = ascdesc

    def OrderByState(self):
        self.history.append(["orderby"])
        self.ColumnState()

    def GroupByByState(self):
        self.history.append(["groupby"])
        self.ColumnState()

    def HavingState(self, target=None):
        self.history.append(["having"])
        
        if target is not None:
            having = target
        else:
            having = self.having_predictor.forward(self.q_emb_var, self.q_len, self.hs_emb_var, self.hs_len, self.col_emb_var, self.col_len, self.col_name_len, self.col_idx)
        if having:
            self.ColumnState()
    
    def KeyWordState(self, target=None):
        
        STATES =[WhereState, GroupByState, OrderByState]
        if target is not None:
            num_kw, keywords = target
        else:
            num_kw, kws = self.keyword_predictor.forward(self.q_emb_var,self.q_len,self.hs_emb_var,self.hs_len, self.kw_emb_var, self.kw_len)
            kws = np.argsort(kws) [:num_kw]
            #We want the keywords in the same order as much as possible
            #Keywords are added FILO queue, so sort it in reverse
            key_words = sorted(kws, reverse=True) 

        #Add other states to the list
        for key_word in key_words:
            STATES[key_word]()
        #First state should be a select state
        self.SelectState()


    def AndOrState(self, target=None):
    
        def update_sql(self):
            if self.kw == 'where':
                self.sql.WHERE[-1].cond_op = self.history[-1]
            elif self.kw == 'having':
                self.sql.HAVING[-1].cond_op = self.history[-1]
        if target is not None:
            andor = target
        else:
            andor = self.andor_predictor.forward(self.q_emb_var, self.q_len, self.hs_emb_var, self.hs_len, self.col_emb_var, self.col_len, self.col_name_len, self.col_idx)
            andor = SQL_COND_OPS[torch.argmax(andor)]

        self.history.append([andor])
        self.update_sql()
    
    def OPState(self, target=None):
        if target is not None:
            op = target
        else:
            op = self.op_predictor.forward(self.q_emb_var, self.q_len, self.hs_emb_var, self.hs_len, self.col_emb_var, self.col_len, self.col_name_len, self.col_idx)
            op = SQL_OPS[torch.argmax(op)]

        self.history.append([op])

    def AggState(self, target=None):

        def update_sql(self):
            if self.kw == 'select':
                #type, table, col name

                col, agg = *self.history[-2:]
                #get_column function
                type_, table_, col_name_ = *col
                self.sql.COLS.append([ColumnSelect(col, agg)])
            elif self.kw == 'orderby':
                col, agg = self.history[-2:]
                self.sql.ORDERBY.append([ColumnSelect(col, agg)])

        if target is not None:
            agg = target
        else:
            agg = self.agg_predictor.forward(self.q_emb_var, self.q_len, self.hs_emb_var, self.hs_len, self.col_emb_var, self.col_len, self.col_name_len, self.col_idx)
            agg = SQL_AGG[torch.argmax(agg)]

        self.history.append([agg])
        self.update_sql()
    
    def ValueState(self):
        def update_sql(self):
            if self.kw == 'having':
                col, agg, op, value = self.history[-4:]
                self.sql.HAVING.append([Condition(col, op, value,agg=agg)])
            elif self.kw == 'where':
                col, op, value = self.history[-3:]
                self.sql.WHERE.append([Condition(col, op, value)])
            return sql
        self.history.append(['VALUE'])
        self.update_sql()

    def ColState(self, column):
        self.history.append([column])
        if self.kw == 'groupby':
            self.sql.GROUPBY.append([column])
    
    def ColumnState(self, target=None):

        if target is not None:
            num_cols, cols = target
        else:
            num_cols, cols = self.col_predictor.forward(self.q_emb_var, self.q_len, self.hs_emb_var, self.hs_len, self.col_emb_var, self.col_len, self.col_name_len, self.col_idx)
            
            cols = cols[:num_cols]

        self.kw = *self.history[-1]

        if self.kw == 'groupby' and len(cols)>0:
            self.HavingState()
        if self.kw == 'orderby':
            self.AscDescState()

        for i, col in enumerate(cols):
            
            if self.kw in ('where','having'):
                #If we predict multiple columns in where or having, we need to also predict and/or
                if num_cols>1 and i>0 :
                    self.AndOrState()
                #We need the value and comparison operation in where/having clauses
                self.ValueState()
                self.OPState()

            if self.kw in ('orderby','select','having'):
                #Each column should have an aggregator
                self.AggState()

            column = StateDatabase.database.get_column_from_idx(col)
            self.ColState(column)
    
            
    def GetSQL(question, database):
        self.sql = database
        self.question = question
        



if __name__ == '__main__':
    from models.dataloader import SpiderDataset
    spider = SpiderDataset(data_path='train.json', tables_path='tables.json',exclude_keywords=["between","-"]) 
    sample = spider[0]

    sqlParser = SQLParser()
    StateDatabase.database = sample['db']
    while len(sqlParser.stack)>0:
        sqlParser.take_action(sample)

    print(sqlParser.history)
    print(sqlParser.sql)
    print(sqlParser.sql.generate_history())