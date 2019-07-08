import numpy as np
from sql import *
#np.random.seed(0)
#TODO: All of this could be implemented prettier by recursion...

class TestModel():
    def __init__(self, num_states):
        self.num_states = num_states
    def forward(self):
        return 2, [np.random.randint(self.num_states),np.random.randint(self.num_states)] 

class StateDatabase():
    #TODO: set the database as a static class variable. This is very unelegant though..
    database = None

class SelectState():
    def next_state(self, history, sql,kw):
        history += ["select"]
        return [ColumnState()], history, sql , kw

class WhereState():
    def next_state(self, history, sql,kw):
        history += ["where"]

        new_states = [ ColumnState()]
        return new_states, history, sql, kw

class AscDescState():
    model = TestModel(5)
    def next_state(self, history, sql,kw):

        _, ascdesc = self.model.forward()
        ascdesc = SQL_ORDERBY_OPS[ascdesc[0]]
        history += [ascdesc]

        sql.ORDERBY_OP = ascdesc

        new_states = []
        #TODO: we might want to predict the limit value
        # if 'limit' in ascdesc:
        #     new_states = [ValueState()]
        return new_states, history, sql, kw


class GroupByState():
    def next_state(self, history, sql,kw):
        history += ["groupby"]

        new_states = [ColumnState()]
        return new_states, history, sql, kw

class HavingState():
    model = TestModel(2)
    def next_state(self, history, sql,kw, target=None):
        history += ["having"]

        if target is not None:
            having = target
        else:
            having,_ = self.model.forward()
        if having:
            new_states = [ColumnState()]
        else:
            new_states = [] 
        return new_states, history, sql, kw
        
class OrderByState():
    def next_state(self, history, sql,kw):
        history += ["orderby"]

        return [ColumnState()], history, sql, kw


class KeyWordState():
    STATES = [WhereState, GroupByState, OrderByState]
    model = TestModel(len(STATES))
    
    
    def forward(self):
        return 3, np.random.choice(3,3,replace=False)
    
    def next_state(self, history, sql,kw, target=None):

        if target is not None:
            num_kw, keywords = target
        else:
            num_kw, kws = self.forward()
            kws = np.argsort(kws) [:num_kw]
            #We want the keywords in the same order as much as possible
            #Keywords are added FILO queue, so sort it in reverse
            key_words = sorted(kws, reverse=True) 

        #Add other states to the list
        new_states = [self.STATES[key_word]() for key_word in key_words]
        #First state should be a select state
        new_states += [SelectState()]
        
        return new_states, history, sql, kw


class AndOrState():
    model = TestModel(2)
    
    def update_sql(self, history, sql, kw):
        if kw == 'where':
            sql.WHERE[-1].cond_op = history[-1]
        elif kw == 'having':
            sql.HAVING[-1].cond_op = history[-1]
        return sql
    def next_state(self, history, sql, kw, target=None):

        if target is not None:
            andor = target
        else:
            _, andor = self.model.forward()
            andor = SQL_COND_OPS[andor[0]]

        history += [andor]

        sql = self.update_sql(history, sql, kw)
        return [], history, sql, kw

class OPState():

    model = TestModel(3)
    def next_state(self, history, sql,kw, target=None):
        
        if target is not None:
            op = target
        else:
            _, op = self.model.forward()
            op = SQL_OPS[op[0]]

        history += [op]
        return [], history, sql, kw

class AggState():
    model = TestModel(3)

    def update_sql(self, history, sql, kw):
        
        if kw == 'select':
            col, agg = history[-2:]
            sql.COLS += [ColumnSelect(col, agg)]
        elif kw == 'orderby':
            col, agg = history[-2:]
            sql.ORDERBY += [ColumnSelect(col, agg)]

        
        return sql
    def next_state(self, history, sql,kw, target = None):
        if target is not None:
            agg = target
        else:
            _, agg = self.model.forward()
            agg = SQL_AGG[agg[0]]

        history += [agg]

        sql = self.update_sql(history, sql, kw)
        return [], history, sql, kw
class ValueState():

    def update_sql(self, history, sql, kw):
        if kw == 'having':
            col, agg, op, value = history[-4:]
            sql.HAVING += [Condition(col, op, value,agg=agg)]
        elif kw == 'where':
            col, op, value = history[-3:]
            sql.WHERE += [Condition(col, op, value)]
        return sql
    def next_state(self, history, sql,kw):
        history += ['VALUE']

        sql = self.update_sql(history, sql, kw)
        return [], history, sql, kw

class ColState():
    def __init__(self,column):
        self.column = column
    def next_state(self, history, sql,kw):
        history += [self.column]

        if kw == 'groupby':
            sql.GROUPBY += [self.column]
        return [], history, sql, kw

class ColumnState():

    model = TestModel(2)
    def next_state(self, history, sql,kw, target=None):

        if target is not None:
            num_cols, cols = target
        else:
            num_cols, cols = self.model.forward()
            
            cols = cols[:num_cols]
        #sql = self.update_sql(sql, history, cols)
        new_states = []

        kw = history[-1]

        if kw == 'groupby' and len(cols)>0:
            new_states += [HavingState()]
        if kw == 'orderby':
            new_states += [AscDescState()]

        for i, col in enumerate(cols):
            
            if kw in ('where','having'):
                #If we predict multiple columns in where or having, we need to also predict and/or
                if num_cols>1 and i>0 :
                    new_states += [AndOrState()]
                #We need the value and comparison operation in where/having clauses
                new_states += [ValueState(), OPState()]

            if kw in ('orderby','select','having'):
                #Each column should have an aggregator
                new_states += [AggState()]

            
            column = StateDatabase.database.get_column_from_idx(col)
            new_states += [ColState(column)]

        return new_states, history, sql, kw
