from sql import SQLStatement
from states import KeyWordState, StateDatabase

class SQLParser():
    
    def __init__(self):
        self.stack = [KeyWordState()]
        self.history = []
        self.sql = SQLStatement()
        self.kw =""



    def take_action(self, sample):
        state = self.stack.pop()
        new_states, history, sql, kw = state.next_state(self.history, self.sql, self.kw)
        
        self.history = history
        self.sql = sql
        self.kw = kw
        print(kw)
        if new_states:
            self.stack += new_states

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