import re
from itertools import zip_longest
import shlex

SQL_OPS = ['=','>','<','>=','<=','!=','NOT LIKE','LIKE']
SQL_AGG = ['max','min','count','sum','avg','']
SQL_COND_OPS = ['AND','OR','']
SQL_ORDERBY_OPS = ['DESC LIMIT','ASC LIMIT','DESC','ASC','LIMIT','']
SQL_DISTINCT_OP = ['distinct','']
SQL_KEYWORDS = ['where','group by','order by']

#We need these to convert sql keywords into words that exists in our embeddings
#...Maybe not since nltk splits "!=" to "!","=" and the asc, desc exists in the glove embeddings. Maybe they aren't as good as the true word?
SQL_AGG_dict = {'max':'maximum', 'min':'minimum', 'count':'count','sum':'sum', 'avg':'average'}
SQL_ORDERBY_OPS_dict = {'DESC LIMIT':'descending limit', 'ASC LIMIT':'ascending limit', 'DESC':'descending', 'ASC':'ascending', 'LIMIT':'limit'}
SQL_OPS_dict = {'=':'=','>':'>','<':'<','>=':'> =','<=':'< =','!=':'! =','NOT LIKE':'not like','LIKE':'like'}
class SQLStatement():
    """
    Class
    """
    def __init__(self, query = None, database=None):
        self.COLS = []
        self.WHERE = []
        self.GROUPBY = []
        self.ORDERBY = []
        self.ORDERBY_OP = []
        self.TABLE = ""
        self.HAVING = []
        self.LIMIT_VALUE = ""
        self.database = database
        if isinstance(query, dict):
            self.from_dict(query)
        elif isinstance(query, str):
            self.from_query(query)
    
    def __eq__(self, other):
        if not isinstance(other, SQLStatement):
            return NotImplemented
        #The order might be different between two statements, so compare the clauses as sets
        return (set(self.COLS)==set(other.COLS) 
             and set(self.WHERE)==set(other.WHERE) 
             and set(self.GROUPBY)==set(other.GROUPBY)
             and set(self.ORDERBY)==set(other.ORDERBY)
             and set(self.ORDERBY_OP)==set(other.ORDERBY_OP)
             and self.TABLE == other.TABLE
             and set(self.HAVING)==set(other.HAVING)
             and self.LIMIT_VALUE==other.LIMIT_VALUE)
        

    @property
    def keywords(self):
        keywords = []
        if self.WHERE:
            keywords += ['where']
        if self.GROUPBY:
            keywords += ['group by']
        if self.ORDERBY:
            keywords += ['order by']
        return keywords            
        
    @property
    def and_ors(self):
        andors = []
        for condition in self.WHERE:
            if condition.cond_op:
                andors += [condition.cond_op]
        for condition in self.HAVING:
            if condition.cond_op:
                andors += [condition.cond_op]

        return andors              

    def from_query(self, query):

       #Remove closing ;
        query = query.replace(';','')

        #TODO: What about multiply (* is also used as wildcard)
        if ' - ' in query or ' + ' in query or ' / ' in query:
            raise NotImplementedError(f"Doesn't support arithmetic in query : {query}")
        
        #TODO: fix so we also support between in our model
        if 'BETWEEN' in query:
            raise NotImplementedError(f'Statement doesn"t support between {query}')

        #Remove any aliasing, since it's uneeded in single table queries, and only causes problems
        if 'AS ' in query:
            aliases = re.findall(r'(?<=AS ).\w+', query)
            for alias in aliases:
                query = query.replace(f'{alias}.','')
                query = query.replace(f'AS {alias}','')

        #Find the clauses used in the query
        clauses = ['SELECT','FROM','WHERE','GROUP BY','HAVING','ORDER BY']
        clauses = [clause for clause in clauses if clause in query]
        #Split query into different clauses
        query_split = re.split(f'({"|".join(clauses)})',query)[1:]


        #We have to read the table first, to link the columns correctly. This should always be clause number 2
        self.TABLE = str.lower(query_split[3]).strip()

        #loop over splits two at a time, since each clause have [claus,value]
        for i in range(0,len(query_split),2):
            clause = query_split[i]
            statement = query_split[i+1]
            if clause == 'SELECT':
                for column in statement.split(','):
                    agg, distinct = '',''
                    column = str.lower(column).strip()

                    #For some queries the query is distinct(column), where distinct is not an aggregator...
                    if 'distinct(' in column:
                        distinct, column = column.split('(')
                        column = column.replace('(','').replace(')','')

                    #Check if there is an aggregator in the selection
                    if '(' in column:
                        agg = column.split('(')[0]
                        #Some queries has distinct count ( distinct column), so we remove the redundant first distinct
                        agg = agg.replace('distinct','').strip()
                        column = column.split('(')[1].split(')')[0].strip()
                    # Columns with aggregators might include the distinct keyword
                    if 'distinct' in column:
                        distinct, column = column.split()
                    
                    column = self.database.get_column(column, self.TABLE)
                    self.COLS.append(ColumnSelect(column, agg=agg, distinct=distinct))
            
            elif clause == 'WHERE':
                
                #Find any AND/OR operators
                conditional_ops = re.findall('( AND | OR )',statement)
                #Split statements into individual statements
                conditions = re.split(' AND | OR ',statement)
                #Combine the statement and AND/OR
                for condition, conditional_op in zip_longest(conditions, conditional_ops, fillvalue=""):
                    #shlex doesn't split substrings in quotes, e.g 
                    #'book = "long title with multiple words"' -> ['book','=','long title with multiple words']
                    column, op, value = re.findall(r'(.*\(.*?\)|\'.*?\'|".*?"|NOT LIKE|\S+)',condition)             

                    column = str.lower(column).strip()
                    conditional_op = conditional_op.strip()
                    column = self.database.get_column(column, self.TABLE)
                    self.WHERE.append(Condition(column, op, value, conditional_op))

            elif clause == 'GROUP BY':
                for column in statement.split(','):
                    column = str.lower(column).replace('(','').replace(')','').strip()
                    column = self.database.get_column(column, self.TABLE)
                    #TODO: technically, groupby never has distinct or aggregator,
                    #but it makes it easier if all variables of the sql has same format
                    column = ColumnSelect(column, '', '')
                    self.GROUPBY.append(column)

            elif clause == 'HAVING':

                #Find any AND/OR operators
                conditional_ops = re.findall('( AND | OR )',statement)
                #Split statements into individual statements
                conditions = re.split(' AND | OR ',statement)
                #Combine the statement and AND/OR
                for condition, conditional_op in zip_longest(conditions, conditional_ops, fillvalue=""):

                    column, op, value = re.findall(r'(.*\(.*?\)|\'.*?\'|".*?"|NOT LIKE|\S+)',condition)

                    agg, distinct = '', ''
                    column = str.lower(column)
                    #Check if statement contains an aggregator
                    if '(' in column:
                        agg = column.split('(')[0].strip()
                        column = column.split('(')[1].split(')')[0]
                    # Columns with aggregators might include the distinct keyword
                    if 'distinct' in column:
                        distinct, column = column.split()

                    conditional_op = conditional_op.strip()
                    column = self.database.get_column(column, self.TABLE)
                    self.HAVING.append(Condition(column, op, value, conditional_op, agg, distinct))

            elif clause == 'ORDER BY':

                for column in statement.split(','):

                    agg, distinct = '', ''

                    #Check if statement contains an aggregator
                    if '(' in column:
                        agg = column.split('(')[0].strip()
                        
                        col = column.split('(')[1].split(')')[0].strip()
                        
                    
                    else:
                        col = column.split()[0].strip()
                    col = str.lower(col)
                    if 'distinct' in col:
                        distinct, col = col.split()
                    
                    #Find order by ops (asc, desc, ...), but ignore the final blank '' op
                    orderby_op = re.findall(f'({"|".join(SQL_ORDERBY_OPS[:-1])})', column)
                    if orderby_op:
                        self.ORDERBY_OP += orderby_op

                        if 'LIMIT' in orderby_op[0] :
                            self.LIMIT_VALUE = re.findall(r'\d+',column)[0]
                    else:
                        self.ORDERBY_OP += [""]

                    agg = str.lower(agg)
                    column = self.database.get_column(col, self.TABLE)
                    column = ColumnSelect(column, agg, distinct)
                    self.ORDERBY.append(column)
                          
    def from_dict(self, query_dict1):
        """
        Create an SQLStatement from a dict with the structure from SPIDER
        """
        raise DeprecationWarning('from_dict method is not up to date')
        query_dict = query_dict1['raffle_query']
        for key in query_dict:
            if key == "SELECT":
                
                columns = iter(query_dict[key])
                for column in columns:
                    agg, distinct = '',''
                    column = str.lower(column).strip()
                    #Some commas might appear 
                    if column == ',':
                        continue
                    #Some queries has aggregator and column split into two different entries
                    if column in SQL_AGG:
                        agg = column
                        #Get the column from next index
                        col = next(columns)
                        col = str.lower(col.replace('(','').replace(')','').strip())
                        if 'distinct' in col:
                            distinct, col = col.split()

                    #
                    elif column == 'distinct':
                        distinct = column
                        col = str.lower(next(columns))
                        if col in SQL_AGG:
                            agg = col
                            col = next(columns)
                    #Check if there is an aggregator in the selection
                    elif '(' in column:
                        agg = column.split('(')[0]
                        col = column.split('(')[1].split(')')[0]
                        # Columns with aggregators might include the distinct keyword
                        if 'distinct' in col:
                            distinct, col = col.split()
                    else:
                        col = column
                    column = self.database.get_column(col, self.TABLE)
                    self.COLS.append(ColumnSelect(column, agg=agg, distinct=distinct))
            
            elif key == "FROM":
                self.TABLE = str.lower(query_dict[key][0])
            
            elif key == "GROUP BY":
                for column in query_dict[key]:
                    column = str.lower(column).replace('(','').replace(')','').strip()
                    column = self.database.get_column(column, self.TABLE)
                    self.GROUPBY.append(column)
            
            elif key == "WHERE":

                conditions = query_dict[key][0]
                
                #TODO: fix so we also support between in our model
                if 'BETWEEN' in conditions:
                    raise NotImplementedError('Statement doesn"t support between')

                #Find any AND/OR operators
                conditional_ops = re.findall('(AND|OR)',conditions)
                #Split statements into individual statements
                conditions = re.split('AND|OR',conditions)
                #Combine the statement and AND/OR
                for condition, conditional_op in zip_longest(conditions, conditional_ops, fillvalue=""):
                    #shlex doesn't split substrings in quotes, e.g 
                    #'book = "long title with multiple words"' -> ['book','=','long title with multiple words']
                    
                    condition_list = shlex.split(condition)
                    #with negated conditions ("not like") we can't just split by whitespace, since we need to combine "not" and "like"
                    if len(condition_list)>3:
                        column, op, value = condition_list[0], ' '.join(condition_list[1:3]), condition_list[3]
                    else:
                        column, op, value = condition_list
                    #TODO: value might contain semicolon
                    value = value.replace(';','')
                    column = str.lower(column)
                    column = self.database.get_column(column, self.TABLE)
                    self.WHERE.append(Condition(column, op, value, conditional_op))

            elif key == "ORDER BY":
                statements = iter(query_dict[key])
                for statement in statements:
                    statement = str.lower(statement)
                    agg = ''
                    #Check if statement contains an aggregator
                    if '(' in statement:
                        agg = statement.split('(')[0]
                        col = statement.split('(')[1].split(')')[0]

                    elif statement in SQL_AGG:
                        agg = statement
                        #Get the column from next index
                        col = next(statements)
                        col = str.lower(col.replace('(','').replace(')','').strip())
                        if 'distinct' in col:
                            distinct, col = col.split()

                        #TODO: Can order by have distinct?
                    #Some of the statements has desc/asc as a seperate entry in the list
                    #So make sure the statement is a column and not just the operator
                    elif statement in SQL_ORDERBY_OPS:
                        self.ORDERBY_OP = statement
                        continue
                    
                    else:
                        col = statement.split()[0]

                    column = self.database.get_column(col, self.TABLE)
                    column = ColumnSelect(column, agg)
                    self.ORDERBY.append(column)
                    
                    #Set the DESC/ASC op if statement contains it
                    orderby_op = re.findall('(DESC|ASC)',statement)
                    if orderby_op:
                        self.ORDERBY_OP = orderby_op[0]
                
            elif key == "HAVING":
                conditions = query_dict[key][0]
                #Find any AND/OR operators
                conditional_ops = re.findall('(AND|OR)',conditions)
                #Split statements into individual statements
                conditions = re.split('AND|OR',conditions)
                #Combine the statement and AND/OR
                for condition, conditional_op in zip_longest(conditions, conditional_ops, fillvalue=""):
                    column, op, value = condition.split()
                    
                    agg = ''
                    #Check if statement contains an aggregator
                    if '(' in column:
                        agg = column.split('(')[0]
                        col = column.split('(')[1].split(')')[0]
                        #TODO: Can order by have distinct?
                    #Some of the statements has desc/asc as a seperate entry in the list
                    #So make sure the statement is a column and not just the operator
                    elif statement not in SQL_ORDERBY_OPS:
                        col = column
                    column = str.lower(col)
                    column = self.database.get_column(col, self.TABLE)
                    self.HAVING.append(Condition(column, agg, op, value, conditional_op))

    def generate_history(self):
        
        history_dict = {'col':[], 'agg':[], 'andor':[], 'keyword':[], 'op':[], 'value':[], 'having':[], 'decasc':[]}

        history = ['select']
        history_dict['keyword'] += [history.copy()]
        history_dict['col'] += [history.copy()]
        for column in self.COLS:

            history += [' '.join(column.column.to_list())]
            history_dict['agg'] += [history.copy()]
            if column.agg:
                history += [column.agg]
        
        if self.WHERE:
            history += ['where']
            history_dict['col'] += [history.copy()]
            for condition in self.WHERE:
                history += [' '.join(condition.column.to_list())]
                
                history_dict['op'] += [history.copy()]
                history += [condition.op]

                history_dict['value'] += [history.copy()]
                history += [condition.value]

                if condition.cond_op:
                    history_dict['andor'] += [history.copy()]
                    history += [condition.cond_op]

        if self.GROUPBY:
            history += ['group by']
            history_dict['col'] += [history.copy()]
            for groupby in self.GROUPBY:
                history += [' '.join(groupby.column.to_list())]

        history_dict['having'] += [history.copy()]
        if self.HAVING:
            history += ['having']
            history_dict['col'] += [history.copy()]
            for condition in self.HAVING:
                history += [' '.join(condition.column.to_list())]

            
                history_dict['agg'] += [history.copy()]
                #only add aggregator if not '', since it will otherwise cause nans
                if condition.column_select.agg:
                    history += [condition.column_select.agg]
                
                history_dict['op'] += [history.copy()]
                history += [condition.op]

                history_dict['value'] += [history.copy()]
                history += [condition.value]

                if condition.cond_op:
                    history_dict['andor'] += [history.copy()]
                    history += [condition.cond_op]

        if self.ORDERBY:
            history += ['order by']
            history_dict['col'] += [history.copy()]
            for orderby in self.ORDERBY:
                history += [' '.join(orderby.column.to_list())]

                history_dict['agg'] += [history.copy()]
                if orderby.agg:
                    history += [orderby.agg]

        
        for orderby_op in self.ORDERBY_OP:
            history_dict['decasc'] += [history.copy()]                
            if orderby_op:
                history += [orderby_op]
        return history, history_dict

    def __str__(self):
        """Convert object to string representation of SQL"""
        string_cols = [str(col) for col in self.COLS]
        string_where = [str(where) for where in self.WHERE]
        string_group = [str(group) for group in self.GROUPBY]
        string_having = [str(having) for having in self.HAVING]
        string_order = [f"{str(order)} {orderop}" for order, orderop in zip_longest(self.ORDERBY, self.ORDERBY_OP,fillvalue="")]

        if not self.TABLE :
            self.TABLE = self.COLS[0].column.table_name

        sql_string = f"SELECT {','.join(string_cols)} FROM {self.TABLE}"

        if string_where:
            sql_string += f" WHERE {' '.join(string_where)}"
        if string_group:
            sql_string += f" GROUP BY {','.join(string_group)}"
        if string_having:
            sql_string += f" HAVING {' '.join(string_having)}"
        if string_order:
            sql_string += f" ORDER BY {','.join(string_order)}"
        
        sql_string += f" {self.LIMIT_VALUE}"
        #Fix to remove unwanted spaces
        sql_string = sql_string.replace('( ','(').replace('  ',' ')
        
        return sql_string

class ColumnSelect():
    column = None
    agg = ""
    distinct = ""

    def __init__(self, column, agg="", distinct=""):
        self.column = column
        self.agg = agg
        self.distinct = distinct

    def __str__(self):

        assert self.column, "column name can't be empty"
        assert self.agg in SQL_AGG, f"{self.agg} is not an SQL aggregator"
        assert self.distinct in SQL_DISTINCT_OP, f"{self.distinct} is not an SQL distinct operator"

        #Parenthesis should only be added if we have an aggregator
        if self.agg:
            return f"{self.agg}({self.distinct} {self.column})"
    
        return f"{self.distinct} {self.column}"

    def __eq__(self, other):
        if not isinstance(other, ColumnSelect):
            return NotImplemented
        
        return str(self) == str(other)


    def __hash__(self):
        return hash(str(self))
class Condition():
    column_select = None
    op = ""
    value = ""
    cond_op = ""

    @property
    def column(self):
        return self.column_select.column
    
    @property
    def agg(self):
        return self.column_select.agg

    def __init__(self, column, op, value, cond_op="", agg="", distinct=""):
        self.column_select = ColumnSelect(column, agg, distinct)
        self.op = op
        self.value = value
        self.cond_op = cond_op


    def __str__(self):
        assert self.op in SQL_OPS, f"{self.op} is not an SQL operation"
        assert self.cond_op in SQL_COND_OPS, f"{self.cond_op} is not an SQL conditional operator (AND/OR)"

        #Like statement assumes that the value is a substring, so add wildcards before and after
        if "like" in self.op:
            return f"{self.column_select} {self.op} %{self.value}% {self.cond_op}"

        return f"{self.column_select} {self.op} {self.value} {self.cond_op}"
    
    def __eq__(self, other):
        if not isinstance(other, Condition):
            return NotImplemented
        
        return str(self) == str(other)


    def __hash__(self):
        return hash(str(self))

#####################
# Meta data classes #
#####################

SQL_TYPES = ("text","number", "others")

class DataBase():

    def __init__(self, data):
        self.tables = []
        self.tables_dict = {}

        if isinstance(data, dict):
            self.from_dict(data)
    
    def from_dict(self, data_dict):
        self.db_name = data_dict['db_id']
        self.primary_keys = data_dict['primary_keys']
        columns_all = []
        column_types_all = []
        table_names = data_dict['table_names_original']
        for (table_id, column_name), column_type in zip(data_dict['column_names_original'], data_dict['column_types']):
            column_name = str.lower(column_name)
            #Add new index to list as we encounter new tables
            if table_id >= len(columns_all):
                columns_all.append([])
                column_types_all.append([])
            #there is a table id -1 which we ignore for now
            if table_id>=0:
                columns_all[table_id].append(column_name)
                column_types_all[table_id].append(column_type)
        for table_name, columns, column_types in zip(table_names, columns_all, column_types_all):
            table_name =  str.lower(table_name)
            #Add the * column to all tables
            columns.append('*')
            column_types.append('text')

            table = Table(table_name, columns, column_types)
            self.tables.append(table)
            self.tables_dict[table_name] = table

    def to_list(self):
        """Returns a list of [column_name, column_type, table] for all columns in the table"""
        lst = []
        for table in self.tables:
            lst += [[col_name, col_type, table.table_name] for col_name, col_type in zip(table.columns, table.column_types)]
        return lst

    def get_column(self, column, table):
        #TODO: this will fail, if the predicted columns aren't all from the same table...
        return self.tables_dict[table].column_dict[column]
    def get_column_from_idx(self, idx):
        columns = []
        for table in self.tables:
            columns += list(table.column_dict.values())
        return columns[idx]

    # def get_idx_from_column(self, column):
    #     for column in 
class Table():

    def __init__(self, table_name, column_names, column_types):
        self.column_dict = {}

        for column_name, column_type in zip(column_names, column_types):
            col = Column(column_name, table_name, column_type)
            self.column_dict[column_name] = col
        self.table_name = table_name
        self.columns = column_names
        self.column_types = column_types

class Column():

    def __init__(self, column_name, table_name, column_type, database_name=""):
        self.column_name = column_name
        self.table_name = table_name
        self.column_type = column_type
        self.database_name = database_name
    
    def __str__(self):
        return self.column_name
    
    def __repr__(self):
        return str(self)

    def to_list(self):
        return [self.column_name, self.column_type, self.table_name]
        
if __name__ == "__main__":
    import json


    tables = json.load(open('tables.json'))
    data = json.load(open('train.json'))
    databases = {}
    for i in range(len(tables)):
        table = tables[i]
        databases[table['db_id']] = DataBase(table)
    
    for i in range(len(data)):
        try:
            sql = SQLStatement(data[i]['query'], databases[data[i]['db_id']])
            v1 = str.lower(str(sql).replace(' ',''))
            v2 = str.lower(data[i]['query'].replace(' ','')).replace(';','')
            if  v1 != v2 :
                print("#######")
                print(v1)
                print(v2)
                print("#######")
        except NotImplementedError:
            print(f"failed at {data[i]['query']}")
        

    
    print(sql)

