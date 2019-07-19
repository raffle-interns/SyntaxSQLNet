
from models.col_predictor import ColPredictor
from models.keyword_predictor import KeyWordPredictor
from models.andor_predictor import AndOrPredictor
from models.agg_predictor import AggPredictor
from models.distinct_predictor import DistinctPredictor
from models.op_predictor import OpPredictor
from models.having_predictor import HavingPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.limit_value_predictor import LimitValuePredictor
from models.value_predictor import ValuePredictor

models = {'column': ColPredictor, 
          'keyword': KeyWordPredictor, 
          'andor': AndOrPredictor, 
          'agg': AggPredictor, 
          'distinct': DistinctPredictor, 
          'op': OpPredictor, 
          'having': HavingPredictor, 
          'desasc': DesAscLimitPredictor, 
          'limitvalue': LimitValuePredictor,
          'value':ValuePredictor
        }
models_inverse = {'ColPredictor': 'column', 
                  'KeyWordPredictor': 'keyword', 
                  'AndOrPredictor': 'andor', 
                  'AggPredictor': 'agg', 
                  'DistinctPredictor': 'distinct', 
                  'OpPredictor': 'op', 
                  'HavingPredictor': 'having', 
                  'DesAscLimitPredictor': 'desasc', 
                  'LimitValuePredictor': 'limitvalue',
                  'ValuePredictor': 'value'
                }
