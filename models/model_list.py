from models.col_predictor import ColPredictor
from models.keyword_predictor import KeyWordPredictor
from models.andor_predictor import AndOrPredictor
from models.agg_predictor import AggPredictor
from models.op_predictor import OpPredictor
from models.having_predictor import HavingPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor

models = {'column': ColPredictor, 'keyword': KeyWordPredictor, 'andor': AndOrPredictor, 'agg': AggPredictor, 'op': OpPredictor, 'having': HavingPredictor, 'desasc': DesAscLimitPredictor}
models_inverse = {'ColPredictor': 'column', 'KeyWordPredictor': 'keyword', 'AndOrPredictor': 'andor', 'AggPredictor': 'agg', 'OpPredictor': 'op', 'HavingPredictor': 'having', 'DesAscLimitPredictor': 'desasc'}
