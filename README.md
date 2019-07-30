Pytorch implementation of [SyntaxSQLNet: Syntax Tree Networks for Complex and Cross-DomainText-to-SQL Task
](https://arxiv.org/abs/1810.05237).

## Improvements
Our model contains several improvements over the original model:
1. The values in WHERE and HAVING conditions, aswell as the value for LIMIT are ignored in the SPIDER evaluation. To be more usefull in practise, our model includes a module for predicting these values. The module is similar to the column predictor, but selects one or more tokens from the question. 
2. A module for predicting the DISTINCT keyword
3. Added the BETWEEN operator
4. Improved the column predictor, to make it possible to predict the same column multiple times.

With these changes, our model achives the following accuracy on easy+medium questions, where we include the values.

| Component   | Accuracy |
|-------------|----------|
| SELECT      | 72.5%    |
| WHERE       | 48.6%    |
| GROUP BY    | 63.5%    |
| ORDER BY    | 65.9%    |
| HAVING      | 88.9%    |
| LIMIT value | 94.9%    |
| KEYWORDS    | 94.4%    |
| **Total**   | 49.0%    |


## Setup
1. Python >= 3.6
2. Install dependencies using ``pip install -r requirements.txt``


### Data
The data for the model can be downloaded from [Spider Dataset website](https://yale-lily.github.io/spider). 
Note that this model only focuses on easy and medium difficulty queries, meaning that we don't include multi table queries, like joins or sub-queries.

To generate augmented data, you also need to download ``wikisql_tables.json`` from [here](https://drive.google.com/file/d/13I_EqnAR4v2aE-CWhJ0XQ8c-UlGS9oic/view?usp=sharing)

The pretrained embeddings can be downloaded from the [Glove website](https://nlp.stanford.edu/projects/glove/)


## Training
Run ``python train.py`` to train each module
It takes the following arguments:
```
  --num_layers        Number of layers in the LSTMs
  --lr                Learnign rate
  --num_epochs 
                      Number of epochs to train the model
  --batch_size 
  --name_postfix 
                      Optional postfix of the model name
  --gpu 
  --hidden_dim 
  --save              Save the model during training
  --dropout 
  --embedding_dim 
                      Dimension of the embeddings
  --num_augmentation 
                      Number of additional augmented questions to generate
  --N_word            Number of trained tokens for the embedding, this just
                        corresponds to the name
  --model             Select a model from {column,keyword,andor,agg,distinct,op,having,desasc,limitvalue,value}
```

# Testing
Run ``python test.py`` to generate the test results
