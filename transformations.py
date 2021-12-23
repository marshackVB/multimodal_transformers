import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder,  PowerTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from transformers import AutoTokenizer
from datasets import Dataset
import warnings


# https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html


class AutoTokenize(BaseEstimator, TransformerMixin):
    def __init__(self, model_type, max_length, batch_size) -> None:
        self.model_type = model_type
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.feature_names = ['attention_mask', 'input_ids']


    def tokenize(self, batch):
        return self.tokenizer(batch['concatenated_text'], 
                              padding='max_length', 
                              truncation=True, 
                              max_length=self.max_length,
                              return_tensors="np")


    def fit(self, X, y=None):
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X.fillna("Missing", inplace=True)
        X['concatenated_text'] = X.apply(lambda row: '. '.join(row.values.astype(str)), axis=1)

        tokens = Dataset.from_pandas(X['concatenated_text'].to_frame())
        tokens = tokens.map(self.tokenize, batched=True, batch_size=self.batch_size)
        tokens.set_format("pandas", columns=self.feature_names)

        return tokens[:]


    def get_feature_names(self):
        return self.feature_names


def get_transformer(model_type, text_cols, numeric_cols, categorical_cols, max_length, tokenizer_batch_size):

    categorical = OneHotEncoder(sparse=False)

    numeric= make_pipeline(SimpleImputer(), 
                           PowerTransformer(method='yeo-johnson', standardize=True))

    text = make_pipeline(AutoTokenize(model_type, max_length, tokenizer_batch_size))

    transformer = ColumnTransformer(transformers=[('categorical_transform ', categorical , categorical_cols),
                                                  ('numerical_transform ',   numeric,      numeric_cols),
                                                  ('text_transform',         text,         text_cols)], 
                                                   remainder='drop')

    return transformer


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names