import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


'''
The transformations are applied in the dataframe in the following order:
1 - Data cleaning: remove NaN and split into categorical and numerical features
2 - Remove unwanted features in numerical features
3 - Remove unwanted features in categorical features
4 - Feature scaling in numerical features (This one is created by using StandardScaler from sklearn)
5 - Concat two features set
6 - Feature selction (Optional)
7 - Drop NaN that might appear from the scaling tranformation
'''

SELECTED_FEATURES = [
       'total_mediaids', 'age_without_access', 'idade', 'mobile_web_time',
       'video_info_time_spent_0_5', 'total_dependents',
       'total_active_dependents', 'total_cancels', 'month_subs',
       'assinatura_age']


PAYMENT_TYPE = ['BOLETO WEB', 'DEBITO AUTOMATICO', 'IN APP PURCHASE', 'CARTAO DE CREDITO']

# CATEGORICAL_FEATURES = ['sexo'] + PAYMENT_TYPE
CATEGORICAL_FEATURES = ['tipo_de_cobranca', 'sexo']


class Encoder(BaseEstimator, TransformerMixin):
    ''' 
    Clean data according the procedures studied in the notebook analyses-02. In short: 
    (i) Drops Nan; (ii) split data in categorical and numerical features; 
    (iii) 1-hot-enconding of categorical features; (iv) Get a unique categorical features
    of an user in a period of 16 weeks; (v)  Get a unique numerical features of an user 
    in a period of 16 weeks; (vi) Average the numerical features in a period of 16-week;
    (vii) contat both feature set;
    -----
    Methods
    ------------------------
    > fit(df)
    Parameters:
    df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def __init__(self):
        self.label_binarizer = LabelBinarizer()
        self.label_encoder = LabelEncoder()
        pass
    
    def fit(self, X):
        # Hot-encoding in categorical features:
        self.label_encoder.fit(X[:,1])
        self.label_binarizer.fit(X[:,0])
    
        return self

    def transform(self, X):
        
        # sexo1hot = pd.DataFrame(data=self.label_encoder.transform(df.sexo), index=df.index, columns=['sexo'])
        # sexo1hot = self.label_encoder.transform(df.sexo)
        # cobranca1hot = pd.DataFrame(data=self.label_binarizer.transform(df.tipo_de_cobranca), 
        #                             index=df.index, columns=df.tipo_de_cobranca.unique())
        
        # categorical = pd.concat([sexo1hot, cobranca1hot], axis=1)
        # numerical = df.drop(columns=['sexo', 'tipo_de_cobranca', 'cidade', 'estado'])

        a_1 = self.label_encoder.transform(X[:,1])
        a_2 = self.label_binarizer.transform(X[:,0])

        return np.column_stack((a_1, a_2))

class DataCleaning(BaseEstimator, TransformerMixin):
    ''' 
    Clean data according the procedures studied in the notebook analyses-02. In short: 
    (i) Drops Nan; (ii) split data in categorical and numerical features; 
    (iii) 1-hot-enconding of categorical features; (iv) Get a unique categorical features
    of an user in a period of 16 weeks; (v)  Get a unique numerical features of an user 
    in a period of 16 weeks; (vi) Average the numerical features in a period of 16-week;
    (vii) contat both feature set;
    -----
    Methods
    ------------------------
    > fit(df)
    Parameters:
    df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def __init__(self, categorical_features=CATEGORICAL_FEATURES):
        self.categorical_features = categorical_features
    
    def fit(self, df):
        # Ensure the order of column in the test set is in the same order than in train set
        df_buffer = df.loc[df['week'] <= 17]
        categorical_agg = df_buffer[CATEGORICAL_FEATURES].groupby(['user']).agg('max')
        numerical_agg = df_buffer.drop(columns=CATEGORICAL_FEATURES).groupby(['user']).agg('mean')
        
        # categorical_features_unique = df_buffer.groupby(['user']).agg('max')

        # # Average of the numerical features:
        # df_buffer = numerical_features.loc[numerical_features['week'] < 17]
        # numerical_features_average = df_buffer.groupby(['user']).agg('mean')

        # Drop week because it is not relevant
        # categorical_agg.drop(columns=['week'], inplace=True)
        # categorical_features_unique.drop(columns=['week'], inplace=True)
        
        self.df_out = pd.concat([numerical_agg, categorical_agg], axis=1)

        return self

    def transform(self, df):
        
        return self.df_out


class CategoricalProcessing(BaseEstimator, TransformerMixin):
    ''' 
    Clean data according the procedures studied in the notebook analyses-02. In short: 
    (i) Drops Nan; (ii) split data in categorical and numerical features; 
    (iii) 1-hot-enconding of categorical features; (iv) Get a unique categorical features
    of an user in a period of 16 weeks; (v)  Get a unique numerical features of an user 
    in a period of 16 weeks; (vi) Average the numerical features in a period of 16-week;
    (vii) contat both feature set;
    -----
    Methods
    ------------------------
    > fit(df)
    Parameters:
    df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''
    
    def fit(self, df):
        return self

    def transform(self, df):
        categorical_features = df
        # Remove NaN:
        # df_clean = df.dropna(how='any', inplace=False)
        
        # Get categorical and numerical features
        # categorical_features = df_clean[['sexo', 'tipo_de_cobranca']]
        # numerical_features = df_clean.drop(columns=['sexo', 'tipo_de_cobranca', 'cidade', 'estado'])
        
        # Hot-encoding in categorical features:
        sexo_1hot = categorical_features.sexo.map({'F': 0, 'M': 1})
        cobranca_1hot = pd.get_dummies(categorical_features.tipo_de_cobranca)
        # self._dummies_frame = pd.get_dummies(cobranca_1hot)
        # cobranca_1hot.reindex(columns = self._dummies_frame.columns, fill_value=0)
        
        # Add a missing column in test set with default value equal to 0
        for payment in set(PAYMENT_TYPE) - set(cobranca_1hot.columns):
            cobranca_1hot[payment] = 0

        # Ensure the order of column in the test set is in the same order than in train set
        categorical_features_1hot = pd.concat([df['week'], sexo_1hot, cobranca_1hot], axis=1)
        df_buffer = categorical_features_1hot.loc[categorical_features_1hot['week'] < 17]
        categorical_features_unique = df_buffer.groupby(['user']).agg('max')

        return categorical_features_unique.values

class Imputer(BaseEstimator, TransformerMixin):
    ''' feat_clean.head()
    Nothing yet
    -----
    Methods
    ------------------------
    > fit(df)
    Parameters:
    df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    - X: numpy array
    
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self._imp = SimpleImputer(missing_values=np.nan, strategy=self.strategy)

    def fit(self, df, y=None):
        self._imp.fit(df)
        self._columns = df.columns
        self._index = df.index
        return self

    def transform(self, df):
        # Average of the numerical features:
        out = self._imp.transform(df)
        # return pd.DataFrame(data=out, columns=self._columns, index=self._index)
        return out
        


class AggregateUserInfo(BaseEstimator, TransformerMixin):
    ''' 
    Clean data according the procedures studied in the notebook analyses-02. In short: 
    (i) Drops Nan; (ii) split data in categorical and numerical features; 
    (iii) 1-hot-enconding of categorical features; (iv) Get a unique categorical features
    of an user in a period of 16 weeks; (v)  Get a unique numerical features of an user 
    in a period of 16 weeks; (vi) Average the numerical features in a period of 16-week;
    (vii) contat both feature set;
    -----
    Methods
    ------------------------
    > fit(df)
    Parameters:
    df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''
    
    def fit(self, df):
        return self

    def transform(self, df):
        # Average of the numerical indexfeatures:
        df_buffer = df.loc[df['week'] < 17]
        df = df_buffer.groupby(['user']).agg('mean')

        return df.values



class RemoveFeatures(BaseEstimator, TransformerMixin):
    ''' 
    Remove unwanted features from the dataframes;
    -----
    Initialized parameters:
    - features: str or list cointaining the field that ought to be removed. Default: 'week'.

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def __init__(self, features='week'):
        self.features = features

    def fit(self, df):
        return self

    def transform(self, df):
        
        return {'numerical': df['numerical'].drop(columns=self.features),
                'categorical': df['categorical'].drop(columns=self.features)}

class FeatureScaling(BaseEstimator, TransformerMixin):
    ''' 
    Scale features by standardization;
    -----
    Initialized parameters:
    - type: str cointaining the scaling method. Default: 'std'.
        - 'std': StandardScaler()

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    self

    Atrributes:
    self._scaler: saved object that sould be used along with the trained model.
    
    > transform(df)

    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def __init__(self, type='std'):
        self.type = type

    def fit(self, df):
        self._scaler = StandardScaler().fit(df['numerical'])
        
        return self

    def transform(self, df):
        if self.type == 'std':
            df_std = self._scaler.transform(df['numerical'])
            df_std = pd.DataFrame(data=df_std, 
                                  columns=df['numerical'].columns, 
                                  index=df['numerical'].index)
        
        return {'numerical': df_std, 'categorical': df['categorical']}

class MergeFeatures(TransformerMixin):
    ''' 
    Concat the numerical and categorical dataframes into a single one.
    -----

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    - dataframe: a daframe with both feature set.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def fit(self, df):
        return self

    def transform(self, df):
        return pd.concat([df['numerical'], df['categorical']], axis=1)

class DropNaN(TransformerMixin):
    ''' 
    Drop any row from the dataframe that contains a NaN.
    -----

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - df: a dataframe
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: a dataframe
    -----
    Returns:
    - dataframe: a daframe withou NaN.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def fit(self, df):
        return self

    def transform(self, df):
        return df.dropna()

class DataFrameSelector(BaseEstimator, TransformerMixin):
    ''' 
    Select the relevant features.
    -----
    Initialized parameters:info_data.values
    - attribute_names: str or list of str containing the fields the should be kept

    Atrributes:
    self.attribute_names: attribute names to get from the DB.

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - df: a dataframe.
    -----
    Returns:
    self

    > transform(df)

    Parameters:
    - df: a dataframe.
    -----
    Returns:
    - df: a dataframe.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''
    def __init__(self, attribute_names, drop=False):
        self.attribute_names = attribute_names
        self.drop = drop

    def fit(self, df):
        return self

    def transform(self, df):
        if self.drop:
            return df.drop(columns=self.attribute_names).values
        else:
            return df[self.attribute_names].values


class GetLables(TransformerMixin):
    ''' 
    Get the labels following the user index in the feature dataframe.
    -----append('week')append('week')
    Methods
    ------------------------
    > fit(df_user, df_features)
    Parameters:
    - df_user: dataframe containing the user's data
    - df_features: dataframe the outta be used as the feature set. It MUST contain
    the user's name as index.
    -----
    Returns:
    self

    > transform(df_user, df_features)
    
    Parameters:
    - df_user: dataframe containing the user's data
    - df_features: dataframe the outta be used as the feature set. It MUST contain
    the user's name as index.
    -----
    Returns:
    - df: a dataframe.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def fit(self, df_user, df_features):
        return self

    def transform(self, df_user, df_features):
        df_user_clean = df_user.loc[df_features.index.unique()]
        return df_user_clean.loc[df_features.index]