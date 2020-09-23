The source code documentation
===

The most important parts here are the [`transforms.py`](transforms.py), in which the objects for construction of the pipeline are defined; the [`experiment.py`](experiment.py) contains a set of functions to provide an easier way to cross-validate the models results; the [`app.py`](app.py) is pretty straightfoward and contains only the wrapper to deploy the model as web service using flask.

# The transforms (`transforms.py`)

To see how it works, take a look into the [example-01.py](example-01.py). It demonstrates how to use the classes of transformers and how to wrap everything into a **Pipeline** for `sklearn`. By the way, all the transformers developed here are based on the [scikit-learn structure](https://scikit-learn.org/stable/modules/classes.html), using the base classe to construct our models. If you are familiar with sklearn, we incorporate the functionality of the `.fit()`, `.transform()` and `.fit_transform()` methods to ensure easiliy operation with advanced functionality of the `Pipeline` class or even the `set_params()` or `get_params()`.

**class DataCleaning(BaseEstimator, TransformerMixin):**
 
    > Clean data according the procedures studied in the notebook analyses-02. In short: 
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
    df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.

**class RemoveFeatures(BaseEstimator, TransformerMixin):**
    
    > Remove unwanted features from the dataframes;
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


**class FeatureScaling(BaseEstimator, TransformerMixin):**
    > Scale features by standardization;
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

**class MergeFeatures(TransformerMixin):**

    > Concat the numerical and categorical dataframes into a single one.
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

**class DropNaN(TransformerMixin):**

    > Drop any row from the dataframe that contains a NaN.
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

**class FeatureSelection(TransformerMixin):**

    > Select the relevant features.
    -----
    Initialized parameters:
    - features: str or list of str containing the fields the should be kept

    Atrributes:
    self.features: feature names.

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

**class GetLables(TransformerMixin):**  
    
    > Get the labels following the user index in the feature dataframe.
    -----
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