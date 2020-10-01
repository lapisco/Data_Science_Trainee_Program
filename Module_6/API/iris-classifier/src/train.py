import argparse
from os import path, mkdir

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

import joblib


DIR_NAME = path.dirname(__file__)


def load_data():
    df_data = pd.read_csv(path.join('..', 'data', 'iris.data'))

    X = df_data.iloc[:,:4].values
    y = df_data.iloc[:,4].values
    
    return X, y


def transform(X):
    """
    docstring
    """
    tf_std = StandardScaler().fit(X)

    pipe = Pipeline(
        [
            ('standard_scaler', tf_std), 
        ]
    )

    return pipe


def train(args):
    """
    Nothing yet
    """

    # data and lablels
    X, y = load_data()

    # transformations
    tf = transform(X)    
    clf = MLPClassifier(max_iter=2000)

    X_tf = tf.transform(X)
    clf.fit(X_tf, y)

    # Save models
    dump_folder = path.join(args['output_folder'], args['experiment_name'])
    
    if not path.exists(dump_folder):
        mkdir(dump_folder)
    
    # dump model
    filename = 'model_mlp_{}_v0.1.pkl'.format(args['model_name_tag'])
    joblib.dump(clf, filename=path.join(dump_folder, filename))
    # dump transformer
    filename = 'tf_std_{}_v0.1.pkl'.format(args['model_name_tag'])
    joblib.dump(tf, filename=path.join(dump_folder, filename))

    

if __name__ == "__main__":
    # Asterisk arguments
    parser = argparse.ArgumentParser(description='Iris classifier training 0.0.1')
    parser.add_argument('--experiment_name',  required=True, type=str)
    parser.add_argument('--output_folder',  default=path.join(DIR_NAME, 'models'), type=str)
    parser.add_argument('--model_name_tag',  required=True, type=str)
    args = vars(parser.parse_args())    
    
    train(args)