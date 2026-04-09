from pathlib import Path
import yaml
from src.utils.train_test_split import get_train_test_data
import joblib
from src.preprocessing.data_cleaning import drop_outliers
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np


parent = Path(__file__).resolve().parents[2]
with open(parent/'src/config.yaml') as c:
        config = yaml.safe_load(c)

def _get_dataset():
    dataset_path = parent/f'{config['dataset']['path']}'

    df = pd.read_csv(str(dataset_path),encoding='ISO-8859-1')
    labels = df['booking_complete']
    df = df.drop(columns=['booking_complete'])
    
    return drop_outliers(df,labels)

def build_and_save_Cat_pipeline():
    df,labels = _get_dataset()

    X_train,X_test,y_train,y_test = get_train_test_data(df,labels)
    preprocess_pipeline = joblib.load(parent/f'{config['pipeline']['preprocessing_cat']}')
    model_path = parent/f'{config['model']['cat_boost']['path']}'
    model= joblib.load(model_path)
    pipeline = Pipeline([
         ('feature_preprocess',preprocess_pipeline),
         ('model',model)
    ])
    joblib.dump(pipeline,parent/f'{config['model']['cat_boost']['pipeline']}')

def build_and_save_pipeline():
    df,labels = _get_dataset()

    X_train,X_test,y_train,y_test = get_train_test_data(df,labels)
    preprocess_pipeline = joblib.load(parent/f'{config['pipeline']['preprocessing_cat']}')
    model_path = parent/f'{config['model']['cat_boost']['path']}'
    model= joblib.load(model_path)
    pipeline = Pipeline([
         ('feature_preprocess',preprocess_pipeline),
         ('model',model)
    ])
    joblib.dump(pipeline,parent/f'{config['model']['cat_boost']['pipeline']}')

if __name__ in '__main__':
     build_and_save_Cat_pipeline()
    