from src.preprocessing.FeatureEngineer import FeatureEngineer
from src.preprocessing.preprocessing import *
from sklearn.pipeline import Pipeline
import yaml
from pathlib import Path
import pandas as pd
from src.utils.train_test_split import get_train_test_data
import joblib
from src.preprocessing.data_cleaning import drop_outliers

def create_pipeline(df,labels,is_frequency_encode=True,is_cat=False):

    feature_engineer = FeatureEngineer(is_frequency_encode)
    df_transformed = feature_engineer.fit_transform(df,labels)

    pipeline = Pipeline([
        ('feature_engineer',feature_engineer),
        ('preprocessing',create_preprocessing_pipeline(df_transformed,is_cat))
    ])
    pipeline.set_output(transform='pandas')
    return pipeline

parent = Path(__file__).resolve().parents[2]
with open(parent/'src/config.yaml') as c:
        config = yaml.safe_load(c)

def _get_dataset():
    dataset_path = parent/f'{config['dataset']['path']}'

    df = pd.read_csv(str(dataset_path),encoding='ISO-8859-1')
    labels = df['booking_complete']
    df = df.drop(columns=['booking_complete'])
    
    return drop_outliers(df,labels)

def build_and_save_pipeline(is_frequency_encode= True, is_cat=False):
    '''
    Build and save the feature engineering and preprocessing pipeline for British
    Airways customer buying behaviour dataset.
    Initialize with is_frequency_encode to either frequency encode the high cardinality features 
    ['departure','arrival','booking_origin'] or not using is_frequency_encode.
    Initialize with is_cat flag, if it is true,a preprocessing
    pipeline is created which does not one hot encode the string based categorical features.
    If not the string based categorical features are one hot encoded.

    Args:
        is_frequency_encode: Either True or False
        is_cat: True or False. If True no one hot encoding
    '''
    df,labels = _get_dataset()

    X_train,X_test,y_train,y_test = get_train_test_data(df,labels)
    pipeline = create_pipeline(X_train,y_train,is_frequency_encode,is_cat)
    X_train = pipeline.fit_transform(X_train,y_train)
    pipeline_path = None

    if is_cat:
        pipeline_path = parent/f'{config['pipeline']['preprocessing_cat']}'
    else:
        pipeline_path = parent/f'{config['pipeline']['preprocessing']}'

    joblib.dump(pipeline,pipeline_path)


def get_preprocessing_pipeline(is_cat=False):
    '''
    Get the fitted and saved preprocessing pipeline.
    Initialize with is_cat flag, if it is true,a preprocessing
    pipeline is returned, which does not one hot encode the string based categorical features.
    If not a preprocessing pipeline is returned where the string based
    categorical features are one hot encoded.

    Args:
        is_cat: True or False. If True no one hot encoding
    '''
    pipeline_path = None
    if is_cat:
        pipeline_path = parent/f'{config['pipeline']['preprocessing_cat']}'
    else:
        pipeline_path = parent/f'{config['pipeline']['preprocessing']}'
    return joblib.load(pipeline_path)

if __name__ == '__main__':
     build_and_save_pipeline(True,False)
     build_and_save_pipeline(False,True)
