from src.preprocessing.FeatureEngineer import FeatureEngineer
from src.preprocessing.preprocessing import *
from sklearn.pipeline import Pipeline
import yaml
from pathlib import Path
import pandas as pd
from src.utils.train_test_split import get_train_test_data
import joblib

def create_pipeline(df,is_cat=False):

    pipeline = Pipeline([
        ('feature_engineer',FeatureEngineer()),
        ('preprocessing',create_preprocessing_pipeline(df,is_cat))
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
    return df,labels

def build_and_save_pipeline(is_cat=False):
    df,labels = _get_dataset()

    pipeline = create_pipeline(df,is_cat)
    X_train,X_test,y_train,y_test = get_train_test_data(df,labels)
    pipeline.fit(X_train,y_train)

    pipeline_path = None

    if is_cat:
        pipeline_path = parent/f'{config['pipeline']['preprocessing_cat']}'
    else:
        pipeline_path = parent/f'{config['pipeline']['preprocessing']}'

    joblib.dump(pipeline,pipeline_path)



if __name__ == '__main__':
     build_and_save_pipeline(False)
     build_and_save_pipeline(True)
