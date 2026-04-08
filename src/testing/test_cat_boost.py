
import pandas as pd
from src.preprocessing.preprocessing import *
from src.preprocessing.feature_engineering import engineer_features
import joblib
from pathlib import Path
import yaml
from sklearn.metrics import classification_report

yaml_path = Path(__file__).resolve().parents[1]/'config.yaml'

with open(yaml_path) as c:
    config = yaml.safe_load(c)


def test_cat_model():
    parent_path  = Path(__file__).resolve().parents[2]
    df = pd.read_csv(str(parent_path/config['dataset']['path']),encoding='ISO-8859-1')
    model_path = parent_path/config['model']['cat_boost']['pipeline']
    model_pipeline = joblib.load(model_path)


    test_sample = df.sample(20)
    test_X = test_sample.drop(columns=['booking_complete'])
    test_y = test_sample['booking_complete']
    pred = model_pipeline.predict(test_X)
    print(classification_report(test_y,pred))




if __name__ in "__main__":
    test_cat_model()