import pandas as pd
import yaml
from pathlib import Path
from src.training.LogisticRegressionTrainer import LogisticRegressionTrainer
from src.training.RandomForestTrainer import RandomForestTrainer
from src.training.XGBoostTrainer import XGBoostTrainer
from src.training.CatBoostTrainer import CatBoostTrainer
from src.preprocessing.data_cleaning import drop_outliers
from src.utils.logger import logger

parent = Path(__file__).resolve().parents[2]

with open(parent/'src/config.yaml') as c:
    config = yaml.safe_load(c)
dataset_path = parent/f'{config['dataset']['path']}'

def get_dataset():
    df = pd.read_csv(str(dataset_path),encoding='ISO-8859-1')
    labels = df['booking_complete']
    df = df.drop(columns=['booking_complete'])
    return drop_outliers(df,labels)

def train_logistic_regression(df:pd.DataFrame,labels:pd.Series):
    log_trainer = LogisticRegressionTrainer(df,labels)
    log_trainer.train()
    log_trainer.predict()

def train_rf(df:pd.DataFrame,labels:pd.Series):
    rf_trainer = RandomForestTrainer(df,labels)
    rf_trainer.train()
    rf_trainer.predict()

def train_xgb(df:pd.DataFrame,labels:pd.Series):
    xgb_trainer = XGBoostTrainer(df,labels)
    xgb_trainer.train()
    xgb_trainer.predict()

def train_rf(df:pd.DataFrame,labels:pd.Series):
    cat_trainer = CatBoostTrainer(df,labels)
    cat_trainer.train()
    cat_trainer.predict()


if __name__ in '__main__':
    df,labels = get_dataset()
    train_xgb(df,labels)