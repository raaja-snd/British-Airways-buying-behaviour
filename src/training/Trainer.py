import pandas as pd
import yaml
from pathlib import Path
from src.training.LogisticRegressionTrainer import LogisticRegressionTrainer
from src.utils.logger import logger

parent = Path(__file__).resolve().parents[2]

with open(parent/'src/config.yaml') as c:
    config = yaml.safe_load(c)
dataset_path = parent/f'{config['dataset']['path']}'

def get_dataset():
    df = pd.read_csv(str(dataset_path),encoding='ISO-8859-1')
    labels = df['booking_complete']
    df = df.drop(columns=['booking_complete'])
    return df,labels

def train_logistic_regression(df:pd.DataFrame,labels:pd.Series):
    log_trainer = LogisticRegressionTrainer(df,labels)
    log_trainer.train()
    log_trainer.predict()


if __name__ in '__main__':
    df,labels = get_dataset()
    train_logistic_regression(df,labels)