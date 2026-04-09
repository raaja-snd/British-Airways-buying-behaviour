import yaml
from pathlib import Path
from src.preprocessing.preprocessing import *
from src.preprocessing.feature_engineering import engineer_features
from src.utils.train_test_split import get_train_test_data
from src.training.BaseTrainer import BaseTrainer
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
import joblib
from src.utils.logger import logger

class LogisticRegressionTrainer(BaseTrainer):

    def __init__(self,df:pd.DataFrame,labels:pd.Series):
        super().__init__()
        self.parent_path = Path(__file__).resolve().parents[2]
        with open(self.parent_path/'src/config.yaml') as c:
            config = yaml.safe_load(c)
        self.preprocessing_pipeline_path = config['pipeline']['preprocessing']
        self.df = df
        self.labels = labels
        self.X_train,self.X_test,self.y_train,self.y_test = get_train_test_data(
            self.df,
            self.labels
            )
        self.preproccessing_pipeline = joblib.load(self.preprocessing_pipeline_path)
        logger.info('Initialized Logistic Regression Trainer')
        
    
    def train(self):
        super().train()

        X_train_processed = self.preproccessing_pipeline.transforme(self.X_train)

        self.log_reg = LogisticRegressionCV(
        cv=5,penalty='l2',
        solver='liblinear',
        scoring='roc_auc',
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        n_jobs=-1)
        logger.info('Starting Logistic Regression Trainer')
        self.log_reg.fit(X_train_processed,self.y_train)
        logger.info('Training complete')

    
    def predict(self):
        super().predict()
        logger.info('Predicting')
        X_test_processed = self.preproccessing_pipeline.transforme(self.X_test)
        prediction = self.log_reg.predict(X_test_processed)
        report = classification_report(self.y_test,prediction)
        print(report)


