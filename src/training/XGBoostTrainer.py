from src.utils.train_test_split import get_train_test_data
from src.training.BaseTrainer import BaseTrainer
from src.preprocessing.Pipeline import get_preprocessing_pipeline
import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.utils.logger import logger

class XGBoostTrainer(BaseTrainer):

    def __init__(self,df:pd.DataFrame,labels:pd.Series):
        super().__init__()
        
        self.df = df
        self.labels = labels
        self.X_train,self.X_test,self.y_train,self.y_test = get_train_test_data(
            self.df,
            self.labels
            )
        self.preproccessing_pipeline = get_preprocessing_pipeline()
        logger.info('Initialized XGBoost Trainer')
        
    
    def train(self):
        super().train()

        X_train_processed = self.preproccessing_pipeline.transform(self.X_train)

        xg_params = {
            'n_estimators':[100,300,500,1000],
            'learning_rate':[0.01, 0.05, 0.1, 0.2],
            'max_depth':[2,4,7,9],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [1, 3, 5.7, 7]
            }
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        xgb = XGBClassifier(objective='binary:logistic',random_state=42,use_label_encoder=False)

        self.xgb_cv = RandomizedSearchCV(
            xgb,
            param_distributions=xg_params,
            n_iter=20,
            scoring='f1',
            cv=cv_strategy,
            random_state=42,
            n_jobs=-1,
            error_score='raise'
        )

        logger.info('Starting XGBoost Trainer')
        self.xgb_cv.fit(X_train_processed,self.y_train)
        logger.info('Training complete')

    
    def predict(self):
        super().predict()
        logger.info('Predicting')
        X_test_processed = self.preproccessing_pipeline.transform(self.X_test)
        xg_pred = self.xgb_cv.best_estimator_.predict(X_test_processed)
        report = classification_report(self.y_test,xg_pred)
        print(report)


