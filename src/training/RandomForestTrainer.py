from src.utils.train_test_split import get_train_test_data
from src.training.BaseTrainer import BaseTrainer
from src.preprocessing.Pipeline import get_preprocessing_pipeline
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.utils.logger import logger

class RandomForestTrainer(BaseTrainer):

    def __init__(self,df:pd.DataFrame,labels:pd.Series):
        super().__init__()
        
        self.df = df
        self.labels = labels
        self.X_train,self.X_test,self.y_train,self.y_test = get_train_test_data(
            self.df,
            self.labels
            )
        self.preproccessing_pipeline = get_preprocessing_pipeline()
        logger.info('Initialized Random Forest Trainer')
        
    
    def train(self):
        super().train()

        X_train_processed = self.preproccessing_pipeline.transform(self.X_train)

        rf = RandomForestClassifier(class_weight='balanced',random_state=42,n_jobs=-1)

        rf_parameter_dist = {
            'n_estimators':[100,200,300,500,1000],
            'max_depth':[10,20,30,50,75],
            'min_samples_split':[2,5,10],
            'min_samples_leaf':[1,2,4],
            'max_features':['sqrt','log2'],
            'bootstrap':[True,False]
        }

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self.rfcv =  RandomizedSearchCV(
                    rf,
                    rf_parameter_dist,
                    n_iter=20,
                    random_state=42,
                    cv=cv_strategy,
                    scoring='f1',
                    n_jobs=-1
                )

        logger.info('Starting Random Forest Trainer')
        self.rfcv.fit(X_train_processed,self.y_train)
        logger.info('Training complete')

    
    def predict(self):
        super().predict()
        logger.info('Predicting')
        X_test_processed = self.preproccessing_pipeline.transform(self.X_test)
        rf_pred = self.rfcv.best_estimator_.predict(X_test_processed)
        report = classification_report(self.y_test,rf_pred)
        print(report)


