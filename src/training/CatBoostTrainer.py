from src.utils.train_test_split import get_train_test_data
from src.training.BaseTrainer import BaseTrainer
from src.preprocessing.Pipeline import get_preprocessing_pipeline
import pandas as pd
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.utils.logger import logger

class CatBoostTrainer(BaseTrainer):

    def __init__(self,df:pd.DataFrame,labels:pd.Series):
        super().__init__()
        
        self.df = df
        self.labels = labels
        self.X_train,self.X_test,self.y_train,self.y_test = get_train_test_data(
            self.df,
            self.labels
            )
        self.preproccessing_pipeline = get_preprocessing_pipeline(is_cat=True)
        logger.info('Initialized CatBoost Trainer')
        
    
    def train(self):
        super().train()

        X_train_processed = self.preproccessing_pipeline.transform(self.X_train)
        cat_features_model = X_train_processed.select_dtypes(include=['str','object']).columns.to_list()

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cat_params = {
            'iterations': [500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 7],
            'auto_class_weights': ['Balanced', 'SqrtBalanced'] # Awesome built-in imbalance handling
        }

        cat = CatBoostClassifier(verbose=0, random_state=42,thread_count=-1)

        self.cat_cv = RandomizedSearchCV(
            cat,
            param_distributions=cat_params,
            n_iter=15,
            scoring='f1',
            cv=cv_strategy,
            n_jobs=-1
        )
                
        # If you use raw data, specify cat_features=[index_of_categorical_columns]
        logger.info('Starting CatBoost Trainer')
        self.cat_cv.fit(X_train_processed, self.y_train,cat_features=cat_features_model,early_stopping_rounds=50)
        logger.info('Training complete')

    
    def predict(self):
        super().predict()
        logger.info('Predicting')
        X_test_processed = self.preproccessing_pipeline.transform(self.X_test)
        cat_pred = self.cat_cv.best_estimator_.predict(X_test_processed)
        report = classification_report(self.y_test,cat_pred)
        print(report)


