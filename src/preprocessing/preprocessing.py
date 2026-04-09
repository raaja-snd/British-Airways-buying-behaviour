from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, SplineTransformer, FunctionTransformer, OneHotEncoder
import numpy as np

def cyclic_name_handler(transformer,inputFeatures):
    return [f'{inputFeatures[0]}_sin',f'{inputFeatures[1]}_cosine']

# Helper function for sin and cosie encoding
def encode_hours(X):
    #Encode cyclic features using sin and cosine encoding
    return np.column_stack([np.sin(2*np.pi*X/24), np.cos(2*np.pi*X/24)])

def encode_days(X):
    return np.column_stack([np.sin(2*np.pi*X/7), np.cos(2*np.pi*X/7)])


def create_preprocessing_pipeline(df,is_cat=False):
    '''
    Create the preprocessing pipeline. If the flag is_cat is true,a preprocessing
    pipeline is created which does not one hot encode the string based categorical features.
    If not the string based categorical features are one hot encoded.

    Args:
        df(pd.Dataframe): British Airways customer buying behavior dataset
        is_cat: True or False. If True no one hot encoding
    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline
    '''
    num_continuous_cols = ['num_passengers', 'purchase_lead', 'length_of_stay',
         'flight_duration', 'extra_services_count']
    cyclic_cols = ['flight_hour','flight_day']
    cat_cols = df.select_dtypes(exclude='number').columns.to_list()

    preprocess_pipeline = None

    if is_cat:

        preprocess_pipeline = ColumnTransformer([
                ('continuous',StandardScaler(),num_continuous_cols),
                ('day_cyclic',FunctionTransformer(encode_days,feature_names_out=cyclic_name_handler),cyclic_cols[1]),
                ('hour_cyclic',FunctionTransformer(encode_hours,feature_names_out=cyclic_name_handler),cyclic_cols[0])
            ],remainder='passthrough')
    else:

        preprocess_pipeline = ColumnTransformer([
                    ('continuous',StandardScaler(),num_continuous_cols),
                    ('day_cyclic',FunctionTransformer(encode_days,feature_names_out=cyclic_name_handler),cyclic_cols[1]),
                    ('hour_cyclic',FunctionTransformer(encode_hours,feature_names_out=cyclic_name_handler),cyclic_cols[0]),
                    ('categorical',OneHotEncoder(drop='first',sparse_output=False),cat_cols)
                ],remainder='passthrough')
    return preprocess_pipeline
