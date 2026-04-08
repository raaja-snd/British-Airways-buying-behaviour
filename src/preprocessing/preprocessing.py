from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, SplineTransformer, FunctionTransformer, OneHotEncoder
import numpy as np

def cyclic_name_handler(transformer,inputFeatures):
    return [f'{inputFeatures[0]}_sin',f'{inputFeatures[1]}_cosine']

# Helper function for sin and cosie encoding
def encode_hours(X):
    return np.column_stack([np.sin(2*np.pi*X/24), np.cos(2*np.pi*X/24)])

def encode_days(X):
    return np.column_stack([np.sin(2*np.pi*X/7), np.cos(2*np.pi*X/7)])
