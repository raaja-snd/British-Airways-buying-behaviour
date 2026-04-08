
import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer

def create_base_features(df):
    #Map flight day in text to number in day of week
    days_mapping = {
        'Mon':1,
        'Tue':2,
        'Wed':3,
        'Thu':4,
        'Fri':5,
        'Sat':6,
        'Sun':7
    }
    df['flight_day'] = df['flight_day'].map(days_mapping)

    # Type of the stays (short, vacation, etc)
    stay_condition = [df['length_of_stay'] <= 15,df['length_of_stay'] <= 60, df['length_of_stay'] <= 180]
    df['stay_type'] = np.select(stay_condition,['short','vacation','temporary_residence'],default='residence')

    # Passenger grouping based on number of passengers
    passenger_condition = [df['num_passengers'] > 2,df['num_passengers'] == 2]
    df['passenger_kind'] = np.select(passenger_condition,['group','pair'],default='solo') 

    # 1. Traveling in am or pm  from flight_hour
    # 2. Whether travelling weekend or weekday from flight_day
    df['travel_am_pm'] = np.where(df['flight_hour']<12,'am','pm')
    df['weekend'] = np.where(df['flight_day'] <=5,0,1) 

    # Long or Short travel from fliht duration
    df['travelling_kind'] = np.where(df['flight_duration'] <=4.5,'short','long')

    # Split route into Departure and arrival features
    df[['departure','arrival']] = df['route'].str.extract(r'(.{3})(.{3})')
    df.drop(columns=['route'],inplace=True)

    # Category of leads from teh purchase_lead
    lead_condition = [
        df['purchase_lead'] <= 7,
        df['purchase_lead'] <= 30,
        df['purchase_lead'] <= 90
    ]
    df['lead_category'] = np.select(lead_condition,['last_minute','short_lead','medium_lead'],default='long_lead')

    # Period of travel (morning afternoon,etc)
    flight_period_condition = [
        df['flight_hour'] <6,
        df['flight_hour'] <12,
        df['flight_hour'] <18
    ]
    df['flight_period'] = np.select(flight_period_condition,['early_morning','morning','afternoon'],default='evening')
    return df

## Interaction Features

def create_interaction_features(df):

    # Passengers taking extra services (ordinal)
    df['extra_services_count'] = df[['wants_extra_baggage','wants_preferred_seat','wants_in_flight_meals']].sum(axis=1)

    # Passengers booking and travel behavior
    booking_conditions = [
        (df['purchase_lead'] < 7) & (df['stay_type'] == 'short'),
        (df['purchase_lead'] >= 7) & (df['stay_type'] == 'short'),
        (df['purchase_lead'] < 7) & (df['stay_type'].isin(['vacation','temporary_residence','residence'])),
        (df['purchase_lead'] >= 7) & (df['stay_type'].isin(['vacation','temporary_residence','residence']))
    ]
    df['booking_behavior'] = np.select(booking_conditions,['urgent','planned_vacation','urgent_long_stay','planned_long_stay'],default='standard')
    return df

## Feature Engineering

def engineer_features(df):
    df = df.copy(deep=True)
    df = create_base_features(df)
    df = create_interaction_features(df)
    return df

def get_feature_enginerring_pipeline():
    feature_engineering_pipeline = FunctionTransformer(engineer_features,validate=False)
    feature_engineering_pipeline.set_output(transform='pandas')
