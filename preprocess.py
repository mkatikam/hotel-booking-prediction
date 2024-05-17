from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.onehot = OneHotEncoder(handle_unknown='ignore')
        self.numerical_columns = [
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
            'no_of_week_nights', 'required_car_parking_space', 'lead_time',
            'arrival_year', 'arrival_month', 'arrival_date', 'repeated_guest',
            'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
            'avg_price_per_room', 'no_of_special_requests'
        ]
        self.categorical_columns = [
            'type_of_meal_plan', 'room_type_reserved', 'market_segment_type'
        ]

    def fit(self, X, y=None):
        self.imputer.fit(X[self.numerical_columns])
        self.scaler.fit(self.imputer.transform(X[self.numerical_columns]))
        self.onehot.fit(X[self.categorical_columns])
        return self

    def transform(self, X):
        X_num = self.imputer.transform(X[self.numerical_columns])
        X_num = self.scaler.transform(X_num)
        X_cat = self.onehot.transform(X[self.categorical_columns]).toarray()
        X_num_df = pd.DataFrame(X_num, columns=self.numerical_columns)
        X_cat_df = pd.DataFrame(X_cat, columns=self.onehot.get_feature_names_out())
        return pd.concat([X_num_df, X_cat_df], axis=1)


def make_prediction(data, pipeline, input_data):
    data['booking_status'] = data['booking_status'].map({1: 0, 2: 1})
    pipeline = Pipeline([
        ('preprocessor', Preprocessor()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['booking_status'], random_state=42)
    y_train = train_data['booking_status']
    X_train = train_data.drop(['booking_status', 'Booking_ID'], axis=1)
    y_test = test_data['booking_status']
    X_test = test_data.drop(['booking_status', 'Booking_ID'], axis=1)
    pipeline.fit(X_train, y_train)
    y_pred_for_all = pipeline.predict(train_data)
    y_pred = pipeline.predict(input_data)
    y_pred_proba = pipeline.predict_proba(input_data)[:, 1]

    return y_pred, y_pred_for_all



def data_preprocess(data):
    # pipeline = Pipeline([
    #     ('preprocessor', Preprocessor()),
    #     ('classifier', RandomForestClassifier(random_state=42))
    # ])

    train_data = data.drop(columns=['Booking_ID', 'booking_status'])
    test_data = data['booking_status']

    for col in train_data.select_dtypes(include=['float64', 'int64']).columns:
        train_data[col].fillna(train_data[col].mean(), inplace=True)

    # Categorical columns: Mode imputation
    for col in train_data.select_dtypes(include=['object']).columns:
        train_data[col].fillna(train_data[col].mode()[0], inplace=True)
    
    for col in train_data.select_dtypes(include=['float64', 'int64']):
        train_data[col].fillna(train_data[col].median(), inplace=True)
    
    numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

    # Check if there are numerical columns identified
    if len(numerical_cols) > 0:
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Scale the numerical columns
        train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
    # pipeline.fit(train_data, test_data)
    return train_data, test_data