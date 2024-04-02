from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

class DataEncoderDecoder():
    def __init__(self):
        self.cat_cols = None
        self.cont_cols = None
        self.label_encoders = {}
        self.scaler = None

    def fit_encode(self, df):
        # Identify categorical and continuous columns
        self.cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.cont_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
        
        # Encode categorical columns
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Scale continuous columns
        self.scaler = StandardScaler()
        df[self.cont_cols] = self.scaler.fit_transform(df[self.cont_cols])

        return df

    def decode(self, df_encoded):
        # Inverse scaling for continuous columns
        df_encoded[self.cont_cols] = self.scaler.inverse_transform(df_encoded[self.cont_cols])
        
        # Inverse encoding for categorical columns
        for col, le in self.label_encoders.items():
            df_encoded[col] = le.inverse_transform(df_encoded[col].astype(int))
        
        return df_encoded