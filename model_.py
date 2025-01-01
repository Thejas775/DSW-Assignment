import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle

class BaseModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def load(self, data_path):
        """Load data from file"""
        self.data = pd.read_excel(data_path)
        return self
    
    def preprocess(self, data, is_training=True):
        """Preprocess the data"""
        df = data.copy()
        
        # Convert date
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['transaction_month'] = df['transaction_date'].dt.month
        df['transaction_year'] = df['transaction_date'].dt.year
        
        # Convert term
        df['term'] = df['term'].str.strip().str.replace(' months', '').astype(int)
        
        # Encode categorical variables
        categorical_cols = ['sub_grade', 'home_ownership', 'purpose', 
                          'application_type', 'verification_status']
        
        for col in categorical_cols:
            if is_training:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Select features
        feature_cols = ['cibil_score', 'total_no_of_acc', 'annual_inc', 'int_rate',
                       'loan_amnt', 'installment', 'account_bal', 'emp_length', 
                       'term', 'transaction_month', 'transaction_year'] + categorical_cols
        
        X = df[feature_cols]
        
        # Scale numerical features
        numerical_cols = ['cibil_score', 'total_no_of_acc', 'annual_inc', 'int_rate',
                         'loan_amnt', 'installment', 'account_bal', 'emp_length']
        
        if is_training:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        if 'loan_status' in df.columns:
            y = df['loan_status']
        else:
            y = None
            
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        raise NotImplementedError("Train method must be implemented by child class")
    
    def test(self, test_data_path):
        """Test the model and generate evaluation metrics"""
        test_data = pd.read_excel(test_data_path)
        X_test, y_test = self.preprocess(test_data, is_training=False)
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        return results
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path):
        """Save the model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler
            }, f)
            
    def load_model(self, path):
        """Load the saved model and preprocessors"""
        with open(path, 'rb') as f:
            saved_objects = pickle.load(f)
            self.model = saved_objects['model']
            self.label_encoders = saved_objects['label_encoders']
            self.scaler = saved_objects['scaler']

class LogisticRegressionModel(BaseModel):
    def __init__(self, params=None):
        super().__init__()
        self.params = params or {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }
        
    def train(self, X, y):
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)
        return self

class RandomForestModel(BaseModel):
    def __init__(self, params=None):
        super().__init__()
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
    def train(self, X, y):
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)
        return self

class GradientBoostingModel(BaseModel):
    def __init__(self, params=None):
        super().__init__()
        self.params = params or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        
    def train(self, X, y):
        self.model = GradientBoostingClassifier(**self.params)
        self.model.fit(X, y)
        return self