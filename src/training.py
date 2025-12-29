from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

class MarchMadnessModel:
    # initialize everything
    def __init__(self, model_type='xgboost'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = model_type
    
    # splits the data
    def split_data(self, X, y, test_size=0.2, temporal=True):
        if temporal:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        return X_train, X_test, y_train, y_test
    
    # trains the model
    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(n_estimators=100, max_depth=5)
        
        self.model.fit(X_scaled, y_train)
    
    # evaluates the model
    def evaluate(self, X_test, y_test):
        X_scaled = self.scaler.transform(X_test)
        preds = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        
        return accuracy, auc