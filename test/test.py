import sys
sys.path.append('./open-meteo')
import common

from db.connector import DatabaseConnector
#import preprocess
from utils.preprocess import DataPreprocessor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

class ModelEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return joblib.load(self.model_path)

    def evaluate_model(self, X, y):
     
        y_pred = self.model.predict(X)
        score = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(rmse)
        return score, rmse

if __name__ == "__main__":
    # Load data
    
    class DatabaseHandler:
        def __init__(self):
            self.conn = DatabaseConnector.connect()
            self.data1 = self.conn
            self.data = None
            self.train_data = None
            self.test_data = None

        def preprocess_data(self):
            self.data1, self.data, self.train_data, self.test_data = DataPreprocessor.preprocess(self.data1)
            return self.data1, self.data, self.train_data, self.test_data
    

    db_handler = DatabaseHandler()
    train_add_col, test_add_col, train, test = db_handler.preprocess_data()

    X_train, X_test, y_train, y_test, train_data_2variable, test_data_2variable = DataPreprocessor.preprocessingSplit(train_add_col)

    # Initialize ModelEvaluator instance
    model_evaluator = ModelEvaluator(f'{common.MODEL_PATH}randomforest')

    # Evaluate model
    score_test, rmse_test = model_evaluator.evaluate_model(X_test, y_test)
    print(f'R2  : {score_test}')
    print(f'RMSE : {rmse_test}')
