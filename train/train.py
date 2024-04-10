
import sys
sys.path.append('./open-meteo')

import common
from db.connector import DatabaseConnector

#import preprocess
from utils.preprocess import DataPreprocessor
import os
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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

class ModelTrainer:
    def __init__(self):
        pass

    def ARIMA(self, train_data):
        order=(10,1,10)
        model = ARIMA(train_data, order=order)
        model_fit2 = model.fit()
        return model_fit2

    def SARIMA(self, train_data):
        order=(5,1,7)
        seasonal_order =(0,0,0,0)
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        return model_fit

    def SARIMAX(self, train_data, test_data, train_data_2variable, test_data_2variable):
        order = (5,1,7)
        seasonal_order = (1, 1, 2, 9)
        model_sarimax_exog = SARIMAX(train_data, exog=train_data_2variable, order=order,seasonal_order=seasonal_order).fit()
        ts_pred_exog = model_sarimax_exog.predict(start=test_data.index[0], end=test_data.index[-1],exog=test_data_2variable)
        rmse = mean_squared_error(test_data.values, ts_pred_exog.values, squared=False)
        return rmse, ts_pred_exog

    def LR(self, X_train, y_train, X_test, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model

    def RF(self, X_train, y_train, X_test, y_test):
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model

class Model:
    @staticmethod
    def save_model(model, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        joblib.dump(model, filepath)
        return filepath


db_handler = DatabaseHandler()
train_add_col, test_add_col, train, test = db_handler.preprocess_data()

X_train, X_test, y_train, y_test, train_data_2variable, test_data_2variable = DataPreprocessor.preprocessingSplit(train_add_col)


trainer = ModelTrainer()
Model_ARIMA = trainer.ARIMA(train)
Model_SARIMA = trainer.SARIMA(train)
Model_SARIMAX, _ = trainer.SARIMAX(train, test, train_data_2variable, test_data_2variable)
Model_RegressionLineaire = trainer.LR(X_train, y_train, X_test, y_test)
Model_RandomForest = trainer.RF(X_train, y_train, X_test, y_test)

model_register = Model()
model_register.save_model(Model_ARIMA, common.MODEL_PATH, 'arima')
model_register.save_model(Model_SARIMA, common.MODEL_PATH, 'sarima')
model_register.save_model(Model_SARIMAX, common.MODEL_PATH, 'sarimax')
model_register.save_model(Model_RegressionLineaire, common.MODEL_PATH, 'regression')
model_register.save_model(Model_RandomForest, common.MODEL_PATH, 'randomforest')
