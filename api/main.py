import sys
sys.path.append('./open-meteo')

import common

from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
from datetime import datetime
import pandas as pd
import sqlite3
from loguru import logger
import uvicorn
from typing import Optional

class InputData(BaseModel):
    step: int
    start_date: str  
    end_date: str    

class PredictionHandler:
    def __init__(self):
        # Load the ARIMA model
        self.model = joblib.load(f'{common.MODEL_PATH}arima')

    def create_dataframe(self, start_dates, prediction_values):
        df = pd.DataFrame({'date': start_dates, 'prediction': prediction_values})
        return df

    def save_to_sqlite(self, df, database_path=common.DB_PRED):
        try:
            conn = sqlite3.connect(database_path)
            df.to_sql('pred_db', conn, if_exists='append', index=False)
            conn.close()
            logger.info("Predictions successfully saved in the database.")
        except Exception as e:
            logger.error(f"Error while saving predictions: {str(e)}")

    def predict(self, input_data: InputData):
        try:
            logger.info(f"Predictions requested for dates {input_data.start_date} to {input_data.end_date}")
            start_date = datetime.strptime(input_data.start_date, "%Y-%m-%d %H:%M:%S")
            end_date = datetime.strptime(input_data.end_date, "%Y-%m-%d %H:%M:%S")
            start_dates = []
            prediction_values = []
            current_date = start_date
            while current_date <= end_date:
                prediction = self.model.predict(start=current_date, end=current_date, dynamic=False)
                start_dates.append(current_date)
                prediction_values.append(prediction[0])
                current_date += pd.Timedelta(hours=3)

            logger.info("Predictions successfully made")
            df_predictions = self.create_dataframe(start_dates, prediction_values)
            self.save_to_sqlite(df_predictions)

            return {"predictions": df_predictions.to_dict(orient='records')}

        except Exception as e:
            logger.error(f"Error during predictions: {str(e)}")
            return {"error": str(e)}

    def fetch_csv_data(self, start_date=None, end_date=None):
        try:
            df = pd.read_csv(f'{common.CURRENT_PATH}data/hourly_dataframe.csv')
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            if start_date and end_date:
                range_time = (df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)
            elif start_date:
                range_time = df['Timestamp'] >= start_date
            elif end_date:
                range_time = df['Timestamp'] <= end_date
            else:
                return df.to_dict(orient='records')

            filtered_data = df.loc[range_time]
            return filtered_data.to_dict(orient='records')

        except Exception as e:
            return {"error": str(e)}

app = FastAPI()
logger.add(common.LOG_PATH)
prediction_handler = PredictionHandler()

@app.get("/")
def root():
    logger.info("Root endpoint called")
    return {"message": "Working API"}

@app.post("/predict")
async def predict(input_data: InputData):
    return prediction_handler.predict(input_data)

@app.get("/predict_range")
async def fetch_predictions(start_date: Optional[str] = Query(None, description="Date de début au format YYYY-MM-DD"),
                            end_date: Optional[str] = Query(None, description="Date de fin au format YYYY-MM-DD")):
    try:
        logger.info("Fetching predictions from the database")
        conn = sqlite3.connect(common.DB_PRED)
        if start_date and end_date:
            query = f"SELECT * FROM pred_db WHERE date BETWEEN '{start_date}' AND '{end_date}'"
        elif start_date:
            query = f"SELECT * FROM pred_db WHERE date >= '{start_date}'"
        elif end_date:
            query = f"SELECT * FROM pred_db WHERE date <= '{end_date}'"
        else:
            query = "SELECT * FROM pred_db"

        df_predictions = pd.read_sql_query(query, conn)
        conn.close()

        predictions_list = df_predictions.to_dict(orient='records')
        logger.info("Predictions successfully fetched from the database")
        return {"predictions": predictions_list}

    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        return {"error": "Error fetching predictions"}

@app.get("/predict_uploads")
async def fetch_combined_predictions(start_date: Optional[str] = Query(None, description="Date de début au format YYYY-MM-DD"),
                                     end_date: Optional[str] = Query(None, description="Date de fin au format YYYY-MM-DD")):
    try:
        logger.info("Fetching data")
        predictions_data_csv = prediction_handler.fetch_csv_data(start_date, end_date)

        if "error" in predictions_data_csv:
            logger.error(f"Error fetching CSV data: {predictions_data_csv['error']}")
            return predictions_data_csv

        conn = sqlite3.connect(common.DB_PRED)
        if start_date and end_date:
            query = f"SELECT * FROM pred_db WHERE date BETWEEN '{start_date}' AND '{end_date}'"
        elif start_date:
            query = f"SELECT * FROM pred_db WHERE date >= '{start_date}'"
        elif end_date:
            query = f"SELECT * FROM pred_db WHERE date <= '{end_date}'"
        else:
            query = "SELECT * FROM pred_db"

        df_predictions_db = pd.read_sql_query(query, conn)
        conn.close()

        predictions_list_db = df_predictions_db.to_dict(orient='records')

        for item in predictions_data_csv:
            item['temperature_2m'] = str(item['temperature_2m'])

        combined_predictions = {"predictions_csv": predictions_data_csv, "predictions_db": predictions_list_db}

        logger.info("Data successfully fetched")
        return combined_predictions

    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        return

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)