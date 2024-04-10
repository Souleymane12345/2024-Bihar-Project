import sys
sys.path.append('./open-meteo')

import common

import pandas as pd
import sqlite3
from io import StringIO

class DatabaseConnector:
    def __init__(self, db_file = common.DB):
        self.db_file = db_file

    def connect(self):
        connection = sqlite3.connect(self.db_file)
        data_test = pd.read_sql('SELECT * FROM temperature_db', connection)
        connection.close()

        csv_buffer = StringIO()
        data_test.to_csv(csv_buffer, index=False)
        print(data_test)

        csv_buffer.seek(0)
        print("Connected")
        return csv_buffer

