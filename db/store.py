import sqlite3

import sys
sys.path.append('./open-meteo')

import csv
import common


class TemperatureDataHandler:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS temperature (
                                date TEXT,
                                temperature_2m REAL
                            )''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS humidity (
                                relative_humidity_2m REAL
                            )''')
        self.conn.commit()

    def save_temperature(self, date, temperature_2m):
        self.cursor.execute('''INSERT INTO temperature (date, temperature_2m) VALUES (?, ?)''', (date, temperature_2m))
        self.conn.commit()

    def save_humidity(self, relative_humidity_2m):
        self.cursor.execute('''INSERT INTO humidity (relative_humidity_2m) VALUES (?)''', (relative_humidity_2m,))
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

    def merge_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS temperature_db AS
                                SELECT temperature.date, temperature.temperature_2m, humidity.relative_humidity_2m
                                FROM temperature
                                INNER JOIN humidity ON temperature.rowid = humidity.rowid''')
        self.conn.commit()
        print("Table temperature_db created successfully.")


data_handler = TemperatureDataHandler(common.DB)

# Read and save temperature data

with open(f'{common.CURRENT_PATH}hourly_dataframe.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        date = row['date']
        temperature_2m = float(row['temperature_2m'])
        data_handler.save_temperature(date, temperature_2m)

# Read and save humidity data
with open(f'{common.CURRENT_PATH}hourly_dataframe_add_col.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        relative_humidity_2m = float(row['relative_humidity_2m'])
        data_handler.save_humidity(relative_humidity_2m)

# Merge tables
data_handler.merge_tables()

# Close connection
data_handler.close_connection()
