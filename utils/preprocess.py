import pandas as pd 
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_file):
        self.data_file = data_file

    def preprocess(self):
        data = pd.read_csv(self.data_file, index_col=None)
        data['date'] = pd.to_datetime(data['date'])
        data['date'] = data['date'].dt.strftime('%Y-%m-%d %H:%M')
        data_copy = data.copy()
        data.drop(columns=['relative_humidity_2m'], inplace=True)
        data['date'] = pd.to_datetime(data['date'])
        data_copy['date'] = pd.to_datetime(data_copy['date'])
        data.set_index('date', inplace=True)
        data_copy.set_index('date', inplace=True)
        resampled_data_mean = data.resample('3h').mean()
        resampled_data_mean.index.names = ['Timestamp']

        resampled_data_mean_2variables = data.resample('3h').mean()
        resampled_data_mean_2variables.index.names = ['Timestamp']

        train_size = int(len(resampled_data_mean) * 0.8)
        train_data = resampled_data_mean[:train_size]
        test_data = resampled_data_mean[train_size:]

        return resampled_data_mean_2variables, resampled_data_mean, train_data, test_data



class DataPreprocessorSplit:
    def __init__(self):
        pass

    @staticmethod
    def preprocessingSplit(dataset, target_column):
        df_copy = dataset.copy()
        ml_df = dataset.copy()
        
        train_size_2variable = int(len(df_copy) * 0.8)
        train_data_2variable = df_copy[:train_size_2variable]
        test_data_2variable = df_copy[train_size_2variable:]

        for i in range(1, 10):
            ml_df[f"lag_{i}"] = ml_df[target_column].shift(i)
        ml_df.dropna(inplace=True)
        
        X = ml_df.drop(target_column, axis=1)
        y = ml_df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        return X_train, X_test, y_train, y_test, train_data_2variable, test_data_2variable

