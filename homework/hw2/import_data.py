
# модуль import_data.py по импорту датасета
import pandas as pd

#Загрузка файла .csv в pandas 
def load_data_csv(dataset_path='https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'):
    return pd.read_csv(dataset_path)

#Загрузка файла .json в pandas 
def load_data_json(dataset_path):
    return pd.read_json(dataset_path)


