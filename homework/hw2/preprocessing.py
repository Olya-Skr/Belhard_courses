import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype


# Отчет о пропущенных значениях
def report_empty_values(df):
    print('Отчет:')
    if df.isnull().any().any() == False:
      report=print('Нет пропущенных значений')
    if df.isnull().any().any() == True:
      report=prim_df.isnull().sum()
      print('Количество пропущенных занчений:')
    return (report)

# Заполнение пропущенных значений средним либо частым
def filling(df):
  if df.isnull().any().any() == True:
    column=df.columns.tolist()
    for c in column:
      if is_numeric_dtype(df[c]):
        df.fillna({c:df[c].mean()}, inplace=True)
      else:
        df.fillna({c:df[c].mode()[0]}, inplace=True)
  else:
    print('Нет пропущенных значений для заполнения')

