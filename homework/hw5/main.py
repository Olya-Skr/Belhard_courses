import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('employee_survey.csv', sep=",")
# Просмотр датасета
#display(df)
df.head()
df.info()
df.describe()

# Проверка на пустые значения и удаление дубликатов
print("\nПропущенные значения:")
print(df.isnull().sum())
df = df.drop_duplicates()

# Масштабирование признаков: PhysicalActivityHours, SleepHours
data_for_scaling=df[["PhysicalActivityHours","SleepHours"]]
# Нормализация
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_for_scaling)
data_for_scaling = scaler.transform(data_for_scaling)
data_for_scaling_new = pd.DataFrame(data_for_scaling,columns=['PhysicalActivityHours', 'SleepHours'])
data_for_scaling_new.head()

# Формируем датасет с обновленными полями
df = df.drop(['PhysicalActivityHours','SleepHours'],axis=1)
df=pd.concat([df,data_for_scaling_new],axis=1)
df.head()

# Обработка категориальных признаков
category_df = df.select_dtypes(include='object')
category_df.head()

# Кодирование данных через OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
encoder_ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
data_ohe = pd.DataFrame()
data_ohe=pd.concat([data_ohe,encoder_ohe.fit_transform(category_df)],axis=1)
data_ohe.head()

# Новый набор данных с закодировннными категориями
categorical_features = category_df.columns.tolist()
print('Закодированные столбцы:\n',categorical_features)
print('\nОбновленный датасет:')
new_df=pd.concat([df,data_ohe],axis=1).drop(columns=categorical_features)
new_df.head()

# Для просмотра корреляции попроще закодируем данные через LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder_le = LabelEncoder()
for each in categorical_features:
    category_df[each] = encoder_le.fit_transform(category_df[each])
    category_df_check=encoder_le.classes_

numeric_df = df.select_dtypes(include='number')
new_df_cor =pd.merge(numeric_df, category_df,  left_index=True, right_index=True)
new_df_cor.head()

# Кореляция переменных
plt.figure(figsize=(20, 8))
sns.heatmap(new_df_cor.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Разделение данных на набор зависимых переменных и метку
data = new_df.drop(["TrainingHoursPerYear"],axis=1)
target = new_df["TrainingHoursPerYear"]

# Убираем бесполезные переменные
X=data.drop(["EmpID","CommuteDistance","NumCompanies"],axis=1)
Y=target

# Проверка распределения целевой метки
kol_target=Y.nunique()
print('Количество уникальных:\n',kol_target)
result_target = new_df.groupby('TrainingHoursPerYear', observed=False)['TrainingHoursPerYear'].count()
print('\nПроверка распределения целевой метки:\n',result_target)
plt.hist(target,bins=10)
plt.xlabel('Значение метки')
plt.ylabel('Количество')
plt.title('')
plt.show()


# Градиентный бустинг
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=43,stratify=Y)

# Создание и обучение модели Gradient Boosting Regressor
GBR_model = GradientBoostingRegressor(max_depth = 2,n_estimators = 50) # параметры
GBR_model.fit(X_train, y_train)

# Предсказание на тестовой выборке
GBR_y_pred = GBR_model.predict(X_test)

# Вычисление метрик
GBR_mae = mean_absolute_error(y_test, GBR_y_pred)
GBR_mse = mean_squared_error(y_test, GBR_y_pred)
GBR_r2 = r2_score(y_test, GBR_y_pred)

print(f'MAE: {GBR_mae:.4f}') # средняя абсолютная ошибка
print(f'MSE: {GBR_mse:.4f}') # среднюю квадратическую ошибк
print(f'R²: {GBR_r2:.4f}') # коэффициент детерминации

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 10))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(GBR_y_pred)), GBR_y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()


# Регрессор Light GBM
from lightgbm import LGBMRegressor

# Создание и обучение модели LGBMRegressor
LGBMR_model = LGBMRegressor(max_depth=-1,n_estimators=50,force_col_wise='true')
LGBMR_model.fit(X_train, y_train)

# Предсказание на тестовой выборке
LGBMR_y_pred = LGBMR_model.predict(X_test)

# Вычисление метрик
LGBMR_mae = mean_absolute_error(y_test, LGBMR_y_pred)
LGBMR_mse = mean_squared_error(y_test, LGBMR_y_pred)
LGBMR_r2 = r2_score(y_test, LGBMR_y_pred)

print(f'MAE: {LGBMR_mae}')
print(f'MSE: {LGBMR_mse}')
print(f'R²: {LGBMR_r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(LGBMR_y_pred)), LGBMR_y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()


# XGBoost - Экстремальный градиентный бустинг.
from xgboost import XGBRegressor

# Определение модели
XGB_model = XGBRegressor(max_depth=3, n_estimators=50)
XGB_model.fit(X_train, y_train)

# Предсказание на тестовой выборке
XGB_y_pred = XGB_model.predict(X_test)

# Вычисление метрик
XGB_mae = mean_absolute_error(y_test, XGB_y_pred)
XGB_mse = mean_squared_error(y_test, XGB_y_pred)
XGB_r2 = r2_score(y_test, XGB_y_pred)

print(f'MAE: {XGB_mae}')
print(f'MSE: {XGB_mse}')
print(f'R²: {XGB_r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(XGB_y_pred)), XGB_y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()


#Гребневая регрессия (L2-регуляризацией)
from sklearn.linear_model import RidgeCV
L2_model = RidgeCV(alphas=10.0) #0.1, 1.0, 10.0
L2_model.fit(X_train, y_train)

#Предсказание на тестовой выборке
L2_y_pred =L2_model.predict(X_test)

# Вычисление метрик
L2_mae = mean_absolute_error(y_test, L2_y_pred)
L2_mse = mean_squared_error(y_test, L2_y_pred)
L2_r2 = r2_score(y_test, L2_y_pred)

print(f'MAE: {L2_mae}')
print(f'MSE: {L2_mse}')
print(f'R²: {L2_r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 10))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(L2_y_pred)), L2_y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()


# Регрессия LASSO (L1-регуляризация)
from sklearn.linear_model import Lasso

# Создание и обучение модели регрессии Lasso
L1_model = Lasso(alpha=0.1)  # Здесь alpha - гиперпараметр регуляризации L1
L1_model.fit(X_train, y_train)

#Предсказание на тестовой выборке
L1_y_pred = L1_model.predict(X_test)

# Вычисление метрик
L1_mae = mean_absolute_error(y_test, L1_y_pred)
L1_mse = mean_squared_error(y_test, L1_y_pred)
L1_r2 = r2_score(y_test, L1_y_pred)

print(f'MAE: {L1_mae}')
print(f'MSE: {L1_mse}')
print(f'R²: {L1_r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 10))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(L1_y_pred)), L1_y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()

#pip install catboost
from catboost import CatBoostRegressor

# Создание и обучение модели CatBoost
CB_model = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=3, random_state=55, verbose=0)
CB_model.fit(X_train, y_train)

#Предсказание на тестовой выборке
CB_y_pred = CB_model.predict(X_test)

# Вычисление метрик
CB_mae = mean_absolute_error(y_test, CB_y_pred)
CB_mse = mean_squared_error(y_test, CB_y_pred)
CB_r2 = r2_score(y_test, CB_y_pred)

print(f'MAE: {CB_mae}')
print(f'MSE: {CB_mse}')
print(f'R²: {CB_r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 10))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(CB_y_pred)), CB_y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()


# Сравнение метрик качества моделей
print(f'Сравнение метрик качества моделей:')
results_metrics = pd.DataFrame([
  ['Градиентный бустинг XGBoost', GBR_mae,GBR_mse,GBR_r2],
  ['Light_GBM', LGBMR_mae,LGBMR_mse,LGBMR_r2],
  ['Экстремальный градиентный бустинг XGBoost', XGB_mae,XGB_mse,XGB_r2],
  ['Гребневая регрессия L2', L2_mae,L2_mse,L2_r2],
  ['Регрессия LASSO L1', L1_mae,L1_mse,L1_r2],
  ['CatBoost', CB_mae,CB_mse,CB_r2]],
  columns=['model','mae', 'mse','r2',])
display(results_metrics)

# Визуализируем ошибки
from matplotlib import pyplot as plt
import seaborn as sns
results_metrics.plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)
