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

# Разделение данных на набор зависимых переменных и метку
data = df.drop(["JobSatisfaction"],axis=1)
target = df["JobSatisfaction"]

# Вывод столбцов с категориальными признаками
category_df = data.select_dtypes(include='object')
category_df.head()

# Кодирование данных
categorical_features = category_df.columns.tolist()
print('Столбцы для кодирования:\n',categorical_features)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for each in categorical_features:
    category_df[each] = encoder.fit_transform(category_df[each])
    category_df_check=encoder.classes_

numeric_df = data.select_dtypes(include='number')

# Новый набор данных с закодировннными категориями
new_data =pd.merge(numeric_df, category_df, left_index=True, right_index=True)
new_df=pd.merge(new_data, target, left_index=True, right_index=True)
new_df.head()

# Кореляция переменных
plt.figure(figsize=(20, 8))
sns.heatmap(new_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Убираем некоторые переменные
X=new_data.drop(["EmpID","Age","CommuteDistance","NumCompanies","TeamSize","NumReports","TrainingHoursPerYear"],axis=1)
Y=target

# Проверка распределения целевой метки
result_target = df.groupby('JobSatisfaction', observed=False)['JobSatisfaction'].count()
print('\nПроверка распределения целевой метки:\n',result_target)
plt.hist(target)
plt.xlabel('Числовые значения')
plt.ylabel('Количество')
plt.title('')
plt.show()

# Градиентный бустинг
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=177)

# Создание и обучение модели Gradient Boosting Regressor
model = GradientBoostingRegressor(max_depth = 4,n_estimators = 222,loss='squared_error', random_state=0) # параметры
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'*****************')
print(f'Метрики качества:')
print(f'*****************')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 10))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()


# Регрессор Light GBM без подбора параметров
from lightgbm import LGBMRegressor

# Создание и обучение модели LGBMRegressor
model = LGBMRegressor(random_state=0,force_col_wise='true')
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()


# XGBoost - Экстремальный градиентный бустинг.
from xgboost import XGBRegressor

# Определение модели
model = XGBRegressor(max_depth = 3,  n_estimators = 222)
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()

#Гребневая регрессия (ридж-регрессия)
from sklearn.linear_model import RidgeCV
model = RidgeCV()
model.fit(X_train, y_train)

#Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 10))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()


# Регрессор catboost
from catboost import CatBoostRegressor

# Создание и обучение модели CatBoost для регрессии
model = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=6, random_state=123, verbose=0)
model.fit(X_train, y_train)

#Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 10))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()

# Регрессия LASSO
from sklearn.linear_model import Lasso

# Создание и обучение модели регрессии Lasso
model = Lasso(alpha=0.1)  # Здесь alpha - гиперпараметр регуляризации L1
model.fit(X_train, y_train)

#Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(20, 10))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel("Актуальное значение")
plt.ylabel("Предсказанное значение")
plt.title("Актуальное vs. Предсказанное (Lasso Regression)")
plt.show()