import import_data as im
import visualization as vi
import preprocessing as pr

# Загрузка данных (стандартное значение датасета задано)
df = im.load_data_csv();

# Просмотр датасета и вывод Имен полей
display(df)
df.head()
print(df.info())
print(df.columns)

##################################################### Предобработка
# Удаления дубликатов
df = df.drop_duplicates()

# Отчет о пропущенных значениях
report_empty_values(df)

# Заполнение пропущенных значений средним либо частым
filling(df)

##################################################### Визуализации
# Гистограмма для просмотра всех числовых полей 
vi.visual_all(df)
# Гистограмма для просмотра возвраста пассажиров
vi.visual_field(df['Age'])

# Сводные таблица Демонстрация статистики выживших в зависимости от пола и класса 
vi.visual_piv('Sex','Pclass','Survived') #

# Круговая диаграмма Выживших мужчины и женщины
vi.visual_circle(df['Sex'])

# Построение парных графиков  
vi.visual_pair(df)

