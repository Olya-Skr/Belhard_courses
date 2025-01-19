import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Гистограмма Для просмотра заданных полей
def visual_field(x_column):
  plt.hist(x_column)
  plt.xlabel('Числовые значения')
  plt.ylabel('Количество')
  plt.title('')
  plt.show()   

# Гистограмма для просмотра всех числовых полей
def visual_all(df):
  # группировки данных по method requires numerical or datetime columns  
  df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
  # цикл для прорисовки гистограм и назначению список осей на рисунке
  for ax in plt.gcf().get_axes():
    ax.set_xlabel('Числовые значения')
    ax.set_ylabel('Частота')
    ax.set_title(ax.get_title())
    plt.show()

#Сводные таблица
def visual_piv(df,y_index,x_column,val):
  pivot = df.pivot_table(
  index=y_index,
  columns=x_column,
  values=val,
  aggfunc=np.average)
  sns.heatmap(pivot)
  plt.show()

#Круговая диаграмма 
def visual_circle(x_column):
  unique_list=list(set(list(x_column)))
  d=[]
  for val in unique_list: 
    d.append(list(x_column).count(val)) 
  plt.pie (d, labels=list(set(x_column)))  
  plt.show()

# Построение парных графиков при помощи sns
def visual_pair(df):
  sns.pairplot(df)
  plt.show()