# Домашняя работа №6:

Взять произвольный датасет, описать его,провести анализ EDA, предварительную обработку данных. Решить задачу сегментации или анализа временного ряда при помощи не менее 5-ти подходов ML. Решить задачу поиска аномалий. Провести визуализации.  

## Датасет:

_wine.csv_ Этот набор данных относится к различным вариантам вина и описывает количество различных химических веществ, присутствующих в вине, и их влияние на его качество.  

Колонки:  
_**fixed acidity**_ - кислотность  
_**volatile acidity**_ - летучесть  
_**citric acid**_ - лимонная кислота  
_**residual sugar**_ - сахар  
_**chlorides**_ - хлориды  
_**free sulfur dioxide**_ - свободный диоксид серы  
_**total sulfur dioxide**_ - общий диоксид серы  
_**density**_ - плотность  
_**pH**_ - pH  
_**sulphates**_ - сульфаты  
_**alcohol**_ - алкоголь   
_**quality**_ - качество  
_**type**_ - тип  

## Задача:

Обнаружения кластеров в наборе данных

## Модели:

✓ Кластеризация K-средних (KMeans)  
✓ Кластеризация со средним сдвигом (MeanShift)  
✓ Спектральная кластеризация (SpectralClustering)  
✓ Агломеративная кластеризация (AgglomerativeClustering)  
✓ Иерархическая через дендрограмму  
✓ DBSCAN плотностной алгоритм  
