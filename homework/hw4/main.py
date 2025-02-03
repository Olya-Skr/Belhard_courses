import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# выгружаем набор данных из OpenML
mnist = fetch_openml('MNIST_784', version=1, as_frame=True)
df=mnist.frame

# изучаем данные
print('Количество уникальных признаков и их перечень: ',len(mnist.target.unique()),':',set(mnist.target))
print('\nСтатистические сведения о датафрейме:\n',df.describe())
result_target = df.groupby('class', observed=False)['class'].count()
print('\nПроверка распределения классов:\n',result_target)
X = df.drop('class', axis=1)  # Признаки
y=df['class'] # Метки
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.axis('off')
    plt.imshow(X.iloc[i].values.reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Метка: {}'.format(y.iloc[i]))

# создание обучающих наборов для классификатора
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=186)
print('Размер тренеровочного набора признаков: ', X_train.shape)
print('Размер тренеровочного набора меток: ',y_train.shape)

# создание классификаторов:
'''
    Метод опорных векторов (Support Vector Machines)
    Метод k-ближайших соседей (K-Nearest Neighbors)
    Классификатор дерева решений (Decision Tree Classifier)
    Наивный байесовский метод (Naive Bayes)
    Линейный дискриминантный анализ (Linear Discriminant Analysis)
'''
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SVC_model = svm.SVC()
KNN_model = KNeighborsClassifier()
DTC_model = DecisionTreeClassifier()
GNB_model = GaussianNB()
LDA_model = LinearDiscriminantAnalysis()


# обучение классификатора
SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
DTC_model.fit(X_train, y_train)
GNB_model.fit(X_train, y_train)
LDA_model.fit(X_train, y_train)

# составление прогнозов
SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)
DTC_prediction = DTC_model.predict(X_test)
GNB_prediction = GNB_model.predict(X_test)
LDA_prediction = LDA_model.predict(X_test)

# оценка производительности классификатора
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Оценка точности — простейший вариант оценки работы классификатора
print('Точность метода опорных векторов:',accuracy_score(SVC_prediction, y_test))
print('Точность метода k-ближайших соседей:',accuracy_score(KNN_prediction, y_test))
print('Точность метода дерева решений:',accuracy_score(DTC_prediction, y_test))
print('Точность Наивный байесовский метод:',accuracy_score(GNB_prediction, y_test))
print('Точность Линейный дискриминантный анализ:',accuracy_score(LDA_prediction, y_test))

print('\nНиже матрицы ошибок и отчёт о классификации\n')
print('Отчет метода опорных векторов:\n',classification_report(SVC_prediction, y_test))
disp = ConfusionMatrixDisplay(confusion_matrix(SVC_prediction, y_test))
disp.plot()
plt.show()

# настройка параметров для дерева
from sklearn.model_selection import GridSearchCV  
DTC_model.get_params().keys() 

grid_search= GridSearchCV( DTC_model,
    {"max_depth": [10,20,50]}, 
    cv = 4, # количество ё разбиений на кросс-валидацию
    scoring = 'accuracy' # выбор метрики ошибки
    )

# Обучение GridSearchCV на обучающих данных
grid_search.fit(X_train, y_train)
print("Лучший параметр:", grid_search.best_params_)

# Сделаем предсказания на тестовом наборе данных с использованием лучшей модели
best_model = grid_search.best_estimator_
DTC_prediction_best = best_model.predict(X_test)

# Оценим качество модели
print('Точность метода с подобранными параметрами:',accuracy_score(DTC_prediction_best, y_test))
print('Точность метода с подобранными параметрами:',accuracy_score(DTC_prediction, y_test))
