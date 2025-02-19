# Домашняя работа №5:

Реализовать минимум 5 регрессоров, сравнить метрики между собой, выбрать лучший для Вашего датасета.


## Датасет:

_employee_survey.csv_ содержит результат годового опроса персонала в том числе с оценкой удовлетворенности работой.

Колонки:  
_**EmpID**_ - Уникальный идентификатор для каждого сотрудника  
_**Gender**_  - Пол (e.g., Male, Female, Other)  
_**Age**_  - Возраст   
_**MaritalStatus**_  - Семейное положение (e.g., Single, Married, Divorced, Widowed)  
_**JobLevel**_  - Уровень должности сотрудника (e.g., Intern/Fresher, Junior, Mid, Senior, Lead)  
_**Experience**_  - Cтаж работы  
_**Dept**_  - Подразделение, в котором работает сотрудник (e.g., IT, HR, Finance, Marketing, Sales, Legal, Operations, Customer Service)  
_**EmpType**_  - Тип занятости (e.g., Full-Time, Part-Time, Contract)  
_**WLB**_  - Рейтинг баланса между работой и личной жизнью (scale from 1 to 5)  
_**WorkEnv**_  - Рейтинг рабочей среды (scale from 1 to 5)  
_**PhysicalActivityHours**_  - Количество часов физической активности в неделю  
_**Workload**_  - Рейтинг рабочей нагрузки (scale from 1 to 5)  
_**Stress**_  - Рейтинг уровня стресса (scale from 1 to 5)  
_**SleepHours**_  - Количество часов сна в сутки  
_**CommuteMode**_  - Способ передвижения (e.g., Car, Public Transport, Bike, Walk, Motorbike)  
_**CommuteDistance**_  - Расстояние (km)  
_**NumCompanies**_  - Количество различных компаний, в которых работал сотрудник  
_**TeamSize**_  - Размер команды, частью которой является сотрудник  
_**NumReports**_  - Количество отчетности (только для уровней Senior и Lead)  
_**EduLevel**_  - Уровень образования, достигнутый сотрудником (e.g., High School, Bachelor, Master, PhD)  
_**haveOT**_  - Индикатор наличия у сотрудника сверхурочной работы (True/False)  
_**TrainingHoursPerYear**_  - Количество часов обучения, полученных в год  
_**JobSatisfaction**_  - Рейтинг удовлетворенности работой (scale from 1 to 5).  

## Цель для регриссинного анализа:
Предсказание потенциальной длительности обучения сотрудника в год, которое потребуется выделить компании.  
Целевой меткой для предсказания выбрана _**TrainingHoursPerYear**_

## Модели машинного обучения:
✓ Регрессор Gradient Boosting  
✓ Регрессор LGBM  
✓ Экстремальный градиентный бустинг  
✓ Гребневая регрессия   
✓ Регрессор CatBoost  
✓ Регрессия Лассо  

## Вывод
Заключение по оценке точности моделей, представленой в ноутбуке _dz5.ipynb_, говорит о том, что наилучшей моделью для выполнения указанной задачи является регрессор **LightGBM**.