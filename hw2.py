import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
import numpy as np


def load_data(file_path):
    return pd.read_csv(file_path)


def scatter_diagram(y_true, y_pred, num_points=50):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_points), y_true[:num_points], color='blue', label='Истинные значения')
    plt.scatter(range(num_points), y_pred[:num_points], color='red', label='Предсказанные значения')
    plt.xlabel('Индекс')
    plt.ylabel('Значения')
    plt.title(f'Истинные и предсказанные значения(первые {num_points} точек)')
    plt.legend()
    plt.show()


def line_graph(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_true)), y_true, label='Истинные значения', marker='o')
    plt.plot(range(len(y_pred)), y_pred, label='Предсказанные значения', marker='x')
    plt.xlabel('Наблюдения')
    plt.ylabel('Значения')
    plt.title('Сравнение истинных и предсказанных значений')
    plt.legend()
    plt.grid(True)
    plt.show()


def histogram(data):
    fig, ax = plt.subplots(1, figsize=(18, 5))

    # Нормальное распределение
    ax.hist(data, bins=30, alpha=0.7, color='blue')
    ax.set_title('Гистограмма')

    plt.tight_layout()
    plt.show()


def missing_values_report(df):
    missing_report = df.isnull().sum()
    missing_report = missing_report[missing_report > 0].sort_values(ascending=False)
    return missing_report


def fill_missing_values(df, method='mean'):
    if method == 'mean':
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(df)
        return imp_mean.transform(df)
    elif method == 'median':
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_median.fit(df)
        return imp_median.transform(df)
    elif method == 'most_frequent':
        imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp_most_frequent.fit(df)
        return imp_most_frequent.transform(df)


#Пример использования missing_values_report()
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names

target_names = iris.target_names

df = pd.DataFrame(data=X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

print(missing_values_report(df))
print('\n')


#Пример использования fill_missing_values()
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, 4, 9]]
print(fill_missing_values(X, 'mean'))
print('\n')


#Пример использования load_data()
file_path = 'data(1).csv'
data = load_data(file_path)
print(data)
print('\n')


#Пример использования scatter_diagram() и line_graph()
y_pred = [0.2, 2.3, 4.5, 3.1]
y_true = [0.18, 3.0, 4.51, 5.2]

scatter_diagram(y_true, y_pred, 4)
line_graph(y_true, y_pred)


#Пример использования histogram()
data = np.random.normal(loc=0, scale=1, size=1000)
histogram(data)
