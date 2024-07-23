from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np


def preprocess_data(df, target_column, numeric_features, categorical_features, scaler='Standard'):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if scaler == 'Standard':
        numeric_transformer = StandardScaler()
    elif scaler == 'MinMax':
        numeric_transformer = MinMaxScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


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
