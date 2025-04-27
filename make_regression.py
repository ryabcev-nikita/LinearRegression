import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import seaborn as sns

import matplotlib

matplotlib.use('TkAgg')  # или 'Qt5Agg'

# Загрузка данных
data = pd.read_excel('data_sber.xlsx')
dff = pd.DataFrame(data)
df = dff.iloc[:, 5:dff.shape[1]]
df = df.dropna()
df_begin = df

print("Предварительные загруженные данные")
print(df)


# Функция для удаления выбросов с помощью IsolationForest
def delete_blowouts_with_isolation_forest(data_frame):
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(max_samples=58, random_state=12)
    clf.fit(data_frame)

    data_frame['anomaly'] = clf.predict(data_frame)
    print("Датафрейм с аномалиями")
    print(data_frame)
    data_frame = data_frame[data_frame.anomaly == 1]
    data_frame = data_frame.drop(columns='anomaly').reset_index(drop=True)
    print("Очищенный датафрейм")
    print(data_frame)
    return data_frame


# Функция для удаления выбросов с использованием IQR
def remove_outliers_iqr(data_frame):
    for column in data_frame.columns:
        Q1 = data_frame[column].quantile(0.25)
        Q3 = data_frame[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data_frame = data_frame[(data_frame[column] >= lower_bound) & (data_frame[column] <= upper_bound)]
    print("Данные после удаления выбросов с использованием IQR:")
    print(data_frame.describe())
    print(data_frame)
    return data_frame.reset_index(drop=True)


# Функция для удаления выбросов с использованием Z-оценки
def remove_outliers_zscore(data_frame, threshold=3):
    from scipy import stats
    z_scores = np.abs(stats.zscore(data_frame))
    data_frame = data_frame[(z_scores < threshold).all(axis=1)]
    print("\nДанные после удаления выбросов с использованием Z-оценки:")
    print(data_frame.describe())
    print(data_frame)
    return data_frame.reset_index(drop=True)


# Удаление выбросов с использованием IQR
data_no_outliers_iqr = remove_outliers_iqr(df.copy())
df = data_no_outliers_iqr.copy()


def draw_diagram_scattering(data_frame):
    # Построение диаграммы рассеяния для каждой пары факторов после удаления выбросов
    for column in data_frame.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=column, y='R_stock', color='blue', label='Данные')


n = len(df)  # количество наблюдений
k = df.shape[1]  # количество факторов
ddf = n - k - 1  # Степени свободы
alpha = 0.05  # уровень значимости

from scipy.stats import f
from statsmodels.stats.diagnostic import het_goldfeldquandt


# Функция для построения и вывода результатов регрессионной модели
def run_regression(y, X, is_const):
    if is_const:
        X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print(f"Результаты регрессии для {y.name} от {', '.join(X.columns)}:\n")
    print(model.summary())
    print("\n" + "=" * 80 + "\n")

    # 1. Визуальная проверка гомоскедастичности
    plt.figure(figsize=(10, 6))
    plt.scatter(model.fittedvalues, model.resid)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    # 2. Тест Голдфельда-Квандта
    # Степени свободы
    df1 = n - k - 1
    df2 = n - k - 1
    gq_test = het_goldfeldquandt(y, X)
    gq_labels = ['F-statistic', 'p-value']
    gq_results = dict(zip(gq_labels, gq_test))
    print("\n")
    print(f"Результаты теста Голдфельда-Квандта:")
    print(f"F-statistic = {gq_results.get('F-statistic')}, p-value = {gq_results.get('p-value')}\n")
    # Находим критическое значение F
    f_critical = f.ppf(1 - alpha, df1, df2)
    print(f"F-табл = {f_critical}\n")
    if f_critical > gq_results.get('F-statistic'):
        print("В остатках модели присутствует гомоскедатичность\n")
    else:
        print("В остатках модели присутствует гетероскедатичность\n")

    # Гистограмма остатков
    residuals = model.resid
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=10, edgecolor='k', alpha=0.7)
    plt.title('Гистограмма остатков')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')

    # Q-Q график
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q график остатков')
    plt.tight_layout()
    plt.show()

    return model


def score_regression(model_regression, y, X):
    from scipy.stats import t

    # Критическое значение t
    t_critical = t.ppf(1 - alpha / 2, ddf)
    print(f"\nКритическое значение t при уровне значимости {alpha}: {t_critical}")
    # Оценка качества модели
    r_squared = model_regression.rsquared
    adj_r_squared = model_regression.rsquared_adj
    f_statistic = model_regression.fvalue
    f_p_value = model_regression.f_pvalue

    # Средняя относительная ошибка аппроксимации
    predictions = model_regression.predict(X)
    mae = 1 / n * (np.sum(np.abs(predictions - y) / np.mean(y))) * 100  # Средняя абсолютная ошибка в процентах

    # Вывод результатов
    print(f"Коэффициент детерминации (R²): {r_squared:.4f}")
    print(f"Скорректированный коэффициент детерминации (Adj R²): {adj_r_squared:.4f}")
    print(f"F-критерий Фишера: {f_statistic:.4f}")
    print(f"p-значение F-критерия: {f_p_value:.4e}")
    print(f"Средняя относительная ошибка аппроксимации: {mae:.4f}%")

    return predictions


def draw_regression_y_values(y, predictions):
    # Фактические значения
    actual_values = y
    # Предсказанные значения
    predicted_values = predictions
    # Создание графика
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values.index, actual_values, label='Фактические значения', color='blue', linewidth=2)
    plt.plot(predicted_values.index, predicted_values, label='Предсказанные значения', color='orange', linestyle='--',
             linewidth=2)

    # Настройка графика
    plt.title('Фактические и предсказанные значения')
    plt.xlabel('Индекс')
    plt.ylabel('Значения')
    plt.legend()
    plt.grid()
    plt.show()


y_var = df.iloc[:, 0]
X_var = df[['R_imoex', 'R_usd', 'R_Inflation', 'R_IPP']]

model_result = run_regression(y_var, X_var, False)
predicts = score_regression(model_result, y_var, X_var)
draw_regression_y_values(y_var, predicts)


import warnings
warnings.filterwarnings("ignore")

R_moex_mean = df['R_imoex'].mean()
print(R_moex_mean)

R_inflation_mean = df['R_Inflation'].mean()
print(R_inflation_mean)

R_ipp_mean = df['R_IPP'].mean()
print(R_ipp_mean)

R_usd_mean = df['R_usd'].mean()
print(R_usd_mean)

R_stock_mean = df['R_stock'].mean()
print(R_stock_mean)

# 4) Вычисление частных коэффициентов корреляции
def partial_correlation(x, y, z):
    """Calculate the partial correlation between x and y controlling for z."""
    model_x = sm.OLS(x, z).fit()
    residual_x = model_x.resid
    model_y = sm.OLS(y, z).fit()
    residual_y = model_y.resid
    return np.corrcoef(residual_x, residual_y)[0, 1]

# Создание матрицы частных корреляций
partial_corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
partial_corr_matrix.fillna(0, inplace=True) # убираем нули на всякий случай

for var1 in partial_corr_matrix.index:
    for var2 in partial_corr_matrix.columns:
        if var1 != var2:
            other_vars = df.drop(columns=[var1, var2])
            partial_corr = partial_correlation(df[var1], df[var2], other_vars)
            partial_corr_matrix.loc[var1, var2] = partial_corr

# Заменяем NaN на 0
partial_corr_matrix.fillna(0, inplace=True)

# Вывод матрицы частных корреляций
print("\nМатрица частных корреляций:")
print(partial_corr_matrix)

coef =  [1.1050, -0.2865, 0.2824, -0.3174]

print(R_moex_mean/R_stock_mean)
beta_coef = [ coef[0]*(R_moex_mean/R_stock_mean), coef[1]*R_usd_mean/R_stock_mean, coef[3]*R_inflation_mean/R_stock_mean, coef[2]*R_ipp_mean/R_stock_mean]
print("Бета-коэффициенты")
print(beta_coef)

R_square = model_result.rsquared

print("Коэффициент детерминации")
print(R_square)

delta_coef = [beta_coef[0]*partial_corr_matrix['R_stock'][1],
              beta_coef[1]*partial_corr_matrix['R_stock'][2],
              beta_coef[2]*partial_corr_matrix['R_stock'][3],
              beta_coef[3]*partial_corr_matrix['R_stock'][4],]

print("Дельта коэффициенты")
for i in range(len(delta_coef)):
    print(delta_coef[i])

from scipy.stats import t

df_new = df_begin[['R_stock', 'R_imoex']]
df_new = remove_outliers_zscore(df_new)

n = len(df_new)  # количество наблюдений
k = df_new.shape[1] # количество факторов
ddf = n - k - 1  # Степени свободы
alpha = 0.1 # уровень значимости

X_v = df_new[['R_imoex']]
y_v = df_new.iloc[:, 0]
model_result_pair = run_regression(y_v, X_v, True)
X_v = sm.add_constant(X_v)
predicts = score_regression(model_result_pair, y_v, X_v)
draw_regression_y_values(y_v, predicts)
print("\n")

t_value = t.ppf(1 - alpha/2, ddf)
std_errors = model_result_pair.bse
confidence_intervals = []
for i in range(len(model_result_pair.params)):
    lower_bound = model_result_pair.params[i] - t_value * std_errors[i]
    upper_bound = model_result_pair.params[i] + t_value * std_errors[i]
    confidence_intervals.append((lower_bound, upper_bound))

print("Стандартные ошибки")
print(std_errors)
print("t-табл")
print(t_value)

for i, (lower, upper) in enumerate(confidence_intervals):
    print(f"Доверительный интервал для коэффициента {i}: ({lower}, {upper})")

# Предсказание на месяц вперед при R_imoex равным 80% от максимального за рассматриваемый период
new_data = pd.DataFrame({
    'R_imoex': [df_new['R_imoex'].max()*0.8],
    'const': 1
})

print(new_data)

predictions = model_result_pair.get_prediction(new_data)

print("Точечное предсказание = ",predictions.predicted_mean)

pred_summary = predictions.summary_frame(alpha=alpha)
lower_bound = pred_summary['obs_ci_lower']
upper_bound = pred_summary['obs_ci_upper']

# Визуализация
plt.figure(figsize=(10, 6))  # Увеличение размера графика
plt.scatter(df_new['R_imoex'], y_v, label='Данные', color='blue')  # Исходные данные
plt.scatter(new_data['R_imoex'], predictions.predicted_mean, color='red',label=f'Точечное значение')

# Заполнение доверительного интервала
plt.fill_between(new_data['R_imoex'], lower_bound, upper_bound, color='grey', alpha=0.5, label='Доверительный интервал', linewidth=5)

# Подписи границ интервала
plt.text(new_data['R_imoex'].values[0], lower_bound.values[0], f'Нижняя граница: {lower_bound.values[0]:.2f}',
         horizontalalignment='right', fontsize=10, color='black')

plt.text(new_data['R_imoex'].values[0], predictions.predicted_mean[0], f'Точечное предсказание: {predictions.predicted_mean[0]:.2f}',
         horizontalalignment='right', fontsize=10, color='black')

plt.text(new_data['R_imoex'].values[0], -1, f'R_imoex: {new_data['R_imoex'].values[0]:.2f}',
         horizontalalignment='right', fontsize=10, color='black')

plt.text(new_data['R_imoex'].values[0], upper_bound.values[0], f'Верхняя граница: {upper_bound.values[0]:.2f}',
         horizontalalignment='right', fontsize=10, color='black')

# Настройки графика
plt.xlabel('R_imoex')
plt.ylabel('R_stock')
plt.legend()
plt.title('Предсказание с доверительным интервалом')
plt.grid(True)  # Добавление сетки для лучшей читаемости
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit

# Предполагается, что df_begin уже определен и содержит нужные данные
df_new = df_begin[['R_stock', 'R_imoex']]
n = len(df_new)

# Определение переменных
x = df_new[['R_imoex']].values.flatten()  # Независимая переменная
y = df_new['R_stock'].values  # Зависимая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Полиномиальная регрессия
degree = 3  # Степень полинома
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
X_test_poly = poly_features.transform(X_test.reshape(-1, 1))

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)

# Показательная регрессия
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Подгонка показательной модели
params_exp, _ = curve_fit(exp_func, x, y)
y_pred_exp = exp_func(X_test, *params_exp)

# Гиперболическая регрессия
def hyperbolic_func(x, a, b):
    return a / x + b

# Подгонка гиперболической модели
# Избегаем деления на ноль, фильтруя только положительные значения
x_positive = x[x > 0]
y_positive = y[x > 0]
params_hyper, _ = curve_fit(hyperbolic_func, x_positive, y_positive)
y_pred_hyper = hyperbolic_func(X_test, *params_hyper)

# Вычисление метрик для всех моделей
def print_metrics(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # Вычисление средней относительной ошибки
    mre = 1/n * np.sum(np.abs((y_true - y_pred) / y_true)) * 100 # Умножаем на 100 для процента
    print(f"{model_name}: R² = {r2:.4f}, MAE = {mae:.4f}, MRE = {mre:.4f}%")

print_metrics(y_test, y_pred_poly, "Полиномиальная регрессия")
print_metrics(y_test, y_pred_exp, "Показательная регрессия")
print_metrics(y_test, y_pred_hyper, "Гиперболическая регрессия")

# Построение графиков
plt.figure(figsize=(12, 8))

# Исходные данные
plt.scatter(x, y, label='Данные', color='blue')

# Полиномиальная регрессия
plt.scatter(X_test, y_pred_poly, label='Полиномиальная регрессия', color='red')
x_range_poly = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_range_poly = model_poly.predict(poly_features.transform(x_range_poly))
plt.plot(x_range_poly, y_range_poly, color='red', linewidth=2)

# Показательная регрессия
plt.scatter(X_test, y_pred_exp, label='Показательная регрессия', color='orange')
plt.plot(np.sort(X_test), exp_func(np.sort(X_test), *params_exp), color='orange', linewidth=2)

# Гиперболическая регрессия
plt.scatter(X_test, y_pred_hyper, label='Гиперболическая регрессия', color='purple')
plt.plot(np.sort(X_test), hyperbolic_func(X_test, *params_hyper), color='purple', linestyle='--')

# Настройки графика
plt.xlabel('R_imoex')
plt.ylabel('R_stock')
plt.title('Регрессии на данных R_stock и R_imoex')
plt.legend()
plt.grid()
plt.show()