import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import chi2, f, t

matplotlib.use('TkAgg')  # или 'Qt5Agg'

data = pd.read_excel('data_sber.xlsx')

dff = pd.DataFrame(data)

df = dff.iloc[:, 5:dff.shape[1]]
df = df.dropna()
print(df)

# Заменяем экселевские запятые на точки для чисел если вдруг пандас не сделал этого
for column in df.columns[1:]:  # Start from the second column (index 1)
    df[column] = df[column].astype(str).str.replace(',', '.').astype(float)
print(df)


# Функция для удаления выбросов с использованием IQR
def remove_outliers_iqr(df_old):
    for column in df_old.columns:
        Q1 = df_old[column].quantile(0.25)
        Q3 = df_old[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_old = df_old[(df_old[column] >= lower_bound) & (df_old[column] <= upper_bound)]
    return df_old


# Функция для удаления выбросов с использованием Z-оценки
def remove_outliers_zscore(df_old, threshold=3):
    from scipy import stats
    z_scores = np.abs(stats.zscore(df_old))
    return df[(z_scores < threshold).all(axis=1)]


# Удаление выбросов с использованием IQR
data_no_outliers_iqr = remove_outliers_iqr(df.copy())
print("Данные после удаления выбросов с использованием IQR:")
print(data_no_outliers_iqr.describe())
print(data_no_outliers_iqr)

df = data_no_outliers_iqr.copy()

# Построение диаграммы рассеяния для каждой пары факторов после удаления выбросов
for column in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=column, y='R_stock', color='blue', label='Данные')

df = df.iloc[:, 1:df.shape[1]]
print(df)

# 1) Построение матрицы межфакторных корреляций
correlation_matrix = df.corr()
print("Матрица межфакторных корреляций:")
print(correlation_matrix)

# Визуализация матрицы межфакторных корреляций
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Матрица межфакторных корреляций')
plt.show()

# Определитель матрицы
determinant = np.linalg.det(correlation_matrix)
print(f"\nОпределитель матрицы: {determinant}")

# 2) Вычисление статистики Фаррара-Глоубера
n = len(df)  # количество наблюдений
k = df.shape[1]  # количество факторов
alpha = 0.1 # уровень значимости
farrar_glober_statistic = -1*(n-1-1/6*(2*k+5)) * math.log(determinant)
print(f"\nСтатистика Фаррара-Глоубера: {farrar_glober_statistic}")

# Табличное значение для Хи-квадрат
df_chi2 = (k - 1) * (k - 1)  # степени свободы
chi2_table_value = chi2.ppf(1 - alpha, df_chi2)
print(f"\nТабличное значение Хи-квадрат: {chi2_table_value}")


# 3) Вычисление обратной матрицы
inverse_matrix = np.linalg.inv(correlation_matrix)
print("\nОбратная матрица:")
print(inverse_matrix)

# Визуализация обратной матрицы
plt.figure(figsize=(8, 6))
sns.heatmap(inverse_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Обратная матрица')
plt.show()

# Степени свободы
df1 = k
df2 = n-k-1

# Находим критическое значение F
f_critical = f.ppf(1 - alpha, df1, df2)

print(f"Критическое значение F при уровне значимости {alpha}: {f_critical}")
# Вычисление F-критериев
f_statistics = np.diag(inverse_matrix)
print("\nF-критерии:")
print(f_statistics)

# Визуализация F-критериев
plt.figure(figsize=(8, 4))
sns.barplot(x=df.columns, y=f_statistics)
plt.title('F-критерии')
plt.ylabel('Значение F-критерия')
plt.axhline(y=0, color='r', linestyle='--')  # Уровень 0
plt.show()

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

# Визуализация матрицы частных корреляций
plt.figure(figsize=(8, 6))
sns.heatmap(partial_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Матрица частных корреляций')
plt.show()

ddf = n - k - 1  # Степени свободы

# Функция для расчета t-критерия
def calculate_t_statistic(r, n, k):
    return (r * np.sqrt(n - k - 1)) / np.sqrt(1 - r**2)

# Расчет t-критериев
t_statistics = pd.DataFrame(index=partial_corr_matrix.index, columns=partial_corr_matrix.columns)

for i in range(len(partial_corr_matrix)):
    for j in range(len(partial_corr_matrix)):
        if i != j:
            r = partial_corr_matrix.iloc[i, j]  # Используем iloc для доступа по индексам
            t_statistics.iloc[i, j] = calculate_t_statistic(r, n, k)

print()
print("t-критерии для коэффициентов частной корреляции:")
print(t_statistics)
# Критическое значение t
t_critical = t.ppf(1 - alpha/2, ddf)

print(f"\nКритическое значение t при уровне значимости {alpha}: {t_critical}")