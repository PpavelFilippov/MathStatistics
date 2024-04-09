import numpy as np
from scipy.stats import chi2

def chi_square_test(observed, expected):
    chi2_stat = np.sum((observed - expected)**2 / expected)
    return chi2_stat

def chi_square_test_hypothesis(chi2_stat, df, alpha):
    critical_value = chi2.ppf(1 - alpha, df)
    return chi2_stat < critical_value

# Параметры распределений
sample_sizes = [20, 100]
distributions = {
    'Normal': {'generator': np.random.normal, 'params': {'loc': 0, 'scale': 1}},
    'Student`s (k=3)': {'generator': np.random.standard_t, 'params': {'df': 3}},
    'Uniform': {'generator': np.random.uniform, 'params': {'low': -np.sqrt(3), 'high': np.sqrt(3)}}
}

# Уровень значимости
alpha = 0.05

# Результаты проверки гипотезы
results = {}

for distribution, dist_params in distributions.items():
    for n in sample_sizes:
        # Генерация выборки
        sample = dist_params['generator'](size=n, **dist_params['params'])
        
        # Вычисление частот
        observed, _ = np.histogram(sample, bins='auto')
        
        # Ожидаемые частоты для равномерного распределения
        if distribution == 'Uniform':
            expected = np.repeat(n / len(observed), len(observed))
        else:
            expected = np.repeat(n / len(observed), len(observed))
        
        # Вычисление статистики chi-square
        chi2_stat = chi_square_test(observed, expected)
        
        # Проверка гипотезы
        df = len(observed) - 1
        hypothesis_result = chi_square_test_hypothesis(chi2_stat, df, alpha)
        
        # Сохранение результатов
        results[(distribution, n)] = {
            'observed': observed,
            'expected': expected,
            'chi2_stat': chi2_stat,
            'df': df,
            'hypothesis_result': hypothesis_result
        }

# Вывод результатов
for key, result in results.items():
    distribution, n = key
    print(f"Distribution: {distribution}, Power: {n}")
    print(f"Calculated freq: {result['observed']}")
    print(f"Expected freq: {result['expected']}")
    print(f"Statistics chi-square: {result['chi2_stat']}")
    print(f"free counts number: {result['df']}")
    print(f"Hypothesis result: {result['hypothesis_result']}")
    print()
