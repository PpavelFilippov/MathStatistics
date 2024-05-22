import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate

def chi_square_test(alpha, k, distribution, size, name):
  sample = distribution.rvs(size)
  bins = np.linspace(min(sample), max(sample), k+1)

  frequency, _ = np.histogram(sample, bins=bins)

  p = [distribution.cdf(bins[i+1])-distribution.cdf(bins[i]) for i in range(k)]
  chi2_statistic = np.sum((frequency - np.array(p)*size)**2 / (np.array(p)*size))

  chi2_critical = stats.chi2.ppf(q=1-alpha, df=k-1)

  result = chi2_statistic < chi2_critical
  n = frequency.sum()

  print(f'\n{name} distribution, Sample Size: {size}\n')
  print('chi2_statistic', chi2_statistic)
  print('chi2_critical', chi2_critical)
  print('H0 accepted', result)

  for i in range(k):
   print(
    '\\(',
    '(', f'{bins[i]:.4f}', f'{bins[i+1]:.4f}', ')',
    '\\)',
    '&', f'{frequency[i]:}',
    '&', f'{p[i]:.4f}',
    '&', f'{n * p[i]:.4f}',
    '&', f'{frequency[i] - n * p[i]:.4f}',
    '&', f'{(frequency[i] - n * p[i]) ** 2 / (n * p[i]):.4f}',
    '\\ \\hline')

alpha = 0.05
k = 10
batches = [
  ('Нормальное', stats.norm()),
  ('Стьюдента', stats.t(10)),
  ('Равномерное', stats.uniform()),
]
sizes = [20, 100]

for name, batch in batches:
  for size in sizes:
    chi_square_test(alpha, 5 if size == 20 else 8, batch, size, name)