#%%
import pandas as pd
import os
import re
os.chdir("/Users/pavelfilippov/Downloads")

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
def extract_number(filename):
    # Регулярное выражение для нахождения числа в начале строки
    match = re.match(r'^(-?[\d\.]+)', filename)
    if match:
        return float(match.group(1))
    return None
#%%
data = {}
for root, dirs, files in os.walk("..\rawData"):
    for file in files:
        voltage = extract_number(file)
        if voltage in data:
            data[voltage] = pd.concat([data[extract_number(file)], pd.read_csv("..\\data\\" + file, sep=' ').iloc[:, 2]])
        else:
            data[voltage] = pd.read_csv("..\\data\\" + file, sep=' ').iloc[:, 2]
data = pd.DataFrame(data)
        

#%%
# Номер рассматриваемого вольтажа
num_voltage = 5
plt.boxplot([data.iloc[:1024, num_voltage], data.iloc[1024:2048, num_voltage], data.iloc[2048:3072, num_voltage], data.iloc[3072:4096, num_voltage], data.iloc[4096:, num_voltage]], labels=["72", "321", "526", "953", "974"])
plt.xlabel("Ячейки остановки")
plt.ylabel("U вых")
plt.grid()
plt.savefig("fig2")
#%%
def get_least_squares_params(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1

def least_module(parameters, x, y) -> float:
    alpha_0, alpha_1 = parameters
    return sum([abs(y[i] - alpha_0 - alpha_1 * x[i])
                for i in range(len(x))])

def get_least_module_params(x, y):
    beta_0, beta_1 = get_least_squares_params(x, y)
    result = scipy.optimize.minimize(least_module, [beta_0, beta_1], args=(x, y), method='SLSQP')
    return result.x[0], result.x[1]

def least_squares_method(x, y):
    beta_0, beta_1 = get_least_squares_params(x, y)
    print(f"beta_0 = {round(beta_0, 3)}\t beta_1 = {round(beta_1, 3)}")
    # print(f"beta_0 = {beta_0}\t beta_1 = {beta_1}")
    return [beta_0 + beta_1 * p
            for p in x]

def least_modules_method(x, y):
    alpha_0, alpha_1 = get_least_module_params(x, y)
    print(f"alpha_0 = {round(alpha_0, 3)}\t alpha_1 = {round(alpha_1, 3)}")
    # print(f"alpha_0 = {alpha_0}\t alpha_1 = {alpha_1}")
    return [alpha_0 + alpha_1 * p
            for p in x]
#%%
x = np.array(sorted(data.columns.values))
y_mean = np.array(list(data.mean().sort_index()))
y_max = np.array(list(data.max().sort_index()))
y_min = np.array(list(data.min().sort_index()))

y_mnk_mean = least_squares_method(x, y_mean)
y_mnk_max = least_squares_method(x, y_max)
y_mnk_min = least_squares_method(x, y_min)

y_mnm_mean = least_modules_method(x, y_mean)
y_mnm_max = least_modules_method(x, y_max)
y_mnm_min = least_modules_method(x, y_min)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(x, y_mnk_mean, color="blue", label="Линия регрессии")
ax[0].plot(x, y_mnk_max, color="red", label="Граница регрессии")
ax[0].plot(x, y_mnk_min, color="red")
ax[0].set_xlabel("Входное напряжение")
ax[0].set_ylabel("Выходное напряжение")
ax[0].legend()
ax[0].set_title("МНК")

ax[1].plot(x, y_mnm_mean, color="blue", label="Линия регрессии")
ax[1].plot(x, y_mnm_max, color="red", label="Граница регрессии")
ax[1].plot(x, y_mnm_min, color="red")
ax[1].set_xlabel("Входное напряжение")
ax[1].set_ylabel("Выходное напряжение")
ax[1].legend()
ax[1].set_title("МНМ")
plt.savefig("fig4")
#%%
# Номер рассматриваемого вольтажа
num_voltage = 9
fig, axs = plt.subplots(5, 1, figsize=(10, 15))

axs[0].hist(data.iloc[:1024, num_voltage], bins=11)
axs[0].set_title('300')
axs[0].set_xlabel("Выходное напряжение")

axs[1].hist(data.iloc[1024:2048, num_voltage], bins=11)
axs[1].set_title('361')
axs[1].set_xlabel("Выходное напряжение")

axs[2].hist(data.iloc[2048:3072, num_voltage], bins=11)
axs[2].set_title('608')
axs[2].set_xlabel("Выходное напряжение")

axs[3].hist(data.iloc[3072:4096, num_voltage], bins=11)
axs[3].set_title('670')
axs[3].set_xlabel("Выходное напряжение")

axs[4].hist(data.iloc[4096:, num_voltage], bins=11)
axs[4].set_title('933')
axs[4].set_xlabel("Выходное напряжение")
fig.tight_layout()
plt.savefig("fig3")