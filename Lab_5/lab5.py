import numpy as np 
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

a = -1.8
b = 2
n = 20

fig1 = plt.figure()
x = np.linspace(a, b, n)
e = np.random.normal(0, 1, n)
y = 2 + 2 * x + e
plt.scatter(x, y)
A = np.vstack([x, np.ones(len(x))]).T

b_sq, a_sq = np.linalg.lstsq(A, y, rcond=None)[0]
x_p = np.linspace(a, b, 100)
y_p = x_p * a_sq + b_sq
plt.plot(x_p, y_p)
fig1.savefig('K_1.png')


def err_func(params, x, y):
    return np.sum(np.abs(params[0] + params[1] * x - y))

a_abs, b_abs = minimize(err_func, [0,0], args=(x, y)).x


fig2 = plt.figure()
plt.scatter(x, y)

x_p = np.linspace(a, b, 100)
y_p = x_p * a_abs + b_abs
plt.plot(x_p, y_p)
fig2.savefig('M_1.png')

y_mod = y.copy()
y_mod[0] += 10
y_mod[-1] -= 10
A_mod = np.vstack([x, np.ones(len(x))]).T

b_sq_mod, a_sq_mod = np.linalg.lstsq(A_mod, y_mod, rcond=None)[0]

def residuals(params):
    return np.abs(params[0] + params[1] * x - y_mod)

a_abs_mod, b_abs_mod = minimize(err_func, [0, 0], args=(x, y_mod)).x
fig3 = plt.figure()
plt.scatter(x, y_mod)

x_p = np.linspace(a, b, 100)
y_p = x_p * a_sq_mod + b_sq_mod
plt.plot(x_p, y_p)
fig3.savefig('K_2.png')

fig4 = plt.figure()

plt.scatter(x, y_mod)

x_p = np.linspace(a, b, 100)
y_p = x_p * a_abs_mod + b_abs_mod
plt.plot(x_p, y_p)
fig4.savefig('M_2.png')


print(f"МНК без возмущений: a = {a_sq}, b = {b_sq}")
print(f"МНМ без возмущений: a = {a_abs}, b = {b_abs}")

print(f"МНК с возмущенииями: a = {a_sq_mod}, b = {b_sq_mod}")
print(f"МНМ с возмущенииями: a = {a_abs_mod}, b = {b_abs_mod}")
