import statistics
import math
import scipy
import tabulate
import scipy.stats as stats
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib import transforms
from matplotlib.patches import Ellipse


class Lab1:
    def __init__(self):
        self.sizes = np.array([20, 60, 100])
        self.iterations = 1000
        self.rhos = np.array([0, 0.5, 0.9])

    def multivar_normal(self, size, rho):
        return stats.multivariate_normal.rvs([0, 0], [[1.0, rho], [rho, 1.0]], size=size)

    def mixed_multivar_normal(self, size, rho):
        arr = 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + \
            0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)
        return arr

    def quadrant_coef(self, x, y):
        x_med = np.median(x)
        y_med = np.median(y)
        n = np.array([0,0,0,0])

        for i in range(len(x)):
            if x[i] >= x_med and y[i] >= y_med:
                n[0] += 1
            elif x[i] < x_med and y[i] >= y_med:
                n[1] += 1
            elif x[i] < x_med:
                n[2] += 1
            else:
                n[3] += 1

        return (n[0] + n[2] - n[1] - n[3]) / len(x)

    def pprint(self, arr):
        st = "& "
        for a in arr:
            st += str(a)
            st += ' & '
            #print("& " + a, end=' ')
        return st

    def run(self):
        for size in self.sizes:
            for rho in self.rhos:
                mean, sq_mean, disp = self.generate_stats(self.multivar_normal, size, rho)
                print(f"Normal\t Size = {size}\t Rho = {rho}\t Mean = {self.pprint(mean)}\t Squares mean = {self.pprint(sq_mean)}\t Dispersion = {self.pprint(disp)}")

            mean, sq_mean, disp = self.generate_stats(self.mixed_multivar_normal, size, 0)
            print(f"Mixed\t Size = {size}\t Mean = {self.pprint(mean)}\t Squares mean = {self.pprint(sq_mean)}\t Dispersion = {self.pprint(disp)}")

            self.draw_ellipse(size)

    def generate_stats(self, distr_generator, size, rho):
        names = {"pearson": list(), "spearman": list(), "quadrant": list()}

        for i in range(self.iterations):
            multi_var = distr_generator(size, rho)
            x = multi_var[:, 0]
            y = multi_var[:, 1]

            names['pearson'].append(stats.pearsonr(x, y)[0])
            names['spearman'].append(stats.spearmanr(x, y)[0])
            names['quadrant'].append(self.quadrant_coef(x, y))

        mean = list()
        sq_mean = list()
        disp = list()
        for val in names.values():
            mean.append(np.median(val))
            sq_mean.append(np.median([val[k] ** 2 for k in range(self.iterations)]))
            disp.append(statistics.variance(val))

        return np.around(mean, decimals=4), np.around(sq_mean, decimals=4), np.around(disp, decimals=4)

    def build_ellipse(self, x, y, ax, n_std=3.0):
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        rad_x = np.sqrt(1 + pearson)
        rad_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', edgecolor='red')
        scale_x = np.sqrt(cov[0, 0]) * 3.0
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * 3.0
        mean_y = np.mean(y)

        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def draw_ellipse(self, size):
        fig, ax = plt.subplots(1, 3)
        titles = [f"rho = {rho}" for rho in self.rhos]

        for i in range(len(self.rhos)):
            sample = self.multivar_normal(size, self.rhos[i])
            x, y = sample[:, 0], sample[:, 1]
            self.build_ellipse(x, y, ax[i])
            ax[i].grid()
            ax[i].scatter(x, y, s=5)
            ax[i].set_title(titles[i])

        plt.suptitle(f"Size {size}")
        plt.show()

class Lab2:
    def __init__(self):
        self.a = -1.8
        self.b = 2
        self.step = 0.2

    def ref_func(self, x):
        return 2 * x + 2

    def depend(self, x):
        return [self.ref_func(xi) + stats.norm.rvs(0, 1) for xi in x]

    def get_least_squares_params(self, x, y):
        beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
        beta_0 = np.mean(y) - beta_1 * np.mean(x)
        return beta_0, beta_1

    def least_module(self, parameters, x, y) -> float:
        alpha_0, alpha_1 = parameters
        return sum([abs(y[i] - alpha_0 - alpha_1 * x[i])
                    for i in range(len(x))])

    def get_least_module_params(self, x, y):
        beta_0, beta_1 = self.get_least_squares_params(x, y)
        result = scipy.optimize.minimize(self.least_module, [beta_0, beta_1], args=(x, y), method='SLSQP')
        return result.x[0], result.x[1]

    def least_squares_method(self, x, y):
        beta_0, beta_1 = self.get_least_squares_params(x, y)
        print(f"beta_0 = {beta_0}\t beta_1 = {beta_1}")
        return [beta_0 + beta_1 * p
                for p in x]

    def least_modules_method(self, x, y):
        alpha_0, alpha_1 = self.get_least_module_params(x, y)
        print(f"alpha_0 = {alpha_0}\t alpha_1 = {alpha_1}")
        return [alpha_0 + alpha_1 * p
                for p in x]

    def plot(self, x, y, name: str) -> None:
        y_mnk = self.least_squares_method(x, y)
        y_mnm = self.least_modules_method(x, y)

        mnk_dist = sum((self.ref_func(x)[i] - y_mnk[i]) ** 2 for i in range(len(y)))
        mnm_dist = sum(abs(self.ref_func(x)[i] - y_mnm[i]) for i in range(len(y)))
        print(f"MNK distance = {mnk_dist}\t MNM distance = {mnm_dist}")

        plt.plot(x, self.ref_func(x), color="blue", label="Standart")
        plt.plot(x, y_mnk, color="green", label="MNK")
        plt.plot(x, y_mnm, color="red", label="MNM")
        plt.scatter(x, y, c="navy", label="Sample")
        plt.xlim([self.a, self.b])
        plt.grid()
        plt.legend()
        plt.title(name)
        plt.show()

    def run(self):
        x = np.arange(self.a, self.b, self.step)
        y = self.depend(x)
        self.plot(x, y, "Distribution without perturbation")
        y[0] += 10
        y[-1] -= 10
       # y = self.depend(x)
        self.plot(x, y, "Distribution with perturbation ")

if __name__ == "__main__":
    lab1 = Lab1()
    lab2 = Lab2()
    #lab1.run()
    lab2.run()