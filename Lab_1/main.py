from scipy import stats as st
from matplotlib import pyplot as plt
import seaborn as sns
import math
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from prettytable import PrettyTable

names = ["Normal distribution", "Cauchy distribution", "Student distribution", "Poisson distribution",
         "Uniform distribution"]


class Distribution:
    def __init__(self, name=None, size=0, repeat_num=1000):
        self.a = None
        self.b = None
        self.name = name
        self.size = size
        self.random_numbers = None
        self.density = None
        self.pdf = None
        self.cdf = None
        self.repeat_num = repeat_num
        self.x = None

    def __repr__(self):
        return f"{self.name}\nOn interval: [{self.a}, {self.b}]\nSize: {self.size}\nRandom numbers: " \
               f"{self.random_numbers}\nDensity: {self.density}\n\n"

    def set_distribution(self):
        if self.name == names[0]:
            self.random_numbers = st.norm.rvs(size=self.size)
            self.density = st.norm()
        elif self.name == names[1]:
            self.random_numbers = st.cauchy.rvs(size=self.size)
            self.density = st.cauchy()
        elif self.name == names[2]:
            self.random_numbers = st.t.rvs(size=self.size, df = 3)
            self.density = st.t(df = 3)
        elif self.name == names[3]:
            self.random_numbers = st.poisson.rvs(mu=10, size=self.size)
            self.density = st.poisson(10)  # mu = 10
        elif self.name == names[4]:
            a = -math.sqrt(3)
            step = 2 * math.sqrt(3)
            self.random_numbers = st.uniform.rvs(size=self.size, loc=a, scale=step)
            self.density = st.uniform(loc=a, scale=step)

    def set_x_pdf(self):
        if self.name == names[3]:
            self.x = np.arange(self.density.ppf(0.01), self.density.ppf(0.99))
            self.pdf = self.density.pmf(self.x)
        else:
            self.x = np.linspace(self.density.ppf(0.01), self.density.ppf(0.99), num=100)
            self.pdf = self.density.pdf(self.x)

    def set_x_cdf_pdf(self, param: str):
        self.x = np.linspace(self.a, self.b, self.repeat_num)
        if self.name == names[0]:
            self.pdf = st.norm.pdf(self.x)
            self.cdf = st.norm.cdf(self.x)
        elif self.name == names[1]:
            self.pdf = st.cauchy.pdf(self.x)
            self.cdf = st.cauchy.cdf(self.x)
        elif self.name == names[2]:
            self.pdf = st.t.pdf(self.x, df = 3)
            self.cdf = st.t.cdf(self.x, df = 3)
        elif self.name == names[3]:
            if param == 'kde':
                self.x = np.linspace(self.a, self.b, -self.a + self.b + 1)
            self.pdf = st.poisson(10).pmf(self.x)
            self.cdf = st.poisson(10).cdf(self.x)
        elif self.name == names[4]:
            a = -math.sqrt(3)
            step = 2 * math.sqrt(3)
            self.x = np.linspace(self.a, self.b, self.repeat_num)
            self.pdf = st.uniform.pdf(self.x, loc=a, scale=step)
            self.cdf = st.uniform.cdf(self.x, loc=a, scale=step)

    def set_a_b(self, a, b, a_poisson, b_poisson):
        if self.name == names[3]:
            self.a, self.b = a_poisson, b_poisson
        else:
            self.a, self.b = a, b
            
colors = ["deepskyblue", "limegreen", "tomato", "blueviolet", "orange"]


def build_histogram(dist_names, sizes):
    labels = ["size", "distribution"]
    line_type = "k--"
    for i, dist_name in enumerate(dist_names):
        for size in sizes:
            dist = Distribution(dist_name, size)
            dist.set_distribution()
            fig, ax = plt.subplots(1, 1)
            ax.hist(dist.random_numbers, density=True, alpha=0.7, histtype='stepfilled', color=colors[i])
            dist.set_x_pdf()
            ax.plot(dist.x, dist.pdf, line_type)
            ax.set_xlabel(labels[0] + ": " + str(size))
            ax.set_ylabel(labels[1])
            ax.set_title(dist_name)
            plt.grid()
            plt.show()
          
repeat_num = 1000

if __name__ == "__main__":
    # initial conditions
    sizes = [[20, 50, 1000]]

    build_histogram(names, sizes[0]) 