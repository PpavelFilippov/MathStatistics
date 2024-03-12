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


          
repeat_num = 1000

def build_boxplot(dist_names, sizes):
    for dist_name in dist_names:
        tips = []
        for size in sizes:
            dist = Distribution(dist_name, size)
            emission_share(dist, dist.repeat_num)
            tips.append(dist.random_numbers)
        draw_boxplot(dist_name, tips)


def mustache(distribution):
    q_1, q_3 = np.quantile(distribution, [0.25, 0.75])
    return q_1 - 3 / 2 * (q_3 - q_1), q_3 + 3 / 2 * (q_3 - q_1)


def count_emissions(distribution):
    x1, x2 = mustache(distribution)
    filtered = [x for x in distribution if x > x2 or x < x1]
    return len(filtered)


def emission_share(dist, repeat_num):
    count = 0
    for i in range(repeat_num):
        dist.set_distribution()
        arr = sorted(dist.random_numbers)
        count += count_emissions(arr)
    count /= (dist.size * repeat_num)
    dist.set_distribution()
    print(f"{dist.name} Size {dist.size}: {count}")


def draw_boxplot(dist_name, data):
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=data, palette='pastel', orient='h')
    sns.despine(offset=10)
    plt.xlabel("x")
    plt.ylabel("n")
    plt.title(dist_name)
    plt.show()
    
a, b = -4, 4
a_poisson, b_poisson = 6, 14
coefs = [0.5, 1, 2]


if __name__ == "__main__":
    # initial conditions
    sizes =  [20, 100]

    build_boxplot(names, sizes)  # lab 3

  