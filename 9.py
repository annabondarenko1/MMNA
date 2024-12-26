import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc

class MonteCarloIntegrator:
    def __init__(self, start_point, end_point, function, tolerance, sigma, max_val=None, method="monte_carlo", grid=None, seed=None):
        self.start_point = start_point
        self.end_point = end_point
        self.function = function
        self.tolerance = tolerance
        self.sigma = sigma
        self.max_val = max_val
        self.method = method
        self.grid = grid
        self.seed = seed

    def _test(self, val):
        err = abs(quad(self.function, self.start_point, self.end_point)[0] - val)
        if err <= self.tolerance:
            print("ALL GOOD")
            print(f"Error = {err}")
            print(f"Integral = {val}")
        else:
            print("Can't achieve that tolerance")
            print(f"Error = {err}")
            print(f"Integral = {val}")

    def _monte_carlo(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        length = self.end_point - self.start_point
        c = int(np.ceil(((self.sigma * length / self.tolerance) ** 2) / 12))
        pts = self.start_point + np.random.rand(c) * length
        val = 0
        for i in range(c):
            val += self.function(pts[i])
        val *= length / c
        self._test(val)

    def _geomic_monte_carlo(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        length = self.end_point - self.start_point
        volume = length * self.max_val
        c = int(np.ceil(((self.sigma * length / self.tolerance) ** 2) / 12))
        print(c)
        if self.grid == "Uniform":
            x = self.start_point + np.random.rand(c) * length
            y = np.random.rand(c) * self.max_val
        elif self.grid == "Sobol":
            s = qmc.Sobol(d=2, scramble=False)
            pts = s.random(c)
            x = [p[0] * length + self.start_point for p in pts]
            y = [p[1] * self.max_val for p in pts]
        else:
            s = qmc.Sobol(d=2, scramble=True)
            pts = s.random(c)
            x = [p[0] * length + self.start_point for p in pts]
            y = [p[1] * self.max_val for p in pts]
        val = 0
        for i in range(c):
            val += (self.function(x[i]) >= y[i])
        val *= volume / c
        self._test(val)

    def run(self):
        if self.method == "monte_carlo":
            self._monte_carlo()
        elif self.method == "geomic_monte_carlo":
            self._geomic_monte_carlo()

def func(x):
    return np.exp(x)

def main():
    start_point = 0.1
    end_point = 2
    tolerance = 5e-2
    max_val = np.exp(3)
    sigma = 3
    method = "geomic_monte_carlo"
    grid = "Sobol"
    integrator = MonteCarloIntegrator(start_point, end_point, func, tolerance, sigma, max_val, method, grid, seed=None)
    integrator.run()

if __name__ == "__main__":
    main()
