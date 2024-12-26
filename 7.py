import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad
import scipy.special

class Integrator:
    def __init__(self, start_point, end_point, function, count_of_points, tolerance, method):
        self.start_point = start_point
        self.end_point = end_point
        self.function = function
        self.count_of_points = count_of_points
        self.tolerance = tolerance
        self.method = method

    def _replace_point(self, point):
        return ((self.start_point + self.end_point) + (self.end_point - self.start_point) * point) / 2

    def _mult(self, x, roots):
        val = 1
        for y in roots:
            val *= (x - y)
        return val

    def _test(self, val):
        r = quad(self.function, self.start_point, self.end_point)
        d = np.abs(r[0] - val)
        if d < self.tolerance:
            print("ALL GOOD")
        else:
            print("Can't achieve that tolerance")
        print(f'dif={d}')
        print(f'result={r[0]}')
        print(f'val={val}')

    def newton_cotes(self):
        pts = np.linspace(-1, 1, self.count_of_points + 1, dtype=np.float64)
        fvals = np.array([self.function(self._replace_point(p)) for p in pts])
        val = 0
        for i in range(self.count_of_points + 1):
            roots = pts.copy()
            roots = np.delete(roots, i)
            p = Polynomial.fromroots(roots).integ()
            val += fvals[i] * (p(1) - p(-1)) / self._mult(pts[i], roots)
        val *= (self.end_point - self.start_point) / 2
        self._test(val)

    def gaus(self):
        pts = scipy.special.p_roots(self.count_of_points)[0]
        fvals = np.array([self.function(self._replace_point(p)) for p in pts])
        val = 0
        for i in range(self.count_of_points):
            pol_der = (scipy.special.lpn(self.count_of_points, pts[i])[1][-1])**2
            c = 2 / ((1 - pts[i]**2) * pol_der)
            val += c * fvals[i]
        val *= (self.end_point - self.start_point) / 2
        self._test(val)

    def clenshaw_curtis(self):
        m = self.count_of_points // 2
        z = [1 - 4*k**2 for k in range(m)]
        val = 0
        for j in range(self.count_of_points + 1):
            w = 1 / 2
            mm = 2 * j * np.pi / self.count_of_points
            c = 0
            for k in range(1, m):
                c += mm
                w += np.cos(c) / z[k]
            w *= 4 / self.count_of_points
            if not j or j == self.count_of_points:
                w /= 2
            val += w * self.function(self._replace_point(np.cos(mm / 2)))
        val *= (self.end_point - self.start_point) / 2
        self._test(val)

    def run(self):
        if self.method == "newton_cotes":
            self.newton_cotes()
        elif self.method == "gaus":
            self.gaus()
        elif self.method == "clenshaw_curtis":
            self.clenshaw_curtis()

def func(x):
    return np.exp(x)

def main():
    count_of_points = 30
    a, b = -10, 10
    eps = 1e-5
    method = "clenshaw_curtis"
    integrator = Integrator(a, b, func, count_of_points, eps, method)
    integrator.run()

if __name__ == "__main__":
    main()
