import numpy as np


class Interpolation:
    def __init__(self, function, a=-1.0, b=1.0, degree=2, tolerance=1e-10, uniform_grid=True):
        self.function = function
        self.a = a
        self.b = b
        self.degree = degree
        self.tolerance = tolerance
        self.uniform_grid = uniform_grid
        self.points = self._create_points()

    def _create_points(self):
        if self.uniform_grid:
            return np.linspace(self.a, self.b, self.degree + 1, dtype=np.float64)
        else:
            root = [0] * self.degree + [1]
            points = np.polynomial.chebyshev.chebroots(root)
            points[0] = self.a
            points[-1] = self.b
            return points


class VandermondeInterpolation(Interpolation):
    def interpolate(self):
        vander = np.vander(self.points, self.degree + 1, increasing=True)
        r_part = np.array([self.function(x) for x in self.points])

        res = np.linalg.solve(vander, r_part)
        self._validate(res)

        return res

    def _validate(self, coefficients):
        for i, point in enumerate(self.points):
            val = sum(coefficients[j] * point ** j for j in range(len(coefficients)))
            if np.abs(val - self.function(point)) >= self.tolerance:
                print("Can't achieve that tolerance")
                return

        print("ALL GOOD")
        print(coefficients)


class LagrangeInterpolation(Interpolation):
    def interpolate(self):
        res = np.zeros(self.degree + 1)

        for i in range(len(self.points)):
            roots = np.delete(self.points, i)
            p = np.poly1d(np.poly(roots))
            val = np.prod(self.points[i] - roots)
            res += p * self.function(self.points[i]) / val

        self._validate(res)
        return res

    def _validate(self, coefficients):
        for i, point in enumerate(self.points):
            val = np.polyval(coefficients, point)
            if np.abs(val - self.function(point)) >= self.tolerance:
                print("Can't achieve that tolerance")
                return

        print("ALL GOOD")
        print(coefficients)


class OrthogonalPolynomialInterpolation(Interpolation):
    def __init__(self, function, a=-1.0, b=1.0, degree=2, tolerance=1e-10, integral_function=None, uniform_grid=True):
        super().__init__(function, a, b, degree, tolerance, uniform_grid)
        self.integral_function = integral_function

    def interpolate(self):
        G = np.zeros((self.degree + 1, self.degree + 1))

        for i in range(self.degree + 1):
            for j in range(self.degree + 1):
                G[i, j] = self.integral_function(i + j)

        L = np.linalg.inv(np.linalg.cholesky(G))
        l = [np.poly1d(L[i, :i + 1][::-1]) for i in range(self.degree + 1)]

        G = np.zeros((self.degree + 1, self.degree + 1))
        r_part = np.zeros(self.degree + 1)

        for i in range(self.degree + 1):
            for j in range(self.degree + 1):
                G[i, j] = np.polyval(l[j], self.points[i])
            r_part[i] = self.function(self.points[i])

        x_mas = np.linalg.solve(G, r_part)
        res = np.poly1d([0])

        for i, coef in enumerate(x_mas):
            res += coef * l[i]

        self._validate(res)
        return res

    def _validate(self, polynomial):
        for i, point in enumerate(self.points):
            val = np.polyval(polynomial, point)
            if np.abs(val - self.function(point)) >= self.tolerance:
                print("Can't achieve that tolerance")
                return

        print("ALL GOOD")
        print(polynomial)


def func(x):
    return np.abs(x)


def integral_1(val):
    return 0 if val % 2 else 2 / (val + 1)


def main():
    a, b = -1, 1
    degree = 2
    tolerance = 1e-10
    method = "ortho"

    if method == "vandermonde":
        interp = VandermondeInterpolation(func, a, b, degree, tolerance, False)
        result = interp.interpolate()
    elif method == "lagrange":
        interp = LagrangeInterpolation(func, a, b, degree, tolerance, True)
        result = interp.interpolate()
    elif method == "ortho":
        interp = OrthogonalPolynomialInterpolation(func, a, b, degree, tolerance, integral_1, False)
        result = interp.interpolate()


if __name__ == "__main__":
    main()
