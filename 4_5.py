import numpy as np
from numpy.polynomial.polynomial import Polynomial


class MultipNewtonInterpolator:

    def __init__(self,
                 start_point=-1.0,
                 end_point=1.0,
                 degree=30,
                 tolerance=1e-2,
                 uniform_grid=True,
                 random_seed=None):

        self.start_point = start_point
        self.end_point = end_point
        self.degree = degree
        self.tolerance = tolerance
        self.uniform_grid = uniform_grid
        self.random_seed = random_seed

        self.n = self.degree + 1
        self.points = None
        self.polynomial = None

    def _func(self, x: float, step: int) -> float:

        r = step % 4
        if r == 0:
            return np.sin(x)
        elif r == 1:
            return np.cos(x)
        elif r == 2:
            return -np.sin(x)
        else:
            return -np.cos(x)

    def _create_points(self) -> np.ndarray:

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        count_of_points = self.n

        n_init = int((count_of_points + 1) / 2)

        if self.uniform_grid:
            points = np.linspace(self.start_point, self.end_point, n_init, dtype=np.float64)
        else:
            root = [0 for _ in range(n_init)]
            root.append(1)
            points = np.polynomial.chebyshev.chebroots(root)
            points[0] = self.start_point
            points[-1] = self.end_point

        points_left = count_of_points - n_init

        randomized = np.random.choice(n_init, points_left)
        points = np.sort(np.append(points, points[randomized]))

        return points

    def build_polynomial(self):

        self.points = self._create_points()
        n = len(self.points)

        l = [[self._func(x, 0) for x in self.points]]

        for i in range(1, n):
            div = []
            for j in range(0, n - i):
                dx = self.points[j] - self.points[j + i]
                if abs(dx) < 1e-10:
                    div.append(self._func(self.points[j], i) / np.math.factorial(i))
                else:
                    div.append((l[i - 1][j] - l[i - 1][j + 1]) / dx)
            l.append(div)

        res_poly = Polynomial([l[0][0]])

        for i in range(1, n):
            factor_poly = Polynomial.fromroots(self.points[:i])
            res_poly += l[i][0] * factor_poly

        self.polynomial = res_poly

    def check_polynomial(self):

        if self.polynomial is None:
            raise ValueError("Сначала вызовите build_polynomial().")

        unique_points, counts = np.unique(self.points, return_counts=True)

        success = True
        for i, point in enumerate(unique_points):
            multiplicity = counts[i]

            for k in range(multiplicity):

                derivative_poly = self.polynomial.deriv(k)

                val = 0
                for power, coeff in enumerate(derivative_poly):
                    val += coeff * point**power

                ref_val = self._func(point, k)
                if abs(val - ref_val) >= self.tolerance:
                    success = False

        if success:
            print("YES (MultipNewton)")
            print(self.polynomial)
        else:
            print("NO (MultipNewton)")
            print(self.polynomial)

    def run(self):

        self.build_polynomial()
        self.check_polynomial()


class NewtonInterpolator:

    def __init__(self,
                 start_point=-1.0,
                 end_point=1.0,
                 degree=14,
                 tolerance=1e-5,
                 uniform_grid=True):

        self.start_point = start_point
        self.end_point = end_point
        self.degree = degree
        self.tolerance = tolerance
        self.uniform_grid = uniform_grid

        self.n = self.degree + 1
        self.points = None
        self.f_val = None
        self.polynomial = None

    def _func(self, x: float) -> float:

        return abs(x)

    def _create_points(self) -> np.ndarray:

        if self.uniform_grid:
            points = np.linspace(self.start_point, self.end_point, self.n, dtype=np.float64)
        else:
            root = [0 for _ in range(self.n)]
            root.append(1)
            points = np.polynomial.chebyshev.chebroots(root)
            points[0] = self.start_point
            points[-1] = self.end_point

        return points

    def _mult(self, idx: int, pts: np.ndarray) -> float:

        val = 1.0
        for i in range(idx):
            val *= (pts[idx] - pts[i])
        return val

    def build_polynomial(self):

        self.points = self._create_points()
        self.f_val = np.array([self._func(x) for x in self.points])

        dividers = [1.0]
        res_poly = Polynomial([self.f_val[0]])
        p = Polynomial([1.0])

        for i in range(1, self.n):

            p *= Polynomial.fromroots([self.points[i - 1]])

            for j in range(i):
                dividers[j] *= (self.points[j] - self.points[i])

            dividers.append(self._mult(i, self.points))

            val = 0.0
            for j in range(i + 1):
                val += self.f_val[j] / dividers[j]

            res_poly += p * val

        self.polynomial = res_poly

    def check_polynomial(self):

        if self.polynomial is None:
            raise ValueError("Сначала вызовите build_polynomial().")

        corr = True
        for i in range(self.n):
            x_i = self.points[i]

            poly_val = 0.0
            for power, coeff in enumerate(self.polynomial):
                poly_val += coeff * (x_i**power)

            if abs(poly_val - self._func(x_i)) >= self.tolerance:
                corr = False

        if corr:
            print("YES (Newton)")
            print(self.polynomial)
        else:
            print("NO (Newton)")
            print(self.polynomial)

    def run(self):

        self.build_polynomial()
        self.check_polynomial()


def main():

    print("=== MultipNewtonInterpolator ===")
    multi_interp = MultipNewtonInterpolator(start_point=-1,
                                            end_point=1,
                                            degree=30,
                                            tolerance=1e-2,
                                            uniform_grid=True,
                                            random_seed=42)
    multi_interp.run()

    print("\n=== NewtonInterpolator ===")
    newt_interp = NewtonInterpolator(start_point=-1,
                                     end_point=1,
                                     degree=14,
                                     tolerance=1e-5,
                                     uniform_grid=False)
    newt_interp.run()


if __name__ == "__main__":
    main()
