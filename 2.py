import numpy as np

class RemezApproximator:
    def __init__(self, function, a=-1.0, b=1.0, degree=1, tolerance=1e-5):
        self.function = function
        self.a = a
        self.b = b
        self.degree = degree
        self.tolerance = tolerance


        self.p_coef = None
        self.max_dev = None

    @staticmethod
    def _poly_value(x, coef):

        val = 0.0
        for i, c in enumerate(coef):
            val += c * x**i
        return val

    def _maximize_scan(self, p_coef, num_points=1000):

        x_vals = np.linspace(self.a, self.b, num_points)
        #Считаем |f(x) - p(x)| для всех x на сетке
        errors = np.array([
            abs(self.function(x) - self._poly_value(x, p_coef))
            for x in x_vals
        ])
        max_index = np.argmax(errors)
        return x_vals[max_index]

    def build_approx(self):

        points = np.sort(np.random.rand(self.degree + 2) * (self.b - self.a) + self.a)

        vander = np.column_stack((
            np.vander(points, self.degree + 1),
            np.array([(-1)**i for i in range(self.degree + 2)])
        ))
        F_val = np.array([self.function(x) for x in points])
        err = self.tolerance * 100

        while err > self.tolerance:
            res = np.linalg.solve(vander, F_val)

            p_coef = res[-2::-1]
            d = res[-1]
            new_point = self._maximize_scan(p_coef)
            func_res = self.function(new_point) - self._poly_value(new_point, p_coef)
            ind = np.argwhere(points > new_point)

            if ind.shape[0]:
                ind = ind.min()
                if vander[ind, -1] * d * func_res < 0:
                    if ind == 0:
                        vander[1:, :] = vander[:-1, :]
                        F_val[1:] = F_val[:-1]
                        vander[ind, -1] = -vander[ind, -1]
                        points[1:] = points[:-1]
                    else:
                        ind -= 1
            else:
                ind = self.degree + 1
                if vander[ind, -1] * d * func_res < 0:
                    vander[:-1, :] = vander[1:, :]
                    F_val[:-1] = F_val[1:]
                    vander[ind, -1] = -vander[ind, -1]
                    points[:-1] = points[1:]


            vander[ind, :-1] = np.array([new_point**i for i in range(self.degree, -1, -1)])
            F_val[ind] = self.function(new_point)
            points[ind] = new_point

            err = np.abs(np.abs(func_res) - np.abs(d))

        self.p_coef = p_coef
        self.max_dev = abs(func_res)

        return self.p_coef, self.max_dev

def main():
    eps = 1e-5
    deg = 3
    a, b = -1, 1

    remez_approx = RemezApproximator(function=np.sin, a=a, b=b, degree=deg, tolerance=eps)

    p_vals, max_dev = remez_approx.build_approx()

    print(f"max_d = {max_dev}")
    print(f"coefs = {p_vals}")

if __name__ == "__main__":
    main()
