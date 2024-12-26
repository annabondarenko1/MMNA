import numpy as np

class SplineApproximator:
    def __init__(self, start_point, end_point, function, count_of_points, uniform_grid=True, seed=None):
        self.start_point = start_point
        self.end_point = end_point
        self.function = function
        self.count_of_points = count_of_points
        self.uniform_grid = uniform_grid
        self.seed = seed
        self.result = None

    def _run_through_method(self, side_diag, main_diag, r_part):
        n = main_diag.shape[0]
        x = np.zeros(n)
        alpha = np.zeros(n - 1)
        betta = np.zeros(n - 1)
        alpha[0] = -side_diag[0] / main_diag[0]
        betta[0] = r_part[0] / main_diag[0]
        for i in range(1, n - 1):
            div = side_diag[i] * alpha[i - 1] + main_diag[i]
            alpha[i] = -side_diag[i] / div
            betta[i] = (r_part[i] - side_diag[i] * betta[i - 1]) / div
        x[-1] = (r_part[-1] - side_diag[-1] * betta[-1]) / (side_diag[-1] * alpha[-1] + main_diag[-1])
        for i in range(n - 2, -1, -1):
            x[i] = alpha[i] * x[i + 1] + betta[i]
        return x

    def _function_delta(self, f_point, s_point, t_point, f_segment, s_segment):
        f_val = (self.function(f_point) - self.function(s_point)) / f_segment
        s_val = (self.function(s_point) - self.function(t_point)) / s_segment
        return 6 * (f_val - s_val)

    def _create_r_part(self, points, segments):
        vals = []
        for i in range(segments.shape[0] - 1):
            vals.append(
                self._function_delta(points[i + 2], points[i + 1], points[i], segments[i + 1], segments[i])
            )
        return np.array(vals)

    def build_spline(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.uniform_grid:
            points = np.linspace(self.start_point, self.end_point, self.count_of_points, dtype=np.float64)
        else:
            points = np.sort(np.random.rand(self.count_of_points)) * (self.end_point - self.start_point) + self.start_point
            points[0] = self.start_point
            points[-1] = self.end_point
        segments = points[1:] - points[:-1]
        f_part = self._create_r_part(points, segments)
        main_diag = 2 * (segments[:-1] + segments[1:])
        side_diag = segments[1:-1]
        res = self._run_through_method(side_diag, main_diag, f_part)
        res = np.append(res, 0)
        res = np.insert(res, 0, 0)
        polin = []
        for i in range(1, self.count_of_points):
            func_point = self.function(points[i - 1])
            deriv = ((self.function(points[i]) - func_point) / segments[i - 1]
                     - (res[i] + 2 * res[i - 1]) * segments[i - 1] / 6)
            diff = (res[i] - res[i - 1]) / (2 * segments[i - 1])
            a_3 = (res[i] - res[i - 1]) / (6 * segments[i - 1])
            a_2 = res[i - 1] / 2 - diff * points[i - 1]
            a_1 = deriv - points[i - 1] * res[i - 1] + diff * points[i - 1]**2
            a_0 = (func_point
                   - points[i - 1] * deriv
                   + res[i - 1] * points[i - 1]**2 / 2
                   - diff * points[i - 1]**3 / 3)
            polin.append([a_3, a_2, a_1, a_0])
        self.result = np.array(polin)
        return points, self.result

    def run(self):
        pts, p = self.build_spline()
        return pts, p

def main():
    def F(x):
        return np.sin(4 * x)
    a, b = -1, 1
    n = 9
    approximator = SplineApproximator(a, b, F, n, True, 42)
    _, res = approximator.run()
    print(res)

if __name__ == "__main__":
    main()
