import numpy as np
from scipy.linalg import eigh_tridiagonal
from itertools import combinations
from functools import reduce
from operator import mul


class OrthonormalPolynomials:

    def __init__(self, max_degree=10, tolerance=1e-5, method='Eigenvalues'):

        self.max_degree = max_degree
        self.n = max_degree + 1
        self.tolerance = tolerance
        self.method = method

        self.L = None

    def integral_of_power(self, deg: int) -> float:

        if deg % 2 == 1:
            return 0.0
        else:
            return 2.0 / (deg + 1)

    def dot_product(self, vecA: np.ndarray, vecB: np.ndarray, shift=0) -> float:
        result = 0.0
        for i, a_i in enumerate(vecA):
            for j, b_j in enumerate(vecB):
                deg = i + j + shift
                result += a_i * b_j * self.integral_of_power(deg)
        return result

    def check_orthonormal(self):
        if self.L is None:
            raise ValueError("Сначала вызовите build_orthonormal_system()")

        flag = True
        for i in range(self.n):
            #Хотим норму ~ 1
            norm_i = self.dot_product(self.L[i, :i + 1], self.L[i, :i + 1], shift=0)
            if abs(abs(norm_i) - 1.0) >= self.tolerance:
                flag = False
                break

            #Хотим ортогональность с j-й строкой
            for j in range(i + 1, self.n):
                cross = self.dot_product(self.L[i, :i + 1], self.L[j, :j + 1], shift=0)
                if abs(cross) >= self.tolerance:
                    flag = False

        if flag:
            print("YES")
            print(self.L)
        else:
            print("NO")

    def build_gram_matrix_method(self):

        G = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                G[i, j] = self.integral_of_power(i + j)

        cholG = np.linalg.cholesky(G)
        self.L = np.linalg.inv(cholG)

    def build_recur_method(self):
        L = np.zeros((self.n, self.n))
        L[0, 0] = np.sqrt(0.5)  # Нормировка первого полинома
        beta = 0.0

        for i in range(1, self.n):
            alpha = self.dot_product(L[i - 1], L[i - 1], shift=1)


            L[i, 1:i + 1] = L[i - 1, :i]

            L[i, :i] -= alpha * L[i - 1, :i]

            if beta != 0.0:

                L[i, :i - 1] -= beta * L[i - 2, :i - 1]

            #нормировка
            beta = np.sqrt(self.dot_product(L[i], L[i], shift=0))
            L[i, :i + 1] /= beta

        self.L = L

    def build_eigenvalues_method(self):

        L = np.zeros((self.n, self.n))
        L[0, 0] = np.sqrt(0.5)

        alpha_list = []
        beta_list = []
        beta_val = 0.0

        for i in range(1, self.n):
            alpha = self.dot_product(L[i - 1], L[i - 1], shift=1)
            alpha_list.append(alpha)

            roots = eigh_tridiagonal(np.array(alpha_list),
                                     np.array(beta_list))[0]

            for j in range(i):
                sum_val = 0.0
                for combo in combinations(roots, i - j):
                    sum_val += reduce(mul, combo, 1.0)
                L[i, j] = ((-1) ** (i - j)) * sum_val

            L[i, i] = 1.0


            norm_i = np.sqrt(self.dot_product(L[i], L[i], shift=0))
            L[i, :i + 1] /= norm_i

            beta_val = self.dot_product(L[i], L[i - 1], shift=1)
            beta_list.append(beta_val)

        self.L = L

    def build_orthonormal_system(self):

        if self.method == 'Gram':
            self.build_gram_matrix_method()
        elif self.method == 'Recur':
            self.build_recur_method()
        elif self.method == 'Eigenvalues':
            self.build_eigenvalues_method()
        else:
            raise ValueError("Unknown method: choose among 'Gram', 'Recur', 'Eigenvalues'.")


def main():
    ortho_poly = OrthonormalPolynomials(max_degree=10, tolerance=1e-5, method='Eigenvalues')
    ortho_poly.build_orthonormal_system()
    ortho_poly.check_orthonormal()


if __name__ == "__main__":
    main()
