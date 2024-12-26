class RootFinding:

    def __init__(self, start_point=10, tolerance=1e-5):
        self.start_point = start_point
        self.tolerance = tolerance

    def function(self, x: float) -> float:
        return 5 * x**2 + 10 * x + 7

    def function_deriv(self, x: float) -> float:
        return 10 * x + 10

    def test_solution(self, x: float) -> None:
        if abs(self.function(x)) < self.tolerance:
            print("ALL GOOD")
            print(f"f(x) = {self.function(x)}")
        else:
            print("Can't achieve that tolerance")

    def newton_method(self) -> None:
        x = self.start_point
        while abs(self.function(x) / self.function_deriv(x)) >= self.tolerance:
            x = x - self.function(x) / self.function_deriv(x)

        self.test_solution(x)

    def run(self) -> None:
        self.newton_method()


def main():
    rf = RootFinding(start_point=10, tolerance=1e-5)
    rf.run()


if __name__ == "__main__":
    main()
