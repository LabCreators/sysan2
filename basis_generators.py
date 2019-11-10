from numpy.polynomial import Polynomial as pm


class BasicBasisGenerator:
    def __init__(self, degree):
        self.degree = degree

    def basis_smoothed_chebyshev(self, degree):
        if not degree:
            return pm([1])
        elif degree == 1:
            return pm([-1, 2])
        else:
            pm([-2, 4]) * self.basis_smoothed_chebyshev(degree - 1) - self.basis_smoothed_chebyshev(degree - 2)

    def basis_smoothed_legendre(self, degree):
        if not degree:
            return pm([1])
        elif degree == 1:
            return pm([-1, 2])
        else:
            return (pm([-2 * degree - 1, 4 * degree + 2]) * self.basis_smoothed_legendre(degree - 1) -
                    degree * self.basis_smoothed_legendre(degree - 2)) / (degree + 1)

    def basis_laguerre(self, degree):
        if not degree:
            return pm([1])
        elif degree == 1:
            return pm([1, -1])
        else:
            return pm([2 * degree + 1, -1]) * self.basis_laguerre(degree - 1) - pow(degree, 2) * self.basis_laguerre(degree - 2)

    def basis_hermite(self, degree):
        if not degree:
            return pm([1])
        elif degree == 1:
            return pm([0, 2])
        else:
            return pm([0, 2]) * self.basis_hermite(degree - 1) - 2 * (degree - 1) * self.basis_hermite(degree - 2)


class BasisGenerator(BasicBasisGenerator):
    def __init__(self, degree, type):
        super().__init__(degree)
        self.type = type

    def _init_generator(self):
        generators_mapping = {'laguerre': self.basis_laguerre,
                              'hermite': self.basis_hermite,
                              'smoothed_legandr': self.basis_smoothed_legendre,
                              'smoothed_chebyshev': self.basis_smoothed_chebyshev}

        return generators_mapping

    def generate(self):
        generator = self._init_generator().get(self.type)
        for deg in range(self.degree):
            yield generator(deg)