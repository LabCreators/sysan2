import pandas as pd
from basis_generators import BasisGenerator
from optimization_methods import *
from functools import reduce


class Solve(object):

    def __init__(self, d):
        self.n = d['samples']
        self.deg = d['dimensions']
        self.filename_input = d['input_file']
        self.filename_output = d['output_file']
        self.dict = d['output_file']
        self.p = list(map(lambda x:x+1,d['degrees'])) # on 1 more because include 0
        self.weights = d['weights']
        self.poly_type = d['poly_type']
        self.splitted_lambdas = d['lambda_multiblock']
        self.solving_method = d['method']
        self.eps = 1E-6
        self.norm_error = 0.0
        self.error = 0.0

    def _prepare_data(self):
        new_cols = list(itertools.chain(*[['X{}'.format(i + 1)] * self.deg[i] if i != len(self.deg) - 1 else ['Y'] * self.deg[i]
                               for i in range(len(self.deg))]))
        new_cols = list(itertools.chain(*np.unique([[el + str(i)
                                                     for i in range(1, new_cols.count(el) + 1)] for el in new_cols])))
        dt = pd.read_csv(self.filename_input, sep='\t', header=None).astype(float)
        dt.columns = new_cols
        return dt

    def _minimize_equation(self, A, b):
        """
        Finds such vector x that |Ax-b|->min.
        :param A: Matrix A
        :param b: Vector b
        :return: Vector x
        """
        if self.solving_method == 'LSTM':
            return np.linalg.lstsq(A,b)[0]
        elif self.solving_method == 'conjucate':
            return conjugate_gradient_method(A.T*A, A.T*b, self.eps)
        elif self.solving_method == 'coordDesc':
            return coordinate_descent(A, b, self.eps)

    def _norm_data(self, data):
        normalized_data = data.copy(deep=True)
        agg = normalized_data.agg([min, max])

        for col in normalized_data.columns:
            min_val = agg.loc['min', col]
            max_val = agg.loc['max', col]
            normalized_data[col] = normalized_data[col].apply(lambda x: (x - min_val) / (max_val - min_val))

        return normalized_data

    def _create_train_dataset(self, data):
        '''
        build matrix X and Y
        :return:
        '''
        X = data.loc[:, [el for el in data.columns if el.find('X') != -1]]
        Y = data.loc[:, [el for el in data.columns if el.find('Y') != -1]]

        return X, Y

    def _get_B(self, Y):
        if self.weights == 'average':
            Y = (Y.max(axis=1) + Y.min(axis=1)) / 2 #arguable, may be need not to normalize Y before this operation
            return np.tile(Y.values.reshape(self.n, 1), (1, self.deg[-1]))
        elif self.weights == 'width_interval':
            np.tile((Y.max(axis=1) - Y.min(axis=1)).values.reshape(self.n, 1), (1, self.deg[-1])) #arguable (same as above)
        else:
            return Y

    def _evaluate_polynom(self, coefs, x):
        return sum([np.array(coef) * pow(x, i) for i, coef in enumerate(coefs)])

    def _get_A(self, data, polynoms_degrees):
        A = pd.DataFrame()
        for i, degree in enumerate(polynoms_degrees):
            gen = BasisGenerator(degree, self.poly_type)
            coefs = list(map(lambda x: x, list(gen.generate())[-1]))
            A = pd.concat([A, data.apply(lambda x: self._evaluate_polynom(coefs, x))], axis=1)
        return A

    def _get_lambdas(self, A, Y):
        lambdas = pd.DataFrame(columns=['lambda_{}'.format(i) for i in range(1, self.deg[-1])])

        for i, j in itertools.product(range(self.deg[-1]), range(len(self.deg) - 1)):
            use_cols = [el for el in A.columns if el.find('X{}'.format(j + 1)) != -1]
            train_data = A.loc[:, use_cols]
            a = train_data.T * Y.loc[:, Y.columns[i]]
            lambdas.loc[i, lambdas.columns[j]] = [self._minimize_equation(a.T.values, Y.loc[:, Y.columns[i]])]
        return lambdas

    def _get_psi(self, A, lambdas):
        psi = [[A.loc[:, [el for el in A.columns if el.find('X{}'.format(i + 1)) != -1]
                ] * lambdas.loc[j, 'lambda_{}'.format(i + 1)][0] for i in range(len(self.deg)-1)] for j in range(self.deg[-1])]
        return psi

    def _get_A1(self, psi, y):
        return [[self._minimize_equation(psi[i][j][:], y.loc[:, y.columns[i]]) for j in range(len(self.deg)-1)]
                for i in range(self.deg[-1])]

    def _get_Fi(self, psi, a1):
        fi = np.array([[psi[i][j] * a1[i][j] for j in range(len(self.deg)-1)] for i in range(self.deg[-1])])
        fi = [reduce(lambda x, y: pd.concat([x, y], axis=1), fi[i]) for i in range(self.deg[-1])]
        return fi

    def _get_coefs(self, fi, y):
        return [self._minimize_equation(np.dot(fi[i].T, fi[i]),
                                np.dot(fi[i].T, y.iloc[:, i])) for i in range(self.deg[-1])]

    #TODO Fitness function for normalize version
    def _get_fitness_function(self, fi, y, coefs):
        fitness = [np.dot(fi[i], coefs[i]) for i in range(self.deg[-1])]
        norm_error = [(y.iloc[:, i] - fitness[i]) for i in range(self.deg[-1])]
        return fitness, norm_error

    #TODO try to put into get_B not normalized Y
    def _save_data(self, data, norm_data, A, lambdas, psi, A1, y_new, y_new_normalized, c,
                  fit_res, errors, nor_errors):
        with pd.ExcelWriter(self.filename_output) as writer:
            data.to_excel(writer, sheet_name='Вхідні дані')
            norm_data.to_excel(writer, sheet_name='Нормалізовані вхідні дані')
            A.to_excel(writer, sheet_name='Матриця А')
            lambdas.to_excel(writer, sheet_name='Значення лямбд')
            for i in range(psi.shape[0]):
                temp = reduce(lambda x, y: pd.concat([x, y], axis=1), psi[i])
                temp.to_excel(writer, sheet_name='PSI{}'.format(i + 1))
            A1.to_excel(writer, sheet_name='матриця А1')
            y_new.to_excel(writer, sheet_name='Перебудовані Y')
            y_new_normalized.to_excel(writer, sheet_name='Перебудовані Y нормалізовані')
            c.to_excel(writer, sheet_name='Коефіцієнти c')
            fit_res.to_excel(writer, sheet_name='Побудований прогноз')
            errors.to_excel(writer, sheet_name='Похибки')
            nor_errors.to_excel(writer, sheet_name='Норми похибок')

    def print_data(self, data, norm_data, A, lambdas, psi, A1, y_new, y_new_normalized, c,
                   fit_res, errors, nor_errors):
        print('Вхідні дані')
        print(data.to_string())
        print('-------------------------')
        print('Нормалізовані вхідні дані')
        print(norm_data.to_string())
        print('-------------------------')
        print('Матриця А')
        print(A.to_string())
        print('-------------------------')
        print('Значення лямбд')
        print(lambdas.to_string())
        print('-------------------------')
        for i in range(psi.shape[0]):
            temp = reduce(lambda x, y: pd.concat([x, y], axis=1), psi[i])
            print('PSI{}'.format(i + 1))
            print(temp.to_string())
            print('---------------------')
        print('матриця А1')
        print(A1.to_string())
        print('-------------------------')
        print('Перебудовані Y')
        print(y_new.to_string())
        print('-------------------------')
        print('Перебудовані Y нормалізовані')
        print(y_new_normalized.to_string())
        print('-------------------------')
        print('Коефіцієнти c')
        print(c.to_string())
        print('-------------------------')
        print('Побудований прогноз')
        print(fit_res.to_string())
        print('-------------------------')
        print('Похибки')
        print(errors.to_string())
        print('-------------------------')
        print('Норми похибок')
        print(nor_errors.to_string())

    def main(self, print_=False):
        prepared_data = self._prepare_data()
        normalized_data = self._norm_data(prepared_data)
        train_data, target = self._create_train_dataset(normalized_data)
        A = self._get_A(train_data, self.deg)
        b = self._get_B(target)
        lambdas = self._get_lambdas(A, b)
        psi = self._get_psi(A, lambdas)
        A1 = self._get_A1(psi, b)
        Fi = self._get_Fi(psi, A1)
        coefs = self._get_coefs(Fi, b)
        fitnes_result, error = self._get_fitness_function(Fi, b, coefs)

        self._save_data(prepared_data, normalized_data, A, lambdas, np.array(psi), pd.DataFrame(A1), b,
                  b, pd.DataFrame(coefs), pd.DataFrame(fitnes_result).T, pd.DataFrame(error).T,
                  pd.DataFrame(pd.DataFrame(error).T.apply(lambda x: np.linalg.norm(x))).T)

        if print_:
            self.print_data(prepared_data, normalized_data, A, lambdas, np.array(psi), pd.DataFrame(A1), b,
                        b, pd.DataFrame(coefs), pd.DataFrame(fitnes_result).T, pd.DataFrame(error).T,
                        pd.DataFrame(pd.DataFrame(error).T.apply(lambda x: np.linalg.norm(x))).T)

        return [prepared_data, normalized_data, A, lambdas, np.array(psi), pd.DataFrame(A1), b,
                        b, pd.DataFrame(coefs), pd.DataFrame(fitnes_result).T, pd.DataFrame(error).T,
                        pd.DataFrame(pd.DataFrame(error).T.apply(lambda x: np.linalg.norm(x))).T]