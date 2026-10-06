"""Check the dense oracle independently of the Rust comparisons."""
import csv
from decimal import Decimal as D, localcontext
import importlib.util
from pathlib import Path
import unittest

import adjudicate as oracle


class OracleTests(unittest.TestCase):
    def test_inverse_with_row_pivot(self):
        with localcontext() as context:
            context.prec = 80
            a = [[D(0), D(2)], [D(3), D(1)]]
            product = oracle.multiply(a, oracle.inverse(a))
            self.assertLess(max(abs(product[i][j] - D(i == j)) for i in range(2) for j in range(2)), D('1e-70'))

    def test_known_three_dimensional_square_root(self):
        with localcontext() as context:
            context.prec = 80
            expected = [[D(v) for v in row] for row in [[2, 1, 0], [1, 2, 0], [0, 0, 4]]]
            actual = oracle.square_root(oracle.multiply(expected, expected))
            self.assertLess(max(abs(actual[i][j] - expected[i][j]) for i in range(3) for j in range(3)), D('1e-50'))
            values, _ = oracle.eigen(expected)
            self.assertLess(max(abs(a - b) for a, b in zip(sorted(values), [1, 3, 4])), D('1e-50'))

    def test_agrees_with_independent_closed_form_on_all_two_dimensional_cases(self):
        source = Path(__file__).resolve().parents[1] / 'generate.py'
        spec = importlib.util.spec_from_file_location('closed_form', source)
        closed = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(closed)
        rows = csv.DictReader(line for line in oracle.INPUT.read_text().splitlines() if not line.startswith('#'))
        count = 0
        with localcontext() as context:
            context.prec = 80
            for row in rows:
                if row['dim'] != '2':
                    continue
                def data(key):
                    values = list(map(D, row[key].split(';')))
                    return [values[i:i + 2] for i in range(0, len(values), 2)]
                with localcontext() as closed_context:
                    closed_context.prec = 120
                    expected = closed.dense(data('q'), data('score'), list(map(D, row['scales'].split(';'))), D(row['ridge']), D(row['threshold']), 2)
                actual = oracle.solve(row)
                matrix = list(map(D, actual[2].split(';')))
                for a, b in zip(matrix, [expected[0], expected[1], expected[1], expected[2]]):
                    self.assertLess(abs(a - b), D('1e-45'))
                self.assertLess(abs(D(actual[3]) - expected[3]), D('1e-45'))
                self.assertEqual(actual[1], expected[4])
                count += 1
        self.assertEqual(count, 6)


if __name__ == '__main__':
    unittest.main()
