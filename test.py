import unittest
from code.grid_optimizer import *
from code.model import *
from code.simulation import *
from code.help_functions import *
from code.look_up_table import look_up_table, convert_index_to_state, convert_state_to_index
from main import *

# set constants: prices, state-space, decision-space
# and max expected change rate of consumption
# HERE THEY ARE SET AGAIN, SINCE THE TESTS ARE BASED
# ON THESE SPECIFIC VALUES!
P_e = 20
P_i = 10
P_b = 5
U = [-2, -1, 0, 1, 2]
O = list(range(11))
V = list(range(11))
B = list(range(11))
V_max_change = 4
B_max_charge = max(B)


class TestSimulation(unittest.TestCase):
    pass


class TestOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = GridModel(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b, U=U, O=O, V=V, B=B,
                               V_max_change=V_max_change, distribution="binom")
        self.optimizer = GridOptimizer(model=self.model)
        self.optimizer.calculate_cost_to_go_matrix_sequence(depth=2)


class TestGridModel(unittest.TestCase):
    """
    Checks returns of all f and L function for given states.
    Checks that distribution is correctly set
    """

    @classmethod
    def setUpClass(self):
        L_i = lambda x, y: x[0] * y[0]
        L_b = lambda x, y: x[1] * y[1]
        L_e = lambda x, y: x[2] * y[2]
        self.model1 = GridModel(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b, U=U, O=O, V=V, B=B,
                                V_max_change=V_max_change, distribution="binom")
        L_i = lambda x, y: x[0]
        L_b = lambda x, y: x[1]
        L_e = lambda x, y: x[2]
        self.model2 = GridModel(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b, U=U, O=O, V=V, B=B,
                                V_max_change=V_max_change, distribution="uniform")
        self.state = [
            (1, 2, 3),
            (3, 2, 5),
            (0, 0, 1),
            (10, 7, 3),
            (0, 0, 1),
            (3, 5, 0),
            (50, 0, 10),
            (0, 40, 0)
        ]
        self.len = len(self.state)

    def get_x(self, i, j):
        assert i < self.len and j < self.len, \
            "Index must be smaller than list length."
        return self.state[i], self.state[j]

    def test_set_distribution(self):
        self.assertIsInstance(self.model1.distribution_name, str)
        self.assertEqual(self.model1.distribution_name, "binom")
        self.assertIsInstance(self.model2.distribution_name, str)
        self.assertEqual(self.model2.distribution_name, "uniform")
        # test assert-Error
        with self.assertRaises(ValueError):
            self.model1.distribution_name = "nothing"
            self.model1.set_distribution()

    def test_L(self):
        x0, x1 = self.get_x(0, 1)
        self.assertEqual(self.model1.L(x0, x1), 22)
        self.assertEqual(self.model2.L(x0, x1), 6)
        x0, x1 = self.get_x(1, 2)
        self.assertEqual(self.model1.L(x0, x1), 5)
        self.assertEqual(self.model2.L(x0, x1), 10)
        x0, x1 = self.get_x(2, 3)
        self.assertEqual(self.model1.L(x0, x1), 3)
        self.assertEqual(self.model2.L(x0, x1), 1)
        x0, x1 = self.get_x(3, 4)
        self.assertEqual(self.model1.L(x0, x1), 3)
        self.assertEqual(self.model2.L(x0, x1), 20)
        x0, x1 = self.get_x(4, 5)
        self.assertEqual(self.model1.L(x0, x1), 0)
        self.assertEqual(self.model2.L(x0, x1), 1)
        x0, x1 = self.get_x(5, 6)
        self.assertEqual(self.model1.L(x0, x1), 150)
        self.assertEqual(self.model2.L(x0, x1), 8)
        x0, x1 = self.get_x(6, 7)
        self.assertEqual(self.model1.L(x0, x1), 0)
        self.assertEqual(self.model2.L(x0, x1), 60)

    def test_f(self):
        x0, u, v = self.state[0], 1, 1
        self.assertTupleEqual(self.model1.f(x0, u, v), (2, 3, 1))
        x0, u, v = self.state[1], 3, -2
        self.assertTupleEqual(self.model1.f(x0, u, v), (6, 0, 9))
        x0, u, v = self.state[2], 2, 3
        self.assertTupleEqual(self.model1.f(x0, u, v), (2, 3, 0))
        x0, u, v = self.state[3], -8, 1
        self.assertTupleEqual(self.model1.f(x0, u, v), (2, 8, 1))
        x0, u, v = self.state[4], 10, 5
        self.assertTupleEqual(self.model1.f(x0, u, v), (10, 5, 1))
        # test assert-Error
        x0, u, v = self.state[6], -20, 6
        with self.assertRaises(ValueError):
            self.model1.f(x0, u, v)


class TestHelpFunctions(unittest.TestCase):
    """
    Checks returns of all help functions for given states.
    """

    @classmethod
    def setUpClass(self):
        # sets some possible states.
        self.state = [
            (1, 2, 3),
            (3, 2, 5),
            (0, 0, 1),
            (12, 7, 2),
            (6, 3, 0),
            (3, 5, 0),
            (50, 0, 10),
            (0, 40, 0)
        ]
        self.len = len(self.state)
        self.max_charge = B_max_charge

    def get_x(self, i, j):
        assert i < self.len and j < self.len, \
            "Index must be smaller than list length."
        return self.state[i], self.state[j]

    def test_produce_O(self):
        x0, x1 = self.get_x(0, 1)
        self.assertEqual(produce_O(x0, x1), 2)
        x0, x1 = self.get_x(1, 2)
        self.assertEqual(produce_O(x0, x1), 1.5)
        x0, x1 = self.get_x(2, 3)
        self.assertEqual(produce_O(x0, x1), 6)
        x0, x1 = self.get_x(3, 4)
        self.assertEqual(produce_O(x0, x1), 9)
        x0, x1 = self.get_x(4, 5)
        self.assertEqual(produce_O(x0, x1), 4.5)
        x0, x1 = self.get_x(6, 7)
        self.assertEqual(produce_O(x0, x1), 25)
        for i in range(100):
            for j in range(100):
                x0, x1 = (i, 0, 0), (j, 0, 0)
                self.assertGreaterEqual(produce_O(x0, x1), 0)

    def test_overflow_O(self):
        x0, x1 = self.get_x(0, 1)
        self.assertListEqual(overflow_O(x0, x1), [0, .25])
        x0, x1 = self.get_x(1, 2)
        self.assertListEqual(overflow_O(x0, x1), [1.5, 0])
        x0, x1 = self.get_x(2, 3)
        np.testing.assert_array_almost_equal(overflow_O(x0, x1), [0, 25. / 24])
        x0, x1 = self.get_x(3, 4)
        self.assertListEqual(overflow_O(x0, x1), [0, 6])
        x0, x1 = self.get_x(4, 5)
        self.assertListEqual(overflow_O(x0, x1), [1. / 6, 0])
        x0, x1 = self.get_x(6, 7)
        self.assertListEqual(overflow_O(x0, x1), [1., 0])

    def test_deficit_O(self):
        x0, x1 = self.get_x(0, 1)
        self.assertEqual(deficit_O(x0, x1), .25)
        x0, x1 = self.get_x(1, 2)
        self.assertEqual(deficit_O(x0, x1), 0.)
        x0, x1 = self.get_x(2, 3)
        self.assertAlmostEqual(deficit_O(x0, x1), 49. / 24)
        x0, x1 = self.get_x(3, 4)
        self.assertEqual(deficit_O(x0, x1), 0.)
        x0, x1 = self.get_x(4, 5)
        self.assertAlmostEqual(deficit_O(x0, x1), 2. / 3)
        x0, x1 = self.get_x(6, 7)
        self.assertAlmostEqual(deficit_O(x0, x1), 16)

    def test_battery_usage(self):
        x0, x1 = self.get_x(0, 1)
        self.assertEqual(battery_usage(x0, x1, self.max_charge), .25)
        x0, x1 = self.get_x(1, 2)
        self.assertEqual(battery_usage(x0, x1, self.max_charge), 0.)
        x0, x1 = self.get_x(2, 3)
        self.assertEqual(battery_usage(x0, x1, self.max_charge), 1.)
        x0, x1 = self.get_x(3, 4)
        self.assertEqual(battery_usage(x0, x1, self.max_charge), 0.)
        x0, x1 = self.get_x(4, 5)
        self.assertAlmostEqual(battery_usage(x0, x1, self.max_charge), 1. / 6)
        x0, x1 = self.get_x(6, 7)
        self.assertEqual(battery_usage(x0, x1, self.max_charge), 10)


class TestLookUpTable(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # sets some examples
        self.state_space = [np.arange(4, 6), np.arange(5, 10), np.arange(1, 7)]
        self.dim0 = (1, 1)
        self.dim1 = (1, 2)
        self.dim2 = (1, 1, 1)
        self.dim3 = (2, 5, 6)
        self.dim4 = (3, 3, 4, 2)
        self.m0 = look_up_table(self.dim0)
        self.m1 = look_up_table(self.dim1)
        self.m2 = look_up_table(self.dim2)
        self.m3 = look_up_table(self.dim3)
        self.m4 = look_up_table(self.dim4)

    def test_type(self):
        # checks for right types
        # m0
        self.assertIsInstance(self.m0, np.ndarray)
        self.assertEqual(self.m0.dtype, tuple)
        self.assertIsInstance(self.m0[0, 0][0], np.int8)
        # m1
        self.assertIsInstance(self.m1, np.ndarray)
        self.assertEqual(self.m1.dtype, tuple)
        self.assertIsInstance(self.m1[0, 1][1], np.int8)
        # m2
        self.assertIsInstance(self.m2, np.ndarray)
        self.assertEqual(self.m2.dtype, tuple)
        self.assertIsInstance(self.m2[0, 0, 0][1], np.int8)
        # m3
        self.assertIsInstance(self.m3, np.ndarray)
        self.assertEqual(self.m3.dtype, tuple)
        self.assertIsInstance(self.m3[0, 1, 4][2], np.int8)
        # m4
        self.assertIsInstance(self.m4, np.ndarray)
        self.assertEqual(self.m4.dtype, tuple)
        self.assertIsInstance(self.m4[2, 1, 2, 0][2], np.int8)

    def test_dimensions(self):
        # checks for right dimension
        # m0
        self.assertTupleEqual(self.m0.shape, self.dim0)
        # m1
        self.assertTupleEqual(self.m1.shape, self.dim1)
        # m2
        self.assertTupleEqual(self.m2.shape, self.dim2)
        # m3
        self.assertTupleEqual(self.m3.shape, self.dim3)
        # m4
        self.assertTupleEqual(self.m4.shape, self.dim4)

    def test_entrees(self):
        # checks the entrees of the returns
        # m0
        m = np.empty(self.dim0, dtype=tuple)
        m[:] = [[(0, 0)]]
        np.testing.assert_array_equal(self.m0, m)
        # m1
        m = np.empty(self.dim1, dtype=tuple)
        m[:] = [[(0, 0), (0, 1)]]
        np.testing.assert_array_equal(self.m1, m)
        # m2
        m = np.empty(self.dim2, dtype=tuple)
        m[:] = [[[(0, 0, 0)]]]
        np.testing.assert_array_equal(self.m2, m)
        # m3
        for i in range(self.dim3[0]):
            for j in range(self.dim3[1]):
                for k in range(self.dim3[2]):
                    self.assertTupleEqual(self.m3[(i, j, k)], (i, j, k))
        # m4
        for i in range(self.dim4[0]):
            for j in range(self.dim4[1]):
                for k in range(self.dim4[2]):
                    for l in range(self.dim4[3]):
                        self.assertTupleEqual(
                            self.m4[(i, j, k, l)], (i, j, k, l))

    def test_convert_index_to_state(self):
        # checks type and output for some examples
        state = convert_index_to_state((1, 1, 1), self.state_space)
        self.assertIsInstance(state, tuple)
        self.assertTupleEqual(state, (5, 6, 2))
        for x in state:
            self.assertIsInstance(x, (int, np.integer))
        state = convert_index_to_state((0, 4, 3), self.state_space)
        self.assertIsInstance(state, tuple)
        self.assertTupleEqual(state, (4, 9, 4))
        for x in state:
            self.assertIsInstance(x, (int, np.integer))
        state = convert_index_to_state((1, 3, 5), self.state_space)
        self.assertIsInstance(state, tuple)
        self.assertTupleEqual(state, (5, 8, 6))
        for x in state:
            self.assertIsInstance(x, (int, np.integer))

    def test_convert_state_to_index(self):
        # checks type and output for some examples
        index = convert_state_to_index((5, 6, 2), self.state_space)
        self.assertIsInstance(index, tuple)
        self.assertTupleEqual(index, (1, 1, 1))
        for ind in index:
            self.assertIsInstance(ind, (int, np.integer))
        index = convert_state_to_index((4, 9, 4), self.state_space)
        self.assertIsInstance(index, tuple)
        self.assertTupleEqual(index, (0, 4, 3))
        for ind in index:
            self.assertIsInstance(ind, (int, np.integer))
        index = convert_state_to_index((5, 8, 6), self.state_space)
        self.assertIsInstance(index, tuple)
        self.assertTupleEqual(index, (1, 3, 5))
        for ind in index:
            self.assertIsInstance(ind, (int, np.integer))


if __name__ == '__main__':
    unittest.main()
