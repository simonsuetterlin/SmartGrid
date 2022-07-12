import unittest
from main import *
from grid_optimizer import *
from simulation import *
from help_functions import *

class TestSimulation(unittest.TestCase):
    pass


class TestOptimizer(unittest.TestCase):
    @classmethod
    def _setUpClass(self):
        self.model = Model(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b, U=U, O=O, V=V, B=B, distribution="binom")
        self.optimizer = GridOptimizer(model=self.model)
        self.optimizer.calculate_cost_to_go_matrix_sequence(depth = 5)
    

class TestModel(unittest.TestCase):
    pass

# Diese Tests sind bissle obsolet!
# würde ich rauslassen, wollte das die nur einmal im Github drinnen sind.
"""
class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # sets some possible states.
        global P_i
        global P_e
        global P_b
        global B_max_charge
        self.P_i = P_i
        self.P_b = P_b
        self.P_e = P_e
        self.B_max_charge = 10
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
    
    def get_x(self, i, j):
        assert i < self.len and j < self.len,\
        "Index must be smaller than list length."
        return self.state[i], self.state[j]
    
    def test_L_i(self):
        
        x0, x1 = self.get_x(0, 1)
        self.assertEqual(L_i(x0, x1), 2 * self.P_i)
        x0, x1 = self.get_x(1, 2)
        self.assertEqual(L_i(x0, x1), 1.5 * self.P_i)
        x0, x1 = self.get_x(2, 3)
        self.assertEqual(L_i(x0, x1), 6 * self.P_i)
        x0, x1 = self.get_x(3, 4)
        self.assertEqual(L_i(x0, x1), 9 * self.P_i)
        x0, x1 = self.get_x(4, 5)
        self.assertEqual(L_i(x0, x1), 4.5 * self.P_i)
        x0, x1 = self.get_x(6, 7)
        self.assertEqual(L_i(x0, x1), 25 * self.P_i)
        
    def test_L_b(self):
        x0, x1 = self.get_x(0, 1)
        self.assertEqual(L_b(x0, x1), .25 * self.P_b)
        x0, x1 = self.get_x(1, 2)
        self.assertEqual(L_b(x0, x1), 0.)
        x0, x1 = self.get_x(2, 3)
        self.assertEqual(L_b(x0, x1), 1. * self.P_b)
        x0, x1 = self.get_x(3, 4)
        self.assertEqual(L_b(x0, x1), 0.)
        x0, x1 = self.get_x(4, 5)
        self.assertAlmostEqual(L_b(x0, x1), 1./6 * self.P_b)
        x0, x1 = self.get_x(6, 7)
        self.assertEqual(L_b(x0, x1), 10 * self.P_b)
        
    def test_L_e(self):
        x0, x1 = self.get_x(0, 1)
        self.assertEqual(L_e(x0, x1), 0.)
        x0, x1 = self.get_x(1, 2)
        self.assertEqual(L_e(x0, x1), 0.)
        x0, x1 = self.get_x(2, 3)
        self.assertAlmostEqual(L_e(x0, x1), 25./24 * self.P_e)
        x0, x1 = self.get_x(3, 4)
        self.assertEqual(L_e(x0, x1), 0.)
        x0, x1 = self.get_x(4, 5)
        self.assertAlmostEqual(L_e(x0, x1), .5 * self.P_e)
        x0, x1 = self.get_x(6, 7)
        self.assertEqual(L_e(x0, x1), 6. * self.P_e)
"""

class TestHelpFunctions(unittest.TestCase):
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
        self.max_charge = 10
    
    def get_x(self, i, j):
        assert i < self.len and j < self.len,\
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
                x0, x1 = (i,0,0), (j,0,0)
                self.assertGreaterEqual(produce_O(x0,x1), 0)
                
    def test_overflow_O(self):
        x0, x1 = self.get_x(0, 1)
        self.assertListEqual(overflow_O(x0, x1), [0, .25])
        x0, x1 = self.get_x(1, 2)
        self.assertListEqual(overflow_O(x0, x1), [1.5, 0])
        x0, x1 = self.get_x(2, 3)
        np.testing.assert_array_almost_equal(overflow_O(x0, x1), [0, 25./24])
        x0, x1 = self.get_x(3, 4)
        self.assertListEqual(overflow_O(x0, x1), [0, 6])
        x0, x1 = self.get_x(4, 5)
        self.assertListEqual(overflow_O(x0, x1), [1./6, 0])
        x0, x1 = self.get_x(6, 7)
        self.assertListEqual(overflow_O(x0, x1), [1., 0])
        
    def test_deficit_O(self):
        x0, x1 = self.get_x(0, 1)
        self.assertEqual(deficit_O(x0, x1), .25)
        x0, x1 = self.get_x(1, 2)
        self.assertEqual(deficit_O(x0, x1), 0.)
        x0, x1 = self.get_x(2, 3)
        self.assertAlmostEqual(deficit_O(x0, x1), 49./24)
        x0, x1 = self.get_x(3, 4)
        self.assertEqual(deficit_O(x0, x1), 0.)
        x0, x1 = self.get_x(4, 5)
        self.assertAlmostEqual(deficit_O(x0, x1), 2./3)
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
        self.assertAlmostEqual(battery_usage(x0, x1, self.max_charge), 1./6)
        x0, x1 = self.get_x(6, 7)
        self.assertEqual(battery_usage(x0, x1, self.max_charge), 10)


class TestLookUpTable(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # sets some examples
        self.dim0 = (1,1)
        self.dim1 = (1,2)
        self.dim2 = (1,1,1)
        self.dim3 = (2, 5, 6)
        self.dim4 = (3, 3, 4, 2)
        self.m0 = look_up_table(self.dim0)
        self.m1 = look_up_table(self.dim1)
        self.m2 = look_up_table(self.dim2)
        self.m3 = look_up_table(self.dim3)
        self.m4 = look_up_table(self.dim4)
        
    @classmethod
    def tearDownClass(self):
        pass
    
    def test_type(self):
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
        # m0
        m = np.empty(self.dim0, dtype=tuple)
        m[:] = [[(0,0)]]
        np.testing.assert_array_equal(self.m0, m)
        # m1
        m = np.empty(self.dim1, dtype=tuple)
        m[:] = [[(0,0), (0,1)]]
        np.testing.assert_array_equal(self.m1, m)
        # m2
        m = np.empty(self.dim2, dtype=tuple)
        m[:] = [[[(0,0,0)]]]
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



if __name__ == '__main__':
    unittest.main()