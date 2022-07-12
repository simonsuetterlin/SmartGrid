import unittest
from main import *
from grid_optimizer import *
from simulation import *
from help_functions import *


class TestMain(unittest.TestCase):
    pass


class TestSimulation(unittest.TestCase):
    pass


class TestOptimizer(unittest.TestCase):
    
    @classmethod
    def _setUpClass(self):
        self.model = Model(L_list=[L_i, L_e, L_b], P_i=P_i, P_e=P_e, P_b=P_b, U=U, O=O, V=V, B=B, distribution="binom")
        self.optimizer = GridOptimizer(model=self.model)
        self.optimizer.calculate_cost_to_go_matrix_sequence(depth = 5)
    
class TestHelpFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # sets some possible states.
        self.state = [(1,2,3), (3,2,5), (0, 0, 0)]
        self.max_charge = 10
    
    def test_produce_O(self):
        for i in range(len(self.state) - 1): 
            
            self.assertGreaterEqual(produce_O(self.))
            self.assertEqual(produce_O, )
    
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
                        self.assertTupleEqual(self.m4[(i, j, k, l)], (i, j, k, l))



if __name__ == '__main__':
    unittest.main()