import unittest
from main import *
from grid_optimizer import *
from simulation import *


class TestUser(unittest.TestCase):

    def test_lookup_table(self):
        self.assertIsNotNone(look_up_table([1,2,3]))

if __name__ == '__main__':
    unittest.main()