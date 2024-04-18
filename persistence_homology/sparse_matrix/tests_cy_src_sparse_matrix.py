import copy
import numpy as np
import random
from itertools import product
import unittest

from cy_src_sparse_matrix import SyncRowsColumnsMatrix
from cy_src_sparse_matrix import ShapeError, DesynchronisationError


random.seed(101)


class Tests(unittest.TestCase):

    def create_random_matrix(self, n, m, nnz):
        src = SyncRowsColumnsMatrix.zeros(n, m)
        for k in range(nnz):
            i = random.choice(range(n))
            j = random.choice(range(m))
            src.rows[i].add(j)
            src.columns[j].add(i)

        src.columns_from_rows()
        return src

    def create_random_numpy_array(self, n, m, nnz):
        arr = np.zeros((n, m), dtype='int8')

        for k in range(nnz):
            i = random.choice(range(n))
            j = random.choice(range(m))
            arr[i, j] = 1

        return arr

    def create_random_pair(self, n, m, nnz):
        src = SyncRowsColumnsMatrix.zeros(n, m)
        arr = np.zeros((n, m), dtype='int8')
        for k in range(nnz):
            i = random.choice(range(n))
            j = random.choice(range(m))
            src.rows[i].add(j)
            src.columns[j].add(i)

            arr[i, j] = 1

        return src, arr

    def test_equals(self):
        n = 10
        m = 20
        nnz = 47
        src, arr = self.create_random_pair(n, m, nnz)
        self.assertEqual(src, SyncRowsColumnsMatrix.from_numpy_array(arr))

        self.assertTrue(numpy_arrays_equal(arr, src.to_numpy()))

    def test_copy(self):
        n = 10
        m = 20
        nnz = 47
        src = self.create_random_matrix(n, m, nnz)

        src_copy = src.copy()

        self.assertEqual(src, src_copy)
        self.assertIsNot(src, src_copy)

    def test_validate_synchronisation(self):
        n = 10
        m = 20
        nnz = 47
        src = self.create_random_matrix(n, m, nnz)
        self.assertTrue(src.validate_synchronisation())
        src.number_of_columns += 1

        with self.assertRaises(ShapeError):
            src.validate_synchronisation()

        src = self.create_random_matrix(n, m, nnz)
        src.rows = [set() for _ in range(n)]

        with self.assertRaises(DesynchronisationError):
            src.validate_synchronisation()

        src = self.create_random_matrix(n, m, nnz)
        src.columns = [set() for _ in range(m)]

        with self.assertRaises(DesynchronisationError):
            src.validate_synchronisation()

    def test_eye(self):
        n = 10
        src = SyncRowsColumnsMatrix.eye(10)
        self.assertTrue(src.is_eye())
        np_eye = np.eye(n)
        np_src = src.to_numpy()
        self.assertTrue(numpy_arrays_equal(np_eye, np_src))
        self.assertEqual(src, SyncRowsColumnsMatrix.from_numpy_array(np_eye))

    def test_zero(self):
        n = 10
        m = 20
        nnz = 47
        src = SyncRowsColumnsMatrix.zeros(n, m)
        self.assertTrue(src.is_zero())

        src = self.create_random_matrix(n, m, nnz)
        self.assertFalse(src.is_zero())

    def test_to_numpy(self):
        n = 10
        m = 20
        src = SyncRowsColumnsMatrix.zeros(n, m)
        np_mat = src.to_numpy()
        np_zeros = np.zeros((n, m))
        self.assertTrue(numpy_arrays_equal(np_mat, np_zeros))

    def test_from_numpy(self):
        n = 10
        m = 20

        np_zeros = np.zeros((n, m))

        src = SyncRowsColumnsMatrix.from_numpy_array(np_zeros)
        self.assertTrue(src.is_zero())

    def test_to_then_from_numpy(self):
        n = 10
        m = 20
        nnz = 47

        src = self.create_random_matrix(n, m, nnz)
        to_from_np = SyncRowsColumnsMatrix.from_numpy_array(src.to_numpy())
        self.assertEqual(to_from_np, src)
        self.assertIsNot(to_from_np, src)

    def test_from_then_to_numpy(self):
        n = 10
        m = 20
        nnz = 47
        np_arr = self.create_random_numpy_array(n, m, nnz)
        src_arr = SyncRowsColumnsMatrix.from_numpy_array(np_arr)
        self.assertTrue(numpy_arrays_equal(np_arr, src_arr.to_numpy()))

    def test_rows_from_columns(self):
        n = 10
        m = 20
        nnz = 47

        src1 = self.create_random_matrix(n, m, nnz)

        src2 = SyncRowsColumnsMatrix()
        src2.columns = copy.deepcopy(src1.columns)
        src2.number_of_columns = m

        src2.number_of_rows = n
        src2.rows_from_columns()
        self.assertTrue(src2.validate_synchronisation())
        self.assertEqual(src1, src2)

    def test_columns_from_rows(self):
        n = 10
        m = 20
        nnz = 47

        src1 = self.create_random_matrix(n, m, nnz)

        src2 = SyncRowsColumnsMatrix()
        src2.rows = copy.deepcopy(src1.rows)
        src2.number_of_rows = n

        src2.number_of_columns = m
        src2.columns_from_rows()
        self.assertTrue(src2.validate_synchronisation())
        self.assertEqual(src1, src2)

    def test_shape(self):
        n = 10
        m = 20
        nnz = 47

        src = self.create_random_matrix(n, m, nnz)
        self.assertEqual(src.shape(), (n, m))

    def test_multiplication(self):

        n = 5
        m = 10
        k = 12
        self.assertNotEqual(n, m)

        # check the results are the same as matrix multiplication
        # as implemented with numpy arrays

        np1 = self.create_random_numpy_array(n, m, 23)
        np2 = self.create_random_numpy_array(m, k, 74)
        np_mult = np1@np2 % 2
        np_mult_src = SyncRowsColumnsMatrix.from_numpy_array(np_mult)

        src1 = SyncRowsColumnsMatrix.from_numpy_array(np1)
        src2 = SyncRowsColumnsMatrix.from_numpy_array(np2)

        src_mult = src1@src2

        self.assertEqual(src_mult, np_mult_src)

        # check that multiplying by identity matrix leaves unchanged
        # both left and right multiplication
        self.assertEqual(src1@SyncRowsColumnsMatrix.eye(m), src1)
        self.assertEqual(SyncRowsColumnsMatrix.eye(n)@src1, src1)

        # check that left and right multiplication by zero matrix
        # results in zero metrix with correct shape
        self.assertEqual(
            src1@SyncRowsColumnsMatrix.zeros(m, k),
            SyncRowsColumnsMatrix.zeros(n, k)
        )

        self.assertEqual(
            SyncRowsColumnsMatrix.zeros(n, n)@src1,
            SyncRowsColumnsMatrix.zeros(n, m)
        )

        # check that multiplying matrices with incompatible shapes
        # raises the ShapeError
        with self.assertRaises(ShapeError):
            src1@src1

    def test_smith_normal_form(self):
        n = 100
        m = 20
        nnz = 470

        src = self.create_random_matrix(n, m, nnz)

        arr = src.to_numpy()
        np_smith_normal_form(arr)

        src.smith_normal_form()
        from_np = SyncRowsColumnsMatrix.from_numpy_array(arr)
        self.assertEqual(from_np, src)

        is_snf = True
        for i in range(src.shape()[0]):
            if src.rows[i] == {i}:
                pass
            elif src.rows[i] == set():
                for j in range(i+1, src.shape()[0]):
                    if src.rows[j] != set():
                        is_snf = False
            else:
                print(src.rows[i])
                is_snf = False

        self.assertTrue(is_snf)


# We will test certain algorithms by checking the result coincides
# with the corresponding numpy implementation, hence we define a few
# helper functions


def swap_rows(i0, i1, A):
    A[[i0, i1], :] = A[[i1, i0], :]


def swap_columns(j0, j1, A):
    A[:, [j0, j1]] = A[:, [j1, j0]]


def np_smith_normal_form(A):
    """
    Input A is a numpy array with dtype=int
    and A[i,j] = 0 or 1 for all i,j
    """
    number_of_rows, number_of_columns = A.shape
    n = 0
    while n < min(number_of_columns, number_of_rows):
        for i, j in product(
            range(n, number_of_rows),
            range(n, number_of_columns)
        ):

            if A[i, j] == 1:
                swap_rows(n, i, A)
                swap_columns(n, j, A)
                break

        for i in range(n+1, number_of_rows):
            if A[i, n] == 1:
                # perform bitwise xor
                A[i] = (A[i] + A[n]) % 2

        for j in range(n+1, number_of_columns):
            if A[n, j] == 1:
                A[:, [j]] = (A[:, [j]] + A[:, [n]]) % 2
        n = n+1

    return None


def numpy_arrays_equal(arr1, arr2):
    return (arr1 == arr2).all()


if __name__ == '__main__':
    unittest.main()
