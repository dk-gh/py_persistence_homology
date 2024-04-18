
import numpy as np
import copy


class DesynchronisationError(Exception):
    pass


class ShapeError(Exception):
    pass


cdef class SyncRowsColumnsMatrix:

    cdef public SyncRowsColumnsMatrix other
    cdef public list rows
    cdef public list columns
    cdef public int number_of_rows
    cdef public int number_of_columns
    cdef public list kernel_elements
    cdef int i
    cdef int j
    cdef int m
    cdef int n

    def __cinit__(self):

        self.rows = []
        self.columns = []
        self.number_of_rows = 0
        self.number_of_columns = 0

    def __init__(self):

        # rows consist of a list of sets of ints
        self.rows = []
        self.number_of_rows = 0

        # columns consists of a list of sets of ints
        self.columns = []
        self.number_of_columns = 0

    def shape(self):
        return self.number_of_rows, self.number_of_columns

    def is_eye(self, validate=False):

        if validate:
            self.validate_synchronisation()

        n, m = self.shape()

        if n != m:
            return False

        for i in range(m):
            if self.rows[i] != {i}:
                return False

        return True

    def is_zero(self, validate=False):

        if validate:
            self.validate_synchronisation()

        for row in self.rows:
            if row:
                return False
        else:
            return True

    def __matmul__(self, other):

        sn, sm = self.shape()
        on, om = other.shape()

        if sm != on:
            raise ShapeError(
                f'cannot multiply matrices of shapes {(sn,sm)} and {(on,om)}'
            )

        n = self.shape()[0]
        m = other.shape()[1]
        mult = SyncRowsColumnsMatrix.zeros(n, m)

        for i, column in enumerate(other.columns):

            for j, row in enumerate(self.rows):
                if len(column.intersection(row)) % 2:
                    mult.columns[i].add(j)
                    mult.rows[j].add(i)

        return mult

    def rows_from_columns(self):

        self.rows = [set() for _ in range(self.number_of_rows)]

        for i, column in enumerate(self.columns):
            for row_index in column:
                self.rows[row_index].add(i)

    def columns_from_rows(self):

        self.columns = [set() for _ in range(self.number_of_columns)]

        for i, row in enumerate(self.rows):
            for column_index in row:
                self.columns[column_index].add(i)


    def __eq__(self, other):

        col_no_eq = self.number_of_columns == other.number_of_columns
        row_no_eq = self.number_of_rows == other.number_of_rows
        col_eq = self.columns == other.columns
        row_eq = self.rows == other.rows
        return (col_no_eq and row_no_eq and col_eq and row_eq)

    def copy(self):
        dc = SyncRowsColumnsMatrix()
        # rows consist of a list of sets of ints
        dc.rows = copy.deepcopy(self.rows)
        dc.number_of_rows = self.number_of_rows
        # columns consists of a list of sets of ints
        dc.columns = copy.deepcopy(self.columns)
        dc.number_of_columns = self.number_of_columns

        return dc

    @classmethod
    def zeros(cls, n, m):
        """
        Creates an n x m matrix with all zero elements

        Parameters
        ----------
        n : int
            number of rows
        m : int
            number of columns

        Returns
        -------
        sm : n x m matrix

        """
        sm = cls()
        sm.number_of_rows = n
        sm.number_of_columns = m
        sm.rows = [set() for _ in range(n)]
        sm.columns = [set() for _ in range(m)]
        return sm

    @classmethod
    def eye(cls, n):
        """
        Creates an n x n identity matrix

        Parameters
        ----------
        n : number of rows and columns

        Returns
        -------
        eye : identity matrix

        """
        eye = cls()
        eye.number_of_rows = n
        eye.number_of_columns = n
        eye.rows = [{i} for i in range(n)]
        eye.columns = [{i} for i in range(n)]
        return eye

    def swap_rows(self, n, m):

        columns_to_update = self.rows[m] ^ self.rows[n]

        for column in columns_to_update:
            if n in self.columns[column]:
                self.columns[column].discard(n)
            else:
                self.columns[column].add(n)

            if m in self.columns[column]:
                self.columns[column].discard(m)
            else:
                self.columns[column].add(m)

        self.rows[n], self.rows[m] = self.rows[m], self.rows[n]

    def swap_columns(self, n, m):

        rows_to_update = self.columns[m] ^ self.columns[n]

        for row in rows_to_update:
            if n in self.rows[row]:
                self.rows[row].discard(n)
            else:
                self.rows[row].add(n)

            if m in self.rows[row]:
                self.rows[row].discard(m)
            else:
                self.rows[row].add(m)

        self.columns[n], self.columns[m] = self.columns[m], self.columns[n]

    cdef c_add_rows_desync(self, int n, int m):
        # cdef int column
        cdef set row_m
        cdef set row_n
        self.rows[n].symmetric_difference_update(self.rows[m])
        #row_m = self.rows[m]
        #row_n = self.rows[n]
        #self.rows[n] = row_m ^ row_n

    cpdef add_rows_desync(self, int n, int m):
        self.c_add_rows_desync(n, m)


    cdef c_add_rows(self, int n, int m):
        cdef int column
        cdef set row_m
        cdef set row_n
        cdef set temp_col
        row_m = self.rows[m]
        row_n = self.rows[n]
        for column in row_m:
            temp_col = self.columns[column]
            if n in temp_col:
                if m in temp_col:
                    temp_col.discard(n)
                else:
                    temp_col.add(n)
            else:
                temp_col.add(n)
            self.columns[column] = temp_col

        for column in row_n - row_m:
            temp_col = self.columns[column]
            if n in temp_col:
                if m in temp_col:
                    temp_col.discard(n)
                else:
                    temp_col.add(n)
            else:
                temp_col.add(n)
            self.columns[column] = temp_col

        self.rows[n].symmetric_difference_update(row_m)
       # self.rows[n] = row_m ^ row_n

    cpdef add_rows(self, int n, int m):
        self.c_add_rows(n, m)

    cdef c_add_columns(self, int n, int m):
        cdef int row
        cdef set column_m
        cdef set column_n
        cdef set temp_row
        
        column_m = self.columns[m]
        column_n = self.columns[n]
        
        for row in column_m:
            temp_row = self.rows[row]
            if n in temp_row:
                if m in temp_row:
                    temp_row.discard(n)
                else:
                    temp_row.add(n)
            else:
                temp_row.add(n)
            self.rows[row] = temp_row

        for row in column_n - column_m:
            temp_row = self.rows[row]
            if n in temp_row:
                if m in temp_row:
                    temp_row.discard(n)
                else:
                    temp_row.add(n)
            else:
                temp_row.add(n)
            self.rows[row] = temp_row

        self.columns[n] = column_m ^ column_n
    
    cpdef add_columns(self, int n, int m):
        self.c_add_columns(n, m)

    def smith_normal_form(self):
        
        n = 0

        while n < min(self.shape()):

            for i, row in enumerate(self.rows):
                if i < n:
                    pass
                elif row:
                    for j in row:
                        if j > n:
                            break

                    self.swap_rows(n, i)
                    self.swap_columns(n, j)
                    break

            for i in self.columns[n]-{n}:
                self.add_rows(i, n)

            for j in self.rows[n]-{n}:
                self.add_columns(j, n)
            n = n+1

    def nnz(self, validate=False):
        '''

        Returns
        -------
        int
        number of non-zero entries in matrix

        '''
        nnz = sum([len(x) for x in self.rows])

        if validate:
            col_nnz = sum([len(x) for x in self.columns])
            if nnz != col_nnz:
                raise DesynchronisationError()

        return nnz

    def density(self):
        nnz = self.nnz()
        n, m = self.shape()

        if n*m > 0:
            n, m = self.shape()
            return nnz/(n*m)

        else:
            return 0

    cdef c_simultaneous_reduce(self, SyncRowsColumnsMatrix other):

        cdef int m
        cdef int i
        cdef int min_index
        cdef int k
        cdef set i_cols
        cdef set cols_with_common_row
        shape = self.shape()
        m = shape[1]
        if other.is_zero():

            for i in range(m):
                if self.columns[i]:
                    i_cols = self.columns[i]

                    min_index = min(i_cols)

                    cols_with_common_row = self.rows[min_index] - {i}

                    for k in cols_with_common_row:
                        self.add_columns(k, i)

        else:
            for i in range(m):
                if self.columns[i]:
                    i_cols = self.columns[i]
                    min_index = min(i_cols)
                    cols_with_common_row = self.rows[min_index] - {i}

                    for k in cols_with_common_row:
                        self.add_columns(k, i)
                        # other.add_rows(i, k)
                        # use the desync version of row
                        # addition -- this does not make corresponding
                        # column updates, and only does fast
                        # symmetric difference operations on the rows.
                        # Below we then bring columns into sync
                        # with other.columns_from_rows()
                        # this gives a roughyl x10 speedup on
                        # on the simultaneous_reduce method
                        other.add_rows_desync(i, k)
            
            other.columns_from_rows()


    def simultaneous_reduce(self, SyncRowsColumnsMatrix other):
        self.c_simultaneous_reduce(other)


    cdef c_row_reduce(self, kernel_elements):
        cdef int i
        cdef int min_col
        cdef int row_index
        if not self.is_zero():
            for i in kernel_elements:
                if not self.rows[i]:
                    pass
                else:
                    min_col = min(self.rows[i])
                    for row_index in self.columns[min_col] - set([i]):
                        self.c_add_rows(row_index, i)
                        # self.number_of_d1_reducing_actions += 1

    cpdef row_reduce(self, kernel_elements):
        self.c_row_reduce(kernel_elements)

    def loc(self, int i, int j):
        """
        lookup a specific value in the matrix

        Parameters
        ----------
        i : int
            i is a row index
        j : int
            j is a column index

        Returns
        -------
        int
            the value of the matrix at [i,j]
        """

        if j in self.rows[i]:
            return 1
        else:
            return 0

    # @property
    def trace(self):
        """
        returns the trace of the matrix

        Returns
        -------
        tr : int
            the trace of the matrix

        """

        tr = 0
        for i in range(min(self.shape())):
            tr += self.loc(i, i)
        return tr

    cpdef validate_synchronisation(self):
        """
        This method checks that both row and column-based
        representations are synchronised, i.e. both represent
        the same underlying matrix.

        This method is generally used in testing during refactoring.

        Raises
        ------
        ShapeError
            DESCRIPTION.
        DesynchronisationError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.shape()[0] != len(self.rows):
            raise ShapeError()

        if self.shape()[1] != len(self.columns):
            raise ShapeError()

        for row_index, row in enumerate(self.rows):
            for column_index in row:
                if row_index not in self.columns[column_index]:
                    raise DesynchronisationError()

        for column_index, column in enumerate(self.columns):
            for row_index in column:
                if column_index not in self.rows[row_index]:
                    raise DesynchronisationError()

        return True

    @classmethod
    def from_numpy_array(cls, array, validate=False):
        """
        Creates a sparse matrix from a numpy array.

        Parameters
        ----------
        array : numpy array
            numpy array with int entries 0,1
        validate : TYPE, optional
            If validate=True then the validate_synchronisation
            method is called. The default is False.

        Returns
        -------
        None.

        """

        scr = SyncRowsColumnsMatrix()
        scr.number_of_rows = array.shape[0]
        scr.number_of_columns = array.shape[1]

        for array_row in array:
            row_set = set()
            for i, value in enumerate(array_row):
                if value == 1:
                    row_set.add(i)
            scr.rows.append(row_set)

        for array_column in array.T:
            column_set = set()
            for i, value in enumerate(array_column):
                if value == 1:
                    column_set.add(i)
            scr.columns.append(column_set)

        if validate:
            scr.validate_synchronisation()

        return scr

    def to_numpy(self, validate=False):
        """
        Outputs the sparse matrix as a numpy array. Uses the row data
        to create the matrix entries.

        Parameters
        ----------
        validate : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        DesynchronisationError
            If validate=True then:
                1) the validate_synchronisation method is called
                2) a second numpy array is created using the column
                   data as well and checks that both numpy arrays
                   are equal.

        Returns
        -------
        array : numpy array
            The matrix is returned as a dense matrix in the form
            of a numpy array.

        """

        array = np.zeros(self.shape(), dtype=int)

        for i, row in enumerate(self.rows):
            for value in row:
                array[i, value] = 1

        if validate:
            self.validate_synchronisation()

            validation_array = np.zeros(self.shape(), dtype=int).T
            for i, column in enumerate(self.columns):
                for value in column:
                    validation_array[i, value] = 1
            validation_array = validation_array.T

            if not (array == validation_array).all():
                raise DesynchronisationError()

        return array

