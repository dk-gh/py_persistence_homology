import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from collections import deque

from .sparse_matrix.cy_src_sparse_matrix import SyncRowsColumnsMatrix
#from .sparse_matrix.src_sparse_matrix import SyncRowsColumnsMatrix


class VietorisRipsComplex():

    def __init__(self, distance_matrix, epsilon_range):

        self.epsilon_range = epsilon_range
        self.distance_matrix = distance_matrix

        self.number_of_vertices = len(self.distance_matrix)

        # These persist over the range of epsilon values
        self.indexed_n_cells = {}
        self.n_cells = {}
        self.index_zero_cells()

        self.stats_record = []
        self.barcode_data = {}

        self.d0 = SyncRowsColumnsMatrix()
        self.d1 = SyncRowsColumnsMatrix()

        # These are reset and recalculated for each epsilon
        self.adjacency_sets = []
        self.cached_partial_n_cells = [None]

        # These are reset for each homology dimension calculated
        self.births_register = {}
        self.deaths_register = {}
        self.kernel_elements = []

    def index_zero_cells(self):
        for v in range(self.number_of_vertices):
            self.add_indexed_n_cells(0, (v,))

    def get_the_living(self):
        return self.births_register.keys() - self.deaths_register.keys()

    def draw_barcode(self):
        persistence_range = []

        for i, rep in enumerate(self.births_register):
            birth_value = self.births_register[rep]
            if self.deaths_register.get(rep):
                death_value = self.deaths_register[rep]
            else:
                death_value = self.epsilon_range[-1]
            persistence_range.append([(birth_value, i+1), (death_value, i+1)])

        for i in persistence_range:
            df = pd.DataFrame(i)
            plt.plot(df[0], df[1])

    def initialise_complex(self):

        self.cached_partial_n_cells = [None]
        self.adjacency_sets = []
        return

    def adjacency_matrix_at_epsilon(self, epsilon):
        self.initialise_complex()

        adj = (self.distance_matrix <= epsilon) & (self.distance_matrix > 0)

        for vertex in range(self.number_of_vertices):
            adjacency_set = set()
            for index in range(self.number_of_vertices):
                if adj[vertex, index]:
                    adjacency_set.add(index)
            self.adjacency_sets.append(adjacency_set)

    def add_indexed_n_cells(self, n, cell):
        if self.indexed_n_cells.get(n) is not None:
            if cell in self.indexed_n_cells[n]:
                pass
            else:
                new_index = len(self.indexed_n_cells[n])
                self.indexed_n_cells[n][cell] = new_index
                self.n_cells[n].append(cell)

        else:
            self.indexed_n_cells[n] = {cell: 0}
            self.n_cells[n] = [cell]

    def compute_one_cells(self):
        one_cells = []
        for v in range(self.number_of_vertices):
            one_cells_containing_v = [(w,) for w in self.adjacency_sets[v]]
            one_cells.append(one_cells_containing_v)
        self.cached_partial_n_cells.append(one_cells)

    def compute_two_cells(self):
        two_cells = []

        one_cells = self.cached_partial_n_cells[1]
        for v in range(self.number_of_vertices):
            one_cells_containing_v = one_cells[v]
            two_cells_contaning_v = []
            k = len(one_cells_containing_v)
            for i in range(k-1):
                for j in range(i+1, k):
                    wi = one_cells_containing_v[i][0]
                    wj = one_cells_containing_v[j][0]
                    if wi in self.adjacency_sets[wj]:
                        two_cells_contaning_v.append((wi, wj))
            two_cells.append(two_cells_contaning_v)
        self.cached_partial_n_cells.append(two_cells)

    def compute_three_cells(self):
        three_cells = []

        two_cells = self.cached_partial_n_cells[2]

        for v in range(self.number_of_vertices):
            two_cells_contaning_v = two_cells[v]
            three_cells_containing_v = set()

            k = len(two_cells_contaning_v)

            for i in range(k-1):
                for j in range(i+1, k):
                    ci = two_cells_contaning_v[i]
                    cj = two_cells_contaning_v[j]
                    if ci[0] == cj[0]:
                        if ci[1] in self.adjacency_sets[cj[1]]:
                            if ci[1] < cj[1]:
                                three_cell = (ci[0], ci[1], cj[1])
                                three_cells_containing_v.add(three_cell)

                            else:
                                three_cell = (ci[0], cj[1], ci[1])
                                three_cells_containing_v.add(three_cell)

                    elif ci[1] == cj[1]:

                        if ci[0] in self.adjacency_sets[cj[0]]:

                            if ci[0] < cj[0]:
                                three_cell = (ci[0], cj[0], ci[1])
                                three_cells_containing_v.add(three_cell)
                            else:
                                three_cell = (cj[0], ci[0], ci[1])
                                three_cells_containing_v.add(three_cell)

                    elif ci[0] == cj[1]:

                        if ci[1] in self.adjacency_sets[cj[0]]:
                            three_cell = (cj[0], cj[1], ci[1])
                            three_cells_containing_v.add(three_cell)

                    elif ci[1] == cj[0]:

                        if ci[0] in self.adjacency_sets[cj[1]]:

                            three_cell = (ci[0], cj[0], cj[1])
                            three_cells_containing_v.add(three_cell)

            three_cells.append(list(three_cells_containing_v))
        self.cached_partial_n_cells.append(three_cells)

    def compute_four_cells(self):
        four_cells = []

        three_cells = self.cached_partial_n_cells[3]

        for v in range(self.number_of_vertices):
            three_cells_contaning_v = three_cells[v]
            four_cells_containing_v = set()

            k = len(three_cells_contaning_v)

            for i in range(k-1):
                for j in range(i+1, k):
                    ci = three_cells_contaning_v[i]
                    cj = three_cells_contaning_v[j]
                    if ci[0] == cj[0]:
                        if ci[1] == cj[1]:
                            if ci[2] in self.adjacency_sets[cj[2]]:
                                if ci[2] < cj[2]:
                                    four_cell = (ci[0], ci[1], ci[2], cj[2])
                                    four_cells_containing_v.add(four_cell)
                                else:
                                    four_cell = (ci[0], ci[1], cj[2], ci[2])
                                    four_cells_containing_v.add(four_cell)
                        elif ci[2] == cj[2]:
                            if ci[1] in self.adjacency_sets[cj[1]]:
                                if ci[1] < cj[1]:
                                    four_cell = (ci[0], ci[1], cj[1], cj[2])
                                    four_cells_containing_v.add(four_cell)
                                else:
                                    four_cell = (ci[0], cj[1], ci[1], ci[2])
                                    four_cells_containing_v.add(four_cell)

                    elif ci[1] == cj[1]:
                        if ci[2] == cj[2]:
                            if ci[0] in self.adjacency_sets[cj[0]]:
                                if ci[0] < cj[0]:
                                    four_cell = (ci[0], cj[0], ci[1], ci[2])
                                    four_cells_containing_v.add(four_cell)
                                else:
                                    four_cell = (cj[0], ci[0], ci[1], ci[2])
                                    four_cells_containing_v.add(four_cell)

            four_cells.append(list(four_cells_containing_v))
        self.cached_partial_n_cells.append(four_cells)

    def compute_n_cells(self, n):
        n_cells = []
        m = n-1
        m_cells = self.cached_partial_n_cells[m]

        for v in range(self.number_of_vertices):
            m_cells_contaning_v = m_cells[v]
            n_cells_containing_v = set()

            k = len(m_cells_contaning_v)

            for i in range(k-1):
                for j in range(i+1, k):
                    ci = m_cells_contaning_v[i]
                    cj = m_cells_contaning_v[j]

                    fi = set(ci)
                    fj = set(cj)
                    fe = fi ^ fj
                    if len(fe) == 2:
                        e = tuple(fe)
                        if e[0] in self.adjacency_sets[e[1]]:
                            new_cell = fi.union(fj)
                            new_cell = tuple(sorted(new_cell))
                            n_cells_containing_v.add(new_cell)

            n_cells.append(list(n_cells_containing_v))
        self.cached_partial_n_cells.append(n_cells)

    def compute_up_to_n_cells(self, n):

        if n == 0:
            return

        for i in range(1, n+1):
            if i == 1:
                self.compute_one_cells()
            elif i == 2:
                self.compute_two_cells()
            elif i == 3:
                self.compute_three_cells()
            elif i == 4:
                self.compute_four_cells()
            else:
                self.compute_n_cells(i)

        return None

    def compute_indexed_n_cells_from_cache(self, n):

        partial_n_cells = self.cached_partial_n_cells[n]

        for vertex, cells in enumerate(partial_n_cells):
            for cell in cells:
                cell = tuple(sorted(list(cell)+[vertex]))
                self.add_indexed_n_cells(n, cell)

        return None

    def compute_boundary_map_d0(self, k):

        if k == 0:
            N = self.number_of_vertices
            self.d0 = SyncRowsColumnsMatrix.zeros(1, N)
            return None

        elif not self.indexed_n_cells.get(k):
            return None

        else:

            k_cells = self.n_cells[k]
            k_minus_one_cells = self.indexed_n_cells[k-1]

            n = len(k_minus_one_cells)
            m = len(k_cells)

            self.d0 = SyncRowsColumnsMatrix.zeros(n, m)

            for i, cell in enumerate(k_cells):
                column_number = i

                for j, v in enumerate(cell):
                    c = list(cell)
                    del c[j]
                    row_number = k_minus_one_cells[tuple(c)]
                    self.d0.columns[column_number].add(row_number)
                    self.d0.rows[row_number].add(column_number)

        return None

    def compute_boundary_map_d1(self, k):

        if not self.d1:
            return None

        if not self.indexed_n_cells.get(k+1):
            return None

        else:
            k_cells = self.indexed_n_cells.get(k)
            k_plus_one_cells = self.n_cells[k+1]

            n = len(k_cells)
            m = len(k_plus_one_cells)

            self.d1 = SyncRowsColumnsMatrix.zeros(n, m)

            for i, cell in enumerate(k_plus_one_cells):
                column_index = i
                for j, v in enumerate(cell):
                    c = list(cell)
                    del c[j]
                    row_number = k_cells[tuple(c)]
                    self.d1.rows[row_number].add(column_index)
                    self.d1.columns[column_index].add(row_number)

        return None

    def update_birth_death_registers(self, homology_generators, epsilon):
        the_living = self.get_the_living()
        births = []
        deaths = []

        for hom_gen in the_living:
            if hom_gen not in homology_generators:
                self.deaths_register[hom_gen] = epsilon
                deaths.append(hom_gen)

        for hom_gen in homology_generators:
            if hom_gen not in self.births_register:
                self.births_register[hom_gen] = epsilon
                births.append(hom_gen)

        return births, deaths

    def d0_column_reduce(self):

        m = self.d0.shape()[1]

        for i in range(m):
            if i_cols := self.d0.columns[i]:
                min_index = min(i_cols)

                cols_with_common_row = self.d0.rows[min_index] - {i}

                for k in cols_with_common_row:
                    self.d0.add_columns(k, i)

    def d1_row_reduce(self, kernel_elements):

        self.number_of_d1_reducing_actions = 0
        if not self.d1.is_zero():

            self.d1.row_reduce(kernel_elements)

            homology_generators = []

            for i in kernel_elements:
                if not self.d1.rows[i]:
                    homology_generators.append(i)
        else:
            homology_generators = kernel_elements

        return homology_generators

    def compute_persistence_homology(self, k, validate=False):
        self.births_register = {}
        self.deaths_register = {}
        self.d0 = SyncRowsColumnsMatrix()
        self.d1 = SyncRowsColumnsMatrix()
        self.kernel_elements = []

        for epsilon in self.epsilon_range:

            print(f'> computing at scale epsilon={epsilon}')
            scale_time_start = time.time()

            generation_stats = {'epsilon': epsilon}
            generation_stats['homology_group_dim'] = k

            self.adjacency_matrix_at_epsilon(epsilon)

            s = time.time()
            self.compute_up_to_n_cells(k+1)
            e = time.time()
            generation_stats['time_compute_cells'] = e-s

            s = time.time()
            if k > 1:
                self.compute_indexed_n_cells_from_cache(k-1)
                self.compute_indexed_n_cells_from_cache(k)
                self.compute_indexed_n_cells_from_cache(k+1)
            elif k == 1:
                self.compute_indexed_n_cells_from_cache(k)
                self.compute_indexed_n_cells_from_cache(k+1)
            elif k == 0:
                self.compute_indexed_n_cells_from_cache(k+1)
            e = time.time()
            generation_stats['time_index_cells'] = e-s

            s = time.time()
            self.compute_boundary_map_d1(k)
            e = time.time()
            generation_stats['time_compute_d1'] = e-s

            s = time.time()
            self.compute_boundary_map_d0(k)
            e = time.time()
            generation_stats['time_compute_d0'] = e-s

            if validate:
                self.validate_boundary_maps()

            generation_stats['d0_shape'] = self.d0.shape()
            generation_stats['d1_shape'] = self.d1.shape()
            generation_stats['d0_nnz'] = self.d0.nnz()
            generation_stats['d1_nnz'] = self.d1.nnz()
            generation_stats['d0_density'] = self.d0.density()
            generation_stats['d1_density'] = self.d1.density()

            s = time.time()
            self.d0.simultaneous_reduce(self.d1)
            e = time.time()
            generation_stats['time_simultaneous_reduce'] = e-s

            if validate:
                self.validate_boundary_maps()

            generation_stats['d0_reduced_shape'] = self.d0.shape()
            generation_stats['d1_reduced_shape'] = self.d1.shape()
            generation_stats['d0_reduced_nnz'] = self.d0.nnz()
            generation_stats['d1_reduced_nnz'] = self.d1.nnz()
            generation_stats['d0_reduced_density'] = self.d0.density()
            generation_stats['d1_reduced_density'] = self.d1.density()

            kernel_elements = deque()
            for i, col in enumerate(self.d0.columns):
                if not col:
                    kernel_elements.appendleft(i)

            generation_stats['number_of_kernel_elements'] = len(
                kernel_elements
            )

            self.kernel_elements.append(kernel_elements)

            s = time.time()
            if self.d1.is_zero():
                homology_generators = set(kernel_elements)
            else:
                homology_generators = self.d1_row_reduce(kernel_elements)
            e = time.time()
            generation_stats['time_row_reduce_d1'] = e-s

            births, deaths = self.update_birth_death_registers(
                homology_generators,
                epsilon
            )

            if validate:
                self.validate_betti_number(homology_generators)

            betti_no = len(self.births_register) - len(self.deaths_register)
            generation_stats['betti_number'] = betti_no

            scale_time_end = time.time()
            print(f'  current betti number b_{k} = {betti_no}')
            print(f'  took {scale_time_end - scale_time_start:.2f} seconds')
            generation_stats['total_time'] = scale_time_end - scale_time_start

            self.stats_record.append(generation_stats)


        # TODO write the births and deaths register to an output file
        # at each scale so that partial results can be retrieved even
        # if the main computation needs to be killed because of time
        # or memory issues
        self.barcode_data[f'{k}'] = {
            'births': self.births_register,
            'deaths': self.deaths_register
        }

        if validate:
            self.validate_kernels()

    def validate_betti_number(self, homology_generators):
        assert len(homology_generators) == len(self.get_the_living())

    def validate_kernels(self):
        """
        The columns of the reduced d0 matrix that are zero correspond
        with elements of the kernel.

        By construction the indices of these kernel element columns
        are always contained in subsequent reduced d0 as epsilon
        increases.

        That is, for any epsilon_n < epsilon_m the columns of the reduced
        d0 matrix that are zero have the following relationship
        Kn.issubset(Km)

        This method checks the subset relationship holds across all
        generations.

        Returns
        -------
        None.

        """

        print('validating kernels...')

        for i, element in enumerate(self.kernel_elements[:-1]):
            assert set(element).issubset(set(self.kernel_elements[i+1]))

    def validate_boundary_maps(self):

        print('validating boundary maps...')
        self.d0.validate_synchronisation()
        self.d1.validate_synchronisation()

        if not self.d1.is_zero():
            assert (self.d0@self.d1).is_zero()

    def validate_indexed_n_cells(self):

        assert self.indexed_n_cells.keys() == self.n_cells.keys()

        for n in self.indexed_n_cells.keys():
            assert len(self.indexed_n_cells[n]) == len(self.n_cells[n])

            for i, cell in enumerate(self.indexed_n_cells[n]):
                assert self.indexed_n_cells[n][cell] == i


if __name__ == '__main__':
    pass
