# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the TERMS of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.


# Last Cleanup: 2026-10-10
import numpy as np
from numba import njit, types, i8, c16
from scipy.sparse import csr_matrix

############################################################
#                      NUMBA COMPILED                     #
############################################################

@njit(types.Tuple((i8[:], i8[:], c16[:], c16[:]))(i8[:,:], i8[:,:], i8[:,:], i8[:,:], i8), cache=True, nogil=True)
def _fill_periodic_matrix(tris: np.ndarray, edges: np.ndarray, tri_to_field: np.ndarray, edge_to_field: np.ndarray, Nfield: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates sparse matrix row, column ids and ones plus the matrix diagonal.

    Args:
        tris (: np.ndarray): The triangle ids
        edges (: np.ndarray): The edge ids
        tri_to_field (: np.ndarray): The triangle to field index mapping
        edge_to_field (: np.ndarray): The edge to field index mapping
        Nfield (int): The number of field points

    Returns:
        np.ndarray: The row ids
        np.ndarray: The column ids
        np.ndarray: The data
        np.ndarray: The diagonal array
    """

    # NUMBERS
    NT = tris.shape[1]
    NE = edges.shape[1]
    nedof = edge_to_field.shape[0]
    face_start = 3 * nedof
    nfdof = tri_to_field.shape[0] - face_start
    N = NT * nfdof + NE * nedof
    
    DIAGONAL = np.ones((Nfield,), dtype=np.complex128)
    ROWS = np.zeros((N,), dtype=np.int64)
    COLS = np.zeros((N,), dtype=np.int64)
    TERMS = np.ones((N,), dtype=np.complex128)
    
    i = 0
    for it in range(NT):
        t1 = tris[0,it]
        t2 = tris[1,it]
        # Nedelec2 stores three edges per edge basis, followed by ALL face
        # bases. The old rows 3/7 selected an edge and omitted the first face
        # basis in the mixed space, and accessed outside first-order arrays.
        for dof in range(nfdof):
            source = tri_to_field[face_start + dof, t1]
            target = tri_to_field[face_start + dof, t2]
            DIAGONAL[target] = 0.
            ROWS[i] = target
            COLS[i] = source
            i += 1
    for ie in range(NE):
        e1 = edges[0,ie]
        e2 = edges[1,ie]
        for dof in range(nedof):
            source = edge_to_field[dof, e1]
            target = edge_to_field[dof, e2]
            DIAGONAL[target] = 0.
            ROWS[i] = target
            COLS[i] = source
            i += 1
    ROWS = ROWS[:i]
    COLS = COLS[:i]
    TERMS = TERMS[:i]
    return ROWS, COLS, TERMS, DIAGONAL


############################################################
#                     PYTHON INTERFACE                    #
############################################################

def gen_periodic_matrix(tris: np.ndarray, 
                        edges: np.ndarray, 
                        tri_to_field: np.ndarray, 
                        edge_to_field: np.ndarray, 
                        linked_tris: dict[int, int], 
                        linked_edges: dict[int, int], 
                        Nfield: int, 
                        phi: complex) -> tuple[csr_matrix, np.ndarray]:
    """This function constructs the periodic boundary matrix

    Args:
        tris (np.ndarray): _description_
        edges (np.ndarray): _description_
        tri_to_field (np.ndarray): _description_
        edge_to_field (np.ndarray): _description_
        linked_tris (dict[int, int]): _description_
        linked_edges (dict[int, int]): _description_
        Nfield (int): _description_
        phi (complex): _description_

    Returns:
        tuple[csr_matrix, np.ndarray]: _description_
    """

    if edge_to_field.ndim != 2 or tri_to_field.ndim != 2:
        raise ValueError("Periodic field mappings must be two-dimensional")
    if tri_to_field.shape[0] < 3 * edge_to_field.shape[0]:
        raise ValueError("Triangle mapping must contain three edges per edge basis")
    for mapping in (edge_to_field, tri_to_field):
        if mapping.size and (mapping.min() < 0 or mapping.max() >= Nfield):
            raise ValueError("Periodic mapping contains an invalid field index")
    tris_array = np.asarray([(tri, linked_tris[tri]) for tri in tris], dtype=np.int64).reshape(-1, 2).T
    edges_array = np.asarray([(edge, linked_edges[edge]) for edge in edges], dtype=np.int64).reshape(-1, 2).T
    ROWS, COLS, TERMS, diagonal = _fill_periodic_matrix(tris_array, edges_array, tri_to_field, edge_to_field, Nfield)
    matrix = csr_matrix((TERMS, (ROWS, COLS)), [Nfield, Nfield], dtype=np.complex128)
    matrix.data.fill(phi)
    matrix.setdiag(diagonal)
    
    return matrix, ROWS
