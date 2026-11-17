"""Periodic constraints cover every basis and reject invalid native-kernel inputs."""
import numpy as np
import pytest

from emerge._emerge.elements.dofsets import ElementSpace
from emerge._emerge.physics.microwave.assembly.emr_microwave import gen_periodic_matrix


@pytest.mark.parametrize("space", list(ElementSpace))
@pytest.mark.parametrize("include_tris,include_edges", [(True, True), (True, False), (False, True), (False, False)])
def test_periodic_mapping_all_bases(space, include_tris, include_edges):
    dofset = space.get_set().set2d
    codes = np.asarray(dofset.codes, dtype=np.int64)
    nedof = dofset.n_edge_dofs
    nlocal = len(codes)
    tri_map = np.arange(nlocal * 2, dtype=np.int64).reshape(nlocal, 2)
    edge_map = tri_map[:nedof].copy()
    phi = np.exp(0.7j)
    matrix, rows = gen_periodic_matrix(
        np.array([0] if include_tris else [], dtype=np.int64),
        np.array([0] if include_edges else [], dtype=np.int64),
        tri_map, edge_map, {0: 1}, {0: 1}, codes, nlocal * 2, phi,
    )
    expected = np.eye(nlocal * 2, dtype=np.complex128)
    local_ids = list(range(3 * nedof, nlocal)) if include_tris else []
    if include_edges:
        local_ids += list(range(nedof))
    for i in local_ids:
        source, target = tri_map[i]
        expected[target, target] = 0
        expected[target, source] = phi
    np.testing.assert_allclose(matrix.toarray(), expected)
    assert set(rows) == {tri_map[i, 1] for i in local_ids}


@pytest.mark.parametrize("bad_map", [np.array([0]), np.array([[0], [1]]), np.array([[0], [1], [6]]), np.array([[0], [1], [-1]])])
def test_invalid_periodic_mapping(bad_map):
    with pytest.raises(ValueError):
        gen_periodic_matrix(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64),
            bad_map, np.array([[0]], dtype=np.int64), {}, {},
            np.array([64, 64, 64], dtype=np.int64), 6, 1.0,
        )
