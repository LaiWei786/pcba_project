# tests/test_stress_tensor.py
from pcba_project.model import PolymerChain
from pcba_project.stress_tensor import gyration_tensor_eigenvalues
import numpy as np

def test_rg_eigenvalues_sum():
    pos = np.random.randn(10,3)
    chain = PolymerChain(pos, b=1.0)
    eigs = gyration_tensor_eigenvalues(chain.positions)
    # trace(S) = sum(eig), trace = Rg^2 * N
    assert np.isclose(np.sum(eigs), chain.radius_of_gyration()**2, atol=1e-6)

