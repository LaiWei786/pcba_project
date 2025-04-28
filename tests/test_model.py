import numpy as np
import pytest
from pcba_project.model import PolymerChain

def test_end_to_end_distance():
    N = 5
    positions = np.zeros((N,3))
    for i in range(N):
        positions[i,0] = i * 2.0  
    chain = PolymerChain(positions, b=2.0)
    assert chain.end_to_end_distance() == pytest.approx((N-1)*2.0)

def test_radius_of_gyration_symmetry():
    # Six points equally spaced on unit circle in XY plane
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    positions = np.vstack([np.cos(angles), np.sin(angles), np.zeros(6)]).T
    chain = PolymerChain(positions, b=1.0)
    eigs = np.linalg.eigvalsh(chain.gyration_tensor())
    eigs = np.sort(eigs)  # ascending

    # Two largest (in-plane) eigenvalues should be equal
    assert np.allclose(eigs[1], eigs[2], atol=1e-6)
    # Smallest (out-of-plane) eigenvalue should be ~0
    assert abs(eigs[0]) < 1e-6


def test_invalid_positions_shape():

    with pytest.raises(AssertionError):
        PolymerChain(np.zeros((10,2)), b=1.0)
