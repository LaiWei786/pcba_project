import numpy as np
import pytest
from pcba_project.random_walk import generate_random_chain, generate_self_avoiding_chain

@pytest.mark.parametrize("factory,b", [
    (generate_random_chain, 1.5),
    (generate_self_avoiding_chain, 1.5),
])
def test_bond_length_constant(factory, b):
    """
    For both random walk and self-avoiding walk, consecutive beads
    must be separated by exactly the bond length b.
    """
    N = 30
    chain = factory(N=N, b=b, seed=0)
    # distances between neighbors
    dists = np.linalg.norm(chain.positions[1:] - chain.positions[:-1], axis=1)
    assert np.allclose(dists, b, atol=1e-8)

def test_random_walk_mean_squared_scaling():
    """
    The mean squared end-to-end distance of a large ensemble of random walks
    should scale as N * b^2 within ~10% tolerance.
    """
    N, b, M = 100, 1.0, 50
    d2s = []
    for seed in range(M):
        chain = generate_random_chain(N=N, b=b, seed=seed)
        d2s.append(chain.end_to_end_distance()**2)
    mean_d2 = np.mean(d2s)
    expected = N * b * b
    rel_error = abs(mean_d2 - expected) / expected
    assert rel_error < 0.1, f"mean R²={mean_d2:.1f}, expected≈{expected:.1f}"

def test_self_avoiding_chain_expands_more_than_random():
    """
    On average, a self-avoiding walk should have a larger ⟨R²⟩ than a simple random walk.
    """
    N, b, M = 50, 1.0, 50
    rw_d2 = []
    saw_d2 = []
    for seed in range(M):
        rw = generate_random_chain(N=N, b=b, seed=seed)
        saw = generate_self_avoiding_chain(N=N, b=b, seed=seed)
        rw_d2.append(rw.end_to_end_distance()**2)
        saw_d2.append(saw.end_to_end_distance()**2)
    assert np.mean(saw_d2) > np.mean(rw_d2), "SAW should yield larger average R² than RW"
