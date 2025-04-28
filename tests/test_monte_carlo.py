import numpy as np
import pytest
from pcba_project.random_walk import generate_random_chain
from pcba_project.monte_carlo import GaussianMetropolisMCChain, truncated_lj_energy

def test_truncated_lj_energy_zero_for_far_pairs():
  
    pos = np.array([[0,0,0],[10,0,0],[20,0,0]])
    E = truncated_lj_energy(pos, eps=1.0, sigma=1.0, r_cut=2.5)
    assert E == pytest.approx(0.0)

def test_mc_energy_decrease_or_stochastic():
 
    chain = generate_random_chain(N=2, b=1.0, seed=0)
    mc = GaussianMetropolisMCChain(sigma_prop=0.0, eps=1.0, sigma_lj=1.0, kT=1.0, r_cut=2.5)
    traj, energies = mc.sample(chain, n_steps=10)

    assert len(traj) == 11
    assert all(e == pytest.approx(energies[0]) for e in energies)

def test_mc_acceptance_rate():

    chain = generate_random_chain(N=5, b=1.0, seed=5)
    mc = GaussianMetropolisMCChain(sigma_prop=0.1, eps=0.1, sigma_lj=1.0, kT=1.0, r_cut=2.5)
    traj, energies = mc.sample(chain, n_steps=200)
    accepts = sum(
        1 for prev, curr in zip(energies, energies[1:]) if curr != prev
    )
    rate = accepts / 200
    assert 0.0 < rate < 1.0
