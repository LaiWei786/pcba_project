import numpy as np
from typing import Tuple, List
from .model import PolymerChain

def truncated_lj_energy(
    positions: np.ndarray,
    eps: float = 1.0,
    sigma: float = 1.0,
    r_cut: float = 2.5
) -> float:
    """
    Compute LJ energy with cutoff: U(r) = 4ε[(σ/r)^12 - (σ/r)^6] - U(r_cut) for r < r_cut.
    Skip adjacent beads.
    """
    N = positions.shape[0]
    E_cut = 4 * eps * ((sigma/r_cut)**12 - (sigma/r_cut)**6)
    E = 0.0
    for i in range(N-1):
        for j in range(i+2, N):
            rij = np.linalg.norm(positions[i] - positions[j])
            if rij < r_cut:
                sr6 = (sigma / rij) ** 6
                E += 4 * eps * (sr6**2 - sr6) - E_cut
    return E

class GaussianMetropolisMCChain:
    """
    Metropolis-Hastings sampler using Gaussian proposals and truncated LJ energy.
    """
    def __init__(
        self,
        sigma_prop: float = 0.5,
        eps: float = 1.0,
        sigma_lj: float = 1.0,
        kT: float = 1.0,
        r_cut: float = 2.5
    ):
        self.sigma_prop = sigma_prop
        self.eps = eps
        self.sigma_lj = sigma_lj
        self.kT = kT
        self.r_cut = r_cut

    def sample(
        self,
        chain: PolymerChain,
        n_steps: int
    ) -> Tuple[List[PolymerChain], List[float]]:
        state = chain.positions.copy()
        E = truncated_lj_energy(state, self.eps, self.sigma_lj, self.r_cut)
        traj   = [PolymerChain(state.copy(), b=chain.b)]
        energies = [E]

        for _ in range(n_steps):
            i = np.random.randint(0, chain.N)
            new_state = state.copy()
            new_state[i] += np.random.normal(scale=self.sigma_prop, size=3)

            E_new = truncated_lj_energy(
                new_state, self.eps, self.sigma_lj, self.r_cut
            )
            delta = E_new - E

            if delta < 0 or np.random.rand() < np.exp(-delta/self.kT):
                state = new_state
                E = E_new

            traj.append(PolymerChain(state.copy(), b=chain.b))
            energies.append(E)

        return traj, energies
