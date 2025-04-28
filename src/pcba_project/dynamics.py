import numpy as np
from typing import List
from .model import PolymerChain

def spring_forces(
    positions: np.ndarray,
    k_spring: float = 1.0,
    b_eq: float = 1.0
) -> np.ndarray:
    N = positions.shape[0]
    F = np.zeros_like(positions)
    for i in range(N):
        if i>0:
            vec = positions[i] - positions[i-1]
            r = np.linalg.norm(vec)
            F[i]   -= k_spring * (r - b_eq) * (vec/r)
        if i<N-1:
            vec = positions[i] - positions[i+1]
            r = np.linalg.norm(vec)
            F[i]   -= k_spring * (r - b_eq) * (vec/r)
    return F

def truncated_lj_forces(
    positions: np.ndarray,
    eps: float = 1.0,
    sigma: float = 1.0,
    r_cut: float = 2.5
) -> np.ndarray:
    N = positions.shape[0]
    F = np.zeros_like(positions)
    E_cut = 4 * eps * ((sigma/r_cut)**12 - (sigma/r_cut)**6)
    for i in range(N-1):
        for j in range(i+2, N):
            rij_vec = positions[i] - positions[j]
            r = np.linalg.norm(rij_vec)
            if r < r_cut and r>0:
                sr6 = (sigma/r)**6
                # dU/dr including shift
                mag = 4*eps*(12*sr6**2/r - 6*sr6/r)
                fij = mag * (rij_vec/r)
                F[i] += fij
                F[j] -= fij
    return F

def overdamped_integrate(
    chain: PolymerChain,
    dt: float,
    n_steps: int,
    k_spring: float = 1.0,
    b_eq: float = 1.0,
    use_lj: bool = False,
    lj_eps: float = 1.0,
    lj_sigma: float = 1.0,
    lj_cut: float = 2.5
) -> List[PolymerChain]:
    """
    Overdamped (no inertia) integration: x←x + dt*F_total.
    """
    pos = chain.positions.copy()
    traj: List[PolymerChain] = [PolymerChain(pos.copy(), b=chain.b)]
    for _ in range(n_steps):
        F = spring_forces(pos, k_spring, b_eq)
        if use_lj:
            F += truncated_lj_forces(pos, lj_eps, lj_sigma, lj_cut)
        pos = pos + dt * F
        traj.append(PolymerChain(pos.copy(), b=chain.b))
    return traj

# alias
velocity_verlet_chain = overdamped_integrate
