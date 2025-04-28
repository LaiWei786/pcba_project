import numpy as np
from .model import PolymerChain

def generate_self_avoiding_chain(
    N: int, b: float = 1.0, max_trials: int = 1000, seed: int = None
) -> PolymerChain:
    """
    Generate a self-avoiding random walk in 3D for N beads.
    Each step is length b. Retry up to max_trials to avoid overlap.
    """
    if seed is not None:
        np.random.seed(seed)

    positions = np.zeros((N, 3))
    for i in range(1, N):
        for trial in range(max_trials):
            vec = np.random.normal(size=3)
            vec /= np.linalg.norm(vec)
            new_pos = positions[i-1] + b * vec
            # check distance to all previous beads
            if np.min(np.linalg.norm(positions[:i] - new_pos, axis=1)) > 0.9 * b:
                positions[i] = new_pos
                break
        else:
            # fallback to unrestricted step
            positions[i] = positions[i-1] + b * vec
    return PolymerChain(positions, b=b)

# alias for simple random walk if needed
def generate_random_chain(
    N: int, b: float = 1.0, seed: int = None
) -> PolymerChain:
    """
    Simple unrestricted random walk of step length b.
    """
    if seed is not None:
        np.random.seed(seed)
    pos = np.zeros((N, 3))
    for i in range(1, N):
        vec = np.random.normal(size=3)
        vec /= np.linalg.norm(vec)
        pos[i] = pos[i-1] + b * vec
    return PolymerChain(pos, b=b)
