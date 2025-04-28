import numpy as np

class PolymerChain:
    """
    A bead-spring polymer chain model.
    Attributes:
        positions: (N,3) array of bead coordinates
        b: float, nominal bond length
    """
    def __init__(self, positions: np.ndarray, b: float = 1.0):
        assert positions.ndim == 2 and positions.shape[1] == 3
        self.positions = positions.copy()
        self.N = positions.shape[0]
        self.b = b

    def end_to_end_distance(self) -> float:
        """Return distance between first and last bead."""
        return np.linalg.norm(self.positions[-1] - self.positions[0])

    def gyration_tensor(self) -> np.ndarray:
        """Compute the radius-of-gyration tensor."""
        com = np.mean(self.positions, axis=0)
        diff = self.positions - com
        return diff.T.dot(diff) / self.N

    def radius_of_gyration(self) -> float:
        """Return radius of gyration: sqrt(trace(S))."""
        return np.sqrt(np.trace(self.gyration_tensor()))
