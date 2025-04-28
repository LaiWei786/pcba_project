# src/pcba_project/stress_tensor.py

import numpy as np

def gyration_tensor_eigenvalues(positions: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of the radius-of-gyration tensor for a set of 3D points.

    Parameters
    ----------
    positions : (N,3) array_like
        Cartesian coordinates of N beads.

    Returns
    -------
    eigs : (3,) ndarray
        The three eigenvalues of the gyration tensor, in ascending order.
    """
    # Center-of-mass
    com = np.mean(positions, axis=0)
    # Displacements from COM
    diff = positions - com  # shape (N,3)
    # Gyration tensor: S = (diff^T @ diff) / N
    S = diff.T.dot(diff) / positions.shape[0]
    # Compute its eigenvalues
    eigs = np.linalg.eigvalsh(S)
    return eigs
