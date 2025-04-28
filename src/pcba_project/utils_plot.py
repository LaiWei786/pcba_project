# src/pcba_project/utils_plot.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .model import PolymerChain
from typing import List, Sequence

def plot_chain_2d(
    chain: PolymerChain,
    ax: plt.Axes = None,
    projection: str = "xy",
    show: bool = True
) -> plt.Axes:
    """
    Plot a 2D projection of the polymer chain.

    Parameters
    ----------
    chain : PolymerChain
        The polymer chain to plot.
    ax : matplotlib Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    projection : {'xy', 'xz', 'yz'}
        Which plane to project onto.
    show : bool
        Whether to call plt.show() at the end.

    Returns
    -------
    ax : matplotlib Axes
    """
    coords = chain.positions
    if projection == "xy":
        xs, ys = coords[:,0], coords[:,1]
        xlabel, ylabel = "X", "Y"
    elif projection == "xz":
        xs, ys = coords[:,0], coords[:,2]
        xlabel, ylabel = "X", "Z"
    elif projection == "yz":
        xs, ys = coords[:,1], coords[:,2]
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError("projection must be 'xy', 'xz', or 'yz'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(xs, ys, '-o', markersize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Chain projection ({projection})")
    ax.axis('equal')
    if show:
        plt.show()
    return ax

def plot_chain_3d(
    chain: PolymerChain,
    ax: plt.Axes = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot the full 3D configuration of the polymer chain.

    Parameters
    ----------
    chain : PolymerChain
    ax : matplotlib 3D Axes, optional
    show : bool

    Returns
    -------
    ax : matplotlib Axes
    """
    coords = chain.positions
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[:,0], coords[:,1], coords[:,2], '-o', markersize=3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Chain configuration")
    if show:
        plt.show()
    return ax

def plot_end_to_end_distance(
    chains: Sequence[PolymerChain],
    ax: plt.Axes = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot end-to-end distance vs. index (time or sample number).

    Parameters
    ----------
    chains : sequence of PolymerChain
    ax : matplotlib Axes, optional
    show : bool

    Returns
    -------
    ax : matplotlib Axes
    """
    distances = [c.end_to_end_distance() for c in chains]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(distances, '-o', markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("End-to-end distance")
    ax.set_title("End-to-end distance vs. step")
    if show:
        plt.show()
    return ax

def plot_energy_trace(
    energies: Sequence[float],
    ax: plt.Axes = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot energy vs. MC step or time step.

    Parameters
    ----------
    energies : sequence of floats
    ax : matplotlib Axes, optional
    show : bool

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(energies, '-')
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy trace")
    if show:
        plt.show()
    return ax

def plot_radius_of_gyration(
    chains: Sequence[PolymerChain],
    ax: plt.Axes = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot radius of gyration vs. index (time or sample number).

    Parameters
    ----------
    chains : sequence of PolymerChain
    ax : matplotlib Axes, optional
    show : bool

    Returns
    -------
    ax : matplotlib Axes
    """
    rgs = [c.radius_of_gyration() for c in chains]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(rgs, '-o', markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Radius of gyration")
    ax.set_title("Radius of gyration vs. step")
    if show:
        plt.show()
    return ax
