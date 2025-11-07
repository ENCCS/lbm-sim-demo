# CONTINUE.md

## Project Overview

This is a two-dimensional lattice-Boltzmann simulation of fluid dynamics, specifically modeling a "wind tunnel" with a barrier. The simulation implements the lattice-Boltzmann method (LBM) to solve fluid flow problems, visualizing the curl of the macroscopic velocity field to show vorticity patterns in the fluid.

Key technologies:
- Python 3
- NumPy for efficient array operations
- Matplotlib for visualization and animation

The simulation models fluid flow through a channel with a central barrier, demonstrating how fluid behaves around obstacles. This is a classic demonstration of computational fluid dynamics using the lattice-Boltzmann method.

## Getting Started

### Prerequisites
- Python
- NumPy
- Matplotlib

### Installation
1. Clone or download the repository
2. Install dependencies: `uv sync`
3. Run the simulation: `python LatticeBoltzmannDemo.py` or `uv run LatticeBoltzmannDemo.py`.

### Basic Usage
The simulation automatically starts when run. It creates a visualization showing:
- Fluid flow patterns (colored by curl/vorticity)
- A central barrier in the channel
- Smooth animation of fluid dynamics over time

### Running Tests
This is a visualization demo rather than a test suite. To verify functionality:
1. Run the script
2. Observe the animation
3. Verify that fluid flows around the barrier and vortices form

## Project Structure

```
.
├── LatticeBoltzmannDemo.py          # Main simulation script
├── README.md                        # Project documentation
└── .continue/                       # Continue-specific configuration
    └── rules/
        └── CONTINUE.md              # This file
```

Key components:
- `LatticeBoltzmannDemo.py`: Contains the complete simulation implementation
- Main simulation functions: `stream()`, `collide()`, `curl()`
- Visualization components: matplotlib animation setup

## Key Concepts

### Lattice-Boltzmann Method
- Discrete velocity model with 9 lattice directions
- Uses distribution functions (n0, nN, nS, etc.) to represent particle populations
- Implements collision and streaming steps to evolve the system
- Simulates fluid behavior through particle interactions

### Physical Parameters
- **Viscosity**: Controls fluid resistance (0.02 in this implementation)
- **Relaxation parameter (omega)**: Governs collision rate
- **Inflow speed (u0)**: Initial fluid velocity (0.1)
- **Grid dimensions**: 80x200 lattice points

### Lattice Directions
The simulation uses 9 discrete velocity directions:
- n0: rest particles (0,0)
- nN, nS: north/south (0,±1)
- nE, nW: east/west (±1,0)
- nNE, nSE, nNW, nSW: diagonals (±1,±1)

## Common Tasks

- ...
