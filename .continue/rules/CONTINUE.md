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
- Matplotlib (`pip install matplotlib`)

### Installation
1. Clone or download the repository
2. Install dependencies: `uv add numpy matplotlib`
3. Run the simulation: `python LatticeBoltzmannDemo.py`

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

## Development Workflow

### Coding Standards
- Uses NumPy for vectorized operations for performance
- Follows lattice-Boltzmann method conventions
- Clear variable naming for physical quantities (ux, uy, rho, etc.)
- Modular function structure for stream and collide operations
- Remove global variables and refactor to use an classes

### Testing Approach
- Write tests using pytest for correctness
- Testing done through visual inspection of simulation output
- Performance can be monitored by setting `performanceData = True`

### Build and Deployment
- Single Python script deployment
- No compilation required
- Visualization output shown directly in matplotlib window

### Contribution Guidelines
Since this is a demonstration script, contributions should:
- Focus on improving documentation clarity
- Suggest performance improvements

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

### Modifying Simulation Parameters
To change the simulation:
1. Adjust `height` and `width` for grid size
2. Modify `viscosity` to change fluid properties
3. Change `u0` to alter inflow speed
4. Edit barrier shape by modifying the `barrier` array initialization

### Customizing Visualization
To change visual appearance:
1. Modify `cmap` parameter in `imshow()` for different color schemes
2. Adjust `norm` parameter to change color scaling
3. Change figure size in `matplotlib.pyplot.figure(figsize=(8,3))`

### Adding New Barriers
To add barriers:
1. Modify the `barrier` array initialization
2. Use NumPy array slicing to create different barrier shapes
3. The barrier system handles bounce-back collisions automatically

### Performance Tuning
To adjust simulation speed:
1. Modify the number of `stream()` and `collide()` steps in `nextFrame()` function
2. Adjust `interval` parameter in `FuncAnimation` for animation speed
3. Set `performanceData = True` to monitor frames per second

## Troubleshooting

### No Visualization Appears
- Ensure matplotlib is properly installed
- Check that Python is running without errors
- Verify the script has proper permissions

### Slow Performance
- Reduce the number of steps in the `for step in range(20):` loop in `nextFrame()`
- Reduce the grid size by modifying `height` and `width`
- Adjust the `interval` parameter in `FuncAnimation`

### Memory Issues
- Reduce the simulation grid size
- Consider using a more efficient data structure if needed
- Monitor memory usage during long simulations

## References

- Original implementation by Daniel V. Schroeder (Weber State University)
- Graham Pullan's lattice-Boltzmann code (http://www.many-core.group.cam.ac.uk/projects/LBdemo.shtml)
- Thomas Pohl's applet (http://thomas-pohl.info/work/lba.html)
- Wagner's lattice-Boltzmann codes (http://www.ndsu.edu/physics/people/faculty/wagner/lattice_boltzmann_codes/)
- Gonsalves' fluid dynamics materials (http://www.physics.buffalo.edu/phy411-506-2004/index.html)
- Succi's "The Lattice Boltzmann Equation" (Oxford University Press)
