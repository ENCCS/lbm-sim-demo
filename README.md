<div align="center">
<h1>2D LBM simulator demo</h1>

[Example of an interactive simulation adding barriers on-the-fly](https://github.com/user-attachments/assets/d2551d02-9c7b-4051-a7f6-c7fa551d5341)
</div>

To run on CPU:

    uv run LatticeBoltzmannDemo.py

and to run with Cupy and GPU support:

    uv run --group gpu LatticeBoltzmannDemo.py


## Credits

This based on the Python version of https://physics.weber.edu/schroeder/fluids/ found [here](https://physics.weber.edu/schroeder/fluids/LatticeBoltzmannDemo.py.txt)
by Daniel V. Schroeder (Weber State University).

## Other prior work
The "wind tunnel" entry/exit conditions are inspired by [Graham Pullan's code](https://web.archive.org/web/20160905093116/http://www.many-core.group.cam.ac.uk/projects/LBdemo.shtml). Additional inspiration from
[Thomas Pohl's applet](https://web.archive.org/web/20110929212754/http://thomas-pohl.info/work/lba.html).

Other portions of code are based on [Wagner](https://web.archive.org/web/20180706163435/https://www.ndsu.edu/physics/people/faculty/wagner/lattice_boltzmann_codes/) and
[Gonsalves](https://web.archive.org/web/20150110053209/http://www.physics.buffalo.edu/phy411-506-2004/index.html); code adapted from [Succi](http://global.oup.com/academic/product/the-lattice-boltzmann-equation-9780199679249).

## See also

For inspiration on possible future studies:

- Study using LLMs and coding in agent-mode for optimization and parallelization
    - Refactoring code
    - Use a normal profiler and use LLMs to fix bottlenecks
    - Compiling hot spots, with Pythran, Numba, Jax, Cupy
    - Source to source compilation (Transpiling) to another language that you are not familiar with (think Julia, Rust, Nim, Zig, Chapel etc.)
    - Parallelizing

- Quantum Computing for Quantum LBM:
    - "Quantum algorithm for lattice Boltzmann (QALB) simulation of incompressible fluids with a nonlinear collision term",
      Wael Itani, Katepalli R. Sreenivasan, Sauro Succi - [_Physics of Fluids_(2024)](https://doi.org/10.1063/5.0176569)
    - "Scaling Quantum Computing Research to a New Milestone", Apurva Tiwari et al. - [arXiv (2025)](https://doi.org/10.48550/arXiv.2504.10870), [Blog post from Ansys (2025)](https://www.ansys.com/blog/scaling-quantum-computing-research)
    - "Quantum lattice Boltzmann method for simulating nonlinear fluid dynamics",
Boyuan Wang, Zhaoyuan Meng, Yaomin Zhao, Yue Yang - [arXiv (2025)](https://doi.org/10.48550/arXiv.2502.16568)
