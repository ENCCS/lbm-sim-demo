# 2D LBM simulator demo

To run on CPU:

    uv run LatticeBoltzmannDemo.py

and to run with Cupy and GPU support:

    uv run --group gpu LatticeBoltzmannDemo.py

## Credits

This is the Python version of https://physics.weber.edu/schroeder/fluids/ here:

https://physics.weber.edu/schroeder/fluids/LatticeBoltzmannDemo.py.txt

## Other prior work
The "wind tunnel" entry/exit conditions are inspired by Graham Pullan's code
(http://www.many-core.group.cam.ac.uk/projects/LBdemo.shtml).  Additional inspiration from
Thomas Pohl's applet (http://thomas-pohl.info/work/lba.html).

Other portions of code are based on Wagner (http://www.ndsu.edu/physics/people/faculty/wagner/lattice_boltzmann_codes/) and
Gonsalves (http://www.physics.buffalo.edu/phy411-506-2004/index.html; code adapted from Succi,
http://global.oup.com/academic/product/the-lattice-boltzmann-equation-9780199679249).

For related materials see http://physics.weber.edu/schroeder/fluids
