# LatticeBoltzmannDemo.py:  a two-dimensional lattice-Boltzmann "wind tunnel" simulation
# Uses numpy to speed up all array handling.
# Uses matplotlib to plot and animate the curl of the macroscopic velocity field.
# Copyright 2013, Daniel V. Schroeder (Weber State University) 2013
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated data and documentation (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# Except as contained in this notice, the name of the author shall not be used in
# advertising or otherwise to promote the sale, use or other dealings in this
# Software without prior written authorization.
# Credits:
# The "wind tunnel" entry/exit conditions are inspired by Graham Pullan's code
# (http://www.many-core.group.cam.ac.uk/projects/LBdemo.shtml).  Additional inspiration from
# Thomas Pohl's applet (http://thomas-pohl.info/work/lba.html).
# Other portions of code are based on Wagner (http://www.ndsu.edu/physics/people/faculty/wagner/lattice_boltzmann_codes/) and
# Gonsalves (http://www.physics.buffalo.edu/phy411-506-2004/index.html; code adapted from Succi,
# http://global.oup.com/academic/product/the-lattice-boltzmann-equation-9780199679249).
# For related materials see http://physics.weber.edu/schroeder/fluids
import time

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np


class LatticeBoltzmannSimulator:
    def __init__(self):
        # Define constants:
        self.height = 80  # lattice dimensions
        self.width = 200
        self.viscosity = 0.02  # fluid viscosity
        self.omega = 1 / (3 * self.viscosity + 0.5)  # "relaxation" parameter
        self.u0 = 0.1  # initial and in-flow speed
        self.four9ths = 4.0 / 9.0  # abbreviations for lattice-Boltzmann weight factors
        self.one9th = 1.0 / 9.0
        self.one36th = 1.0 / 36.0
        self.performanceData = False  # set to True if performance data is desired
        
        # Initialize all the arrays to steady rightward flow:
        self.n0 = self.four9ths * (
            np.ones((self.height, self.width)) - 1.5 * self.u0**2
        )  # particle densities along 9 directions
        self.nN = self.one9th * (np.ones((self.height, self.width)) - 1.5 * self.u0**2)
        self.nS = self.one9th * (np.ones((self.height, self.width)) - 1.5 * self.u0**2)
        self.nE = self.one9th * (np.ones((self.height, self.width)) + 3 * self.u0 + 4.5 * self.u0**2 - 1.5 * self.u0**2)
        self.nW = self.one9th * (np.ones((self.height, self.width)) - 3 * self.u0 + 4.5 * self.u0**2 - 1.5 * self.u0**2)
        self.nNE = self.one36th * (np.ones((self.height, self.width)) + 3 * self.u0 + 4.5 * self.u0**2 - 1.5 * self.u0**2)
        self.nSE = self.one36th * (np.ones((self.height, self.width)) + 3 * self.u0 + 4.5 * self.u0**2 - 1.5 * self.u0**2)
        self.nNW = self.one36th * (np.ones((self.height, self.width)) - 3 * self.u0 + 4.5 * self.u0**2 - 1.5 * self.u0**2)
        self.nSW = self.one36th * (np.ones((self.height, self.width)) - 3 * self.u0 + 4.5 * self.u0**2 - 1.5 * self.u0**2)
        self.rho = self.n0 + self.nN + self.nS + self.nE + self.nW + self.nNE + self.nSE + self.nNW + self.nSW  # macroscopic density
        self.ux = (self.nE + self.nNE + self.nSE - self.nW - self.nNW - self.nSW) / self.rho  # macroscopic x velocity
        self.uy = (self.nN + self.nNE + self.nNW - self.nS - self.nSE - self.nSW) / self.rho  # macroscopic y velocity

        # Initialize barriers:
        self.barrier = np.zeros((self.height, self.width), bool)  # True wherever there's a barrier
        self.barrier[(self.height // 2) - 8 : (self.height // 2) + 8, self.height // 2] = (
            True  # simple linear barrier
        )
        self.barrierN = np.roll(self.barrier, 1, axis=0)  # sites just north of barriers
        self.barrierS = np.roll(self.barrier, -1, axis=0)  # sites just south of barriers
        self.barrierE = np.roll(self.barrier, 1, axis=1)  # etc.
        self.barrierW = np.roll(self.barrier, -1, axis=1)
        self.barrierNE = np.roll(self.barrierN, 1, axis=1)
        self.barrierNW = np.roll(self.barrierN, -1, axis=1)
        self.barrierSE = np.roll(self.barrierS, 1, axis=1)
        self.barrierSW = np.roll(self.barrierS, -1, axis=1)
    
    # Move all particles by one step along their directions of motion (pbc):
    def stream(self):
        self.nN = np.roll(self.nN, 1, axis=0)  # axis 0 is north-south; + direction is north
        self.nNE = np.roll(self.nNE, 1, axis=0)
        self.nNW = np.roll(self.nNW, 1, axis=0)
        self.nS = np.roll(self.nS, -1, axis=0)
        self.nSE = np.roll(self.nSE, -1, axis=0)
        self.nSW = np.roll(self.nSW, -1, axis=0)
        self.nE = np.roll(self.nE, 1, axis=1)  # axis 1 is east-west; + direction is east
        self.nNE = np.roll(self.nNE, 1, axis=1)
        self.nSE = np.roll(self.nSE, 1, axis=1)
        self.nW = np.roll(self.nW, -1, axis=1)
        self.nNW = np.roll(self.nNW, -1, axis=1)
        self.nSW = np.roll(self.nSW, -1, axis=1)
        # Use tricky boolean arrays to handle barrier collisions (bounce-back):
        self.nN[self.barrierN] = self.nS[self.barrier]
        self.nS[self.barrierS] = self.nN[self.barrier]
        self.nE[self.barrierE] = self.nW[self.barrier]
        self.nW[self.barrierW] = self.nE[self.barrier]
        self.nNE[self.barrierNE] = self.nSW[self.barrier]
        self.nNW[self.barrierNW] = self.nSE[self.barrier]
        self.nSE[self.barrierSE] = self.nNW[self.barrier]
        self.nSW[self.barrierSW] = self.nNE[self.barrier]

    # Collide particles within each cell to redistribute velocities (could be optimized a little more):
    def collide(self):
        self.rho = self.n0 + self.nN + self.nS + self.nE + self.nW + self.nNE + self.nSE + self.nNW + self.nSW
        self.ux = (self.nE + self.nNE + self.nSE - self.nW - self.nNW - self.nSW) / self.rho
        self.uy = (self.nN + self.nNE + self.nNW - self.nS - self.nSE - self.nSW) / self.rho
        ux2 = self.ux * self.ux  # pre-compute terms used repeatedly...
        uy2 = self.uy * self.uy
        u2 = ux2 + uy2
        omu215 = 1 - 1.5 * u2  # "one minus u2 times 1.5"
        uxuy = self.ux * self.uy
        self.n0 = (1 - self.omega) * self.n0 + self.omega * self.four9ths * self.rho * omu215
        self.nN = (1 - self.omega) * self.nN + self.omega * self.one9th * self.rho * (omu215 + 3 * self.uy + 4.5 * uy2)
        self.nS = (1 - self.omega) * self.nS + self.omega * self.one9th * self.rho * (omu215 - 3 * self.uy + 4.5 * uy2)
        self.nE = (1 - self.omega) * self.nE + self.omega * self.one9th * self.rho * (omu215 + 3 * self.ux + 4.5 * ux2)
        self.nW = (1 - self.omega) * self.nW + self.omega * self.one9th * self.rho * (omu215 - 3 * self.ux + 4.5 * ux2)
        self.nNE = (1 - self.omega) * self.nNE + self.omega * self.one36th * self.rho * (
            omu215 + 3 * (self.ux + self.uy) + 4.5 * (u2 + 2 * uxuy)
        )
        self.nNW = (1 - self.omega) * self.nNW + self.omega * self.one36th * self.rho * (
            omu215 + 3 * (-self.ux + self.uy) + 4.5 * (u2 - 2 * uxuy)
        )
        self.nSE = (1 - self.omega) * self.nSE + self.omega * self.one36th * self.rho * (
            omu215 + 3 * (self.ux - self.uy) + 4.5 * (u2 - 2 * uxuy)
        )
        self.nSW = (1 - self.omega) * self.nSW + self.omega * self.one36th * self.rho * (
            omu215 + 3 * (-self.ux - self.uy) + 4.5 * (u2 + 2 * uxuy)
        )
        # Force steady rightward flow at ends (no need to set 0, N, and S components):
        def set_inlet_velocity(arr, fraction, direction_factor, u0):
            arr[:, 0] = fraction * (1 + direction_factor * u0 + 4.5 * u0**2 - 1.5 * u0**2)
            
        set_inlet_velocity(self.nE, self.one9th, 3, self.u0)
        set_inlet_velocity(self.nW, self.one9th, -3, self.u0)
        set_inlet_velocity(self.nNE, self.one36th, 3, self.u0)
        set_inlet_velocity(self.nSE, self.one36th, 3, self.u0)
        set_inlet_velocity(self.nNW, self.one36th, -3, self.u0)
        set_inlet_velocity(self.nSW, self.one36th, -3, self.u0)


    # Compute curl of the macroscopic velocity field:
    def curl(self, ux, uy):
        return (
            np.roll(uy, -1, axis=1)
            - np.roll(uy, 1, axis=1)
            - np.roll(ux, -1, axis=0)
            + np.roll(ux, 1, axis=0)
        )


# Create simulator instance
simulator = LatticeBoltzmannSimulator()

# Here comes the graphics and animation...
theFig = plt.figure(figsize=(8, 3))
fluidImage = plt.imshow(
    simulator.curl(simulator.ux, simulator.uy),
    origin="lower",
    norm=plt.Normalize(-0.1, 0.1),
    cmap=plt.get_cmap("jet"),
    interpolation="none",
)  # See http://www.loria.fr/~rougier/teaching/matplotlib/#colormaps for other cmap options
bImageArray = np.zeros((simulator.height, simulator.width, 4), np.uint8)  # an RGBA image
bImageArray[simulator.barrier, 3] = 255  # set alpha=255 only at barrier sites
barrierImage = plt.imshow(
    bImageArray, origin="lower", interpolation="none"
)

# Function called for each successive animation frame:
startTime = time.time()  # frameList = open('frameList.txt','w')		# file containing list of images (to make movie)


def nextFrame(arg):  # (arg is the frame number, which we don't need)
    global startTime
    if simulator.performanceData and (arg % 100 == 0) and (arg > 0):
        endTime = time.time()
        print("%1.1f" % (100 / (endTime - startTime)), "frames per second")
        startTime = endTime
    # frameName = "frame%04d.png" % arg
    # plt.savefig(frameName)
    # frameList.write(frameName + '\n')
    for step in range(20):  # adjust number of steps for smooth animation
        simulator.stream()
        simulator.collide()
    fluidImage.set_array(simulator.curl(simulator.ux, simulator.uy))
    return (fluidImage, barrierImage)  # return the figure elements to redraw


animate = matplotlib.animation.FuncAnimation(theFig, nextFrame, interval=1, blit=True)
plt.show()