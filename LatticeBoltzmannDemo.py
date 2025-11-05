"""
LatticeBoltzmannDemo.py
- A two-dimensional lattice-Boltzmann "wind tunnel" simulation
- Uses numpy to speed up all array handling.
- Uses matplotlib to plot and animate the curl of the macroscopic velocity field.
"""

import time

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

try:
    import cupy as np
except ModuleNotFoundError:
    GPU = False
    import numpy as np
else:
    GPU = True

from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons, RectangleSelector


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
        self.rho = (
            self.n0 + self.nN + self.nS + self.nE + self.nW + self.nNE + self.nSE + self.nNW + self.nSW
        )  # macroscopic density
        self.ux = (self.nE + self.nNE + self.nSE - self.nW - self.nNW - self.nSW) / self.rho  # macroscopic x velocity
        self.uy = (self.nN + self.nNE + self.nNW - self.nS - self.nSE - self.nSW) / self.rho  # macroscopic y velocity

        # Initialize barriers:
        self.barrier = np.zeros((self.height, self.width), bool)  # True wherever there's a barrier
        self.barrier_type = "circle"  # Default barrier type
        if self.barrier_type == "circle":
            self.init_circular_barrier(center_x=self.width // 2, center_y=self.height // 2, radius=10)
        else:
            y_center = self.height // 2
            y_start = y_center - 8
            y_end = y_center + 8
            x_start = y_center
            x_end = y_center + 1

            self.init_barrier(y_start, y_end, x_start, x_end)

    def init_circular_barrier(self, center_x, center_y, radius):
        """Initialize a circular barrier in the center of the domain"""
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[: self.height, : self.width]

        # Create circular barrier using distance from center
        distance_from_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        self.barrier[distance_from_center <= radius] = True
        self._update_barrier_neighbor_arrays()

    def init_barrier(self, y_start, y_end, x_start, x_end):
        """Initialize a rectangular barrier (kept for compatibility)"""
        self.barrier[y_start:y_end, x_start:x_end] = True  # simple linear barrier
        self._update_barrier_neighbor_arrays()

    def _update_barrier_neighbor_arrays(self):
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
        res = np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1) - np.roll(ux, -1, axis=0) + np.roll(ux, 1, axis=0)
        if GPU:
            return res.get()  # To numpy
        else:
            return res


# Create simulator instance
simulator = LatticeBoltzmannSimulator()

# Here comes the graphics and animation...
fig = plt.figure(figsize=(8, 3))
fig.suptitle("2D LBM simulator demo")

gs = GridSpec(3, 2, width_ratios=[1, 4], height_ratios=[4, 2, 1])
ax0_radio = fig.add_subplot(gs[1, 0])
ax0_button = fig.add_subplot(gs[2, 0])
ax1 = fig.add_subplot(gs[:, 1])  # Combine the rows of the second column

fig.text(0.01, 0.8, "Click with mouse and draw")
fig.text(0.01, 0.7, "to add new barriers.")

fluidImage = ax1.imshow(
    simulator.curl(simulator.ux, simulator.uy),
    origin="lower",
    norm=plt.Normalize(-0.1, 0.1),
    cmap=plt.get_cmap("coolwarm"),
    interpolation="none",
)
bImageArray = np.zeros((simulator.height, simulator.width, 4), np.uint8)  # an RGBA image
bImageArray[simulator.barrier, 3] = 255  # set alpha=255 only at barrier sites
if GPU:
    bImageArray = bImageArray.get()
barrierImage = ax1.imshow(bImageArray, origin="lower", interpolation="none")

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


def rect_onselect(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    # Convert to integer coordinates and ensure they're within bounds
    x_start = int(min(x1, x2) + 0.5)
    x_end = int(max(x1, x2) + 0.5)
    y_start = int(min(y1, y2) + 0.5)
    y_end = int(max(y1, y2) + 0.5)

    # Ensure coordinates are within valid range
    x_start = max(0, min(x_start, simulator.width - 1))
    x_end = max(0, min(x_end, simulator.width - 1))
    y_start = max(0, min(y_start, simulator.height - 1))
    y_end = max(0, min(y_end, simulator.height - 1))

    # Update barrier array
    if simulator.barrier_type == "circle":
        xc = (x_start + x_end) // 2
        yc = (y_start + y_end) // 2
        radius = min(x_end - xc, y_end - yc)
        simulator.init_circular_barrier(xc, yc, radius)
    else:
        simulator.init_barrier(y_start, y_end, x_start, x_end)

    # Update visualization
    mask = simulator.barrier
    if GPU:
        mask = mask.get()

    bImageArray[..., 3] = 255 * mask
    barrierImage.set_array(bImageArray)
    plt.draw()


# Set up rectangle selector
rect_selector = RectangleSelector(
    ax1,
    rect_onselect,
    useblit=False,
    props=dict(edgecolor="red", linestyle="--", linewidth=1, fill=False),
    interactive=True,
)


def clear_barrier(_event):
    simulator.barrier[:, :] = False
    simulator._update_barrier_neighbor_arrays()
    # simulator.init_barrier(0, 0, 0, 0)  # Reinitialize barrier arrays
    bImageArray[..., 3] = 0  # Set alpha to 0 everywhere
    barrierImage.set_array(bImageArray)
    plt.draw()


clear_barrier_button = Button(ax0_button, "Clear Barrier")
clear_barrier_button.on_clicked(clear_barrier)


def select_barrier_type(label):
    """Callback function for radio button selection"""
    if label == "Circle":
        simulator.barrier_type = "circle"
    else:  # Rectangle
        simulator.barrier_type = "rectangle"


# Add radio buttons for barrier selection
ax0_radio.set_title("Barrier type")
radio = RadioButtons(ax0_radio, ("Circle", "Rectangle"))
radio.on_clicked(select_barrier_type)

animate = matplotlib.animation.FuncAnimation(fig, nextFrame, interval=1, blit=True)
plt.tight_layout()
plt.show()
