"""
============================================================
Ferrocell Simulation using Magpylib 4.4.0
============================================================

Author: Lori Gardi
Refactor and documentation: ChatGPT

PURPOSE
-------
This script simulates the optical response of a Ferrocell:
a thin ferrofluid film illuminated by multiple light sources
around its perimeter and observed from a fixed viewpoint.

The ferrofluid is assumed to form microscopic chains aligned
with the local magnetic field direction. Each chain behaves
like a specular reflector whose effective surface normal is
defined by the magnetic field direction.

The brightness of a pixel is determined by how closely the
magnetic field direction matches the normal required to
reflect light from a given source into the observer.

The final image is the accumulated RGB contribution from all
light sources.

UNITS
-----
All distances are in millimeters (mm) unless the parameter
'factor' is changed.

DEPENDENCIES
------------
- numpy
- matplotlib
- magpylib version 4.4.0 or newer
"""

import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from datetime import datetime
import os
import shutil


# ============================================================
# COLOR DEFINITIONS
# ============================================================

def Red():
    """RGBA color for red light."""
    return np.array([1, 0, 0, 1])

def Green():
    """RGBA color for green light."""
    return np.array([0, 1, 0, 1])

def Blue():
    """RGBA color for blue light."""
    return np.array([0, 0, 1, 1])

def Yellow():
    """RGBA color for yellow light."""
    return np.array([1, 1, 0, 1])


def Clist_gen(base_colors, N):
    """
    Generate a list of N colors by repeating a base sequence.

    Parameters
    ----------
    base_colors : array of shape (M,4)
        List of RGBA colors used cyclically.
    N : int
        Total number of colors required.

    Returns
    -------
    colors : array of shape (N,4)
        Expanded color list matching number of lights.
    """
    out = base_colors.copy()
    while len(out) < N:
        out = np.vstack((out, base_colors))
    return out[:N]


# ============================================================
# VECTOR GEOMETRY AND OPTICS
# ============================================================

def incident_vector(x, y, z, L):
    """
    Compute the incident vector from a light source to the film.

    The incident vector is defined as:
        I = p_film minus p_light
    """
    Ix = x - L[0]
    Iy = y - L[1]
    Iz = z - L[2]
    Imag = np.sqrt(Ix**2 + Iy**2 + Iz**2)
    return Ix, Iy, Iz, Imag


def reflection_normals(L, x, y, z, O):
    """
    Compute the surface normal required for specular reflection.

    Physics:
    For specular reflection the surface normal N is proportional
    to (R - I) divided by 2, where:

        I is the normalized incident direction
        R is the normalized reflection direction
    """
    Dx, Dy, Dz, _ = incident_vector(x, y, z, L)
    Dmag = np.sqrt(Dx**2 + Dy**2 + Dz**2)
    Dx, Dy, Dz = Dx/Dmag, Dy/Dmag, Dz/Dmag

    Rx = O[0] - x
    Ry = O[1] - y
    Rz = O[2] - z
    Rmag = np.sqrt(Rx**2 + Ry**2 + Rz**2)
    Rx, Ry, Rz = Rx/Rmag, Ry/Rmag, Rz/Rmag

    Nx = (Rx - Dx) / 2
    Ny = (Ry - Dy) / 2
    Nz = (Rz - Dz) / 2

    Nmag = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    return Nx/Nmag, Ny/Nmag, Nz/Nmag


def dot2d(x1, y1, z1, x2, y2, z2):
    """Dot product of two vector fields."""
    return x1*x2 + y1*y2 + z1*z2


# ============================================================
# MAIN FERROCELL SIMULATION
# ============================================================

def run_ferrocell():
    """Main driver function for the Ferrocell simulation."""

    fname = "Ferrocell_Magpylib44"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outdir = "outputs/" + fname + "_" + timestamp
    os.makedirs(outdir, exist_ok=True)

    shutil.copy(__file__, outdir + "/script.py")

    factor = 1.0
    Radius = 25 * factor
    N_points = 400
    N_lights = 36

    observer = np.array([0, 0, 190 * factor])
    film_z = 0.0

    diameter = 8 * factor
    height = 20 * factor
    magnet_z = -height/2 - 3*factor

    sharpness = 50
    logit_k = 7.5 / (10 * factor)
    logit_0 = Radius * 2.5

    magnet = magpy.magnet.Cylinder(
        magnetization=(0, 0, 1),
        dimension=(diameter, height),
        position=(0, 0, magnet_z)
    )

    xs = np.linspace(-Radius, Radius, N_points)
    ys = np.linspace(-Radius, Radius, N_points)
    x, y = np.meshgrid(xs, ys)
    z = np.zeros_like(x) + film_z

    POS = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    B = magpy.getB(magnet, POS).reshape(N_points, N_points, 3)
    Bmag = np.linalg.norm(B, axis=2)

    Mux = B[:, :, 0] / Bmag
    Muy = B[:, :, 1] / Bmag
    Muz = B[:, :, 2] / Bmag

    sel = (x**2 + y**2) <= Radius**2

    theta = np.linspace(0, 2*np.pi, N_lights, endpoint=False)
    L = np.column_stack((
        (Radius + 5) * np.cos(theta),
        (Radius + 5) * np.sin(theta),
        np.zeros_like(theta)
    ))

    colors = Clist_gen(
        np.array([Yellow(), Blue(), Red(), Green()]),
        N_lights
    )

    visible = np.zeros((N_points, N_points, N_lights))

    for i in range(N_lights):
        _, _, _, Imag = incident_vector(x, y, z, L[i])
        Nx, Ny, Nz = reflection_normals(L[i], x, y, z, observer)
        mudotN = dot2d(Mux, Muy, Muz, Nx, Ny, Nz)

        vis = np.exp(-(mudotN**2) / (2*(1/sharpness)**2))
        vis /= np.max(vis)

        atten = 1 - 1/(1 + np.exp(-logit_k*(Imag - logit_0)))
        atten /= np.max(atten)

        visible[:, :, i] = vis * atten

    img = np.zeros((N_points, N_points, 4))
    img[:, :, 3] = 1

    for i in range(N_lights):
        img[:, :, :3] += colors[i][:3] * visible[:, :, i][:, :, None] / N_lights

    mag = np.linalg.norm(img[:, :, :3], axis=2)
    brightnessFactor = 2.5
    img[:, :, :3] /= mag.max()
    img[:, :, :3] *= brightnessFactor
    img[:, :, :3] = np.clip(img[:, :, :3], 0, 1)

    img[:, :, 3] *= sel

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir + "/ferrocell.png", dpi=300, transparent=True)
    plt.close()


if __name__ == "__main__":
    run_ferrocell()
