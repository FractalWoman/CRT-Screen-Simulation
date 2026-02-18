# Copyright © [2026] Lori Gardi
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files, to use, copy, modify, and distribute the software for
# educational and research purposes.
#
# This software is provided "as is", without warranty of any kind.
#
# Python 3.11
# Magpylib 4.4.0

import numpy as np
import magpylib as mag
import math
from PIL import Image

# --------------------------
# PARAMETERS
# --------------------------

# Physical parameters for the magnet
mag_dim = 0.004              # Magnet diameter in meters
mag_radius = mag_dim / 2      # Magnet radius

region_factor = 1.6           # Factor to extend the sampling region around the magnet
voxels_per_axis = 1000        # Resolution of the simulated CRT grid (1000x1000)

# Magnetic field visualization parameters
B_ref = 0.04                  # Reference field magnitude for normalizing brightness
B_threshold = 1e-6            # Minimum field magnitude to be displayed
orientation_axis = 2           # Z-component is used for the CRT image

# Frame settings
N_FRAMES = 4                  # Number of frames (rotated views of the magnet)

# Color parameters
COLOR_CYCLE_SCALE = 275       # Scale factor for RGB cycling based on field magnitude
GAMMA = 0.3                   # Power-law exponent for CRT-like fade (less than 1 = slower fade)

# Distortion parameters (Lorentz-style nonlinear electron beam deflection)
DISTORT_SCALE = 0.012         # Magnitude of distortion applied to the beam
NONLINEAR_ALPHA = 18.0        # Nonlinearity factor for distortion

# --------------------------
# GRID SETUP
# --------------------------

# Define the physical region of the CRT screen
extent = region_factor * mag_dim

# Generate linear arrays along Y and Z axes
y_vals = np.linspace(-extent, +extent, voxels_per_axis)
z_vals = np.linspace(-extent, +extent, voxels_per_axis)

# Create a 2D grid (mesh) representing the CRT screen
Y, Z = np.meshgrid(y_vals, z_vals)

# X-coordinate is fixed (screen plane) at magnet surface
x_fixed = np.full_like(Y, mag_radius)

# Stack grid points into a single array for vectorized field computation
points = np.stack([x_fixed.ravel(), Y.ravel(), Z.ravel()], axis=1)

# --------------------------
# ROTATION FUNCTION
# --------------------------

def rotate_vector(v, axis, angle):
    """
    Rotate a vector `v` around a given `axis` by `angle` radians using Rodrigues' rotation formula.
    Used to simulate the rotating magnet in each frame.
    """
    axis = axis / np.linalg.norm(axis)
    return (
        v * math.cos(angle) +
        np.cross(axis, v) * math.sin(angle) +
        axis * np.dot(axis, v) * (1 - math.cos(angle))
    )

# --------------------------
# INITIAL MAGNETIZATION
# --------------------------

mag_strength = 10  # Arbitrary magnetization magnitude for visualization
initial_mag = np.array([0.0, 0.0, mag_strength])  # Initial vector pointing along Z
rotation_axis = np.array([-1, 0.0, 0.0])           # Axis about which magnet rotates
step = 360 / N_FRAMES                               # Rotation step in degrees per frame

# --------------------------
# EGG-SHAPED VOID PARAMETERS
# --------------------------

# Radii of the egg-shaped void in pixels (Y and Z directions)
egg_radius_y = int(voxels_per_axis * 0.10)
egg_radius_z = int(voxels_per_axis * 0.12)

# Shift applied to egg per frame for realistic CRT displacement
shift_pixels = int(voxels_per_axis * 0.03)

# --------------------------
# FRAME LOOP
# --------------------------

for frame in range(N_FRAMES):
    # --------------------------
    # ROTATE MAGNET FOR CURRENT FRAME
    # --------------------------
    angle = math.radians(frame * step)  # Convert degrees to radians
    mag_vec = rotate_vector(initial_mag, rotation_axis, angle)
    mag_vec /= np.linalg.norm(mag_vec)  # Normalize

    # Create the spherical magnet with rotated magnetization
    sphere_mag = mag.magnet.Sphere(
        magnetization=tuple(mag_vec),
        diameter=mag_dim,
        position=(mag_radius / 10, 0.0, 0.0)  # Slight offset for visualization
    )
    system = mag.Collection(sphere_mag)

    # --------------------------
    # FIRST FIELD EVALUATION (ORIGINAL GRID)
    # --------------------------
    B_all = system.getB(points)  # Compute magnetic field at all screen points

    # Extract and reshape components for grid
    B_x = B_all[:, 0].reshape(voxels_per_axis, voxels_per_axis)
    B_y = B_all[:, 1].reshape(voxels_per_axis, voxels_per_axis)
    B_z = B_all[:, 2].reshape(voxels_per_axis, voxels_per_axis)

    # Compute magnitude (not used directly for display but for distortion)
    B_mag = np.sqrt(B_x**2 + B_y**2 + B_z**2)

    # --------------------------
    # DISTORTION (LORENTZ-STYLE NONLINEAR)
    # --------------------------
    denom = 1 + NONLINEAR_ALPHA * B_mag
    dY = DISTORT_SCALE * (B_z / denom)   # Vertical distortion
    dZ = -DISTORT_SCALE * (B_y / denom)  # Horizontal distortion

    # Apply distortions to the grid
    Y_dist = Y + dY
    Z_dist = Z + dZ

    # Stack distorted points for second field evaluation
    points_dist = np.stack([x_fixed.ravel(), Y_dist.ravel(), Z_dist.ravel()], axis=1)

    # --------------------------
    # SECOND FIELD EVALUATION (WARPED GRID)
    # --------------------------
    B_all = system.getB(points_dist)
    B_component = B_all[:, orientation_axis].reshape(voxels_per_axis, voxels_per_axis)
    B_abs = np.abs(B_component)

    # --------------------------
    # THRESHOLD MASK
    # --------------------------
    valid_mask = B_abs >= B_threshold

    # --------------------------
    # BRIGHTNESS (POWER-LAW)
    # --------------------------
    norm = np.clip(B_abs / B_ref, 0, 1)
    brightness = norm ** GAMMA

    # --------------------------
    # BEAM DENSITY (AREA CHANGE / VOID EFFECT)
    # --------------------------
    dY_dy = np.gradient(dY, axis=0)
    dZ_dz = np.gradient(dZ, axis=1)
    area_change = 1 + dY_dy + dZ_dz
    density = 1 / np.clip(area_change, 0.4, 3.0)  # Prevent extreme values

    # Final intensity combines brightness and density
    intensity = (brightness * density * 255).clip(0, 255).astype(np.uint8)

    # --------------------------
    # PURE RGB CYCLING
    # --------------------------
    phase = np.mod(B_abs * COLOR_CYCLE_SCALE, 2 * math.pi)
    image = np.zeros((voxels_per_axis, voxels_per_axis, 3), dtype=np.uint8)
    red_mask = phase < 2 * math.pi / 3
    green_mask = (phase >= 2 * math.pi / 3) & (phase < 4 * math.pi / 3)
    blue_mask = phase >= 4 * math.pi / 3

    # Assign intensities to each channel
    image[..., 0][valid_mask & red_mask] = intensity[valid_mask & red_mask]
    image[..., 1][valid_mask & green_mask] = intensity[valid_mask & green_mask]
    image[..., 2][valid_mask & blue_mask] = intensity[valid_mask & blue_mask]

    # --------------------------
    # POLARITY SWAP
    # --------------------------
    neg_mask = B_component < 0
    image[neg_mask] = image[neg_mask][:, ::-1]  # Swap colors for negative field

    # --------------------------
    # CHANNEL REORDER (CUSTOM)
    # --------------------------
    image = image[..., [1, 2, 0]]  # Original swap from your first implementation

    # --------------------------
    # APPLY EGG-SHAPED VOID
    # --------------------------
    center_y = voxels_per_axis // 2
    center_z = voxels_per_axis // 2

    if frame == 0:  # horizontal egg, shifted right
        egg_mask = ((np.arange(voxels_per_axis)[:, None] - center_y)**2 / egg_radius_y**2 +
                    (np.arange(voxels_per_axis)[None, :] - (center_z + shift_pixels))**2 / egg_radius_z**2) <= 1
    elif frame == 1:  # vertical egg, shifted down
        egg_mask = ((np.arange(voxels_per_axis)[:, None] - (center_y - shift_pixels))**2 / egg_radius_z**2 +
                    (np.arange(voxels_per_axis)[None, :] - center_z)**2 / egg_radius_y**2) <= 1
    elif frame == 2:  # horizontal egg, shifted left
        egg_mask = ((np.arange(voxels_per_axis)[:, None] - center_y)**2 / egg_radius_y**2 +
                    (np.arange(voxels_per_axis)[None, :] - (center_z - shift_pixels))**2 / egg_radius_z**2) <= 1
    else:  # frame 3 vertical egg, shifted up
        egg_mask = ((np.arange(voxels_per_axis)[:, None] - (center_y + shift_pixels))**2 / egg_radius_z**2 +
                    (np.arange(voxels_per_axis)[None, :] - center_z)**2 / egg_radius_y**2) <= 1

    #image[egg_mask] = 0  # Paint the egg-shaped void black

    # --------------------------
    # SAVE IMAGE
    # --------------------------
    image = np.flipud(image)  # Flip vertically to match screen coordinates
    filename = f"CRT_magnetic_slice_{frame*90:03d}_deg.png"
    Image.fromarray(image).save(filename)
    print(f"Saved {filename}")

print("All PNG images generated.")
