# Copyright © [2025] Lori Gardi
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
mag_dim = 0.002
mag_radius = mag_dim / 2
region_factor = 5
voxels_per_axis = 1000

B_ref = 0.004                  # reference field for normalization
B_threshold = 1e-6
orientation_axis = 2           # Z component

N_FRAMES = 4                   # 0°, 90°, 180°, 270°

# Colour cycling
COLOR_CYCLE_SCALE = 85

# Power-law compression
GAMMA = 0.4                    # <1 = slow fade (CRT-like)

# --------------------------
# GRID
# --------------------------
extent = region_factor * mag_dim
y_vals = np.linspace(-extent, +extent, voxels_per_axis)
z_vals = np.linspace(-extent, +extent, voxels_per_axis)

x_fixed = mag_radius
sphere_center = np.array((0.0, 0.0, 0.0))

# --------------------------
# HELPERS
# --------------------------
def inside_sphere(p, center, radius):
    return np.linalg.norm(p - center) <= radius

def rotate_vector(v, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return (
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * np.dot(axis, v) * (1 - math.cos(angle))
    )

# --------------------------
# INITIAL MAGNETIZATION
# --------------------------
initial_mag = np.array([0.0, 0.0, 1.0])
rotation_axis = np.array([1.0, 0.0, 0.0])
step = 360 / N_FRAMES

# --------------------------
# FRAME LOOP
# --------------------------
for frame in range(N_FRAMES):

    angle = math.radians(frame * step)
    mag_vec = rotate_vector(initial_mag, rotation_axis, angle)
    mag_vec /= np.linalg.norm(mag_vec)

    sphere_mag = mag.magnet.Sphere(
        magnetization=tuple(mag_vec),
        diameter=mag_dim,
        position=(0.0, 0.0, 0.0)
    )
    system = mag.Collection(sphere_mag)

    image = np.zeros((voxels_per_axis, voxels_per_axis, 3), dtype=np.uint8)

    for iy, y in enumerate(y_vals):
        for iz, z in enumerate(z_vals):

            p = np.array([x_fixed, y, z])

            # --------------------------
            # MAGNET CORE
            # --------------------------
            if inside_sphere(p, sphere_center, mag_radius):
                polarity = np.dot(p - sphere_center, mag_vec)
                image[iz, iy] = [255, 0, 0] if polarity >= 0 else [0, 0, 255]
                continue

            # --------------------------
            # FIELD
            # --------------------------
            B_vec = system.getB(p)
            B = B_vec[orientation_axis]
            B_abs = abs(B)

            if B_abs < B_threshold:
                continue

            # --------------------------
            # POWER-LAW BRIGHTNESS
            # --------------------------
            norm = min(B_abs / B_ref, 1.0)
            brightness = norm ** GAMMA
            intensity = int(brightness * 255)

            if intensity == 0:
                continue

            # --------------------------
            # PURE RGB CYCLING
            # --------------------------
            phase = (B_abs * COLOR_CYCLE_SCALE) % (2 * math.pi)

            if phase < 2 * math.pi / 3:
                color = [intensity, 0, 0]
            elif phase < 4 * math.pi / 3:
                color = [0, intensity, 0]
            else:
                color = [0, 0, intensity]

            # Polarity swap
            if B < 0:
                color = color[::-1]

            image[iz, iy] = color

    image = np.flipud(image)

    filename = f"CRT_magnetic_slice_{frame*90:03d}_deg.png"
    Image.fromarray(image).save(filename)
    print(f"Saved {filename}")

print("All PNG images generated.")
