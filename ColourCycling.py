#Copyright © [2025] Lori Gardi

#Permission is hereby granted, free of charge, to any person
#obtaining a copy of this software and associated documentation
#files, to use, copy, modify, and distribute the software for
#educational and research purposes.
#This software is provided "as is", without warranty of any kind.
#This implementation is an independent work and does not claim
#ownership of any external ideas, methods, or experimental results.

#This software is a computer simulation of what you would see when 
#hold a magnet up against a CRT screen. The magnet is rotated 360 degrees.
#An image is stored for each angle. An mp4 video is generated showing
#an animation of the CRT display as the magnet is rotated.

#This code works with Python 3.11 and MagPyLib 4.4.0

import numpy as np
import magpylib as mag
import imageio
import math

# --------------------------
# PARAMETERS
# --------------------------
mag_dim = 0.002                 # sphere diameter (m)
mag_radius = mag_dim / 2
region_factor = 5
voxels_per_axis = 500

B_max_clamp = 0.004             # for brightness scaling
B_threshold = 1e-5
orientation_axis = 2            # Z component

N_FRAMES = 36                   # frames per rotation
FPS = 25

# Colour cycling scale
COLOR_CYCLE_SCALE = 85           # higher = more RGB bands

# --------------------------
# GRID
# --------------------------
extent = region_factor * mag_dim
x_vals = np.linspace(-extent, +extent, voxels_per_axis)
y_vals = np.linspace(-extent, +extent, voxels_per_axis)
z_vals = np.linspace(-extent, +extent, voxels_per_axis)

# Slice at magnet edge
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
# MP4 WRITER
# --------------------------
writer = imageio.get_writer(
    "rotating_sphere_magnet_CRT_RGB.mp4",
    fps=FPS,
    codec="libx264",
    pixelformat="yuv420p"
)

# --------------------------
# ANIMATION LOOP
# --------------------------
for frame in range(N_FRAMES):
    angle = math.radians(frame * step)
    mag_vec = rotate_vector(initial_mag, rotation_axis, angle)
    mag_vec /= np.linalg.norm(mag_vec)

    # Create magnet
    sphere_mag = mag.magnet.Sphere(
        magnetization=tuple(mag_vec),
        diameter=mag_dim,
        position=tuple(sphere_center)
    )
    magnet_system = mag.Collection(sphere_mag)

    # Image buffer
    image = np.zeros((voxels_per_axis, voxels_per_axis, 3), dtype=np.uint8)

    for iy, y in enumerate(y_vals):
        for iz, z in enumerate(z_vals):
            p = np.array([x_fixed, y, z])

            # Magnet core
            if inside_sphere(p, sphere_center, mag_radius):
                polarity = np.dot(p - sphere_center, mag_vec)
                if polarity >= 0:
                    image[iz, iy] = [255, 0, 0]  # pure red core
                else:
                    image[iz, iy] = [0, 0, 255]  # pure blue core
                continue

            # Field vector
            B_vec = magnet_system.getB(p)
            B = B_vec[orientation_axis]

            if abs(B) < B_threshold:
                continue

            # Brightness scaling (clamped)
            brightness = min(abs(B) / B_max_clamp, 1.0)
            intensity = int(brightness * 255)

            # Phase for RGB cycling (unclamped)
            phase = abs(B) * COLOR_CYCLE_SCALE
            phase = phase % (2 * math.pi)  # wrap 0→2pi

            # Hard RGB selection based on phase
            if phase < 2 * math.pi / 3:
                color = [intensity, 0, 0]       # RED
            elif phase < 4 * math.pi / 3:
                color = [0, intensity, 0]       # GREEN
            else:
                color = [0, 0, intensity]       # BLUE

            # Polarity flip
            if B < 0:
                color = color[::-1]  # swap R <-> B

            image[iz, iy] = color

    image = np.flipud(image)
    writer.append_data(image)
    print(f"Frame {frame+1}/{N_FRAMES}")

writer.close()
print("MP4 saved as rotating_sphere_magnet_CRT_RGB.mp4")



