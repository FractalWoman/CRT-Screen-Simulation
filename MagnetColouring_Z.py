import numpy as np
import magpylib as mag
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# -----------------------------------------
# PARAMETERS
# -----------------------------------------
grid_n = 19              # grid resolution (20x20)
plane_y = 0.0            # x–z plane
region = 0.05            # +/- region in meters
arrow_len = 0.003        # constant arrow length
min_norm = 1e-12

magnet_dim = (0.01, 0.01, 0.01)  # 1 cm cube

# -----------------------------------------
# MAGNET
# -----------------------------------------
magnet = mag.magnet.Cuboid(
    magnetization=(, 0, 0),
    dimension=magnet_dim,
    position=(0, 0, 1)
)

system = mag.Collection(magnet)

# -----------------------------------------
# UNIFORM GRID IN X–Z PLANE
# -----------------------------------------
xs = np.linspace(-region, region, grid_n)
zs = np.linspace(-region, region, grid_n)
XX, ZZ = np.meshgrid(xs, zs)

flat_x = XX.ravel()
flat_z = ZZ.ravel()
flat_y = np.full_like(flat_x, plane_y)

points = np.vstack([flat_x, flat_y, flat_z]).T

# -----------------------------------------
# REMOVE POINTS INSIDE MAGNET
# -----------------------------------------
dx, dy, dz = np.array(magnet_dim) / 2

inside = (
    (np.abs(flat_x) <= dx) &
    (np.abs(flat_z) <= dz)
)

points = points[~inside]
flat_x = flat_x[~inside]
flat_z = flat_z[~inside]

# -----------------------------------------
# B-FIELD (DIRECTION ONLY)
# -----------------------------------------
B = system.getB(points)
Bx = B[:, 0]
Bz = B[:, 2]

norm = np.sqrt(Bx**2 + Bz**2)
norm = np.maximum(norm, min_norm)

Ux = (Bx / norm) * arrow_len
Uz = (Bz / norm) * arrow_len

# -----------------------------------------
# DRAW
# -----------------------------------------
fig = plt.figure(figsize=(8, 8), dpi=500)
ax = fig.add_subplot(111)
ax.set_facecolor("white")

# Hollow arrows (coloring book style)
for x, z, u, w in zip(flat_x, flat_z, Ux, Uz):
    arrow = FancyArrowPatch(
        (x, z),
        (x + u, z + w),
        arrowstyle='-|>',
        linewidth=0.6,
        edgecolor='black',
        facecolor='none',
        mutation_scale=40
    )
    ax.add_patch(arrow)

# -----------------------------------------
# MAGNET (TOP RED, BOTTOM BLUE)
# -----------------------------------------
ax.fill(
    [-dx, dx, dx, -dx],
    [0, 0, dz, dz],
    color="red",
    alpha=1.0
)

ax.fill(
    [-dx, dx, dx, -dx],
    [-dz, -dz, 0, 0],
    color="blue",
    alpha=1.0
)

# -----------------------------------------
# FINAL FORMATTING
# -----------------------------------------
ax.set_aspect("equal")
ax.set_xlim(-region, region)
ax.set_ylim(-region, region)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_title("Fun with Fields")

plt.savefig("magnetic_field_coloring_book_page2.png", dpi=500)
plt.close()

print("Saved magnetic_field_coloring_book.png")
