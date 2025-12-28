import numpy as np
import magpylib as mag
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ============================================================
# PARAMETERS
# ============================================================
plane_y = 0.0
region = 0.03               # half-width of plot region (m)
levels = 32                 # number of equipotential lines
arrow_len = 0.003
min_norm = 1e-12

sphere_diameter = 0.01
sphere_radius = sphere_diameter / 2

# Hex/triangular arrow grid
nx = 30
nz = 30
dx = 0.003
dz = dx * np.sqrt(3)/2

# ============================================================
# SPHERICAL MAGNET WITH STRONGER FIELD
# ============================================================
magnetization_strength = 2.0  # double the field
magnet = mag.magnet.Sphere(
    magnetization=(0, 0, magnetization_strength),
    diameter=sphere_diameter,
    position=(0, 0, 0)
)
system = mag.Collection(magnet)

# ============================================================
# HIGH-RES GRID FOR ANALYTIC SCALAR POTENTIAL Φ
# ============================================================
grid_n_potential = 200
x = np.linspace(-region, region, grid_n_potential)
z = np.linspace(-region, region, grid_n_potential)
X, Z = np.meshgrid(x, z)
Y = np.full_like(X, plane_y)

points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
inside = (X**2 + Z**2) <= sphere_radius**2

# Magnetic moment of sphere
m = np.array([0, 0, magnetization_strength]) * (4/3 * np.pi * sphere_radius**3)

r = points
r_norm = np.linalg.norm(r, axis=1)
r_norm[r_norm < sphere_radius] = np.nan
Phi = np.sum(r * m, axis=1) / (4 * np.pi * r_norm**3)
Phi = Phi.reshape(X.shape)
Phi = np.ma.array(Phi, mask=inside)

# Rescale contour levels to full range
phi_min = np.nanmin(Phi)
phi_max = np.nanmax(Phi)
contour_levels = np.linspace(phi_min, phi_max, levels)

# ============================================================
# HEXAGONAL GRID FOR FIELD ARROWS
# ============================================================
x_coords = []
z_coords = []

for i in range(nz):
    shift = dx/2 if i % 2 == 1 else 0
    for j in range(nx):
        x_coords.append(j * dx + shift)
        z_coords.append(i * dz)

x_coords = np.array(x_coords) - (nx-1)*dx/2
z_coords = np.array(z_coords) - (nz-1)*dz/2
flat_x = x_coords
flat_z = z_coords
flat_y = np.full_like(flat_x, plane_y)

arrow_points = np.vstack([flat_x, flat_y, flat_z]).T
inside_arrow = (flat_x**2 + flat_z**2) <= sphere_radius**2
arrow_points = arrow_points[~inside_arrow]
flat_x = flat_x[~inside_arrow]
flat_z = flat_z[~inside_arrow]

B = system.getB(arrow_points)
Bx = B[:, 0]
Bz = B[:, 2]
norm = np.sqrt(Bx**2 + Bz**2)
norm = np.maximum(norm, min_norm)
Ux = (Bx / norm) * arrow_len
Uz = (Bz / norm) * arrow_len

# ============================================================
# DRAW FIGURE
# ============================================================
fig = plt.figure(figsize=(8, 8), dpi=500)
ax = fig.add_subplot(111)
ax.set_facecolor("white")

# Equipotential contours
ax.contour(
    X, Z, Phi,
    levels=contour_levels,
    colors="black",
    linewidths=1.0,
    linestyles="solid"
)

# Field arrows centered around midpoint
for x0, z0, u, w in zip(flat_x, flat_z, Ux, Uz):
    dx_arrow = u / 2
    dz_arrow = w / 2
    start = (x0 - dx_arrow, z0 - dz_arrow)
    end   = (x0 + dx_arrow, z0 + dz_arrow)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle='-|>',
        linewidth=0.6,
        edgecolor='black',
        facecolor='none',
        mutation_scale=20
    )
    ax.add_patch(arrow)

# ============================================================
# WHITE CIRCLE BOUNDARY AROUND THE MAGNET
# ============================================================
circle_boundary = plt.Circle(
    (0, 0), sphere_radius,
    color='black',
    fill=False,
    linewidth=4.0,
    zorder=6
)
ax.add_patch(circle_boundary)

# ============================================================
# SPHERE COLORING
# ============================================================
theta = np.linspace(0, np.pi, 200)
# North pole
x_north = sphere_radius * np.cos(theta)
z_north = sphere_radius * np.sin(theta)
ax.fill(x_north, z_north, color='red', zorder=5)
# South pole
x_south = sphere_radius * np.cos(theta)
z_south = -sphere_radius * np.sin(theta)
ax.fill(x_south, z_south, color='blue', zorder=5)

# ============================================================
# FINAL FORMATTING
# ============================================================
ax.set_aspect("equal")
ax.set_xlim(-region, region)
ax.set_ylim(-region, region)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_title("Stronger Magnetic Field & Expanded Equipotentials (Hex Grid)")

plt.savefig("sphere_field_equipotentials_strong_hexgrid_boundary.png", dpi=500)
plt.close()
print("Saved sphere_field_equipotentials_strong_hexgrid_boundary.png")
