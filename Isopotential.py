import numpy as np
import magpylib as mag
import matplotlib.pyplot as plt

# -----------------------------------------
# PARAMETERS
# -----------------------------------------
grid_n = 200
plane_y = 0.0000
region = 0.02
levels = 32

magnet_dim = (0.01, 0.01, 0.01)

# -----------------------------------------
# MAGNET (N UP, S DOWN)
# -----------------------------------------
magnet = mag.magnet.Cuboid(
    magnetization=(0, 0, 1),   # S → N
    dimension=magnet_dim,
    position=(0, 0, 0)
)

system = mag.Collection(magnet)

# -----------------------------------------
# GRID (X–Z PLANE)
# -----------------------------------------
x = np.linspace(-region, region, grid_n)
z = np.linspace(-region, region, grid_n)
X, Z = np.meshgrid(x, z)
Y = np.full_like(X, plane_y)

points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# -----------------------------------------
# MAGNET MASK
# -----------------------------------------
dx, dy, dz = np.array(magnet_dim) / 2

inside = (
    (np.abs(X) <= dx) &
    (np.abs(Z) <= dz)
)

# -----------------------------------------
# H FIELD
# -----------------------------------------
H = system.getH(points)
Hx = H[:, 0].reshape(X.shape)
Hz = H[:, 2].reshape(Z.shape)

# -----------------------------------------
# SCALAR POTENTIAL Φ (SYMMETRIC INTEGRATION)
# -----------------------------------------
dxg = x[1] - x[0]
dzg = z[1] - z[0]

Phi = np.zeros_like(X)

# integrate from lower-left corner
for i in range(1, grid_n):
    Phi[i, 0] = Phi[i-1, 0] - Hz[i-1, 0] * dzg

for j in range(1, grid_n):
    Phi[:, j] = Phi[:, j-1] - Hx[:, j-1] * dxg

Phi = np.ma.array(Phi, mask=inside)

# -----------------------------------------
# DRAW
# -----------------------------------------
fig = plt.figure(figsize=(8, 8), dpi=500)
ax = fig.add_subplot(111)
ax.set_facecolor("white")

ax.contour(
    X, Z, Phi,
    levels=levels,
    colors="black",
    linewidths=0.7,
    linestyles="solid"   # explicitly force solid lines
)

# -----------------------------------------
# MAGNET (TOP RED, BOTTOM BLUE)
# -----------------------------------------
ax.fill(
    [-dx, dx, dx, -dx],
    [0, 0, dz, dz],
    color="red"
)

ax.fill(
    [-dx, dx, dx, -dx],
    [-dz, -dz, 0, 0],
    color="blue"
)

# -----------------------------------------
# FINAL FORMATTING
# -----------------------------------------
ax.set_aspect("equal")
ax.set_xlim(-region, region)
ax.set_ylim(-region, region)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_title("Magnetic Scalar Potential – Equipotential Curves")

plt.savefig("magnetic_equipotential_coloring_page.png", dpi=500)
plt.close()

print("Saved magnetic_equipotential_coloring_page.png")
