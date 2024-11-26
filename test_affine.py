import numpy as np
from matplotlib import pyplot as plt
from functions import grid_in_plane

np.random.seed(0)

# Pixel data
i = np.arange(11)
I, J = np.meshgrid(i, i)
ijk = np.column_stack((I.ravel(), J.ravel(), np.zeros_like(I.ravel())))
vals = ijk[:,0]*ijk[:,1]*4


# Create rotation matrix based on vector rotation
basis_ijk = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
normal_xyz = np.random.rand(3)
normal_xyz /= np.linalg.norm(normal_xyz)
u = np.array([1, 0, 0]) if abs(normal_xyz[0]) < abs(normal_xyz[1]) else np.array([0, 1, 0])
v = np.cross(normal_xyz, u)
v /= np.linalg.norm(v)
u = np.cross(v, normal_xyz)
u /= np.linalg.norm(u)
basis_xyz = np.column_stack((u, v, normal_xyz))

Q = np.array([[np.dot(basis_xyz[0], basis_ijk[0]), np.dot(basis_xyz[0], basis_ijk[1]), np.dot(basis_xyz[0], basis_ijk[2])],
                [np.dot(basis_xyz[1], basis_ijk[0]), np.dot(basis_xyz[1], basis_ijk[1]), np.dot(basis_xyz[1], basis_ijk[2])],
                [np.dot(basis_xyz[2], basis_ijk[0]), np.dot(basis_xyz[2], basis_ijk[1]), np.dot(basis_xyz[2], basis_ijk[2])]])

# Create pixel dimension scaling matrix
T = np.array([[0.6,0.,0.],
                [0.,0.6,0.],
                [0.,0.,1]])
# T = np.eye(3)
t = np.random.rand(3)*-4
A = np.eye(4)
A[:3, :3] = Q@T
A[:3, 3] = t    # Translation vector

xyz = np.dot(A[:3, :3], ijk.T).T + A[:3, 3]


#%% Triying to figure out how to reconstruct the affine using only the normal vector and the center of the slice
# Basically, we need a matrix such that A@ijk = xyz

# Things that are known
normal = normal_xyz
center = np.mean(xyz, axis=0)
plane_size = 6
pixdim = T[0,0]
basis_ijk = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

# Create a grid in the plane
points, affine = grid_in_plane(center, normal, pixdim, plane_size)


#%% Plot
fig = plt.figure(1, clear=True)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ijk[:, 0], ijk[:, 1], ijk[:, 2], 'bo')   # In-plane grid
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'ro')   # Transformed points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='x')     # Reconstructed from normal and origin

# Plot the basis vectors for ijk
origin = np.zeros((1, 3))
ax.quiver(*origin.T, *basis_ijk[:, 0], color='r', length=1.0, normalize=True, linestyle='dashed')
ax.quiver(*origin.T, *basis_ijk[:, 1], color='g', length=1.0, normalize=True, linestyle='dashed')
ax.quiver(*origin.T, *basis_ijk[:, 2], color='b', length=1.0, normalize=True, linestyle='dashed')

# Plot the basis vectors for xyz
origin += t
ax.quiver(*origin.T, *basis_xyz[:, 0], color='r', length=1.0, normalize=True)
ax.quiver(*origin.T, *basis_xyz[:, 1], color='g', length=1.0, normalize=True)
ax.quiver(*origin.T, *basis_xyz[:, 2], color='b', length=1.0, normalize=True)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()