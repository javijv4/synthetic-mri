import numpy as np
from matplotlib import pyplot as plt


i = np.arange(10)
I, J = np.meshgrid(i, i)
ijk = np.column_stack((I.ravel(), J.ravel(), np.zeros_like(I.ravel())))


affine = np.array([[1, 0, 0, 0],
                   [0, 0.5, 0, 0],
                   [0, 0, 1, 10],
                   [0, 0, 0, 1]])
A = affine[:3, :3]

theta = np.radians(45)
c, s = np.cos(theta), np.sin(theta)
R = np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]])

affine[:3, :3] = np.dot(A, R)



xyz = np.dot(affine[:3, :3], ijk.T).T + affine[:3, 3]


fig = plt.figure(1, clear=True)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ijk[:, 0], ijk[:, 1], ijk[:, 2], 'o')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
