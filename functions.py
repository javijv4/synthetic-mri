import os
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
import monai.transforms as mt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



def readFromNIFTI(segName, correct_ras=True):
    ''' Helper function used by masks2ContoursSA() and masks2ContoursLA(). Returns (seg, transform, pixSpacing). '''
    # Load NIFTI image and its header.
    if os.path.isfile(segName):
        ext = ''
    elif os.path.isfile(segName + '.nii.gz'):
        ext = '.nii.gz'
    elif os.path.isfile(segName + '.nii'):
        ext = '.nii'
    else:
        raise FileNotFoundError('File {} was not found'.format(segName))

    img = mt.LoadImage(image_only=True)(segName + ext)
    seg = img.numpy().astype(float)
    transform = img.affine.numpy()
    pixdim = img.pixdim.numpy()

    return (seg, transform, pixdim)


def calculate_spatial_information(label_data, affine, LV_LABEL=1, RV_LABEL=3, AORTA_LABEL=6):
    lv_centroid = calculate_centroid(label_data, LV_LABEL, affine)
    rv_centroid = calculate_centroid(label_data, RV_LABEL, affine)
    aorta_centroid = calculate_centroid(label_data, AORTA_LABEL, affine)
    lv_long_axis = calculate_lv_long_axis(label_data, LV_LABEL, affine)

    spatial_info = {
        "LV Centroid": lv_centroid,
        "RV Centroid": rv_centroid,
        "LV Long Axis": lv_long_axis,
        "Aorta Centroid": aorta_centroid
    }

    return spatial_info


def grid_in_plane(origin, normal, spacing, plane_size):
    """
    Generates a grid of points in a plane defined by an origin and a normal vector.
    Note that the grid is centered at the origin and the points are spaced the same in both in-plane directions.
    You want to make sure the grid is large enough to cover the entire image (plane_size argument).

    Args:
        origin (numpy.ndarray): A 3-element numpy array representing the coordinates of the origin point of the plane.
        normal (numpy.ndarray): A 3-element numpy array representing the normal vector of the plane.
        plane_size (float): The size of the plane in pixel units

    Returns:
        numpy.ndarray: A 2D array of shape (N, 3) where each row represents the coordinates (in pixel units) of a point in the plane.
    """
    # NORMAL VECTOR IS PERPENDICULAR TO A PLANE OR A SURFACE, and its used to define orientation of plane
    # orthogonal vector check if two VECTORS are perpendicular to each other: if a dot product b = 0 they are orthogonal
    # conver to normal unit vector using numpys built in functions
    normal = normal / np.linalg.norm(normal)

    u = np.array([1, 0, 0]) if abs(normal[0]) < abs(normal[1]) else np.array([0, 1, 0])
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)   # Normalizing vector
    u = np.cross(v, normal)
    u = u / np.linalg.norm(u)   # Normalizing vector
    basis_xyz = np.column_stack((u, v, normal)) # Create basis vectors for the plane

    #linespace generates sequence of of evenly spaced numbers of range
    # creates planesize amount of evenly spaced values from -half_size to half_size
    # creates a 1d array of evenly spaced points
    npoints = int(np.ceil(plane_size / spacing)) + 1  # Calculate number of points to cover plane_size
    lin_space = np.arange(npoints)                    # this ensures the spacing is correct

    # .meshgrid produces coordinate matrices from coordinate vectors
    # takes 1d array of evenly spaced points and combines into 2d grid
    grid_i, grid_j = np.meshgrid(lin_space, lin_space)
    basis_ijk = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    ijk = np.column_stack((grid_i.ravel(), grid_j.ravel(), np.zeros_like(grid_i.ravel())))

    # Create the rotation matrix (this is a formula I learned in undergrad, I can't find a reference for it, but it works!)
    Q = np.array([[np.dot(basis_xyz[0], basis_ijk[0]), np.dot(basis_xyz[0], basis_ijk[1]), np.dot(basis_xyz[0], basis_ijk[2])],
                    [np.dot(basis_xyz[1], basis_ijk[0]), np.dot(basis_xyz[1], basis_ijk[1]), np.dot(basis_xyz[1], basis_ijk[2])],
                    [np.dot(basis_xyz[2], basis_ijk[0]), np.dot(basis_xyz[2], basis_ijk[1]), np.dot(basis_xyz[2], basis_ijk[2])]])


    # Step 1: Scale the plane to the desired spacing (scaling first avoids shearing deformations)
    S = np.eye(3)
    S[0,0] = spacing
    S[1,1] = spacing
    S[2,2] = 1
    ijk_scaled = np.dot(S, ijk.T).T

    # Step 2: Rotate the grid points to align with the desired plane
    ijk_scaled_rot = np.dot(Q, ijk_scaled.T).T

    # Step 3: Find the translation that centers the rotated plane at the origin
    ijk_rot_scaled_center = np.mean(ijk_scaled_rot, axis=0)
    t = origin - ijk_rot_scaled_center

    # Put all the transformations into a single affine matrix
    A = np.eye(4)
    A[:3, :3] = Q@S
    A[:3, 3] = t

    # Compute 3D coordinates for each point on the
    #  creates array of shape (N, N, 3), where each row is a 3D point in the plane centered on origin
    # grid x and y determine x and y cords while u and v orient cords in 3d space to essentially create grind
    points = (A[:3, :3] @ ijk.T).T + A[:3, 3] # This should be the same as xyz

    #  2D array of shape (N, 3) where each row represents the coordinates (in pixel units) of a point in the plane.
    return points, A

def grid_in_plane2(origin, normal, spacing, plane_size):
    # TODO: instead of the number of points, the function should receive the spacing between points
    """
    Generates a grid of points in a plane defined by an origin and a normal vector.
    Note that the grid is centered at the origin and the points are spaced the same in both in-plane directions.
    You want to make sure the grid is large enough to cover the entire image (plane_size argument).

    Args:
        origin (numpy.ndarray): A 3-element numpy array representing the coordinates of the origin point of the plane.
        normal (numpy.ndarray): A 3-element numpy array representing the normal vector of the plane.
        plane_size (float): The size of the plane in pixel units

    Returns:
        numpy.ndarray: A 2D array of shape (N, 3) where each row represents the coordinates (in pixel units) of a point in the plane.
    """

    # IMPORTNAT CALC 3 REMINDERS:
    # NORMAL VECTOR IS PERPENDICULAR TO A PLANE OR A SURFACE, and its used to define orientation of plane
    # orthogonal vector check if two VECTORS are perpendicular to each other: if a dot product b = 0 they are orthogonal
    # conver to normal unit vector using numpys built in functions
    normal = normal / np.linalg.norm(normal)
    if not np.allclose(normal, [1, 0, 0]):
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)   # Normalizing vector
    half_size = plane_size / 2     # Create grid points within the plane

    #linespace generates sequence of of evenly spaced numbers of range
    # creates planesize amount of evenly spaced values from -half_size to half_size
    # creates a 1d array of evenly spaced points
    npoints = int(np.ceil(plane_size / spacing))  # Calculate number of points to cover plane_size
    lin_space = np.linspace(-half_size, half_size, npoints)

    grid_x, grid_y = np.meshgrid(lin_space, lin_space)

    # Compute 3D coordinates for each point on the
    # creates array of shape (N, N, 3), where each row is a 3D point in the plane centered on origin
    # grid x and y determine x and y cords while u and v orient cords in 3d space to essentially create grind
    points = origin + grid_x[..., None] * u + grid_y[..., None] * v

    #  2D array of shape (N, 3) where each row represents the coordinates (in pixel units) of a point in the plane.
    return points.reshape(len(lin_space) * len(lin_space), 3)
    # return points


def interpolate_image(xyz, data, ct_affine):
    """
    Interpolates the given image data based on the provided coordinates.

    Args:
        xyz (tuple of float): The coordinates in the form of (x, y, z) that need to be converted to integer indices.
        data (numpy.ndarray): The image data to be interpolated.

    Returns:
        numpy.ndarray: The interpolated image data with integer values.
    """
    # create voxel coordinates based of real world cords, every ijk now corresponds to a position in ct_affine
    # A voxel coordinate refers to the position of a voxel (3D pixel) in the index space of a 3D image array, typically in integer units that specify its location along each axis in the array.
    #https://nipy.org/nibabel/reference/nibabel.affines.html#nibabel.affines.apply_affine
    ijk = nib.affines.apply_affine(np.linalg.inv(ct_affine), xyz)
    # round everything to integers for floating point number, indexing array doesnt take floating point
    ijk = np.round(ijk).astype(int)

    # Check points outside box
    ijk[:, 0] = np.clip(ijk[:, 0], 0, data.shape[0] - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, data.shape[1] - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, data.shape[2] - 1)

    # Sample data at the nearest indices
    # ijk[:, 0] grabs first column, ijk[:, 1] grabs second column, ijk[:, 2] grabs third column,
    # first col is the x, 2nd is y
    # effectively create three arrays of cords which is x,y,z of all points we want to sample
    interpolated_data = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    # print("Transformed ijk indices:", ijk[:10])

    # Reshape to a 2D grid so we can use in imshow()
    N = int(np.sqrt(len(xyz)))
    # trying to flip the y axis
    # slice_data = interpolated_data.reshape(N, N)
    # slice_data = np.flipud(slice_data)  # Flip along the y-axis
    # return slice_data
    return interpolated_data.reshape(N,N)

def probe_CT_label_image (xyz, data, ct_affine):
    ijk = nib.affines.apply_affine(np.linalg.inv(ct_affine), xyz)
    ijk = np.round(ijk).astype(int)

    # Clip indices to stay within bounds of the data array
    ijk[0] = np.clip(ijk[0], 0, data.shape[0] - 1)
    ijk[1] = np.clip(ijk[1], 0, data.shape[1] - 1)
    ijk[2] = np.clip(ijk[2], 0, data.shape[2] - 1)
    return int(data[ijk[0], ijk[1], ijk[2]])

def save_Nifti(data, affine, file_name):
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, file_name)

def calculate_centroid(label_data, label_value, affine):
    # Calculates the centroid of a specific label in the segmented data.

    # Args:
    #     label_data (numpy.ndarray): The 3D label data where each structure has a unique integer label.
    #     label_value (int): The label value representing the structure (e.g., LV, RV, Aorta).
    #     affine (numpy.ndarray): The affine matrix for converting voxel coordinates to real-world coordinates.

    # Returns:
    #     numpy.ndarray: The (x, y, z) coordinates of the centroid in real-world space.
    # Get coordinates of all voxels with the given label
    coords = np.argwhere(label_data == label_value)
    if coords.size == 0:
        return None

    # Calculate centroid in voxel coordinates
    centroid_voxel = np.mean(coords, axis=0)

    # Convert to real-world coordinates using the affine matrix
    centroid_real = nib.affines.apply_affine(affine, centroid_voxel)
    return centroid_real

def calculate_lv_long_axis(label_data, lv_label, affine):
    """
    Calculates the long axis of the LV using PCA.
    """
    # takes an array and returns the indices where the specified condition is True
    coords = np.argwhere(label_data == lv_label)
    if coords.size == 0:
        return None

    # create instance of PCA class with 3 principal components of xyz
    pca = PCA(n_components=3)
    pca.fit(coords)

    # The first principal component (i.e., the first vector produced by PCA) points in the direction of the maximum variance in the dataset. For the LV region:
    # The first principal component corresponds to the longest dimension of the LV shape.
    # This is why you can use it to estimate the LV long axis.
    long_axis_voxel = pca.components_[0]

    #     explained_variance = pca.explained_variance_ratio_
    # print("Explained Variance Ratio:", explained_variance)
    #  convert the LV long axis vector from voxel space to real-world coordinates using the affine transformation matrix.

    # affine[:3, :3] extracts the top-left 3x3 submatrix of the affine matrix.
# This submatrix represents the rotation and scaling but not the translation.
    long_axis_real = affine[:3, :3] @ long_axis_voxel
    return long_axis_real


"""  PLOTTING FUNCTIONS  """

def plot_short_axis(label_data, affine, lv_centroid, lv_long_axis, plane_size=100, spacing=1.0,
                    out_of_plane_spacing=8.0, number_of_slices=13, plotOn = True, output_file="short_axis.nii"):
    # TODO: make number_of_slices spaced out_of_plane_spacing and return a 3D array. Note that the affine matrix should be the same for all slices.
    # However, the grid_in_plane function will return an affine with a different translation vector for each slice.
    # You will need to modify the affine such that it can transform the 3D array into the correct position.
    if lv_centroid is None or lv_long_axis is None:
        print("Invalid LV centroid or long axis.")
        return None

    slices = []
    slice_affines = []

    for sliceIndex in range(number_of_slices): 
        slice_origin = lv_centroid + (sliceIndex - number_of_slices // 2) * out_of_plane_spacing * lv_long_axis # print(f"Slice {sliceIndex} Origin: {slice_origin}")
        xyz, slice_affine = grid_in_plane(slice_origin, lv_long_axis, spacing, plane_size)                      # print(f"Slice {sliceIndex}: xyz Grid = {xyz[:5]}") 
        slice_affine[:3, 3] = slice_origin
        slice_affines.append(slice_affine)
        slice_data = interpolate_image(xyz, label_data, affine)
        slices.append(slice_data)
        
    slices_3d = np.stack(slices, axis=0)
    middle_affine = slice_affines[number_of_slices // 2]
    unified_affine = affine.copy()
    unified_affine[:3, :3] = middle_affine[:3, :3]
    unified_affine[:3, 3] = middle_affine[:3, 3]

    # Save as a NIfTI file
    nifti_img = nib.Nifti1Image(slices_3d, unified_affine)
    nib.save(nifti_img, output_file)

    if plotOn:
        middle_index = number_of_slices // 2
        plt.figure(figsize=(6, 6))
        plt.imshow(slices_3d[middle_index], cmap='gray', origin='lower')
        plt.title(f"Middle Slice (Slice {middle_index + 1})")
        plt.axis('off')
        plt.show()

    return slices_3d, slice_affines

def plot_2_chamber_view(label_data, affine, lv_centroid, rv_centroid, plane_size=100, spacing=1.0, 
                        out_of_plane_spacing=8.0, number_of_slices=13, plotOn = True):

    if lv_centroid is None or rv_centroid is None:
        print("Invalid LV or RV centroid.")
        return None

    slices = []
    slice_affines = []
    v = rv_centroid - lv_centroid
    v = v / np.linalg.norm(v)

    for slice_index in range(number_of_slices):
        slice_origin = lv_centroid + (slice_index - number_of_slices // 2) * out_of_plane_spacing * v  # print(f"Slice {sliceIndex} Origin: {slice_origin}")
        xyz, slice_affine = grid_in_plane(slice_origin, v, spacing, plane_size) # print(f"Slice {sliceIndex}: xyz Grid = {xyz[:5]}") 
        slice_affine[:3, 3] = slice_origin
        slice_data = interpolate_image(xyz, label_data, affine)
        slices.append(slice_data)
        slice_affines.append(slice_affine)
    slices_3d = np.stack(slices, axis=0)

    if plotOn:
        middle_index = number_of_slices // 2
        plt.figure(figsize=(6, 6))
        plt.imshow(slices_3d[middle_index], cmap='gray', origin='lower')
        plt.title(f"Middle Slice (Slice {middle_index + 1})")
        plt.axis('off')
        plt.show()

    return slices_3d, slice_affine


def main_2chamber_view(seg_file, plane_size=100, spacing=1.0, out_of_plane_spacing=8.0, number_of_slices=13, plotOn = True):
    label_data, affine, _ = readFromNIFTI(seg_file)

    spatial_info = calculate_spatial_information(label_data, affine)

    lv_centroid = spatial_info["LV Centroid"]
    rv_centroid = spatial_info["RV Centroid"]

    if lv_centroid is None or rv_centroid is None:
        print("Could not compute LV or RV centroids.")
        return None

    print("LV Centroid:", lv_centroid)
    print("RV Centroid:", rv_centroid)

    return plot_2_chamber_view(label_data, affine, lv_centroid, rv_centroid, plane_size, spacing, out_of_plane_spacing, number_of_slices, plotOn)




def plot_4_chamber_view(label_data, affine, lv_centroid, rv_centroid, lv_long_axis, plane_size=100, spacing=1.0, 
                         out_of_plane_spacing=8.0, number_of_slices=13, plotOn = True):
    if lv_centroid is None or rv_centroid is None:
        print("Invalid LV or RV centroid.")
        return None, None
    
    slices = []
    slice_affines = []
    v = rv_centroid - lv_centroid
    v = v / np.linalg.norm(v)
    x = np.cross(lv_long_axis, v)
    x = x / np.linalg.norm(x)

    for slice_index in range(number_of_slices):
        slice_origin = lv_centroid + (slice_index - number_of_slices // 2) * out_of_plane_spacing * x
        xyz, slice_affine = grid_in_plane(slice_origin, x, spacing, plane_size)
        slice_affine[:3, 3] = slice_origin
        slice_data = interpolate_image(xyz, label_data, affine)
        slices.append(slice_data)
        slice_affines.append(slice_affine)
    slices_3d = np.stack(slices, axis=0)

    if plotOn:
        middle_index = number_of_slices // 2
        plt.figure(figsize=(6, 6))
        plt.imshow(slices_3d[middle_index], cmap='gray', origin='lower')
        plt.title(f"Middle Slice (Slice {middle_index + 1})")
        plt.axis('off')
        plt.show()

    return slice_data, slice_affines

def main_4chamber_view(seg_file, plane_size=100, spacing=1.0, out_of_plane_spacing = 8.0, number_of_slices = 13, plotOn = True):
    label_data, affine, _ = readFromNIFTI(seg_file)

    spatial_info = calculate_spatial_information(label_data, affine)

    lv_centroid = spatial_info["LV Centroid"]
    rv_centroid = spatial_info["RV Centroid"]
    lv_long_axis = spatial_info["LV Long Axis"]

    if lv_centroid is None or rv_centroid is None:
        print("Could not compute LV or RV centroids.")
        return None

    print("LV Centroid:", lv_centroid)
    print("RV Centroid:", rv_centroid)
    print("LV Long Axis:", lv_long_axis)

    return plot_4_chamber_view(label_data, affine, lv_centroid, rv_centroid, lv_long_axis, plane_size, spacing, out_of_plane_spacing, number_of_slices, plotOn)


def plot_3_chamber_view(label_data, affine, lv_centroid, aorta_centroid, lv_long_axis, plane_size=100, 
                        spacing=1.0, out_of_plane_spacing=8.0, number_of_slices=13, plotOn = True):
    """
    Three chamber view. Have v be a vector between the aorta centroid and the LV centroid, get the normal of V, and the center be the LV centroid
    """
    if lv_centroid is None or aorta_centroid is None:
        print("Invalid LV or Aorta centroid.")
        return None, None

    slices = []
    slice_affines = []
    v = lv_centroid - aorta_centroid
    v = v / np.linalg.norm(v)
    normal = np.cross(v, lv_long_axis)
    normal = normal / np.linalg.norm(normal)

    for slice_index in range(number_of_slices):
        # Adjust origin for each slice
        slice_origin = lv_centroid + (slice_index - number_of_slices // 2) * out_of_plane_spacing * normal
        xyz, slice_affine = grid_in_plane(slice_origin, normal, spacing, plane_size)
        slice_affine[:3, 3] = slice_origin
        slice_data = interpolate_image(xyz, label_data, affine)
        slices.append(slice_data)
        slice_affines.append(slice_affine)
    slices_3d = np.stack(slices, axis=0)

    if plotOn:
        middle_index = number_of_slices // 2
        plt.figure(figsize=(6, 6))
        plt.imshow(slices_3d[middle_index], cmap='gray', origin='lower')
        plt.title(f"Middle Slice (Slice {middle_index + 1})")
        plt.axis('off')
        plt.show()

    return slices_3d, slice_affines


def main_3chamber_view(seg_file, plane_size=100, spacing=1.0, out_of_plane_spacing=8.0, number_of_slices=13, plotOn=True):
    label_data, affine, _ = readFromNIFTI(seg_file)
    spatial_info = calculate_spatial_information(label_data, affine)

    lv_centroid = spatial_info["LV Centroid"]
    aorta_centroid = spatial_info["Aorta Centroid"]
    lv_long_axis = spatial_info["LV Long Axis"]

    if lv_centroid is None or aorta_centroid is None:
        print("Could not compute LV or Aorta centroids.")
        return None

    print("LV Centroid:", lv_centroid)
    print("Aorta Centroid:", aorta_centroid)


    return plot_3_chamber_view(label_data, affine, lv_centroid, aorta_centroid, lv_long_axis,
                               plane_size, spacing, out_of_plane_spacing, number_of_slices, plotOn)


def apply_misalignment(origin, normal, misalignment_level=0.0):

    if misalignment_level > 0:
        # The np.random.randn(3) generates a random vector with three components, each drawn from a normal distribution with a mean of 0 and a standard deviation of 1.
        translation = np.random.randn(3) * misalignment_level
        origin = origin + translation

        # Apply a small random rotation to the normal vector
        rotation = np.random.randn(3) * misalignment_level
        normal = normal + rotation
        normal = normal / np.linalg.norm(normal)

    return origin, normal

#  Trying to add a slider that controls the depth of the view being displayed using Matplotlib Slider
def plot_interactive_view(label_data, affine, lv_centroid, lv_long_axis, plane_size=100, spacing=1.0, num_slices=10):
    slice_data_list = []
    offsets = np.linspace(-plane_size / 2, plane_size / 2, num_slices)

    for offset in offsets:
        slice_origin = lv_centroid + offset * lv_long_axis 
        xyz, slice_affine = grid_in_plane(slice_origin, lv_long_axis, spacing, plane_size)
        slice_data = interpolate_image(xyz, label_data, affine)
        slice_data_list.append(slice_data)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.25)

    img = ax.imshow(slice_data_list[0], cmap='gray', origin='lower')
    ax.set_title("Interactive View")
    ax.axis('off')

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Position: [left, bottom, width, height]
    slider = Slider(ax_slider, 'Slice Depth', 0, num_slices - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        slice_idx = int(slider.val)  
        img.set_data(slice_data_list[slice_idx]) 
        fig.canvas.draw_idle()  

    slider.on_changed(update)  

    plt.show()


def display_views(sa_data=None, la_2CH_data=None, la_3CH_data=None, la_4CH_data=None):
    print("\nDisplaying all views...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    if sa_data is None:
       print("Failed to generate short-axis view.")
       axes[0, 0].axis('off')
    else:
        middle_index = sa_data.shape[0] // 2
        axes[0, 0].imshow(sa_data[middle_index], cmap='gray', origin='lower')
        axes[0, 0].set_title("Short-Axis View (Middle Slice)")
        axes[0, 0].axis('off')

   # 2-Chamber View
    if la_2CH_data is None:
       print("Failed to generate 2-chamber view.")
       axes[0, 1].axis('off') 
    else:
        middle_index = la_2CH_data.shape[0] // 2
        axes[0, 1].imshow(la_2CH_data[middle_index], cmap='gray', origin='lower')
        axes[0, 1].set_title("Two-Chamber View (Middle Slice)")
        axes[0, 1].axis('off')

   # 3-Chamber View
    if la_3CH_data is None:
       print("Failed to generate 3-chamber view.")
       axes[1, 0].axis('off')
    else:
        middle_index = la_3CH_data.shape[0] // 2
        axes[1, 0].imshow(la_3CH_data[middle_index], cmap='gray', origin='lower')
        axes[1, 0].set_title("Three-Chamber View (Middle Slice)")
        axes[1, 0].axis('off')

   # 4-Chamber View
    if la_4CH_data is None:
       print("Failed to generate 4-chamber view.")
       axes[1, 1].axis('off')
    else:
        middle_index = la_4CH_data.shape[0] // 2
        axes[1, 1].imshow(la_4CH_data[middle_index], cmap='gray', origin='lower')
        axes[1, 1].set_title("Four-Chamber View (Middle Slice)")
        axes[1, 1].axis('off')


    plt.tight_layout()
    plt.show()
