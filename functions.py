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


def get_view_normal_origin(spatial_info):
    # SA
    sa_normal = spatial_info["LV Long Axis"]
    sa_origin = spatial_info["LV Centroid"]
    sa_normal_origin = sa_normal, sa_origin

    # 2CH
    v = spatial_info["RV Centroid"] - spatial_info["LV Centroid"]
    v = v / np.linalg.norm(v)
    la_2ch_normal = v
    la_2ch_origin = spatial_info["LV Centroid"]
    la_2ch_normal_origin = la_2ch_normal, la_2ch_origin

    # 3CH
    v = spatial_info["LV Centroid"] - spatial_info["Aorta Centroid"]
    v = v / np.linalg.norm(v)
    normal = np.cross(v, spatial_info["LV Long Axis"])
    normal = normal / np.linalg.norm(normal)
    la_3ch_normal = normal
    la_3ch_origin = spatial_info["LV Centroid"]
    la_3ch_normal_origin = la_3ch_normal, la_3ch_origin

    # 4CH
    v = spatial_info["RV Centroid"] - spatial_info["LV Centroid"]
    v = v / np.linalg.norm(v)
    x = np.cross(spatial_info["LV Long Axis"], v)
    x = x / np.linalg.norm(x)
    la_4ch_normal = x
    la_4ch_origin = spatial_info["LV Centroid"]
    la_4ch_normal_origin = la_4ch_normal, la_4ch_origin

    return sa_normal_origin, la_2ch_normal_origin, la_3ch_normal_origin, la_4ch_normal_origin


def grid_in_plane(origin, normal, spacing, plane_size, plane_vector=None):
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

    # Use plane vector if provided, otherwise create one
    if plane_vector is not None:
        u = plane_vector / np.linalg.norm(plane_vector)
    else:
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


def interpolate_image(xyz, data, data_affine):
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
    ijk = nib.affines.apply_affine(np.linalg.inv(data_affine), xyz)
    # round everything to integers for floating point number, indexing array doesnt take floating point
    ijk = np.round(ijk).astype(int)

    # Check points outside box
    ijk[:, 0] = np.clip(ijk[:, 0], 0, data.shape[0] - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, data.shape[1] - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, data.shape[2] - 1)
    # effectively create three arrays of cords which is x,y,z of all points we want to sample
    interpolated_data = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    # print("Transformed ijk indices:", ijk[:10])

    N = int(np.sqrt(len(xyz)))
    return interpolated_data.reshape(N,N)


def generate_scan_slices(centroid, normal, spacing, plane_size, ct_data, ct_affine, number_of_slices, out_of_plane_spacing, plotOn=False, plane_vector=None):
    
    # Generate data for all slices 
    # slice_affines = np.zeros((number_of_slices, 4, 4))
    slice_affines = []
    slice_datas = []
    for slice_index in range(number_of_slices):
        # Find the origin of the slice by moving along normal vector from centroid
        slice_origin = centroid + (slice_index - number_of_slices // 2) * out_of_plane_spacing * normal

        # Create grid an interpolate data
        slice_grid, slice_affine = grid_in_plane(slice_origin, normal, spacing, plane_size, plane_vector)
        slice_data = interpolate_image(slice_grid, ct_data, ct_affine)

        # Save in arrays
        slice_affines.append(slice_affine)
        slice_datas.append(slice_data)

    scan_data = np.dstack(slice_datas)    # (plane_size, plane_size, number_of_slices)
    if number_of_slices == 1:
        scan_affine = slice_affines[0]
    else:
        base_affine = slice_affines[0].copy()
        slice_direction = (slice_affines[1][:, 3] - slice_affines[0][:, 3]) / out_of_plane_spacing
        base_affine[:3, 2] = slice_direction[:3]
        base_affine[:3, 3] = slice_affines[0][:3, 3]
        scan_affine = base_affine
    if plotOn:
        plot_cardiac_view_slice(scan_data, number_of_slices, "2 Chamber View")
    return scan_data, scan_affine


def save_Nifti(data, affine, spacing, out_of_plane_spacing, file_name):
    nifti_img = nib.Nifti1Image(data, affine)
    header = nifti_img.header
    header.set_zooms((spacing, spacing, out_of_plane_spacing))
    header.set_sform(affine)
    header.set_qform(affine, code=1)
    header['bitpix'] = 32  
    header['scl_slope'] = 1.0 
    header['scl_inter'] = 0.0
    # nifti_img.header = header
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

def plot_cardiac_view_slice(slices_3d, number_of_slices, title=None):
    middle_index = number_of_slices // 2
    plt.figure(figsize=(6, 6))
    plt.imshow(slices_3d[:,:,middle_index], cmap='gray', origin='lower')
    
    if title:
        plt.title(title)
    else:
        plt.title(f"Middle Slice (Slice {middle_index + 1})")
    
    plt.axis('off')
    plt.show()

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
    # The first principal component corresponds to the longest
    #  dimension of the LV shape.
    # This is why you can use it to estimate the LV long axis.
    long_axis_voxel = pca.components_[0]

    #     explained_variance = pca.explained_variance_ratio_
    # print("Explained Variance Ratio:", explained_variance)
    #  convert the LV long axis vector from voxel space to real-world coordinates using the affine transformation matrix.

    # affine[:3, :3] extracts the top-left 3x3 submatrix of the affine matrix.
# This submatrix represents the rotation and scaling but not the translation.
    long_axis_real = affine[:3, :3] @ long_axis_voxel
    long_axis_real = long_axis_real / np.linalg.norm(long_axis_real)
    return long_axis_real



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


"""  PLOTTING FUNCTIONS  """

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
        middle_index = sa_data.shape[-1] // 2
        axes[0, 0].imshow(sa_data[:,:,middle_index], cmap='gray', origin='lower')
        axes[0, 0].set_title("Short-Axis View (Middle Slice)")
        axes[0, 0].axis('off')

   # 2-Chamber View
    if la_2CH_data is None:
       print("Failed to generate 2-chamber view.")
       axes[0, 1].axis('off') 
    else:
        middle_index = la_2CH_data.shape[-1] // 2
        axes[0, 1].imshow(la_2CH_data[:,:,middle_index], cmap='gray', origin='lower')
        axes[0, 1].set_title("Two-Chamber View (Middle Slice)")
        axes[0, 1].axis('off')

   # 3-Chamber View
    if la_3CH_data is None:
       print("Failed to generate 3-chamber view.")
       axes[1, 0].axis('off')
    else:
        middle_index = la_3CH_data.shape[-1] // 2
        axes[1, 0].imshow(la_3CH_data[:,:,middle_index], cmap='gray', origin='lower')
        axes[1, 0].set_title("Three-Chamber View (Middle Slice)")
        axes[1, 0].axis('off')

   # 4-Chamber View
    if la_4CH_data is None:
       print("Failed to generate 4-chamber view.")
       axes[1, 1].axis('off')
    else:
        middle_index = la_4CH_data.shape[-1] // 2
        axes[1, 1].imshow(la_4CH_data[:,:,middle_index], cmap='gray', origin='lower')
        axes[1, 1].set_title("Four-Chamber View (Middle Slice)")
        axes[1, 1].axis('off')
    plt.tight_layout()
    plt.show()


import plotly.graph_objects as go
import plotly.io as pio
def show_point_cloud(points, fig=None, color=None, size=10, cmap='Viridis',
                     opacity=1, marker_symbol='circle', label=None, showscale=False,
                     cmin = None, cmax = None):
    if fig is None:
        fig = go.Figure()
    if len(points.shape) == 1:
        points = points[None]
    fig.add_scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers',
                      marker=dict(
                                    color=color,
                                    size=size,
                                    colorscale=cmap,
                                    opacity=opacity,
                                    symbol=marker_symbol,
                                    showscale=showscale,
                                    cmin = cmin,
                                    cmax = cmax
                                ),
                      name = label
        )
    return fig

    
def show_segmentations(data, affine, fig=None, background=False):
    nlabels = np.unique(data)
    nlabels = nlabels[nlabels != 0]
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'cyan',
              4: 'grey', 5: 'yellow', 6: 'purple', 7: 'magenta',
              8: 'aqua', 9: 'maroon', 10: 'olive', 11: 'lime'}
    
    if fig is None:
        fig = go.Figure()
    for i, label in enumerate(nlabels):
        mask = data == label
        ijk = np.vstack(np.where(mask)).T
        if ijk.ndim != 2 or ijk.shape[1] != 3: 
            continue
        xyz = nib.affines.apply_affine(affine, ijk)
        if xyz.shape[0] == 0:  # Check transformed points
            continue
        show_point_cloud(xyz, fig=fig, opacity=0.9, color=colors[i], size=5)
    if not background:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    return fig

