import os
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
import monai.transforms as mt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.measure import find_contours, label, regionprops
from scipy.spatial.distance import euclidean
from skimage.segmentation import find_boundaries
from skimage.transform import resize
from scipy.spatial import distance
from skimage.morphology import erosion, disk, dilation
from skimage.filters import gaussian



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
    v = spatial_info["Aorta Centroid"] - spatial_info["LV Centroid"]
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

    # Use plane vector if provided, otherwise create one
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
    ijk[:, 0] = np.clip(ijk[:, 0], 0, data.shape[0] - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, data.shape[1] - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, data.shape[2] - 1)
    interpolated_data = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    N = int(np.sqrt(len(xyz)))
    return interpolated_data.reshape(N,N)


def generate_scan_slices(centroid, normal, spacing, plane_size, ct_data, ct_affine, number_of_slices, out_of_plane_spacing, 
                         plotOn=False, misalignment = .1):
    
    # Generate data for all slices 
    # slice_affines = np.zeros((number_of_slices, 4, 4))
    slice_affines = []
    slice_datas = []
    slice_data_misaligned = []
    for slice_index in range(number_of_slices):
        # Find the origin of the slice by moving along normal vector from centroid
        slice_origin = centroid + (slice_index - number_of_slices // 2) * out_of_plane_spacing * normal
        slice_grid, slice_affine = grid_in_plane(slice_origin, normal, spacing, plane_size)
        slice_data = interpolate_image(slice_grid, ct_data, ct_affine)

        translation = np.random.uniform(-misalignment * spacing, misalignment * spacing, size=2)  # x, y shifts
        slice_grid[:, 0] += translation[0]  # Shift x-coordinates
        slice_grid[:, 1] += translation[1]  # Shift y-coordinates

        # slice_grid[:, 0] += np.random.uniform(-misalignment * spacing, misalignment * spacing, size=slice_grid.shape[0])
        # slice_grid[:, 1] += np.random.uniform(-misalignment * spacing, misalignment * spacing, size=slice_grid.shape[0])


        curr_misaligned = interpolate_image(slice_grid, ct_data, ct_affine)
        slice_data_misaligned.append(curr_misaligned)
        slice_affines.append(slice_affine)
        slice_datas.append(slice_data)

    scan_data = np.dstack(slice_datas)    # dstack stacks arrays along third dimension, depth
    slice_data_misaligned = np.dstack(slice_data_misaligned)
    if number_of_slices == 1:
        scan_affine = slice_affines[0]
    else:
        base_affine = slice_affines[0].copy()
        slice_direction = (slice_affines[1][:, 3] - slice_affines[0][:, 3])
        base_affine[:3, 2] = slice_direction[:3]
        base_affine[:3, 3] = slice_affines[0][:3, 3]
        scan_affine = base_affine
    if plotOn:
        plot_cardiac_view_slice(scan_data, number_of_slices, "2 Chamber View")
    return scan_data, scan_affine, slice_data_misaligned #original_coordinates, shifted_coordinates


def save_Nifti(data, affine, file_name = None):
    new_affine = affine.copy()
    new_affine[:2,:] = -new_affine[:2,:]
    affine = new_affine
    new_data = np.swapaxes(data, 0, 1)
    nifti_img = nib.Nifti1Image(new_data, affine)
    header = nifti_img.header
    header.set_qform(affine)
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

    centroid_voxel = np.mean(coords, axis=0)     # Calculate centroid in voxel coordinates
    centroid_real = nib.affines.apply_affine(affine, centroid_voxel)

    # label_name = {1: "LV", 3: "RV", 6: "Aorta"}.get(label_value, "Unknown")
    # fig, ax = plt.subplots(figsize=(6, 6))
    # middle_slice_index = int(centroid_voxel[2]) 
    # slice_data = label_data[:, :, middle_slice_index]
    # ax.imshow(slice_data.T, cmap='gray', origin='lower')
    # ax.scatter(centroid_voxel[0], centroid_voxel[1], color='red', s=100, label="Centroid")
    # ax.set_title(f"Centroid of {label_name} centroid in Middle Slice")
    # plt.show()

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


def apply_misalignment(grid_points, transformation_matrix, spacing, file_name, misalignment_factor=0.1):
    npoints = int(np.sqrt(len(grid_points))) 
    grid_points = grid_points.reshape((npoints, npoints, 3)) 
    

    misaligned_planes = []
    for z_slice in grid_points:
        misalignment = np.random.uniform(-misalignment_factor * spacing, 
                                          misalignment_factor * spacing, 
                                          size=z_slice.shape)
        z_slice_misaligned = z_slice.copy()
        z_slice_misaligned[:, 0:2] += misalignment[:, 0:2]
        misaligned_planes.append(z_slice_misaligned)
    grid_points_misaligned = np.dstack(misaligned_planes)

    grid_volume = grid_points_misaligned.reshape((npoints, npoints, 3))
    nifti_img = nib.Nifti1Image(grid_volume, transformation_matrix)
    nib.save(nifti_img, file_name)

    return grid_points_misaligned



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


def plot_slice_with_endpoints(axis, slice_data, title, endpoints_dict):
    axis.imshow(slice_data, cmap='gray', origin='lower')
    axis.set_title(title, fontsize = 18)
    axis.axis('off')

    for endpoint_type, (endpoints, color, label) in endpoints_dict.items():
        if endpoints is not None and all(endpoint is not None for endpoint in endpoints):
            label_added = False
            for endpoint in endpoints:
                if not label_added:
                    axis.plot(endpoint[1], endpoint[0], marker='o', color=color, markersize=3, label=label)
                    label_added = True
                else:
                    axis.plot(endpoint[1], endpoint[0], marker='o', color=color, markersize=3)
    axis.legend(loc='upper right')

def display_views(paths, Type, misalignment, sa_data=None, la_2CH_data=None, la_3CH_data=None, la_4CH_data=None, la_2CH_affine = None, la_3CH_affine = None, la_4CH_affine = None, ct_affine = None):
    print("\nDisplaying all views...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    endpoints_summary = {}

    if sa_data is not None:    # Short-Axis View
        middle_index = sa_data.shape[-1] // 2
        plot_slice_with_endpoints( axes[0, 0], sa_data[:, :, middle_index], "Short-Axis View (Middle Slice)", {})
    else:
        print("Failed to generate short-axis view.")
        axes[0, 0].axis('off')

    if la_2CH_data is not None:    # 2-Chamber View
        middle_index = la_2CH_data.shape[-1] // 2
        two_ch_slice = la_2CH_data[:, :, middle_index]
        MV_endpoints = find_MV_2CH(slice_data=two_ch_slice, lv_label=1, la_label=4)
        plot_slice_with_endpoints(axes[0, 1], two_ch_slice, "Two-Chamber View (Middle Slice)", 
                                  {'MV': (MV_endpoints, '#FFCB05', 'MV Endpoint')} )
        array_size = (*two_ch_slice.shape, 1)
        valve_array = np.zeros(array_size, dtype=np.uint8)
        create_valve_array(valve_array, MV_endpoints, 1)
        save_Nifti(valve_array, la_2CH_affine, paths[Type]['data'] + f'2CH_valve{"" if Type == "clean" else f"_{misalignment:.2f}"}.nii.gz')
        save_Nifti(valve_array, la_2CH_affine, paths[Type]['bvg'] + f'2CH_valve{"" if Type == "clean" else f"_{misalignment:.2f}"}.nii.gz')
        endpoints_summary['2CH'] = {'MV': MV_endpoints}
    else:
        print("Failed to generate 2-chamber view.")
        axes[0, 1].axis('off')

    if la_3CH_data is not None:    # 3-Chamber View
        middle_index = la_3CH_data.shape[-1] // 2
        three_ch_slice = la_3CH_data[:, :, middle_index]
        AV_endpoints, MV_endpoints = find_MV_AV_3CH(slice_data=three_ch_slice, lv_label=1, la_label=4, aorta_label=6)
        plot_slice_with_endpoints( axes[1, 0], three_ch_slice, "Three-Chamber View (Middle Slice)",
            { 'AV': (AV_endpoints, '#FFCB05', 'AV Endpoint'), 'MV': (MV_endpoints, '#007FFF', 'MV Endpoint')})
        array_size = (*three_ch_slice.shape, 1)
        valve_array = np.zeros(array_size, dtype=np.uint8)
        create_valve_array(valve_array, AV_endpoints, 3)
        create_valve_array(valve_array, MV_endpoints, 1)
        save_Nifti(valve_array, la_3CH_affine, paths[Type]['data'] + f'3CH_valve{"" if Type == "clean" else f"_{misalignment:.2f}"}.nii.gz')
        save_Nifti(valve_array, la_3CH_affine, paths[Type]['bvg'] + f'3CH_valve{"" if Type == "clean" else f"_{misalignment:.2f}"}.nii.gz')
        endpoints_summary['3CH'] = {'AV': AV_endpoints, 'MV': MV_endpoints}
    else:
        print("Failed to generate 3-chamber view.")
        axes[1, 0].axis('off')

    if la_4CH_data is not None:    # 4-Chamber View
        middle_index = la_4CH_data.shape[-1] // 2
        four_ch_slice = la_4CH_data[:, :, middle_index]
        TV_endpoints, MV_endpoints = find_MV_TV_4CH(slice_data=four_ch_slice, lv_label=1, rv_label=3, la_label=4, ra_label=5)
        plot_slice_with_endpoints( axes[1, 1], four_ch_slice, "Four-Chamber View (Middle Slice)", 
                                  {'TV': (TV_endpoints, '#FFCB05', 'TV Endpoint'), 'MV': (MV_endpoints, '#007FFF', 'MV Endpoint') })
        array_size = (*four_ch_slice.shape, 1)
        valve_array = np.zeros(array_size, dtype=np.uint8)
        create_valve_array(valve_array, TV_endpoints, 2)
        create_valve_array(valve_array, MV_endpoints, 1)
        save_Nifti(valve_array, la_4CH_affine, paths[Type]['data'] + f'4CH_valve{"" if Type == "clean" else f"_{misalignment:.2f}"}.nii.gz')
        save_Nifti(valve_array, la_4CH_affine, paths[Type]['bvg'] + f'4CH_valve{"" if Type == "clean" else f"_{misalignment:.2f}"}.nii.gz')
        endpoints_summary['4CH'] = {'TV': TV_endpoints, 'MV': MV_endpoints} 
    else:
        print("Failed to generate 4-chamber view.")
        axes[1, 1].axis('off')

    plt.tight_layout()
    # plt.show()
    return endpoints_summary

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

    # Check data dimensions
    if len(data.shape) == 2:
        # Make ij grid
        i = np.arange(data.shape[0])
        j = np.arange(data.shape[1])
        I, J = np.meshgrid(i, j)
        ij = np.column_stack((I.ravel(), J.ravel()))
        ijk = np.column_stack((ij, np.zeros_like(ij[:,0])))
    elif len(data.shape) == 3:
        # Make ijk grid
        i = np.arange(data.shape[0])
        j = np.arange(data.shape[1])
        k = np.arange(data.shape[2])
        I, J, K = np.meshgrid(i, j, k)
        ijk = np.column_stack((I.ravel(), J.ravel(), K.ravel()))

    xyz = nib.affines.apply_affine(affine, ijk)
    for i, label in enumerate(nlabels):
        show_point_cloud(xyz[data.ravel() == label], fig=fig, opacity=0.9, color=colors[i], size=5)

    if not background:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    return fig


def save_all_nifti_files(sa_data, sa_affine, la_2ch_data, la_2ch_affine,
                         la_3ch_data, la_3ch_affine, la_4ch_data, la_4ch_affine, sa_data_misaligned, 
                         la_2ch_data_misaligned, la_3ch_data_misaligned, la_4ch_data_misaligned, 
                         paths, misalignment):
    save_Nifti(sa_data, sa_affine, paths['clean']['data'] + 'SA.nii.gz')
    save_Nifti(la_2ch_data, la_2ch_affine, paths['clean']['data'] + '2CH.nii.gz')
    save_Nifti(la_3ch_data, la_3ch_affine, paths['clean']['data'] + '3CH.nii.gz')
    save_Nifti(la_4ch_data, la_4ch_affine, paths['clean']['data'] + '4CH.nii.gz')

    save_Nifti(sa_data_misaligned, sa_affine, paths['misaligned']['data'] + f'SA_misaligned_{misalignment:.2f}.nii.gz')
    save_Nifti(la_2ch_data_misaligned, la_2ch_affine, paths['misaligned']['data'] + f'2CH_misaligned_{misalignment:.2f}.nii.gz')
    save_Nifti(la_3ch_data_misaligned, la_3ch_affine, paths['misaligned']['data'] + f'3CH_misaligned_{misalignment:.2f}.nii.gz')
    save_Nifti(la_4ch_data_misaligned, la_4ch_affine, paths['misaligned']['data'] + f'4CH_misaligned_{misalignment:.2f}.nii.gz')

    save_Nifti(sa_data, sa_affine, paths['clean']['bvg'] + 'SA.nii.gz')
    save_Nifti(la_2ch_data, la_2ch_affine, paths['clean']['bvg'] + '2CH.nii.gz')
    save_Nifti(la_3ch_data, la_3ch_affine, paths['clean']['bvg'] + '3CH.nii.gz')
    save_Nifti(la_4ch_data, la_4ch_affine, paths['clean']['bvg'] + '4CH.nii.gz')

    save_Nifti(sa_data_misaligned, sa_affine, paths['misaligned']['bvg'] + f'SA_misaligned_{misalignment:.2f}.nii.gz')
    save_Nifti(la_2ch_data_misaligned, la_2ch_affine, paths['misaligned']['bvg'] + f'2CH_misaligned_{misalignment:.2f}.nii.gz')
    save_Nifti(la_3ch_data_misaligned, la_3ch_affine, paths['misaligned']['bvg'] + f'3CH_misaligned_{misalignment:.2f}.nii.gz')
    save_Nifti(la_4ch_data_misaligned, la_4ch_affine, paths['misaligned']['bvg'] + f'4CH_misaligned_{misalignment:.2f}.nii.gz')

def find_furthest_points(contour):
    max_distance = 0
    endpoints = (None, None)
    for i, p1 in enumerate(contour):
        for j, p2 in enumerate(contour):
            distance = euclidean(p1, p2)
            if distance > max_distance:
                max_distance = distance
                endpoints = (p1, p2)
    return endpoints

def find_overlap(slice_data, label1, label2):
    mask1 = slice_data == label1
    mask2 = slice_data == label2

    boundary1 = find_boundaries(mask1, mode='thick')
    boundary2 = find_boundaries(mask2, mode='thick')
    boundary1_resized = resize(boundary1, slice_data.shape, order=0, mode='constant', preserve_range=True)
    boundary2_resized = resize(boundary2, slice_data.shape, order=0, mode='constant', preserve_range=True)
    overlap_boundary = np.logical_and(boundary1_resized > 0.99, boundary2_resized > 0.99)
    # Smooth out the boundary so the endpoints are more accurate
    overlap_boundary = gaussian(overlap_boundary, sigma=.85)

    contours = find_contours(overlap_boundary, level=0.5)
    if not contours:
        raise ValueError("No contours found in the overlap boundary.")

    largest_contour = max(contours, key=len)

    endpoints = find_furthest_points(largest_contour)
    return endpoints


def create_valve_array(array, endpoints, endpoint_value):
    for point in endpoints:
        if point is not None:  
            y, x = int(point[0]), int(point[1]) 
            if 0 <= y < array.shape[0] and 0 <= x < array.shape[1]:  
                array[y, x] = endpoint_value

    return array

def find_MV_2CH(slice_data, lv_label=1, la_label=4):
    endpoints = find_overlap(slice_data, lv_label, la_label)
    return endpoints

def find_MV_AV_3CH(slice_data, lv_label=1, la_label=4, aorta_label=6):
    AV_endpoints = find_overlap(slice_data, lv_label, aorta_label)
    MV_endpoints = find_overlap(slice_data, lv_label, la_label)
    return AV_endpoints, MV_endpoints

def find_MV_TV_4CH(slice_data, lv_label=1, rv_label=3, la_label=4, ra_label=5):
    TV_endpoints = find_overlap(slice_data, rv_label, ra_label)
    MV_endpoints = find_overlap(slice_data, lv_label, la_label)
    return TV_endpoints, MV_endpoints

