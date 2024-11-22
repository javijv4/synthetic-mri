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


def calculate_spatial_information(label_data, affine):
    # TODO: the labels should be passed as arguments to the function
    LV_LABEL = 1
    RV_LABEL = 2
    AORTA_LABEL = 3

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
    
    # if normal is close to the x axis [1,0,0], create an orthogonal vector
    # else then normal is closer to the y axis, create an orthogonal vector with the y axis 
    if not np.allclose(normal, [1, 0, 0]):
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    
    # make a unit vector (normalize it)
    u = u / np.linalg.norm(u)

    # make v which is orthogonal to u and v
    v = np.cross(normal, u)
    
    # Create grid points within the plane
    half_size = plane_size / 2

    #linespace generates sequence of of evenly spaced numbers of range  
    # creates planesize amount of evenly spaced values from -half_size to half_size
    # creates a 1d array of evenly spaced points
    npoints = int(np.ceil(plane_size / spacing))  # Calculate number of points to cover plane_size
    lin_space = np.linspace(-half_size, half_size, npoints)

    # cretea grid in plane using u and v as axes? 
    # .meshgrid produces coordinate matrices from coordinate vectors
    # takes 1d array of evenly spaced points and combines into 2d grid 
    grid_x, grid_y = np.meshgrid(lin_space, lin_space)
    # grid_y = np.meshgrid(lin_space, lin_space)
    
    # Compute 3D coordinates for each point on the 
    #  creates array of shape (N, N, 3), where each row is a 3D point in the plane centered on origin
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

def plot_short_axis(label_data, affine, lv_centroid, lv_long_axis, plane_size=100, spacing=1.0, plotOn = True):
    if lv_centroid is None or lv_long_axis is None:
        print("Invalid LV centroid or long axis.")
        return None
    
    xyz = grid_in_plane(lv_centroid, lv_long_axis, spacing, plane_size)
    
    slice_data = interpolate_image(xyz, label_data, affine)

    #  transformed_coords = np.dot(affine[:3, :3], xyz.T).T + affine[:3, 3]
    # return slice_data, transformed_coords
    transformed_coords = [1,1,1]
    if plotOn: 
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_data, cmap='gray', origin='lower')
        plt.title("Short-Axis View")
        plt.axis('off')
        plt.show()

    return slice_data, transformed_coords

def plot_2_chamber_view(label_data, affine, lv_centroid, rv_centroid, plane_size=100, spacing=1.0, plotOn = True):

    if lv_centroid is None or rv_centroid is None:
        print("Invalid LV or RV centroid.")
        return None
    
    v = rv_centroid - lv_centroid
    v = v / np.linalg.norm(v) 

    origin = lv_centroid
    normal = v

    xyz = grid_in_plane(origin, normal, spacing, plane_size)
    
    slice_data = interpolate_image(xyz, label_data, affine)
    
    if plotOn:
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_data, cmap='gray', origin='lower')
        plt.title("2-Chamber View")
        plt.axis('off')
        plt.show()

    #  from chatgpt? returning affined data
    transformed_coords = [1,1,1]
    return slice_data, transformed_coords



def main_2chamber_view(seg_file, plane_size=100, spacing=1.0, plotOn = True):
    label_data, affine, _ = readFromNIFTI(seg_file)
    
    spatial_info = calculate_spatial_information(label_data, affine)
    
    lv_centroid = spatial_info["LV Centroid"]
    rv_centroid = spatial_info["RV Centroid"]
    
    if lv_centroid is None or rv_centroid is None:
        print("Could not compute LV or RV centroids.")
        return None

    print("LV Centroid:", lv_centroid)
    print("RV Centroid:", rv_centroid)

    return plot_2_chamber_view(label_data, affine, lv_centroid, rv_centroid, plane_size, spacing, plotOn)

    


def plot_4_chamber_view(label_data, affine, lv_centroid, rv_centroid, lv_long_axis, plane_size=100, spacing=1.0, plotOn = True):
    if lv_centroid is None or rv_centroid is None:
        print("Invalid LV or RV centroid.")
        return None, None
    
    v = rv_centroid - lv_centroid
    v = v / np.linalg.norm(v) 

    # Choose a reference vector (like [0, 0, 1]) to compute the cross-product
    # ref_vector = np.array([0, 0, 1]) if not np.allclose(v, [0, 0, 1]) else np.array([0, 1, 0])
    # x = np.cross(v, ref_vector)

    x = np.cross(lv_long_axis, v)
    x = x / np.linalg.norm(x)

    origin = lv_centroid
    normal = x

    xyz = grid_in_plane(origin, normal, spacing, plane_size)
    
    slice_data = interpolate_image(xyz, label_data, affine)
    
    if plotOn: 
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_data, cmap='gray', origin='lower')
        plt.title("4-Chamber View")
        plt.axis('off')
        plt.show()

    transformed_coords = [1,1,1]
    return slice_data, transformed_coords

def main_4chamber_view(seg_file, plane_size=100, spacing=1.0, plotOn = True):
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


    return plot_4_chamber_view(label_data, affine, lv_centroid, rv_centroid, lv_long_axis, plane_size, spacing, plotOn)


def plot_3_chamber_view(label_data, affine, lv_centroid, aorta_centroid, lv_long_axis, plane_size=100, spacing=1.0, plotOn = True):
    """
    Three chamber view. Have v be a vector between the aorta centroid and the LV centroid, get the normal of V, and the center be the LV centroid
    """
    if lv_centroid is None or aorta_centroid is None:
        print("Invalid LV or Aorta centroid.")
        return None, None
     
    v = lv_centroid - aorta_centroid
    v = v / np.linalg.norm(v) 

    # Use a reference vector (like [0, 0, 1]) for cross-product calculation
    # https://liu.diva-portal.org/smash/get/diva2:1709829/FULLTEXT01.pdf
    ref_vector = np.array([0, 0, 1]) if not np.allclose(v, [0, 0, 1]) else np.array([0, 1, 0])
    normal = np.cross(v, ref_vector)
    normal = normal / np.linalg.norm(normal)

    origin = lv_centroid

    xyz = grid_in_plane(origin, normal, spacing, plane_size)
    

    normal = np.cross(v, lv_long_axis)
    normal = normal / np.linalg.norm(normal)

    origin = lv_centroid
    slice_data = interpolate_image(xyz, label_data, affine)
    
    if plotOn: 
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_data, cmap='gray', origin='lower')
        plt.title("3-Chamber View")
        plt.axis('off')
        plt.show()

    transformed_coords = [1,1,1]
    return slice_data, transformed_coords


def main_3chamber_view(seg_file, plane_size=100, spacing=1.0, plotOn = True):
    label_data, affine, _ = readFromNIFTI(seg_file)
    
    spatial_info = calculate_spatial_information(label_data, affine)
    
    lv_centroid = spatial_info["LV Centroid"]
    aorta_centroid = spatial_info["Aorta Centroid"]
    lv_long_axis = spatial_info["LV Long Axis"]

    if lv_centroid is None or aorta_centroid is None:
        print("Could not compute LV or Aorta centroids.")
        return None

    # sourceFile = open('demo.txt', 'w')
    # print("LV Centroid:", lv_centroid, file = sourceFile)
    print("LV Centroid:", lv_centroid)
    print("Aorta Centroid:", aorta_centroid)
    

    return plot_3_chamber_view(label_data, affine, lv_centroid, aorta_centroid, lv_long_axis, plane_size, spacing, plotOn)


    
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
        slice_origin = lv_centroid + offset * lv_long_axis  # Adjust origin for each slice
        xyz = grid_in_plane(slice_origin, lv_long_axis, spacing, plane_size)
        slice_data = interpolate_image(xyz, label_data, affine)
        slice_data_list.append(slice_data)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.25)

    # Display the first slice initially
    img = ax.imshow(slice_data_list[0], cmap='gray', origin='lower')
    ax.set_title("Interactive View")
    ax.axis('off')

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Position: [left, bottom, width, height]
    slider = Slider(ax_slider, 'Slice Depth', 0, num_slices - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        slice_idx = int(slider.val)  # Get the current slider value
        img.set_data(slice_data_list[slice_idx])  # Update the image data
        fig.canvas.draw_idle()  # Redraw the canvas

    slider.on_changed(update)  # Connect the slider to the update function

    plt.show()