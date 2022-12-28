from typing import Optional, Union
import numpy as np
import nibabel as nib
from scipy import ndimage
from PIL import Image
from os.path import basename
from src.util_plot import plot_localizer_from_axial, plot_localizer_crop_line, plot_localizer_from_cropped


def read_nifti_file(filepath: str) -> [np.ndarray, np.ndarray, nib.nifti1.Nifti1Header]:
    """
    Read and load the NIFTI file from the selected file path.

    Parameters
    ----------
    (Input)

        filepath : str
            - Full file path of the NIFTI file to be read.
    
    (Output)

        scan : np.ndarray
            - The image data array (volume) of the NIFTI file. 
            - A 3D or 4D (colour 3D image) array of image data.
        aff : np.ndarray
            - The affine array, gives the position of the image array data in a reference space.
        header : nib.nifti1.Nifti1Header
            - Image metadata describing the image.
    
    (Internal)

        orig_nii : nibabel.nifti1.Nifti1Image (Nibabel images)
            - The Nibabel image extracted from the selected path (filepath). 
            - Contain image data array, affine array, and image metadata.
    """
    # Read file
    orig_nii = nib.load(filepath)
    # Get raw data
    scan = orig_nii.get_fdata()
    aff = orig_nii.affine
    header = orig_nii.header
    return scan, aff, header

def save_midplane(image_3d: np.ndarray):
    """
    Save the mid-plane of the 3d image to png.

    Parameters
    ----------
    (Input)

        image_3d : np.ndarray
            - The image data array (volume) of the NIFTI/DICOMs.
            - A 3D or 4D (colour 3D image) array of image data.
    
    (Output)

        * No output parameters.
        * PNG images generated and saved in "current_pngs/" folder.
    
    (Internal)

        image_shape : tuple
            - The shape of the 3D image.
        axi_aspect & sag_aspect & cor_aspect : float
            - The aspect ratio of the axes scaling for the axial, sagittal, and coronal plane.
        plane_axi & plane_sag & plane_cor : Pillow image
            - The image from the array for the mid-axial, mid-sagittal, and mid-coronal plane.
    """
    image_shape = image_3d.shape
    axi_aspect = image_shape[0]/image_shape[1]
    sag_aspect = image_shape[2]/image_shape[0]
    cor_aspect = image_shape[1]/image_shape[2]
    image_3d = ((image_3d + abs(np.min(image_3d[:, :, :])))/(np.max(image_3d[:, :, :])+ abs(np.min(image_3d[:, :, :]))))*255
    plane_axi = Image.fromarray(image_3d[:, :, image_shape[2]//2]).convert("L")
    plane_axi.save("current_pngs/plane_axi.png")
    plane_sag = Image.fromarray(image_3d[:, image_shape[1]//2, :]).convert("L")
    plane_sag.save("current_pngs/plane_sag.png")
    plane_cor = Image.fromarray(image_3d[image_shape[0]//2, :, :].T).convert("L")
    plane_cor.save("current_pngs/plane_cor.png")
    print("Files saved.")
    return

def save_crop_3d(newname: str, targetpath: str, image_3d_cropped: np.ndarray, aff_3d: np.ndarray, header_3d: nib.nifti1.Nifti1Header):
    """
    Save the cropped image data array to a NIFTI file (but keeping the affine array and metadata same as the original).

    Parameters
    ----------
    (Input)

        newname : str
            - The name of the new NIFTI file.
        targetpath : str
            - Full file path of the new NIFTI file to save.
        image_3d_cropped : np.ndarray
            - The cropped image data array. 3D or 4D (colour 3D image).
        aff_3d : np.ndarray
            - The original affine array.
        header_3d : nib.nifti1.Nifti1Header
            - Image metadata describing the image.
    
    (Output)

        * No output parameters.
        * NIFTI file generated and saved as "targetpath/newname".
    
    (Internal)

        cropped_nii : nibabel.nifti1.Nifti1Image (Nibabel images)
            - The Nibabel image constructed from the cropped image data array. 
    """
    cropped_nii = nib.Nifti1Image(image_3d_cropped, aff_3d, header_3d)
    nib.save(cropped_nii, targetpath + newname)
    # print("Cropped imaged saved in: ", targetpath)
    # print("Cropped imaged saved as: ", "cropped_" + newname)
    return

def crop_3dimage_n_plot(filepath: str, crop_array, localizer_array, manual_set_localizer) -> [np.ndarray, np.ndarray, nib.nifti1.Nifti1Header]:
    """
    Crop a 3D image according to the defined crop array and plot the crop.

    Parameters
    ----------
    (Input)

        filepath : str
            - Full file path of the NIFTI file to be read and crop.
        crop_array : list
            - User defined pixel locations of the cropping boundary (adjust to the ROI).
        localizer_array : list
            - User defined position of localizer (localizer slice to show).
        manual_set_localizer : bool
            - User defined setting for localizer. 
            - True = use localizer_array to select localizer position; 
            - False = localizer position set to the mid-plane for the original 3D image.
    
    (Output)

        image_3d_cropped : np.ndarray
            - The cropped image data array. 3D or 4D (colour 3D image).
        aff_3d : np.ndarray
            - The original affine array.
        header_3d : nib.nifti1.Nifti1Header
            - Image metadata describing the image.
    
    (Internal)

        orig_image_3d : np.ndarray
            - The original image data array.
        nii_sample_to_plot : str
            - File name of the original NIFTI.
        local_axi_slice & local_cor_slice & local_sag_slice : int
            - Mid-plane location of the cropped image.
            - Values will update the localizer_array if manual_set_localizer set as False.
    """
    image_3d, aff_3d, header_3d = read_nifti_file(filepath)
    orig_image_3d = image_3d
    nii_sample_to_plot = basename(filepath)
    if manual_set_localizer == False:
        local_axi_slice = ((crop_array[2][1]-crop_array[2][0])//2) + crop_array[2][0]
        local_cor_slice = ((crop_array[0][1]-crop_array[0][0])//2) + crop_array[0][0]
        local_sag_slice = ((crop_array[1][1]-crop_array[1][0])//2) + crop_array[1][0]
        localizer_array = [local_axi_slice, local_cor_slice, local_sag_slice]
    print("Plotting: ",nii_sample_to_plot)
    print("Shape (original): ", image_3d.shape)
    #print("Showing localizer slice of: axial=", localizer_array[0], "; coronal=", localizer_array[1], "sagittal=", localizer_array[2])
    plot_localizer_crop_line(image_3d, crop_array, localizer_array)
    print("--------------------------------- Cropped (volume inside the green box) ---------------------------------")
    image_3d_cropped = image_3d[crop_array[1][0]:crop_array[1][1],crop_array[0][0]:crop_array[0][1],crop_array[2][0]:crop_array[2][1]]
    print("Shape (cropped): ", image_3d_cropped.shape)
    print("Showing localizer slice of: Mid-plane")
    plot_localizer_from_cropped(image_3d_cropped, orig_image_3d)
    return image_3d_cropped, aff_3d, header_3d

def crop_3dimage(filepath: str, crop_array) -> [np.ndarray, np.ndarray, nib.nifti1.Nifti1Header]:
    """
    Crop a 3D image according to the defined crop array.

    Parameters
    ----------
    (Input)

        filepath : str
            - Full file path of the NIFTI file to be read and crop.
        crop_array : list
            - User defined pixel locations of the cropping boundary (adjust to the ROI).
    
    (Output)

        image_3d_cropped : np.ndarray
            - The cropped image data array. 3D or 4D (colour 3D image).
        aff_3d : np.ndarray
            - The original affine array.
        header_3d : nib.nifti1.Nifti1Header
            - Image metadata describing the image.
    
    (Internal)

        orig_image_3d : np.ndarray
            - The original image data array.
    """
    image_3d, aff_3d, header_3d = read_nifti_file(filepath)
    orig_image_3d = image_3d
    image_3d_cropped = image_3d[crop_array[1][0]:crop_array[1][1],crop_array[0][0]:crop_array[0][1],crop_array[2][0]:crop_array[2][1]]
    return image_3d_cropped, aff_3d, header_3d

def normalise(volume: np.ndarray, max_n: int, min_n: int) -> np.ndarray:
    """
    Normalise the volume (max-min normalisation).

    Parameters
    ----------
    (Input)

        volume : np.ndarray
            - (image) data array.
        
    (Output)

        volume_n : np.ndarray
            - Normalised (image) data array.
    """
    volume[volume < min_n] = min_n
    volume[volume > max_n] = max_n
    volume = (volume - min_n) / (max_n - min_n)
    volume_n = volume.astype("float32")
    return volume_n

def resize_volume(img,target_volume):
    """
    Resize across z-axis (3D)
    """
    # Set the desired depth
    desired_depth = target_volume[2]
    desired_width = target_volume[0]
    desired_height = target_volume[1]
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    return img

def process_scan(path,target_volume):
    """
    Read and resize volume
    """
    # Read scan
    volume, aff, header = read_nifti_file(path)
    # Normalize
    volume = normalise(volume,400,-200)
    # Resize width, height and depth
    volume = resize_volume(volume,target_volume)
    return volume, aff, header

def resize_imagesize(img,target_imagesize):
    """
    Resize the image (2D/2.5D)
    """
    # Set the desired depth
    desired_width = target_imagesize[0]
    desired_height = target_imagesize[1]
    # Get current image size
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute ratio factor
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def get_volume_from_nifti(path):
    """
    Get the NIFTI pixel data from path and normalise the pixel data.
    """ 
    # Read scan
    volume, aff, header = read_nifti_file(path)
    # Normalize
    volume = normalise(volume,400,-200)
    return volume